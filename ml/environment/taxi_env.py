import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
from ml.graph.builder import RoadNetworkGraph


class TaxiRouteEnv(gym.Env):
    """
    Gymnasium-compatible Reinforcement Learning Environment for Taxi Route Optimization.
    Operates on networkx DiGraph with dynamic traffic and multi-objective rewards.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, graph_builder: RoadNetworkGraph, max_neighbors: int = 8,
                 max_steps: int = 60, reward_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.rng = graph_builder
        self.graph = graph_builder.graph
        if self.graph is None or self.graph.number_of_nodes() == 0:
            self.rng.build_synthetic_grid()
            self.graph = self.rng.graph

        self.max_neighbors = max_neighbors
        self.max_steps = max_steps
        self.node_list = list(self.graph.nodes())
        self.num_nodes = len(self.node_list)
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        
        # Default reward weights
        self.rw = reward_weights or {
            'goal': 100.0,
            'step_cost_multiplier': 1.0,
            'invalid_action': -15.0,
            'loop_penalty': -5.0,
            'max_steps_penalty': -20.0
        }

        # Observation Space (8 continuous normalized features)
        # [current_node_norm, dest_node_norm, distance_to_dest_norm, traffic_factor_norm,
        #  time_of_day_norm, dir_x, dir_y, path_length_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.5, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Action Space: index of outgoing neighbor (0 to max_neighbors-1)
        self.action_space = spaces.Discrete(self.max_neighbors)

        # Internal state
        self.current_node: int = 0
        self.destination_node: int = 0
        self.visited_nodes: List[int] = []
        self.step_count: int = 0
        self.total_distance_km: float = 0.0
        self.total_time_min: float = 0.0
        self.total_cost: float = 0.0
        self.time_of_day: float = 12.0  # 12:00 PM default

    def _get_neighbors(self, node: int) -> List[int]:
        """Return sorted outgoing neighbors of node."""
        return sorted(list(self.graph.successors(node)))

    def get_action_mask(self, node: Optional[int] = None) -> np.ndarray:
        """Return boolean mask of valid outgoing neighbor actions."""
        target = node if node is not None else self.current_node
        nbrs = self._get_neighbors(target)
        mask = np.zeros(self.max_neighbors, dtype=bool)
        mask[:min(len(nbrs), self.max_neighbors)] = True
        return mask

    def _get_obs(self) -> np.ndarray:
        """Construct normalized observation array."""
        curr_idx = self.node_to_idx.get(self.current_node, 0) / max(self.num_nodes - 1, 1)
        dest_idx = self.node_to_idx.get(self.destination_node, 0) / max(self.num_nodes - 1, 1)
        
        c_lat, c_lng = self.graph.nodes[self.current_node]['lat'], self.graph.nodes[self.current_node]['lng']
        d_lat, d_lng = self.graph.nodes[self.destination_node]['lat'], self.graph.nodes[self.destination_node]['lng']
        
        rem_dist = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)
        rem_dist_norm = np.clip(rem_dist / 20.0, 0.0, 1.0)
        
        # Compute direction vector
        dx = d_lng - c_lng
        dy = d_lat - c_lat
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        dir_x = float(dx / norm)
        dir_y = float(dy / norm)
        
        # Traffic factor average near current node
        nbrs = self._get_neighbors(self.current_node)
        if nbrs:
            tf = np.mean([self.graph[self.current_node][nbr].get('traffic_factor', 1.0) for nbr in nbrs])
        else:
            tf = 1.0
            
        tod_norm = self.time_of_day / 24.0
        path_len_norm = np.clip(self.step_count / float(self.max_steps), 0.0, 1.0)
        
        return np.array([curr_idx, dest_idx, rem_dist_norm, tf, tod_norm, dir_x, dir_y, path_len_norm], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if options and 'source' in options and 'destination' in options:
            self.current_node = options['source']
            self.destination_node = options['destination']
        else:
            # Pick random distinct nodes with path existing between them
            valid_pair = False
            attempts = 0
            while not valid_pair and attempts < 100:
                s, d = np.random.choice(self.node_list, size=2, replace=False)
                if nx.has_path(self.graph, s, d) and s != d:
                    self.current_node = s
                    self.destination_node = d
                    valid_pair = True
                attempts += 1
                
            if not valid_pair:
                self.current_node = self.node_list[0]
                self.destination_node = self.node_list[-1]

        self.visited_nodes = [self.current_node]
        self.step_count = 0
        self.total_distance_km = 0.0
        self.total_time_min = 0.0
        self.total_cost = 0.0
        self.time_of_day = options.get('time_of_day', 12.0) if options else 12.0

        info = {
            'source_node': self.current_node,
            'destination_node': self.destination_node,
            'action_mask': self.get_action_mask()
        }
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        nbrs = self._get_neighbors(self.current_node)
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check action validity
        if action < 0 or action >= len(nbrs):
            # Invalid action
            reward += self.rw['invalid_action']
            next_node = self.current_node
        else:
            next_node = nbrs[action]
            edge_data = self.graph[self.current_node][next_node]
            step_cost = edge_data.get('weight', 1.0)
            dist = edge_data.get('distance', 0.1)
            travel_time = edge_data.get('travel_time', 0.5)
            
            self.total_distance_km += dist
            self.total_time_min += travel_time
            self.total_cost += step_cost
            
            # Step penalty based on cost
            reward -= step_cost * self.rw['step_cost_multiplier']
            
            # Loop penalty
            if next_node in self.visited_nodes:
                reward += self.rw['loop_penalty']
                
            self.current_node = next_node
            self.visited_nodes.append(next_node)

        # Check goal reached
        if self.current_node == self.destination_node:
            reward += self.rw['goal']
            terminated = True
            
        # Check max steps limit
        if self.step_count >= self.max_steps and not terminated:
            reward += self.rw['max_steps_penalty']
            truncated = True

        info = {
            'current_node': self.current_node,
            'destination_node': self.destination_node,
            'step_count': self.step_count,
            'total_distance_km': self.total_distance_km,
            'total_time_min': self.total_time_min,
            'total_cost': self.total_cost,
            'visited_nodes': self.visited_nodes,
            'action_mask': self.get_action_mask()
        }

        return self._get_obs(), float(reward), terminated, truncated, info

import math
import random
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from collections import Counter
from typing import Dict, Any, Tuple, Optional, List
from ml.graph.builder import RoadNetworkGraph


class TaxiRouteEnv(gym.Env):
    """
    Gymnasium-compatible Reinforcement Learning Environment for Taxi Route Optimization.
    Operates on networkx DiGraph with geographic directional action mapping,
    dynamic traffic penalties, action masking with recency-window and trap-avoidance cycle prevention,
    and multi-objective rewards.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, graph_builder: RoadNetworkGraph, max_neighbors: int = 8,
                 max_steps: int = 150, reward_weights: Optional[Dict[str, float]] = None,
                 enable_cycle_aware_masking: bool = False,
                 max_node_visits: int = 2,
                 revisit_penalty_lambda: float = 3.0,
                 enable_pbrs: bool = False,
                 pbrs_gamma: float = 0.995,
                 pbrs_scale: float = 8.0,
                 alt_heuristic: Optional[Any] = None):
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

        # Geographic center and scaling factors for Ludhiana bounding box
        self.center_lat = 30.9000
        self.center_lng = 75.8500
        self.lat_scale = 0.10
        self.lng_scale = 0.16

        # Default reward weights
        self.rw = reward_weights or {
            'goal': 120.0,
            'step_cost_multiplier': 0.5,
            'invalid_action': -15.0,
            'loop_penalty': -4.0,
            'max_steps_penalty': -20.0,
            'progress_multiplier': 8.0
        }

        # Observation Space (12 continuous normalized scale-invariant features)
        # 1. curr_lat_norm [-1, 1]
        # 2. curr_lng_norm [-1, 1]
        # 3. dest_lat_norm [-1, 1]
        # 4. dest_lng_norm [-1, 1]
        # 5. rem_dist_norm [0, 1]
        # 6. dir_x (sin bearing) [-1, 1]
        # 7. dir_y (cos bearing) [-1, 1]
        # 8. curr_speed_norm [0.2, 1.0] (road hierarchy)
        # 9. bridge_flag [0, 1] (river/railway crossing indicator)
        # 10. revisit_count_norm [0, 1] (loop/trap indicator)
        # 11. progress_ratio [-1, 1] (macro progress towards destination)
        # 12. traffic_factor_norm [0.5, 3.0]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.2, 0.0, 0.0, -1.0, 0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0], dtype=np.float32),
            dtype=np.float32
        )

        # Action Space: 8 compass sectors (0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
        self.action_space = spaces.Discrete(self.max_neighbors)

        # Internal state
        self.current_node: int = 0
        self.destination_node: int = 0
        self.visited_nodes: List[int] = []
        self.visited_counts: Dict[int, int] = {}
        self.initial_direct_dist: float = 1.0
        self.step_count: int = 0
        self.total_distance_km: float = 0.0
        self.total_time_min: float = 0.0
        self.total_cost: float = 0.0
        self.time_of_day: float = 12.0  # 12:00 PM default

        # Cycle-aware masking configuration and state
        self.enable_cycle_aware_masking = enable_cycle_aware_masking
        self.max_node_visits = max_node_visits
        self.revisit_penalty_lambda = revisit_penalty_lambda
        self.edge_traversals: Dict[Tuple[int, int], int] = {}
        self.recovery_actions: int = 0

        # Potential-Based Reward Shaping (PBRS) configuration and heuristic state
        self.enable_pbrs = enable_pbrs
        self.pbrs_gamma = pbrs_gamma
        self.pbrs_scale = pbrs_scale
        self._alt_heuristic = alt_heuristic

    def _get_alt_heuristic(self):
        """Lazy load or return initialized ALT landmark heuristic instance."""
        if self._alt_heuristic is None:
            from ml.graph.landmarks import ALTLandmarkHeuristic
            self._alt_heuristic = ALTLandmarkHeuristic(self.graph)
        return self._alt_heuristic

    @staticmethod
    def calculate_bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate forward azimuth / bearing in degrees [0, 360) from point 1 to point 2."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lng2 - lng1)
        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        return bearing

    def _get_neighbors(self, node: int) -> List[int]:
        """Return sorted outgoing neighbors of node."""
        return sorted(list(self.graph.successors(node)))

    def get_action_neighbor_map(self, node: int) -> Dict[int, int]:
        """
        Map 8 directional compass sectors to outgoing neighbor nodes with adjacent-only
        dynamic allocation and road-hierarchy prioritization.
        Guarantees:
        1. Primary candidates take their true compass sector.
        2. In the 0.27% of nodes with collisions, secondary edges shift ONLY to adjacent
           unoccupied sectors (offset +/- 1, max 45-degree angle difference).
        3. Never shifts to opposite sectors, preserving strict spatial directional meaning.
        """
        if node not in self.graph:
            return {}
        u_data = self.graph.nodes[node]
        u_lat, u_lng = float(u_data['lat']), float(u_data['lng'])
        successors = list(self.graph.successors(node))
        if not successors:
            return {}

        candidates = []
        for v in successors:
            v_data = self.graph.nodes[v]
            v_lat, v_lng = float(v_data['lat']), float(v_data['lng'])
            bearing = self.calculate_bearing(u_lat, u_lng, v_lat, v_lng)
            sector = int((bearing + 22.5) // 45) % self.max_neighbors
            edge_data = self.graph[node][v]
            speed = float(edge_data.get('speed', 40.0))
            is_bridge = 1.0 if any(k in edge_data.get('name', '').lower() for k in ('bridge', 'flyover', 'overbridge')) else 0.0
            candidates.append((v, bearing, sector, speed, is_bridge))

        # Prioritize major/arterial roads (higher speed, bridges)
        candidates.sort(key=lambda item: (-item[4], -item[3]))

        action_map: Dict[int, int] = {}
        occupied_sectors = set()

        for v, bearing, pref_sector, speed, is_bridge in candidates:
            if pref_sector not in occupied_sectors:
                action_map[pref_sector] = v
                occupied_sectors.add(pref_sector)
            else:
                # Dynamic allocation: assign to the closest unoccupied adjacent sector
                for offset in [1, -1, 2, -2, 3, -3, 4]:
                    s = (pref_sector + offset) % self.max_neighbors
                    if s not in occupied_sectors:
                        action_map[s] = v
                        occupied_sectors.add(s)
                        break

        return action_map

    def get_action_mask(self, node: Optional[int] = None, prev_node: Optional[int] = None,
                        visited_nodes: Optional[List[int]] = None,
                        recency_window: int = 60,
                        enable_cycle_aware_masking: Optional[bool] = None,
                        max_node_visits: Optional[int] = None) -> np.ndarray:
        """
        Return boolean mask of valid outgoing actions.
        Prevents cycles by masking nodes visited within recency_window, capping repeat visits,
        and pruning 2-hop traps. When cycle-aware masking is enabled, enforces strict lifetime
        visit capping and least-visited dead-end recovery. The destination node is never masked out.
        """
        target = node if node is not None else self.current_node
        act_map = self.get_action_neighbor_map(target)
        mask = np.zeros(self.max_neighbors, dtype=bool)

        for act in act_map.keys():
            if act < self.max_neighbors:
                mask[act] = True

        dest = self.destination_node
        enable_cycle = self.enable_cycle_aware_masking if enable_cycle_aware_masking is None else enable_cycle_aware_masking
        max_visits = self.max_node_visits if max_node_visits is None else max_node_visits

        if not enable_cycle:
            if visited_nodes is not None and len(visited_nodes) > 1:
                recent_slice = visited_nodes[-recency_window:] if len(visited_nodes) > recency_window else visited_nodes
                recent_set = set(recent_slice)
                counts = Counter(visited_nodes)

                # Check if any outgoing neighbor is unvisited or the destination
                has_fresh = any(
                    (nbr not in recent_set or nbr == dest) and act < self.max_neighbors
                    for act, nbr in act_map.items()
                )

                if has_fresh:
                    for act, nbr in act_map.items():
                        if act < self.max_neighbors and nbr != dest:
                            if nbr in recent_set or counts[nbr] >= 2:
                                mask[act] = False

                # 2-hop trap check: if neighbor v has all its successors in recent_set, mask v if alternatives exist
                valid_acts = [a for a, m in enumerate(mask) if m and a in act_map]
                if len(valid_acts) > 1:
                    for a in valid_acts:
                        v = act_map[a]
                        if v == dest:
                            continue
                        v_succs = list(self.graph.successors(v))
                        if v_succs and all(s in recent_set and s != dest for s in v_succs):
                            mask[a] = False

            # Immediate backtracking check if only prev_node is specified or as secondary filter
            if prev_node is not None and np.sum(mask) > 1:
                for act, nbr in act_map.items():
                    if nbr == prev_node and act < self.max_neighbors:
                        mask[act] = False

            # Fallback: if all masked (e.g. dead-end cul-de-sac), open all available successors
            if not np.any(mask):
                for act in act_map.keys():
                    if act < self.max_neighbors:
                        mask[act] = True

            return mask

        # Cycle-aware masking branch:
        visited = visited_nodes if visited_nodes is not None else self.visited_nodes
        counts = Counter(visited) if visited else Counter()
        recent_slice = visited[-recency_window:] if (visited and len(visited) > recency_window) else (visited or [])
        recent_set = set(recent_slice)

        has_fresh = any(
            (counts.get(nbr, 0) == 0 or nbr == dest) and act < self.max_neighbors
            for act, nbr in act_map.items()
        )

        for act, nbr in act_map.items():
            if act >= self.max_neighbors:
                continue
            if nbr == dest:
                mask[act] = True
                continue

            if prev_node is not None and nbr == prev_node and len(act_map) > 1:
                mask[act] = False
            elif counts.get(nbr, 0) >= max_visits:
                mask[act] = False
            elif nbr in recent_set and len(act_map) > 1:
                if has_fresh or any(counts.get(o, 0) < counts.get(nbr, 0) for a_k, o in act_map.items() if a_k < self.max_neighbors):
                    mask[act] = False

        # 2-hop trap lookahead
        valid_acts = [a for a, m in enumerate(mask) if m and a in act_map]
        if len(valid_acts) > 1:
            for a in valid_acts:
                v = act_map[a]
                if v == dest:
                    continue
                v_succs = list(self.graph.successors(v))
                if v_succs and all((s in recent_set or counts.get(s, 0) >= max_visits) and s != dest for s in v_succs):
                    mask[a] = False

        # Dead-end recovery fallback: relax to least-visited and least-traversed outgoing edges
        if not np.any(mask):
            valid_avail = [a for a in act_map.keys() if a < self.max_neighbors]
            if valid_avail:
                min_visits = min(counts.get(act_map[a], 0) for a in valid_avail)
                min_acts = [a for a in valid_avail if counts.get(act_map[a], 0) == min_visits]
                min_traversals = min(self.edge_traversals.get((target, act_map[a]), 0) for a in min_acts)
                for a in min_acts:
                    if self.edge_traversals.get((target, act_map[a]), 0) == min_traversals:
                        mask[a] = True
                self.recovery_actions += 1

        return mask

    def get_candidate_edges(self, node: Optional[int] = None,
                            prev_node: Optional[int] = None,
                            visited_nodes: Optional[List[int]] = None,
                            recency_window: int = 60,
                            enable_cycle_aware_masking: Optional[bool] = None,
                            max_node_visits: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return candidate outgoing edges for the specified node with graph-aware features.
        Preserves 100% of outgoing edges (no sector binning or collisions).
        Each candidate contains:
          - 'neighbor': int node ID
          - 'features': np.ndarray of shape (20,) [8 state features + 12 edge features]
          - 'is_valid': bool (masked by cycle/backtrack prevention; destination is always valid)
          - 'is_recovery': bool (True if edge was relaxed via dead-end recovery)
        """
        curr = node if node is not None else self.current_node
        dest = self.destination_node
        if curr not in self.graph:
            return []

        successors = list(self.graph.successors(curr))
        if not successors:
            return []

        successors.sort()

        c_data = self.graph.nodes[curr]
        d_data = self.graph.nodes[dest]
        c_lat, c_lng = float(c_data['lat']), float(c_data['lng'])
        d_lat, d_lng = float(d_data['lat']), float(d_data['lng'])

        # 8 Macro state features
        curr_lat_norm = float(np.clip((c_lat - self.center_lat) / self.lat_scale, -1.0, 1.0))
        curr_lng_norm = float(np.clip((c_lng - self.center_lng) / self.lng_scale, -1.0, 1.0))
        dest_lat_norm = float(np.clip((d_lat - self.center_lat) / self.lat_scale, -1.0, 1.0))
        dest_lng_norm = float(np.clip((d_lng - self.center_lng) / self.lng_scale, -1.0, 1.0))

        if curr == dest:
            rem_dist = 0.0
            dir_to_dest_x, dir_to_dest_y = 0.0, 1.0
            dest_bearing = 0.0
        else:
            rem_dist = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)
            dest_bearing = self.calculate_bearing(c_lat, c_lng, d_lat, d_lng)
            rad_dest = math.radians(dest_bearing)
            dir_to_dest_x = float(math.sin(rad_dest))
            dir_to_dest_y = float(math.cos(rad_dest))

        rem_dist_norm = float(np.clip(rem_dist / 20.0, 0.0, 1.0))
        init_dist = max(self.initial_direct_dist, 0.01)
        progress_ratio = float(np.clip((init_dist - rem_dist) / init_dist, -1.0, 1.0))

        macro_state = [curr_lat_norm, curr_lng_norm, dest_lat_norm, dest_lng_norm,
                       rem_dist_norm, dir_to_dest_x, dir_to_dest_y, progress_ratio]

        # Cycle / visited tracking
        visited = visited_nodes if visited_nodes is not None else self.visited_nodes
        recent_set = set(visited[-recency_window:]) if visited else set()
        counts = Counter(visited) if visited else Counter()

        enable_cycle = self.enable_cycle_aware_masking if enable_cycle_aware_masking is None else enable_cycle_aware_masking
        max_visits = self.max_node_visits if max_node_visits is None else max_node_visits

        has_fresh = any((v not in recent_set or v == dest) for v in successors)
        has_fresh_strict = any((counts.get(v, 0) == 0 or v == dest) for v in successors)

        candidates = []
        for v in successors:
            v_data = self.graph.nodes[v]
            v_lat, v_lng = float(v_data['lat']), float(v_data['lng'])
            edge_data = self.graph[curr][v]

            edge_dist = float(edge_data.get('distance', 0.1))
            edge_speed = float(edge_data.get('speed', 40.0))
            is_bridge = 1.0 if any(k in edge_data.get('name', '').lower() for k in ('bridge', 'flyover', 'overbridge')) else 0.0
            tf = float(edge_data.get('traffic_factor', 1.0))

            edge_bearing = self.calculate_bearing(c_lat, c_lng, v_lat, v_lng)
            angle_diff = math.radians(edge_bearing - dest_bearing)
            bearing_alignment = float(math.cos(angle_diff))
            bearing_sine = float(math.sin(angle_diff))

            nbr_rem = self.rng.haversine_distance(v_lat, v_lng, d_lat, d_lng)
            nbr_rem_norm = float(np.clip(nbr_rem / 20.0, 0.0, 1.0))

            dist_progress = rem_dist - nbr_rem
            dist_progress_norm = float(np.clip(dist_progress / max(edge_dist, 0.01), -2.0, 2.0))

            is_in_recent = 1.0 if (v in recent_set and v != dest) else 0.0
            v_count = counts.get(v, 0)
            revisit_count_norm = float(np.clip(v_count / 3.0, 0.0, 1.0))
            is_backtrack = 1.0 if (prev_node is not None and v == prev_node and v != dest) else 0.0
            is_dest = 1.0 if v == dest else 0.0

            edge_feat = [
                float(np.clip(edge_dist / 2.0, 0.0, 1.0)),
                float(np.clip(edge_speed / 60.0, 0.2, 1.0)),
                is_bridge,
                float(np.clip(tf / 2.0, 0.25, 2.0)),
                bearing_alignment,
                bearing_sine,
                nbr_rem_norm,
                dist_progress_norm,
                is_in_recent,
                revisit_count_norm,
                is_backtrack,
                is_dest
            ]

            full_features = np.array(macro_state + edge_feat, dtype=np.float32)

            is_valid = True
            if v == dest:
                is_valid = True
            else:
                if not enable_cycle:
                    if has_fresh and (v in recent_set or v_count >= 2):
                        is_valid = False
                    if prev_node is not None and v == prev_node and len(successors) > 1:
                        is_valid = False
                else:
                    if prev_node is not None and v == prev_node and len(successors) > 1:
                        is_valid = False
                    elif v_count >= max_visits:
                        is_valid = False
                    elif v in recent_set and len(successors) > 1:
                        if has_fresh_strict or any(counts.get(o, 0) < v_count for o in successors):
                            is_valid = False

            candidates.append({
                'neighbor': v,
                'features': full_features,
                'is_valid': is_valid,
                'is_recovery': False
            })

        # 2-hop trap lookahead
        valid_candidates = [c for c in candidates if c['is_valid']]
        if len(valid_candidates) > 1:
            for c in valid_candidates:
                v = c['neighbor']
                if v == dest:
                    continue
                v_succs = list(self.graph.successors(v))
                if not enable_cycle:
                    if v_succs and all(s in recent_set and s != dest for s in v_succs):
                        c['is_valid'] = False
                else:
                    if v_succs and all((s in recent_set or counts.get(s, 0) >= max_visits) and s != dest for s in v_succs):
                        c['is_valid'] = False

        # Fallback if all masked
        if not any(c['is_valid'] for c in candidates):
            if not enable_cycle:
                for c in candidates:
                    c['is_valid'] = True
            else:
                min_visits = min(counts.get(c['neighbor'], 0) for c in candidates)
                min_cands = [c for c in candidates if counts.get(c['neighbor'], 0) == min_visits]
                min_traversals = min(self.edge_traversals.get((curr, c['neighbor']), 0) for c in min_cands)
                for c in candidates:
                    if counts.get(c['neighbor'], 0) == min_visits and self.edge_traversals.get((curr, c['neighbor']), 0) == min_traversals:
                        c['is_valid'] = True
                        c['is_recovery'] = True
                    else:
                        c['is_valid'] = False
                self.recovery_actions += 1

        return candidates

    def _get_obs(self) -> np.ndarray:
        """Construct continuous, scale-invariant normalized 12-feature observation vector."""
        c_node_data = self.graph.nodes[self.current_node]
        d_node_data = self.graph.nodes[self.destination_node]

        c_lat, c_lng = float(c_node_data['lat']), float(c_node_data['lng'])
        d_lat, d_lng = float(d_node_data['lat']), float(d_node_data['lng'])

        # 1-4. Absolute normalized geographic coordinates
        curr_lat_norm = float(np.clip((c_lat - self.center_lat) / self.lat_scale, -1.0, 1.0))
        curr_lng_norm = float(np.clip((c_lng - self.center_lng) / self.lng_scale, -1.0, 1.0))
        dest_lat_norm = float(np.clip((d_lat - self.center_lat) / self.lat_scale, -1.0, 1.0))
        dest_lng_norm = float(np.clip((d_lng - self.center_lng) / self.lng_scale, -1.0, 1.0))

        if self.current_node == self.destination_node:
            rem_dist_norm = 0.0
            dir_x, dir_y = 0.0, 1.0
            rem_dist = 0.0
        else:
            rem_dist = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)
            rem_dist_norm = float(np.clip(rem_dist / 20.0, 0.0, 1.0))

            # 6-7. Compass bearing unit vector towards target
            bearing = self.calculate_bearing(c_lat, c_lng, d_lat, d_lng)
            rad = math.radians(bearing)
            dir_x = float(math.sin(rad))
            dir_y = float(math.cos(rad))

        # 8. Road hierarchy: normalized mean speed of outgoing edges
        nbrs = list(self.graph.successors(self.current_node))
        if nbrs:
            curr_speed = float(np.mean([self.graph[self.current_node][v].get('speed', 40.0) for v in nbrs]))
        else:
            curr_speed = 40.0
        curr_speed_norm = float(np.clip(curr_speed / 60.0, 0.2, 1.0))

        # 9. Bridge / flyover crossing indicator
        is_bridge = 1.0 if any(any(k in self.graph[self.current_node][v].get('name', '').lower()
                                  for k in ('bridge', 'flyover', 'overbridge')) for v in nbrs) else 0.0

        # 10. Revisit count of current node
        revisit_count = self.visited_counts.get(self.current_node, 1)
        revisit_count_norm = float(np.clip((revisit_count - 1) / 3.0, 0.0, 1.0))

        # 11. Macro progress ratio towards destination
        init_dist = max(self.initial_direct_dist, 0.01)
        progress_ratio = float(np.clip((init_dist - rem_dist) / init_dist, -1.0, 1.0))

        # 12. Local traffic factor
        if nbrs:
            tf = float(np.mean([self.graph[self.current_node][nbr].get('traffic_factor', 1.0) for nbr in nbrs]))
        else:
            tf = 1.0
        tf_norm = float(np.clip(tf, 0.5, 3.0))

        return np.array([curr_lat_norm, curr_lng_norm, dest_lat_norm, dest_lng_norm,
                         rem_dist_norm, dir_x, dir_y, curr_speed_norm, is_bridge,
                         revisit_count_norm, progress_ratio, tf_norm], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options and 'max_steps' in options:
            self.max_steps = int(options['max_steps'])

        if options and 'enable_cycle_aware_masking' in options:
            self.enable_cycle_aware_masking = bool(options['enable_cycle_aware_masking'])
        if options and 'max_node_visits' in options:
            self.max_node_visits = int(options['max_node_visits'])
        if options and 'revisit_penalty_lambda' in options:
            self.revisit_penalty_lambda = float(options['revisit_penalty_lambda'])
        if options and 'enable_pbrs' in options:
            self.enable_pbrs = bool(options['enable_pbrs'])
        if options and 'pbrs_gamma' in options:
            self.pbrs_gamma = float(options['pbrs_gamma'])
        if options and 'pbrs_scale' in options:
            self.pbrs_scale = float(options['pbrs_scale'])

        if options and 'source' in options and 'destination' in options:
            self.current_node = options['source']
            self.destination_node = options['destination']
        else:
            s = int(np.random.choice(self.node_list))
            curr = s
            hops = np.random.randint(5, 30)
            for _ in range(hops):
                succs = list(self.graph.successors(curr))
                if succs:
                    curr = random.choice(succs)
                else:
                    break
            d = curr if curr != s else self.node_list[(self.node_to_idx.get(s, 0) + 1) % self.num_nodes]
            self.current_node = s
            self.destination_node = d

        c_lat = float(self.graph.nodes[self.current_node]['lat'])
        c_lng = float(self.graph.nodes[self.current_node]['lng'])
        d_lat = float(self.graph.nodes[self.destination_node]['lat'])
        d_lng = float(self.graph.nodes[self.destination_node]['lng'])
        self.initial_direct_dist = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)

        self.visited_nodes = [self.current_node]
        self.visited_counts = {self.current_node: 1}
        self.edge_traversals = {}
        self.recovery_actions = 0
        self.step_count = 0
        self.total_distance_km = 0.0
        self.total_time_min = 0.0
        self.total_cost = 0.0
        self.time_of_day = options.get('time_of_day', 12.0) if options else 12.0

        init_potential = 0.0
        if self.enable_pbrs and self.current_node != self.destination_node:
            alt = self._get_alt_heuristic()
            init_potential = -self.pbrs_scale * alt.get_lower_bound(self.current_node, self.destination_node)

        info = {
            'source_node': self.current_node,
            'current_node': self.current_node,
            'destination_node': self.destination_node,
            'action_mask': self.get_action_mask(visited_nodes=self.visited_nodes),
            'recovery_actions': self.recovery_actions,
            'pbrs_potential': float(init_potential)
        }
        return self._get_obs(), info

    def _compute_pbrs_shaping(self, curr_node: int, next_node: int, is_goal: bool) -> Tuple[float, float, float]:
        """
        Compute Potential-Based Reward Shaping (PBRS) term:
            F(s, a, s') = gamma * Phi(s') - Phi(s)
        Where:
            Phi(s) = -pbrs_scale * h_ALT(curr_node, destination_node)
            Phi(s_terminal) = 0.0 (absorbing terminal goal state condition)
        Returns:
            (shaping_f, phi_curr, phi_next)
        """
        if not self.enable_pbrs:
            return 0.0, 0.0, 0.0

        alt = self._get_alt_heuristic()
        h_curr = alt.get_lower_bound(curr_node, self.destination_node)
        phi_curr = -self.pbrs_scale * h_curr

        if is_goal:
            # Absorbing goal state has zero potential (Ng et al. 1999)
            phi_next = 0.0
        else:
            h_next = alt.get_lower_bound(next_node, self.destination_node)
            phi_next = -self.pbrs_scale * h_next

        shaping_f = self.pbrs_gamma * phi_next - phi_curr
        return float(shaping_f), float(phi_curr), float(phi_next)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        act_map = self.get_action_neighbor_map(self.current_node)

        terminated = False
        truncated = False
        reward = 0.0

        if action not in act_map:
            reward += self.rw['invalid_action']
            r_orig_step = self.rw['invalid_action']
            shaping_f = 0.0
            phi_curr = 0.0
            phi_next = 0.0
            next_node = self.current_node
        else:
            next_node = act_map[action]
            edge_tuple = (self.current_node, next_node)
            self.edge_traversals[edge_tuple] = self.edge_traversals.get(edge_tuple, 0) + 1

            edge_data = self.graph[self.current_node][next_node]
            step_cost = edge_data.get('weight', 1.0)
            dist = edge_data.get('distance', 0.1)
            travel_time = edge_data.get('travel_time', 0.5)

            self.total_distance_km += dist
            self.total_time_min += travel_time
            self.total_cost += step_cost

            # Distance calculations
            c_lat = float(self.graph.nodes[self.current_node]['lat'])
            c_lng = float(self.graph.nodes[self.current_node]['lng'])
            n_lat = float(self.graph.nodes[next_node]['lat'])
            n_lng = float(self.graph.nodes[next_node]['lng'])
            d_lat = float(self.graph.nodes[self.destination_node]['lat'])
            d_lng = float(self.graph.nodes[self.destination_node]['lng'])

            prev_rem = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)
            new_rem = self.rng.haversine_distance(n_lat, n_lng, d_lat, d_lng)
            dist_progress = prev_rem - new_rem

            r_base = -step_cost * self.rw.get('step_cost_multiplier', 0.5)
            edge_speed = float(edge_data.get('speed', 40.0))
            if edge_speed >= 50.0:
                r_base += 0.1
            is_bridge = any(k in edge_data.get('name', '').lower() for k in ('bridge', 'flyover', 'overbridge'))
            if is_bridge:
                r_base += 0.5
            if next_node in self.visited_nodes[-60:]:
                r_base += self.rw['loop_penalty']
            prior_visits = self.visited_counts.get(next_node, 0)
            if prior_visits >= 1:
                r_base -= self.revisit_penalty_lambda * prior_visits

            # Baseline Euclidean progress
            r_euclidean = dist_progress * self.rw.get('progress_multiplier', 8.0)
            r_orig_step = r_base + r_euclidean

            is_goal = (next_node == self.destination_node)
            shaping_f, phi_curr, phi_next = self._compute_pbrs_shaping(self.current_node, next_node, is_goal)

            if self.enable_pbrs:
                r_step = r_base + shaping_f
            else:
                r_step = r_orig_step

            reward += r_step
            self.current_node = next_node
            self.visited_nodes.append(next_node)
            self.visited_counts[next_node] = prior_visits + 1

        # Check goal reached
        if self.current_node == self.destination_node:
            reward += self.rw['goal']
            terminated = True

        # Check max steps limit
        if self.step_count >= self.max_steps and not terminated:
            reward += self.rw['max_steps_penalty']
            truncated = True

        r_orig_total = r_orig_step + (self.rw['goal'] if terminated else 0.0) + (self.rw['max_steps_penalty'] if truncated else 0.0)
        prev_node = self.visited_nodes[-2] if len(self.visited_nodes) >= 2 else None
        info = {
            'current_node': self.current_node,
            'destination_node': self.destination_node,
            'step_count': self.step_count,
            'total_distance_km': self.total_distance_km,
            'total_time_min': self.total_time_min,
            'total_cost': self.total_cost,
            'visited_nodes': self.visited_nodes,
            'recovery_actions': self.recovery_actions,
            'r_original': float(r_orig_total),
            'shaping_reward': float(shaping_f),
            'r_shaped': float(reward),
            'pbrs_potential_curr': float(phi_curr),
            'pbrs_potential_next': float(phi_next),
            'action_mask': self.get_action_mask(prev_node=prev_node, visited_nodes=self.visited_nodes)
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def step_to_neighbor(self, next_node: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment by moving directly to an outgoing neighbor node.
        Computes distance, travel time, reward, checks goal termination and step limits.
        """
        self.step_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        if not self.graph.has_edge(self.current_node, next_node):
            reward += self.rw['invalid_action']
            r_orig_step = self.rw['invalid_action']
            shaping_f = 0.0
            phi_curr = 0.0
            phi_next = 0.0
            next_node = self.current_node
        else:
            edge_tuple = (self.current_node, next_node)
            self.edge_traversals[edge_tuple] = self.edge_traversals.get(edge_tuple, 0) + 1

            edge_data = self.graph[self.current_node][next_node]
            step_cost = edge_data.get('weight', 1.0)
            dist = edge_data.get('distance', 0.1)
            travel_time = edge_data.get('travel_time', 0.5)

            self.total_distance_km += dist
            self.total_time_min += travel_time
            self.total_cost += step_cost

            c_lat = float(self.graph.nodes[self.current_node]['lat'])
            c_lng = float(self.graph.nodes[self.current_node]['lng'])
            n_lat = float(self.graph.nodes[next_node]['lat'])
            n_lng = float(self.graph.nodes[next_node]['lng'])
            d_lat = float(self.graph.nodes[self.destination_node]['lat'])
            d_lng = float(self.graph.nodes[self.destination_node]['lng'])

            prev_rem = self.rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)
            new_rem = self.rng.haversine_distance(n_lat, n_lng, d_lat, d_lng)
            dist_progress = prev_rem - new_rem

            r_base = -step_cost * self.rw.get('step_cost_multiplier', 0.5)
            edge_speed = float(edge_data.get('speed', 40.0))
            if edge_speed >= 50.0:
                r_base += 0.1
            is_bridge = any(k in edge_data.get('name', '').lower() for k in ('bridge', 'flyover', 'overbridge'))
            if is_bridge:
                r_base += 0.5
            if next_node in self.visited_nodes[-60:]:
                r_base += self.rw['loop_penalty']
            prior_visits = self.visited_counts.get(next_node, 0)
            if prior_visits >= 1:
                r_base -= self.revisit_penalty_lambda * prior_visits

            # Baseline Euclidean progress
            r_euclidean = dist_progress * self.rw.get('progress_multiplier', 8.0)
            r_orig_step = r_base + r_euclidean

            is_goal = (next_node == self.destination_node)
            shaping_f, phi_curr, phi_next = self._compute_pbrs_shaping(self.current_node, next_node, is_goal)

            if self.enable_pbrs:
                r_step = r_base + shaping_f
            else:
                r_step = r_orig_step

            reward += r_step
            self.current_node = next_node
            self.visited_nodes.append(next_node)
            self.visited_counts[next_node] = prior_visits + 1

        if self.current_node == self.destination_node:
            reward += self.rw['goal']
            terminated = True

        if self.step_count >= self.max_steps and not terminated:
            reward += self.rw['max_steps_penalty']
            truncated = True

        r_orig_total = r_orig_step + (self.rw['goal'] if terminated else 0.0) + (self.rw['max_steps_penalty'] if truncated else 0.0)
        prev_node = self.visited_nodes[-2] if len(self.visited_nodes) >= 2 else None
        info = {
            'current_node': self.current_node,
            'destination_node': self.destination_node,
            'step_count': self.step_count,
            'total_distance_km': self.total_distance_km,
            'total_time_min': self.total_time_min,
            'total_cost': self.total_cost,
            'visited_nodes': self.visited_nodes,
            'recovery_actions': self.recovery_actions,
            'r_original': float(r_orig_total),
            'shaping_reward': float(shaping_f),
            'r_shaped': float(reward),
            'pbrs_potential_curr': float(phi_curr),
            'pbrs_potential_next': float(phi_next),
            'candidate_edges': self.get_candidate_edges(prev_node=prev_node, visited_nodes=self.visited_nodes)
        }
        return self._get_obs(), float(reward), terminated, truncated, info

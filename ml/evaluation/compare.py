import os
import time
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional
from ml.graph.builder import RoadNetworkGraph
from ml.evaluation.baselines import BaselineRouter
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.dqn_agent import DQNAgent


class AlgorithmEvaluator:
    """
    Evaluates and compares Dijkstra, A*, DQN (and PPO if loaded)
    on identical test scenarios without hard-coded numbers.
    """

    def __init__(self, graph_builder: RoadNetworkGraph, dqn_model_path: Optional[str] = None):
        self.rng = graph_builder
        self.graph = graph_builder.graph
        self.baseline_router = BaselineRouter(graph_builder)
        
        self.env = TaxiRouteEnv(graph_builder, max_steps=50)
        self.dqn_agent: Optional[DQNAgent] = None
        self.dqn_loaded = False

        if dqn_model_path and os.path.exists(dqn_model_path):
            try:
                self.dqn_agent = DQNAgent(
                    state_dim=self.env.observation_space.shape[0],
                    action_dim=self.env.action_space.n
                )
                self.dqn_loaded = self.dqn_agent.load_checkpoint(dqn_model_path)
            except Exception as e:
                print(f"Failed to load DQN model from {dqn_model_path}: {e}")

    def route_dqn(self, source_node: int, dest_node: int) -> Dict[str, Any]:
        """
        Generate route using trained DQN agent.
        """
        start_time = time.perf_counter()
        
        if not self.dqn_loaded or self.dqn_agent is None:
            return {
                'algorithm': 'DQN',
                'success': False,
                'error': 'Model not trained yet. Train model first from /training page or CLI.',
                'route_nodes': [],
                'route_coordinates': [],
                'total_distance_km': 0.0,
                'total_time_min': 0.0,
                'total_cost': 0.0,
                'operational_cost': 0.0,
                'step_count': 0,
                'route_efficiency_pct': 0.0,
                'execution_time_ms': 0.0
            }

        obs, info = self.env.reset(options={'source': source_node, 'destination': dest_node})
        path_nodes = [source_node]
        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < 50:
            mask = info.get('action_mask')
            action = self.dqn_agent.select_action(obs, mask=mask, eval_mode=True)
            obs, reward, done, truncated, info = self.env.step(action)
            current_node = info.get('current_node')
            if current_node not in path_nodes:
                path_nodes.append(current_node)
            step += 1

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0
        success = (path_nodes[-1] == dest_node)

        # Build response metrics
        route_coordinates = []
        total_distance_km = 0.0
        total_time_min = 0.0
        total_cost = 0.0
        total_op_cost = 0.0

        for i, u in enumerate(path_nodes):
            lat = float(self.graph.nodes[u]['lat'])
            lng = float(self.graph.nodes[u]['lng'])
            route_coordinates.append({'lat': lat, 'lng': lng, 'node_id': u})

            if i > 0:
                prev_u = path_nodes[i - 1]
                if self.graph.has_edge(prev_u, u):
                    edge_data = self.graph[prev_u][u]
                    total_distance_km += edge_data.get('distance', 0.0)
                    total_time_min += edge_data.get('travel_time', 0.0)
                    total_cost += edge_data.get('weight', 0.0)
                    total_op_cost += edge_data.get('operational_cost', 0.0)

        if len(path_nodes) > 1:
            src_lat, src_lng = route_coordinates[0]['lat'], route_coordinates[0]['lng']
            dst_lat, dst_lng = route_coordinates[-1]['lat'], route_coordinates[-1]['lng']
            direct_dist = RoadNetworkGraph.haversine_distance(src_lat, src_lng, dst_lat, dst_lng)
            efficiency = round(min(direct_dist / max(total_distance_km, 0.001), 1.0) * 100.0, 1)
        else:
            efficiency = 100.0

        return {
            'algorithm': 'DQN',
            'success': success,
            'route_nodes': path_nodes,
            'route_coordinates': route_coordinates,
            'total_distance_km': round(total_distance_km, 2),
            'total_time_min': round(total_time_min, 1),
            'total_cost': round(total_cost, 2),
            'operational_cost': round(total_op_cost, 2),
            'step_count': max(len(path_nodes) - 1, 0),
            'route_efficiency_pct': efficiency,
            'execution_time_ms': round(exec_time_ms, 3)
        }

    def compare_all(self, source_lat: float, source_lng: float,
                    dest_lat: float, dest_lng: float,
                    optimization_objective: str = "balanced") -> Dict[str, Any]:
        """
        Compare Dijkstra, A*, and DQN for given coordinate pair.
        Returns comprehensive comparison report & deterministic explanation.
        """
        s_node = self.rng.find_nearest_node(source_lat, source_lng)
        d_node = self.rng.find_nearest_node(dest_lat, dest_lng)

        dijkstra_res = self.baseline_router.dijkstra_route(s_node, d_node)
        astar_res = self.baseline_router.astar_route(s_node, d_node)
        dqn_res = self.route_dqn(s_node, d_node)

        results = {
            'Dijkstra': dijkstra_res,
            'A*': astar_res,
            'DQN': dqn_res
        }

        # Determine winner based on objective
        valid_results = {k: v for k, v in results.items() if v.get('success', False)}

        if valid_results:
            if optimization_objective == "distance":
                recommended = min(valid_results.items(), key=lambda x: x[1]['total_distance_km'])[0]
            elif optimization_objective == "time":
                recommended = min(valid_results.items(), key=lambda x: x[1]['total_time_min'])[0]
            elif optimization_objective == "cost":
                recommended = min(valid_results.items(), key=lambda x: x[1]['total_cost'])[0]
            else:  # balanced
                recommended = min(valid_results.items(), key=lambda x: (x[1]['total_cost'], x[1]['execution_time_ms']))[0]
        else:
            recommended = "Dijkstra"

        # Generate deterministic explanation
        explanation = self.generate_explanation(results, recommended, optimization_objective)

        return {
            'source': {'lat': source_lat, 'lng': source_lng, 'node_id': s_node},
            'destination': {'lat': dest_lat, 'lng': dest_lng, 'node_id': d_node},
            'optimization_objective': optimization_objective,
            'recommended_algorithm': recommended,
            'explanation': explanation,
            'algorithms': results
        }

    @staticmethod
    def generate_explanation(results: Dict[str, Dict[str, Any]], winner: str, objective: str) -> str:
        """Generate deterministic, data-driven explanation of algorithm comparison results."""
        winner_data = results.get(winner, {})
        if not winner_data.get('success', False):
            return "Unable to calculate a valid route for the selected start and destination locations."

        w_dist = winner_data.get('total_distance_km', 0.0)
        w_time = winner_data.get('total_time_min', 0.0)
        w_cost = winner_data.get('total_cost', 0.0)
        w_exec = winner_data.get('execution_time_ms', 0.0)

        explanation = f"Recommended Algorithm: **{winner}** based on the '{objective}' objective. "
        explanation += f"It achieved a total distance of {w_dist} km in ~{w_time} mins with an estimated combined cost of {w_cost}. "

        if winner == "DQN" and results.get('Dijkstra', {}).get('success'):
            d_time = results['Dijkstra']['total_time_min']
            explanation += f"DQN learned an adaptive policy considering localized traffic factors, computing route in {w_exec:.2f}ms compared to Dijkstra ({d_time} mins travel time)."
        elif winner == "A*":
            explanation += f"A* leveraged the Haversine heuristic to prune search space, executing in {w_exec:.2f}ms with optimal travel cost."
        elif winner == "Dijkstra":
            explanation += f"Dijkstra guaranteed exact global minimum cost across the road network graph."

        return explanation

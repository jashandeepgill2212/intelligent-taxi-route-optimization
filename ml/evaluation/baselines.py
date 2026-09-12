import time
import heapq
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from ml.graph.builder import RoadNetworkGraph


class BaselineRouter:
    """
    Implements Dijkstra and A* shortest-path routing algorithms
    on the RoadNetworkGraph with unified output structures.
    """

    def __init__(self, graph_builder: RoadNetworkGraph):
        self.rng = graph_builder
        self.graph = graph_builder.graph

    def dijkstra_route(self, source_node: int, dest_node: int) -> Dict[str, Any]:
        """
        Execute Dijkstra's algorithm to find optimal path based on edge weights.
        """
        start_time = time.perf_counter()
        
        try:
            path_nodes = nx.dijkstra_path(self.graph, source=source_node, target=dest_node, weight='weight')
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            return self._format_route_response("Dijkstra", path_nodes, exec_time_ms)
        except nx.NetworkXNoPath:
            return self._empty_route_response("Dijkstra", source_node, dest_node)

    def astar_route(self, source_node: int, dest_node: int) -> Dict[str, Any]:
        """
        Execute A* algorithm using Haversine heuristic between current node and destination node.
        """
        start_time = time.perf_counter()
        dest_lat = self.graph.nodes[dest_node]['lat']
        dest_lng = self.graph.nodes[dest_node]['lng']

        def haversine_heuristic(u: int, v: int) -> float:
            u_lat = self.graph.nodes[u]['lat']
            u_lng = self.graph.nodes[u]['lng']
            # Convert heuristic distance (km) to approximate cost
            dist_km = RoadNetworkGraph.haversine_distance(u_lat, u_lng, dest_lat, dest_lng)
            # Cost factor approximation: distance * (w2_distance + w1_time * time_per_km)
            return dist_km * (self.rng.w2_distance + self.rng.w1_time * (60.0 / 40.0))

        try:
            path_nodes = nx.astar_path(self.graph, source=source_node, target=dest_node,
                                       heuristic=haversine_heuristic, weight='weight')
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            return self._format_route_response("A*", path_nodes, exec_time_ms)
        except nx.NetworkXNoPath:
            return self._empty_route_response("A*", source_node, dest_node)

    def _format_route_response(self, algorithm: str, path_nodes: List[int], exec_time_ms: float) -> Dict[str, Any]:
        """Format detailed metrics and coordinates array for calculated route."""
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
                edge_data = self.graph[prev_u][u]
                total_distance_km += edge_data.get('distance', 0.0)
                total_time_min += edge_data.get('travel_time', 0.0)
                total_cost += edge_data.get('weight', 0.0)
                total_op_cost += edge_data.get('operational_cost', 0.0)

        # Efficiency metric: ratio of straight line distance to route distance
        if len(path_nodes) > 1:
            src_lat, src_lng = route_coordinates[0]['lat'], route_coordinates[0]['lng']
            dst_lat, dst_lng = route_coordinates[-1]['lat'], route_coordinates[-1]['lng']
            direct_dist = RoadNetworkGraph.haversine_distance(src_lat, src_lng, dst_lat, dst_lng)
            efficiency = round(min(direct_dist / max(total_distance_km, 0.001), 1.0) * 100.0, 1)
        else:
            efficiency = 100.0

        return {
            'algorithm': algorithm,
            'success': True,
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

    def _empty_route_response(self, algorithm: str, source_node: int, dest_node: int) -> Dict[str, Any]:
        """Return standardized response when no route exists."""
        return {
            'algorithm': algorithm,
            'success': False,
            'error': f"No valid path exists between node {source_node} and node {dest_node}.",
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

import os
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class RoadNetworkGraph:
    """
    Road Network Graph manager for OSMnx real-world maps & synthetic grid fallback.
    Computes multi-objective edge costs considering distance, travel time, traffic factors,
    and operational/fuel costs.
    """

    def __init__(self,
                 w1_time: float = 0.4,
                 w2_distance: float = 0.3,
                 w3_traffic: float = 0.2,
                 w4_operational: float = 0.1,
                 cost_per_km: float = 12.0):
        self.w1_time = w1_time
        self.w2_distance = w2_distance
        self.w3_traffic = w3_traffic
        self.w4_operational = w4_operational
        self.cost_per_km = cost_per_km
        self.graph: Optional[nx.DiGraph] = None
        self.nodes_dict: Dict[int, Dict[str, float]] = {}
        self.is_synthetic: bool = False
        self.city_name: str = ""

    def calculate_edge_cost(self, distance_km: float, speed_kmh: float,
                            traffic_factor: float = 1.0) -> Dict[str, float]:
        """
        Compute edge costs based on configured weights.
        Returns detailed breakdown dictionary.
        """
        # Travel time in minutes
        effective_speed = max(speed_kmh / max(traffic_factor, 0.5), 5.0)
        travel_time_min = (distance_km / effective_speed) * 60.0
        
        # Traffic penalty factor
        traffic_penalty = (traffic_factor - 1.0) * travel_time_min * 1.5
        
        # Operational cost in Currency (e.g. ₹ per km)
        operational_cost = distance_km * self.cost_per_km * (1.0 + 0.2 * (traffic_factor - 1.0))
        
        # Total combined cost
        total_cost = (
            self.w1_time * travel_time_min +
            self.w2_distance * distance_km +
            self.w3_traffic * traffic_penalty +
            self.w4_operational * (operational_cost / 10.0)
        )
        
        return {
            'distance_km': round(distance_km, 3),
            'speed_kmh': round(speed_kmh, 1),
            'effective_speed_kmh': round(effective_speed, 1),
            'travel_time_min': round(travel_time_min, 3),
            'traffic_factor': round(traffic_factor, 2),
            'operational_cost': round(operational_cost, 2),
            'total_cost': round(total_cost, 4)
        }

    def build_synthetic_grid(self, grid_size: int = 10, center_lat: float = 30.9010,
                            center_lng: float = 75.8573, spacing_deg: float = 0.008) -> nx.DiGraph:
        """
        Build a deterministic synthetic road network grid for offline development and testing.
        Grid size 10x10 = 100 intersections/nodes.
        """
        G = nx.DiGraph()
        self.is_synthetic = True
        self.city_name = f"Synthetic {grid_size}x{grid_size} Grid"
        
        node_id = 0
        grid_map = {}
        
        for r in range(grid_size):
            for c in range(grid_size):
                lat = center_lat + (r - grid_size / 2) * spacing_deg
                lng = center_lng + (c - grid_size / 2) * spacing_deg
                G.add_node(node_id, y=lat, x=lng, lat=lat, lng=lng)
                grid_map[(r, c)] = node_id
                self.nodes_dict[node_id] = {'lat': lat, 'lng': lng}
                node_id += 1
                
        # Connect neighboring nodes bidirectional with realistic street speeds
        for r in range(grid_size):
            for c in range(grid_size):
                u = grid_map[(r, c)]
                neighbors = []
                if r > 0: neighbors.append((grid_map[(r - 1, c)], 35.0))  # Avenue
                if r < grid_size - 1: neighbors.append((grid_map[(r + 1, c)], 35.0))
                if c > 0: neighbors.append((grid_map[(r, c - 1)], 45.0))  # Main Boulevard
                if c < grid_size - 1: neighbors.append((grid_map[(r, c + 1)], 45.0))
                
                # Diagonal connections for grid variety
                if r > 0 and c > 0 and (r + c) % 3 == 0:
                    neighbors.append((grid_map[(r - 1, c - 1)], 30.0))
                if r < grid_size - 1 and c < grid_size - 1 and (r + c) % 3 == 0:
                    neighbors.append((grid_map[(r + 1, c + 1)], 30.0))
                
                u_lat, u_lng = G.nodes[u]['lat'], G.nodes[u]['lng']
                
                for v, default_speed in neighbors:
                    v_lat, v_lng = G.nodes[v]['lat'], G.nodes[v]['lng']
                    
                    # Haversine distance
                    dist_km = self.haversine_distance(u_lat, u_lng, v_lat, v_lng)
                    cost_info = self.calculate_edge_cost(dist_km, default_speed, traffic_factor=1.0)
                    
                    G.add_edge(u, v,
                               distance=cost_info['distance_km'],
                               speed=cost_info['speed_kmh'],
                               travel_time=cost_info['travel_time_min'],
                               traffic_factor=cost_info['traffic_factor'],
                               operational_cost=cost_info['operational_cost'],
                               weight=cost_info['total_cost'])
                    
        self.graph = G
        return G

    def load_osm_graph(self, place_name: str = "Ludhiana, India", network_type: str = "drive") -> nx.DiGraph:
        """
        Attempt to download / load OpenStreetMap graph using OSMnx.
        Falls back to synthetic grid if OSMnx or network fails.
        """
        try:
            import osmnx as ox
            print(f"Attempting to download OSM road network for '{place_name}'...")
            # Try fetching drive network
            G_raw = ox.graph_from_place(place_name, network_type=network_type)
            G = nx.DiGraph()
            
            # Convert to strongly connected DiGraph with unified node IDs
            mapping = {old_id: idx for idx, old_id in enumerate(G_raw.nodes())}
            
            for old_id, new_id in mapping.items():
                node_data = G_raw.nodes[old_id]
                lat = node_data.get('y', node_data.get('lat', 0.0))
                lng = node_data.get('x', node_data.get('lon', node_data.get('lng', 0.0)))
                G.add_node(new_id, y=lat, x=lng, lat=lat, lng=lng)
                self.nodes_dict[new_id] = {'lat': lat, 'lng': lng}

            for u_old, v_old, data in G_raw.edges(data=True):
                u, v = mapping[u_old], mapping[v_old]
                length_m = data.get('length', 100.0)
                dist_km = length_m / 1000.0
                max_speed = data.get('maxspeed', 40.0)
                if isinstance(max_speed, list):
                    max_speed = max_speed[0]
                try:
                    speed_kmh = float(str(max_speed).replace(' mph', '').replace(' km/h', ''))
                except ValueError:
                    speed_kmh = 40.0

                cost_info = self.calculate_edge_cost(dist_km, speed_kmh, traffic_factor=1.0)
                G.add_edge(u, v,
                           distance=cost_info['distance_km'],
                           speed=cost_info['speed_kmh'],
                           travel_time=cost_info['travel_time_min'],
                           traffic_factor=cost_info['traffic_factor'],
                           operational_cost=cost_info['operational_cost'],
                           weight=cost_info['total_cost'])

            self.graph = G
            self.is_synthetic = False
            self.city_name = place_name
            print(f"Loaded OSM graph for {place_name} with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
            return G

        except Exception as e:
            print(f"OSMnx load failed ({e}). Falling back to synthetic grid road network...")
            return self.build_synthetic_grid()

    def set_traffic_condition(self, traffic_level: str = "MEDIUM"):
        """
        Update graph edge weights based on simulated traffic condition.
        traffic_level: 'LOW' (1.0x), 'MEDIUM' (1.3x), 'HIGH' (1.8x)
        """
        if self.graph is None:
            return
            
        factors = {'LOW': 1.0, 'MEDIUM': 1.3, 'HIGH': 1.8}
        base_factor = factors.get(traffic_level.upper(), 1.0)
        
        np.random.seed(42)  # Deterministic traffic distribution
        
        for u, v, data in self.graph.edges(data=True):
            # Vary factor per edge slightly to simulate realistic bottleneck corridors
            noise = np.random.uniform(-0.15, 0.15)
            edge_factor = max(1.0, base_factor + noise)
            
            cost_info = self.calculate_edge_cost(
                distance_km=data['distance'],
                speed_kmh=data['speed'],
                traffic_factor=edge_factor
            )
            
            data['traffic_factor'] = cost_info['traffic_factor']
            data['travel_time'] = cost_info['travel_time_min']
            data['operational_cost'] = cost_info['operational_cost']
            data['weight'] = cost_info['total_cost']

    def find_nearest_node(self, lat: float, lng: float) -> int:
        """Find graph node ID nearest to given GPS latitude/longitude."""
        if not self.nodes_dict:
            raise ValueError("Graph nodes dictionary is empty!")
            
        best_node = None
        min_dist = float('inf')
        
        for node_id, coords in self.nodes_dict.items():
            d = self.haversine_distance(lat, lng, coords['lat'], coords['lng'])
            if d < min_dist:
                min_dist = d
                best_node = node_id
                
        return best_node

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in kilometers between two points."""
        R = 6371.0  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(R * c)

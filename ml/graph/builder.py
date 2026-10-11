import os
import json
import logging
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger(__name__)


class RoadNetworkGraph:
    """
    Road Network Graph manager for real-world OpenStreetMap networks
    and synthetic grid simulation fallback.
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
        self.is_synthetic: bool = True
        self.source_type: str = "SYNTHETIC SIMULATION"
        self.city_name: str = "Synthetic 10x10 Grid"
        
        # Internal vectorized caches for ultra-fast node snapping
        self._node_ids_arr: Optional[np.ndarray] = None
        self._lats_arr: Optional[np.ndarray] = None
        self._lngs_arr: Optional[np.ndarray] = None

    def _invalidate_node_cache(self):
        self._node_ids_arr = None
        self._lats_arr = None
        self._lngs_arr = None

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
        Build a deterministic synthetic road network grid for offline development,
        simulations, and fallback testing. Grid size 10x10 = 100 intersections/nodes.
        """
        G = nx.DiGraph()
        self.is_synthetic = True
        self.source_type = "SYNTHETIC SIMULATION"
        self.city_name = f"Synthetic {grid_size}x{grid_size} Grid"
        self.nodes_dict = {}
        self._invalidate_node_cache()
        
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
                    dist_km = self.haversine_distance(u_lat, u_lng, v_lat, v_lng)
                    cost_info = self.calculate_edge_cost(dist_km, default_speed, traffic_factor=1.0)
                    
                    G.add_edge(u, v,
                               distance=cost_info['distance_km'],
                               speed=cost_info['speed_kmh'],
                               travel_time=cost_info['travel_time_min'],
                               traffic_factor=cost_info['traffic_factor'],
                               operational_cost=cost_info['operational_cost'],
                               weight=cost_info['total_cost'],
                               name="Simulation Grid Road")
                    
        self.graph = G
        return G

    def load_osm_graph(self, place_name: str = "Ludhiana, Punjab, India",
                       cache_path: Optional[str] = None,
                       network_type: str = "drive",
                       force_synthetic: bool = False) -> nx.DiGraph:
        """
        Load a real OpenStreetMap road network graph from cache or Overpass/OSMnx.
        Ensures nodes and edges represent actual physical roads with real geographic coordinates.
        Falls back seamlessly to synthetic simulation graph if offline or unavailable.
        """
        if force_synthetic:
            return self.build_synthetic_grid()

        if cache_path is None:
            cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")

        # 1. Try loading from cache file
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                G = self._build_graph_from_osm_json(data, place_name)
                if G is not None and G.number_of_nodes() > 50:
                    self.graph = G
                    self.is_synthetic = False
                    self.source_type = "REAL OSM ROAD NETWORK"
                    self.city_name = place_name
                    self._invalidate_node_cache()
                    logger.info(f"Loaded REAL OSM ROAD NETWORK from cache: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    return G
            except Exception as e:
                logger.warning(f"Failed to load OSM cache ({e}), attempting live fetch...")

        # 2. Try live Overpass fetch
        try:
            import requests
            query = """[out:json][timeout:35];(
              way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](30.84,75.76,30.96,75.93);
            );
            out body;
            >;
            out skel qt;"""
            headers = {'User-Agent': 'TaxiRoute-RL/1.0 (academic project; contact: student@localhost)'}
            endpoints = [
                'https://overpass-api.de/api/interpreter',
                'https://overpass.kumi.systems/api/interpreter'
            ]
            for ep in endpoints:
                try:
                    resp = requests.post(ep, data={'data': query}, headers=headers, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        G = self._build_graph_from_osm_json(data, place_name)
                        if G is not None and G.number_of_nodes() > 50:
                            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                            with open(cache_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f)
                            self.graph = G
                            self.is_synthetic = False
                            self.source_type = "REAL OSM ROAD NETWORK"
                            self.city_name = place_name
                            self._invalidate_node_cache()
                            return G
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Live OSM fetch failed ({e})")

        # 3. Fallback to synthetic grid if live download also unavailable
        logger.warning("Real OSM road network unavailable. Falling back to SYNTHETIC SIMULATION.")
        return self.build_synthetic_grid()

    def _build_graph_from_osm_json(self, data: Dict[str, Any], place_name: str) -> Optional[nx.DiGraph]:
        """Construct a strongly connected NetworkX DiGraph from OpenStreetMap JSON data."""
        elements = data.get('elements', [])
        raw_nodes = {e['id']: (float(e['lat']), float(e['lon'])) for e in elements if e.get('type') == 'node'}
        ways = [e for e in elements if e.get('type') == 'way']
        if not raw_nodes or not ways:
            return None

        G = nx.DiGraph()
        for w in ways:
            w_nodes = w.get('nodes', [])
            tags = w.get('tags', {})
            hwy = tags.get('highway', 'road')
            oneway = tags.get('oneway', 'no') in ('yes', '1', 'true')
            maxspeed = tags.get('maxspeed', '40')
            name = tags.get('name', 'Road')
            try:
                speed = float(str(maxspeed).split()[0])
            except (ValueError, IndexError):
                speed = 50.0 if hwy in ('trunk', 'motorway', 'primary') else 35.0

            for i in range(len(w_nodes) - 1):
                u_id, v_id = w_nodes[i], w_nodes[i + 1]
                if u_id in raw_nodes and v_id in raw_nodes:
                    u_lat, u_lon = raw_nodes[u_id]
                    v_lat, v_lon = raw_nodes[v_id]
                    dist_km = self.haversine_distance(u_lat, u_lon, v_lat, v_lon)
                    if dist_km < 0.0001:
                        continue

                    if not G.has_node(u_id):
                        G.add_node(u_id, lat=u_lat, lng=u_lon)
                    if not G.has_node(v_id):
                        G.add_node(v_id, lat=v_lat, lng=v_lon)

                    cost_info = self.calculate_edge_cost(dist_km, speed, traffic_factor=1.0)
                    G.add_edge(u_id, v_id,
                               distance=cost_info['distance_km'],
                               speed=cost_info['speed_kmh'],
                               travel_time=cost_info['travel_time_min'],
                               traffic_factor=cost_info['traffic_factor'],
                               operational_cost=cost_info['operational_cost'],
                               weight=cost_info['total_cost'],
                               name=name)

                    if not oneway:
                        G.add_edge(v_id, u_id,
                                   distance=cost_info['distance_km'],
                                   speed=cost_info['speed_kmh'],
                                   travel_time=cost_info['travel_time_min'],
                                   traffic_factor=cost_info['traffic_factor'],
                                   operational_cost=cost_info['operational_cost'],
                                   weight=cost_info['total_cost'],
                                   name=name)

        if G.number_of_nodes() == 0:
            return None

        # Extract largest strongly connected component to guarantee path reachability
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G_scc = G.subgraph(largest_scc).copy()

        # Re-index to contiguous integer IDs (0 to N-1) for RL and heuristic routers
        mapping = {old_id: idx for idx, old_id in enumerate(G_scc.nodes())}
        final_G = nx.DiGraph()
        self.nodes_dict = {}

        for old_id, new_id in mapping.items():
            node_attr = G_scc.nodes[old_id]
            final_G.add_node(new_id, lat=node_attr['lat'], lng=node_attr['lng'])
            self.nodes_dict[new_id] = {'lat': node_attr['lat'], 'lng': node_attr['lng']}

        for u_old, v_old, data in G_scc.edges(data=True):
            final_G.add_edge(mapping[u_old], mapping[v_old], **data)

        return final_G

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
        """Find graph node ID nearest to given GPS latitude/longitude using vectorized calculation."""
        if not self.nodes_dict:
            raise ValueError("Graph nodes dictionary is empty!")

        # Initialize or update vectorized coordinates cache
        if self._node_ids_arr is None or len(self._node_ids_arr) != len(self.nodes_dict):
            self._node_ids_arr = np.array(list(self.nodes_dict.keys()), dtype=np.int64)
            self._lats_arr = np.array([c['lat'] for c in self.nodes_dict.values()], dtype=np.float64)
            self._lngs_arr = np.array([c['lng'] for c in self.nodes_dict.values()], dtype=np.float64)

        # Vectorized Haversine distance
        dlat = np.radians(self._lats_arr - lat)
        dlng = np.radians(self._lngs_arr - lng)
        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(np.radians(lat)) * np.cos(np.radians(self._lats_arr)) * np.sin(dlng / 2.0) ** 2)
        min_idx = int(np.argmin(a))
        return int(self._node_ids_arr[min_idx])

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in kilometers between two GPS points."""
        R = 6371.0  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2.0) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(R * c)

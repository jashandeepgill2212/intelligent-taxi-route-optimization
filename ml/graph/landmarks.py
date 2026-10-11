"""
ALT (A*, Landmarks, Triangle Inequality) Heuristic Module for Taxi Route Optimization.
Provides admissible, consistent shortest-path lower bounds on directed road graphs
using precomputed forward and backward distances from canonical landmarks.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import numpy as np


class ALTLandmarkHeuristic:
    """
    Computes directed shortest-path distance lower bounds using 16 canonical landmarks
    and the triangle inequality on directed road network graphs.
    
    For a directed graph with edge weights >= 0:
      Triangle inequality 1: d(L, t) <= d(L, v) + d(v, t)  ==>  d(v, t) >= d(L, t) - d(L, v)
      Triangle inequality 2: d(v, L) <= d(v, t) + d(t, L)  ==>  d(v, t) >= d(v, L) - d(t, L)
      
    Admissible lower bound:
      h_ALT(v, t) = max(0, max_L [d(L, t) - d(L, v)], max_L [d(v, L) - d(t, L)])
      where only finite distance pairs are evaluated.
    """

    # 16 Canonical Geographic & Topological Landmarks for Ludhiana OSM Road Network
    # Spatially distributed across northern, southern, eastern, western perimeters,
    # major highway corridors (NH44, Ferozepur Rd), and central arterial junctions.
    DEFAULT_LANDMARK_NODES = [
        # Peripheral Boundary Landmarks
        12066,  # North perimeter / Jalandhar Bypass corridor
        12149,  # South perimeter / Gill canal crossing
        11442,  # East perimeter / Samrala Chowk / Chandigarh Rd
        3619,   # West perimeter / Ferozepur Road / PAU boundary
        1198,   # South-Central / Gill Road arterial
        659,    # Central Rail / Ludhiana Junction station
        3858,   # City Center / Ferozepur Rd junction
        8585,   # Southwest perimeter
        792,    # Northeast perimeter
        409,    # Northwest boundary
        10476,  # Southeast industrial corridor
        1824,   # Southwest suburban junction
        1169,   # Clock Tower / Old City core
        9980,   # Model Town arterial junction
        9195,   # Dholewal Chowk / GT Road junction
        3257,   # Northern bypass flyover connection
    ]

    def __init__(
        self,
        graph: nx.DiGraph,
        landmarks: Optional[List[int]] = None,
        weight_attr: str = "distance",
        cache_path: Optional[str] = None,
        auto_load: bool = True,
    ):
        self.graph = graph
        self.weight_attr = weight_attr
        self.cache_path = cache_path or os.path.abspath(
            "ml/data/osm_cache/alt_landmarks_16.json"
        )
        self.landmarks: List[int] = []
        
        # Distance lookup tables: landmark -> dict of {node: distance}
        self.d_from_landmark: Dict[int, Dict[int, float]] = {}  # d(L, v)
        self.d_to_landmark: Dict[int, Dict[int, float]] = {}    # d(v, L)
        self.graph_checksum: str = self.compute_graph_checksum(graph)
        self.is_ready: bool = False

        if auto_load and graph is not None and graph.number_of_nodes() > 0:
            self.initialize_landmarks(landmarks)

    @staticmethod
    def compute_graph_checksum(graph: nx.DiGraph) -> str:
        """Compute SHA256 checksum based on graph topology and node/edge counts."""
        if graph is None or graph.number_of_nodes() == 0:
            return "empty_graph"
        hasher = hashlib.sha256()
        hasher.update(str(graph.number_of_nodes()).encode("utf-8"))
        hasher.update(str(graph.number_of_edges()).encode("utf-8"))
        # Sample deterministic nodes to ensure unique graph identification
        sorted_nodes = sorted(list(graph.nodes()))
        sample = sorted_nodes[:: max(1, len(sorted_nodes) // 50)]
        hasher.update(str(sample).encode("utf-8"))
        return hasher.hexdigest()

    def _select_canonical_landmarks(self, requested: Optional[List[int]]) -> List[int]:
        """Validate or select 16 canonical landmarks present in the graph."""
        available_nodes = set(self.graph.nodes())
        selected = []

        if requested:
            for n in requested:
                if n in available_nodes and n not in selected:
                    selected.append(n)

        # Fallback to default landmark set
        if len(selected) < 16:
            for n in self.DEFAULT_LANDMARK_NODES:
                if n in available_nodes and n not in selected:
                    selected.append(n)
                if len(selected) == 16:
                    break

        # If graph is synthetic or lacks default IDs, select uniformly distributed nodes
        if len(selected) < 16:
            all_sorted = sorted(list(available_nodes))
            step = max(1, len(all_sorted) // 16)
            for idx in range(0, len(all_sorted), step):
                node = all_sorted[idx]
                if node not in selected:
                    selected.append(node)
                if len(selected) == 16:
                    break

        return selected[:16]

    def initialize_landmarks(self, landmarks: Optional[List[int]] = None) -> bool:
        """Load precomputed distances from cache or compute and save."""
        self.landmarks = self._select_canonical_landmarks(landmarks)
        if not self.landmarks:
            self.is_ready = False
            return False

        # Attempt to load from valid cache
        if self._load_from_cache():
            self.is_ready = True
            return True

        # Precompute forward and backward Dijkstra shortest paths
        self._precompute_all_distances()
        self._save_to_cache()
        self.is_ready = True
        return True

    def _precompute_all_distances(self):
        """Compute all forward and backward shortest path distances for all landmarks."""
        rev_graph = self.graph.reverse(copy=False)
        self.d_from_landmark = {}
        self.d_to_landmark = {}

        for L in self.landmarks:
            # Forward distances from L to all reachable nodes: d(L, v)
            try:
                fwd_dists = nx.single_source_dijkstra_path_length(
                    self.graph, source=L, weight=self.weight_attr
                )
            except Exception:
                fwd_dists = {L: 0.0}
            self.d_from_landmark[L] = {int(k): float(v) for k, v in fwd_dists.items()}

            # Backward distances from all nodes to L: d(v, L)
            # Computed as forward Dijkstra on the reversed graph
            try:
                bwd_dists = nx.single_source_dijkstra_path_length(
                    rev_graph, source=L, weight=self.weight_attr
                )
            except Exception:
                bwd_dists = {L: 0.0}
            self.d_to_landmark[L] = {int(k): float(v) for k, v in bwd_dists.items()}

    def _save_to_cache(self) -> bool:
        """Save landmark distances and graph metadata to JSON cache."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            data = {
                "graph_checksum": self.graph_checksum,
                "weight_attr": self.weight_attr,
                "num_landmarks": len(self.landmarks),
                "landmarks": self.landmarks,
                "d_from_landmark": {
                    str(L): dists for L, dists in self.d_from_landmark.items()
                },
                "d_to_landmark": {
                    str(L): dists for L, dists in self.d_to_landmark.items()
                },
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception:
            return False

    def _load_from_cache(self) -> bool:
        """Load landmark distances if cache exists and checksum matches."""
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("graph_checksum") != self.graph_checksum:
                return False
            if data.get("weight_attr") != self.weight_attr:
                return False
            cached_landmarks = data.get("landmarks", [])
            if sorted(cached_landmarks) != sorted(self.landmarks):
                return False

            self.landmarks = cached_landmarks
            self.d_from_landmark = {
                int(L): {int(k): float(v) for k, v in dists.items()}
                for L, dists in data.get("d_from_landmark", {}).items()
            }
            self.d_to_landmark = {
                int(L): {int(k): float(v) for k, v in dists.items()}
                for L, dists in data.get("d_to_landmark", {}).items()
            }
            return True
        except Exception:
            return False

    def get_lower_bound(self, current_node: int, dest_node: int) -> float:
        """
        Compute the directed ALT distance lower bound between current_node and dest_node.
        Guarantees:
        1. h_ALT(v, t) == 0.0 when v == t.
        2. h_ALT(v, t) >= 0.0 for all v, t (never negative).
        3. Never returns NaN or +/- infinity.
        4. Handles unreachable landmarks and nodes gracefully.
        """
        if current_node == dest_node:
            return 0.0

        if not self.is_ready or not self.landmarks:
            return 0.0

        best_bound = 0.0

        for L in self.landmarks:
            # Bound 1: d(L, dest) - d(L, current)
            fwd_map = self.d_from_landmark.get(L, {})
            d_L_dest = fwd_map.get(dest_node)
            d_L_curr = fwd_map.get(current_node)
            if d_L_dest is not None and d_L_curr is not None:
                b1 = d_L_dest - d_L_curr
                if b1 > best_bound:
                    best_bound = b1

            # Bound 2: d(current, L) - d(dest, L)
            bwd_map = self.d_to_landmark.get(L, {})
            d_curr_L = bwd_map.get(current_node)
            d_dest_L = bwd_map.get(dest_node)
            if d_curr_L is not None and d_dest_L is not None:
                b2 = d_curr_L - d_dest_L
                if b2 > best_bound:
                    best_bound = b2

        # Sanity check: finite, non-negative
        if not np.isfinite(best_bound) or best_bound < 0.0:
            return 0.0

        return float(best_bound)

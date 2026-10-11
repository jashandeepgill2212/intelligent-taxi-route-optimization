"""
Held-out benchmark dataset of 30 reproducible, reachable origin-destination pairs
on the Ludhiana OpenStreetMap road network.
Used strictly for validation and testing; never sampled during DQN training episodes.

All direct straight-line distances (direct_dist_km) are verified against the canonical
OSM node coordinates via the Haversine formula on the Earth sphere (R=6371.0 km).
"""

import math
from typing import List, Dict, Any, Tuple, Optional


def compute_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute straight-line Haversine distance in kilometers between two lat/lng coordinates."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


BENCHMARK_PAIRS: List[Dict[str, Any]] = [
    {"id": 1, "name": "Dashboard: Gill Rd -> Ludhiana Junction", "source_node": 1198, "dest_node": 659, "category": "Cross-City / Arterial", "direct_dist_km": 10.68},
    {"id": 2, "name": "Ludhiana City Center -> Ludhiana Junction", "source_node": 3858, "dest_node": 659, "category": "Short / Urban Core", "direct_dist_km": 4.80},
    {"id": 3, "name": "PAU Ludhiana -> Ludhiana Main Bus Stand", "source_node": 12105, "dest_node": 10233, "category": "Cross-Railway East-West", "direct_dist_km": 4.48},
    {"id": 4, "name": "Sarabha Nagar -> Ludhiana City Center", "source_node": 10531, "dest_node": 3858, "category": "Medium / West-to-Center", "direct_dist_km": 4.55},
    {"id": 5, "name": "Civil Lines -> Ludhiana Main Bus Stand", "source_node": 4209, "dest_node": 10233, "category": "Medium / North-to-South", "direct_dist_km": 2.43},
    {"id": 6, "name": "Jalandhar Bypass -> Ludhiana City Center", "source_node": 8867, "dest_node": 3858, "category": "Medium / North Radial", "direct_dist_km": 5.15},
    {"id": 7, "name": "Dugri Phase 1 -> Ludhiana Junction", "source_node": 6687, "dest_node": 659, "category": "Long / South-to-North", "direct_dist_km": 8.82},
    {"id": 8, "name": "Focal Point Phase 5 -> PAU Ludhiana", "source_node": 8595, "dest_node": 12105, "category": "Long / East-to-West", "direct_dist_km": 9.84},
    {"id": 9, "name": "BRS Nagar -> Samrala Chowk", "source_node": 4756, "dest_node": 9426, "category": "Long / Cross-City SW-NE", "direct_dist_km": 8.67},
    {"id": 10, "name": "Haibowal Kalan -> Sherpur Chowk", "source_node": 2874, "dest_node": 740, "category": "Long / Cross-City NW-SE", "direct_dist_km": 7.82},
    {"id": 11, "name": "Civil Lines -> PAU Ludhiana", "source_node": 4209, "dest_node": 12105, "category": "Short / West Corridor", "direct_dist_km": 2.73},
    {"id": 12, "name": "Samrala Chowk -> Ludhiana Main Bus Stand", "source_node": 9426, "dest_node": 10233, "category": "Short / East Urban", "direct_dist_km": 2.43},
    {"id": 13, "name": "Random Test Pair 1 (3.3 km)", "source_node": 9308, "dest_node": 9170, "category": "Medium (3-7 km)", "direct_dist_km": 3.27},
    {"id": 14, "name": "Random Test Pair 2 (7.0 km)", "source_node": 732, "dest_node": 4893, "category": "Medium (3-7 km)", "direct_dist_km": 6.97},
    {"id": 15, "name": "Random Test Pair 3 (8.8 km)", "source_node": 10565, "dest_node": 9481, "category": "Long (>7 km)", "direct_dist_km": 8.78},
    {"id": 16, "name": "Random Test Pair 4 (6.2 km)", "source_node": 4353, "dest_node": 4627, "category": "Medium (3-7 km)", "direct_dist_km": 6.22},
    {"id": 17, "name": "Random Test Pair 5 (14.2 km)", "source_node": 8393, "dest_node": 3449, "category": "Long (>7 km)", "direct_dist_km": 14.20},
    {"id": 18, "name": "Random Test Pair 6 (7.4 km)", "source_node": 7249, "dest_node": 6867, "category": "Long (>7 km)", "direct_dist_km": 7.41},
    {"id": 19, "name": "Random Test Pair 7 (12.2 km)", "source_node": 6926, "dest_node": 6393, "category": "Long (>7 km)", "direct_dist_km": 12.22},
    {"id": 20, "name": "Random Test Pair 8 (3.8 km)", "source_node": 8585, "dest_node": 792, "category": "Medium (3-7 km)", "direct_dist_km": 3.82},
    {"id": 21, "name": "Random Test Pair 9 (18.1 km)", "source_node": 10515, "dest_node": 8839, "category": "Long (>7 km)", "direct_dist_km": 18.15},
    {"id": 22, "name": "Random Test Pair 10 (2.8 km)", "source_node": 1032, "dest_node": 11529, "category": "Short (<3 km)", "direct_dist_km": 2.75},
    {"id": 23, "name": "Random Test Pair 11 (8.7 km)", "source_node": 5111, "dest_node": 11210, "category": "Long (>7 km)", "direct_dist_km": 8.65},
    {"id": 24, "name": "Random Test Pair 12 (1.7 km)", "source_node": 8555, "dest_node": 12282, "category": "Short (<3 km)", "direct_dist_km": 1.73},
    {"id": 25, "name": "Random Test Pair 13 (4.9 km)", "source_node": 11776, "dest_node": 10385, "category": "Medium (3-7 km)", "direct_dist_km": 4.92},
    {"id": 26, "name": "Random Test Pair 14 (15.3 km)", "source_node": 11477, "dest_node": 8802, "category": "Long (>7 km)", "direct_dist_km": 15.26},
    {"id": 27, "name": "Random Test Pair 15 (6.6 km)", "source_node": 4545, "dest_node": 3280, "category": "Medium (3-7 km)", "direct_dist_km": 6.64},
    {"id": 28, "name": "Random Test Pair 16 (7.1 km)", "source_node": 9323, "dest_node": 6280, "category": "Long (>7 km)", "direct_dist_km": 7.11},
    {"id": 29, "name": "Random Test Pair 17 (5.2 km)", "source_node": 3619, "dest_node": 11442, "category": "Medium (3-7 km)", "direct_dist_km": 5.18},
    {"id": 30, "name": "Random Test Pair 18 (11.1 km)", "source_node": 4562, "dest_node": 2664, "category": "Long (>7 km)", "direct_dist_km": 11.10},
]

HELD_OUT_NODE_PAIRS = set((p["source_node"], p["dest_node"]) for p in BENCHMARK_PAIRS)

VALIDATION_PAIRS: List[Dict[str, Any]] = [
    {"id": 101, "name": "Val 1 (Short 1.1km)", "source_node": 12066, "dest_node": 1679, "category": "Validation / Short", "direct_dist_km": 1.06},
    {"id": 102, "name": "Val 2 (Short 2.6km)", "source_node": 1169, "dest_node": 9980, "category": "Validation / Short", "direct_dist_km": 2.56},
    {"id": 103, "name": "Val 3 (Short 2.9km)", "source_node": 4304, "dest_node": 12239, "category": "Validation / Short", "direct_dist_km": 2.90},
    {"id": 104, "name": "Val 4 (Medium 3.9km)", "source_node": 4506, "dest_node": 4012, "category": "Validation / Medium", "direct_dist_km": 3.85},
    {"id": 105, "name": "Val 5 (Medium 5.3km)", "source_node": 8935, "dest_node": 1424, "category": "Validation / Medium", "direct_dist_km": 5.26},
    {"id": 106, "name": "Val 6 (Medium 5.3km)", "source_node": 9195, "dest_node": 3257, "category": "Validation / Medium", "direct_dist_km": 5.33},
    {"id": 107, "name": "Val 7 (Long 8.6km)", "source_node": 10476, "dest_node": 1824, "category": "Validation / Long", "direct_dist_km": 8.58},
    {"id": 108, "name": "Val 8 (Long 9.8km)", "source_node": 409, "dest_node": 12149, "category": "Validation / Long", "direct_dist_km": 9.77},
]

ALL_EXCLUDED_TRAIN_PAIRS = HELD_OUT_NODE_PAIRS | set((p["source_node"], p["dest_node"]) for p in VALIDATION_PAIRS)


def calculate_pair_haversine(pair: Dict[str, Any], graph: Any) -> float:
    """Calculate the true straight-line distance in km between the endpoints of a pair using graph node coordinates."""
    s, d = pair["source_node"], pair["dest_node"]
    if s not in graph.nodes or d not in graph.nodes:
        raise KeyError(f"Endpoints ({s}, {d}) not found in graph.")
    s_lat = float(graph.nodes[s]["lat"])
    s_lng = float(graph.nodes[s]["lng"])
    d_lat = float(graph.nodes[d]["lat"])
    d_lng = float(graph.nodes[d]["lng"])
    return round(compute_haversine_km(s_lat, s_lng, d_lat, d_lng), 2)


def validate_benchmark_integrity(pairs: List[Dict[str, Any]], graph: Any) -> List[Dict[str, Any]]:
    """
    Validate that every pair exists in graph, has valid coordinates,
    and stored direct_dist_km matches calculate_pair_haversine within tolerance.
    """
    errors = []
    for p in pairs:
        s, d = p["source_node"], p["dest_node"]
        if s not in graph.nodes:
            errors.append({"id": p["id"], "error": f"Source node {s} not in graph"})
            continue
        if d not in graph.nodes:
            errors.append({"id": p["id"], "error": f"Dest node {d} not in graph"})
            continue
        calc_dist = calculate_pair_haversine(p, graph)
        diff = abs(p["direct_dist_km"] - calc_dist)
        if diff > 0.05:
            errors.append({
                "id": p["id"],
                "error": f"Stored distance ({p['direct_dist_km']:.2f} km) differs from calculated ({calc_dist:.2f} km) by {diff:.2f} km"
            })
    return errors

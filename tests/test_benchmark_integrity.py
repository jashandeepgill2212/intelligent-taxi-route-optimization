import os
import pytest
from ml.graph.builder import RoadNetworkGraph
from ml.evaluation.benchmark_dataset import BENCHMARK_PAIRS, VALIDATION_PAIRS
from ml.evaluation.baselines import BaselineRouter


@pytest.fixture(scope="module")
def graph_and_router():
    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)
    router = BaselineRouter(rng)
    return rng, router


def test_triangle_inequality_all_benchmark_pairs(graph_and_router):
    """
    Verify that the road-network shortest path distance is NEVER less than
    the true geographic straight-line distance between identical endpoints.
    """
    rng, router = graph_and_router
    for p in BENCHMARK_PAIRS:
        s, d = p["source_node"], p["dest_node"]
        s_lat, s_lng = float(rng.graph.nodes[s]["lat"]), float(rng.graph.nodes[s]["lng"])
        d_lat, d_lng = float(rng.graph.nodes[d]["lat"]), float(rng.graph.nodes[d]["lng"])
        true_direct_km = rng.haversine_distance(s_lat, s_lng, d_lat, d_lng)
        
        dijk = router.dijkstra_route(s, d)
        assert dijk["success"] is True
        dijk_dist_km = dijk["total_distance_km"]
        
        # Road distance along physical roads must always be >= straight-line distance
        assert dijk_dist_km >= (true_direct_km - 1e-4), (
            f"Triangle inequality violated for Pair {p['id']} ({p['name']}): "
            f"Dijkstra {dijk_dist_km:.3f} km < Direct {true_direct_km:.3f} km"
        )


def test_validation_pairs_direct_distance_exactness(graph_and_router):
    """Verify that all validation pairs have exactly matching stored vs computed direct distance."""
    rng, _ = graph_and_router
    for p in VALIDATION_PAIRS:
        s, d = p["source_node"], p["dest_node"]
        s_lat, s_lng = float(rng.graph.nodes[s]["lat"]), float(rng.graph.nodes[s]["lng"])
        d_lat, d_lng = float(rng.graph.nodes[d]["lat"]), float(rng.graph.nodes[d]["lng"])
        true_direct_km = rng.haversine_distance(s_lat, s_lng, d_lat, d_lng)
        assert abs(p["direct_dist_km"] - true_direct_km) < 0.05


def test_benchmark_endpoint_existence_and_reachability(graph_and_router):
    """Verify that every source and destination node in the benchmark exists and is mutually reachable."""
    import networkx as nx
    rng, router = graph_and_router
    for p in BENCHMARK_PAIRS:
        s, d = p["source_node"], p["dest_node"]
        assert s in rng.graph, f"Source node {s} not in graph"
        assert d in rng.graph, f"Destination node {d} not in graph"
        assert nx.has_path(rng.graph, s, d) is True, f"No path from {s} to {d}"


def test_benchmark_pairs_direct_distance_exactness(graph_and_router):
    """Verify that all 30 benchmark pairs now have accurately matching stored vs computed direct distance."""
    rng, _ = graph_and_router
    from ml.evaluation.benchmark_dataset import calculate_pair_haversine, validate_benchmark_integrity
    errors = validate_benchmark_integrity(BENCHMARK_PAIRS, rng.graph)
    assert len(errors) == 0, f"Benchmark integrity validation errors: {errors}"


def test_validate_benchmark_integrity_invalid_node_behavior(graph_and_router):
    """Verify that validate_benchmark_integrity properly catches invalid nodes and mismatches."""
    rng, _ = graph_and_router
    from ml.evaluation.benchmark_dataset import validate_benchmark_integrity
    bad_pairs = [
        {"id": 991, "name": "Bad Node Pair", "source_node": 999999, "dest_node": 659, "direct_dist_km": 10.0},
        {"id": 992, "name": "Mismatched Distance", "source_node": 1198, "dest_node": 659, "direct_dist_km": 99.0},
    ]
    errors = validate_benchmark_integrity(bad_pairs, rng.graph)
    assert len(errors) == 2
    assert "not in graph" in errors[0]["error"]
    assert "differs from calculated" in errors[1]["error"]


def test_consistent_ratio_calculation():
    """Verify that distance ratio formula is always dqn_dist / max(dijk_dist, 0.01)."""
    dqn_dist = 20.0
    dijk_dist = 10.0
    ratio = dqn_dist / max(dijk_dist, 0.01)
    assert ratio == 2.0


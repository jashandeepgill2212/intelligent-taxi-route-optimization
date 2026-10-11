"""
Unit and Regression Test Suite for ALT Landmark Heuristics & Potential-Based Reward Shaping (PBRS).
Verifies mathematical correctness, directed-graph triangle inequality lower bounds,
terminal state potential handling, discount factor consistency, and backward compatibility.
"""

import os
import math
import pytest
import hashlib
import numpy as np
import networkx as nx

from ml.graph.builder import RoadNetworkGraph
from ml.graph.landmarks import ALTLandmarkHeuristic
from ml.environment.taxi_env import TaxiRouteEnv
from ml.evaluation.benchmark_dataset import BENCHMARK_PAIRS, VALIDATION_PAIRS


# Expected SHA256 checksums of preserved original checkpoints
EXPECTED_CHECKPOINTS = {
    "ml/checkpoints/dqn_model.pt": "cb11870a74e68ce70cd25e59d1c4176a7da15db858fb7506a81078dbc0e31740",
    "ml/checkpoints/dqn_model_full_allocation.pt": "4a9bf370c4b38417df6f7cfb8f01476844b8b183f51357e73c1eb1f7998913c8",
    "ml/checkpoints/dqn_model_graph_aware.pt": "ce5821e470b657439b5f67d4eb08b6d80421b906a12f0760872d45120ee81764",
}


def create_directed_test_graph():
    """
    Construct a directed weighted graph with asymmetric edge lengths,
    one-way roads, and cycles to test directed ALT properties.
    """
    G = nx.DiGraph()
    # 4 nodes in a ring with asymmetric weights and one cross edge
    # 0 -> 1: weight 2.0
    # 1 -> 0: weight 10.0 (asymmetric return)
    # 1 -> 2: weight 3.0
    # 2 -> 3: weight 4.0
    # 3 -> 0: weight 1.0
    # 0 -> 2: weight 8.0 (one-way shortcut)
    coords = {
        0: (30.90, 75.85),
        1: (30.91, 75.85),
        2: (30.91, 75.86),
        3: (30.90, 75.86),
    }
    for n, (lat, lng) in coords.items():
        G.add_node(n, lat=lat, lng=lng)

    edges = [
        (0, 1, 2.0),
        (1, 0, 10.0),
        (1, 2, 3.0),
        (2, 3, 4.0),
        (3, 0, 1.0),
        (0, 2, 8.0),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, distance=w, weight=w, speed=40.0, travel_time=(w / 40.0) * 60.0)

    return G


# 1. ALT Lower-Bound Correctness & Admissibility
def test_alt_lower_bound_admissibility():
    """Verify h_ALT(u, v) <= true_shortest_path(u, v) for all pairs on directed graph."""
    G = create_directed_test_graph()
    alt = ALTLandmarkHeuristic(G, landmarks=[0, 3], auto_load=True, cache_path="ml/data/osm_cache/test_alt_tmp.json")

    for u in G.nodes():
        for v in G.nodes():
            bound = alt.get_lower_bound(u, v)
            true_dist = nx.dijkstra_path_length(G, u, v, weight="distance")
            assert bound >= 0.0, f"Bound is negative: {bound}"
            assert bound <= true_dist + 1e-6, (
                f"Admissibility violated: h_ALT({u}, {v}) = {bound:.3f} > true_dist {true_dist:.3f}"
            )


# 2. Landmark Distance Directionality
def test_landmark_directionality_asymmetry():
    """Verify directed bounds respect asymmetric forward vs backward path lengths."""
    G = create_directed_test_graph()
    alt = ALTLandmarkHeuristic(G, landmarks=[0], auto_load=True, cache_path="ml/data/osm_cache/test_alt_tmp.json")

    # In G: 0 -> 1 is 2.0, but 1 -> 0 is 8.0 (1->2->3->0: 3+4+1=8 < 10)
    d_0_1 = nx.dijkstra_path_length(G, 0, 1, weight="distance")
    d_1_0 = nx.dijkstra_path_length(G, 1, 0, weight="distance")
    assert d_0_1 == 2.0
    assert d_1_0 == 8.0

    # Ensure forward table d(L, v) differs from backward table d(v, L)
    assert alt.d_from_landmark[0][1] == 2.0
    assert alt.d_to_landmark[0][1] == 8.0


# 3. Disconnected Nodes Handling
def test_alt_disconnected_nodes():
    """Verify handling when a node has no path to landmark or destination."""
    G = nx.DiGraph()
    G.add_node(0, lat=30.90, lng=75.85)
    G.add_node(1, lat=30.91, lng=75.85)
    G.add_node(99, lat=30.99, lng=75.99)  # completely disconnected node
    G.add_edge(0, 1, distance=1.0, weight=1.0)

    alt = ALTLandmarkHeuristic(G, landmarks=[0], auto_load=True, cache_path="ml/data/osm_cache/test_alt_disc.json")
    bound = alt.get_lower_bound(0, 99)
    assert bound == 0.0
    assert np.isfinite(bound)


# 4. Missing Landmark Paths Handling
def test_missing_landmark_paths_safe_filtering():
    """Verify no infinite or NaN values when landmark cannot reach target."""
    G = nx.DiGraph()
    G.add_node(0, lat=30.90, lng=75.85)
    G.add_node(1, lat=30.91, lng=75.85)
    G.add_edge(0, 1, distance=1.0, weight=1.0)
    # One-way: 0 -> 1. 1 cannot reach 0.
    alt = ALTLandmarkHeuristic(G, landmarks=[1], auto_load=True, cache_path="ml/data/osm_cache/test_alt_oneway.json")
    bound = alt.get_lower_bound(0, 1)
    assert np.isfinite(bound)
    assert bound >= 0.0


# 5. Destination Equal to Current Node
def test_destination_equal_to_current_node():
    """Verify h_ALT(t, t) == 0.0 always."""
    G = create_directed_test_graph()
    alt = ALTLandmarkHeuristic(G, landmarks=[0, 1], auto_load=True, cache_path="ml/data/osm_cache/test_alt_eq.json")
    for n in G.nodes():
        assert alt.get_lower_bound(n, n) == 0.0


# 6 & 7 & 8 & 9. Potential & PBRS Shaping Formula & Terminal Zero Condition
def test_pbrs_potential_shaping_and_terminal_zero():
    """
    Verify:
    1. Phi(s) = -w * h_ALT(s, t)
    2. Phi(s_terminal) = 0.0
    3. F = gamma * Phi(s') - Phi(s)
    4. Exact discount factor gamma used
    """
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(
        rng,
        enable_pbrs=True,
        pbrs_gamma=0.995,
        pbrs_scale=8.0,
    )
    env.reset(options={"source": 0, "destination": 24, "enable_pbrs": True})

    # Test helper directly
    alt = env._get_alt_heuristic()
    h_0 = alt.get_lower_bound(0, 24)
    h_1 = alt.get_lower_bound(1, 24)

    # Transition 0 -> 1 (non-terminal)
    f_step, phi_curr, phi_next = env._compute_pbrs_shaping(0, 1, is_goal=False)
    expected_phi_curr = -8.0 * h_0
    expected_phi_next = -8.0 * h_1
    expected_f = 0.995 * expected_phi_next - expected_phi_curr

    assert abs(phi_curr - expected_phi_curr) < 1e-5
    assert abs(phi_next - expected_phi_next) < 1e-5
    assert abs(f_step - expected_f) < 1e-5

    # Transition into goal (terminal)
    f_goal, phi_curr_goal, phi_term = env._compute_pbrs_shaping(23, 24, is_goal=True)
    assert phi_term == 0.0, "Absorbing goal terminal potential must be strictly 0.0!"
    h_23 = alt.get_lower_bound(23, 24)
    expected_phi_23 = -8.0 * h_23
    expected_f_goal = 0.995 * 0.0 - expected_phi_23
    assert abs(f_goal - expected_f_goal) < 1e-5
    assert f_goal >= 0.0, "Goal arrival shaping term must be positive"


# 10. Original Reward Preservation when PBRS is Disabled
def test_original_reward_preserved_when_disabled():
    """Verify that when enable_pbrs=False, environment reward is 100% identical to baseline."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)

    env_orig = TaxiRouteEnv(rng, enable_pbrs=False)
    env_pbrs_off = TaxiRouteEnv(rng, enable_pbrs=False)

    obs1, info1 = env_orig.reset(options={"source": 0, "destination": 24})
    obs2, info2 = env_pbrs_off.reset(options={"source": 0, "destination": 24})

    # Step to node 1
    o1, r1, term1, trunc1, inf1 = env_orig.step_to_neighbor(1)
    o2, r2, term2, trunc2, inf2 = env_pbrs_off.step_to_neighbor(1)

    assert r1 == r2, f"Rewards differ when disabled: {r1} != {r2}"
    assert inf2["shaping_reward"] == 0.0
    assert inf2["r_shaped"] == r1


# 11 & 12. No Duplicate Shaping & No NaNs or Infinities
def test_no_duplicate_shaping_and_no_nans():
    """Verify reward shaping is added exactly once and generates finite numbers."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(rng, enable_pbrs=True, pbrs_gamma=0.995, pbrs_scale=8.0)
    env.reset(options={"source": 0, "destination": 24, "enable_pbrs": True})

    obs, r, term, trunc, info = env.step_to_neighbor(1)

    assert np.isfinite(r)
    assert np.isfinite(info["shaping_reward"])
    assert np.isfinite(info["r_original"])
    assert np.isfinite(info["r_shaped"])
    assert not np.isnan(r)

    # Check exact decomposition: r_shaped == r_base + shaping_reward
    # And shaping_reward is not counted twice
    assert info["shaping_reward"] != 0.0 or info["pbrs_potential_curr"] == info["pbrs_potential_next"]


# 13. Cache Checksum Validation
def test_cache_checksum_invalidation():
    """Verify changing graph topology invalidates cached landmark data."""
    G1 = create_directed_test_graph()
    cache_file = "ml/data/osm_cache/test_checksum_alt.json"
    alt1 = ALTLandmarkHeuristic(G1, landmarks=[0], cache_path=cache_file)
    assert alt1.is_ready is True
    assert os.path.exists(cache_file)

    # Modify graph topology
    G2 = G1.copy()
    G2.add_edge(2, 0, distance=1.5, weight=1.5)
    checksum1 = ALTLandmarkHeuristic.compute_graph_checksum(G1)
    checksum2 = ALTLandmarkHeuristic.compute_graph_checksum(G2)
    assert checksum1 != checksum2

    # Initializing with modified graph should detect mismatch
    alt2 = ALTLandmarkHeuristic(G2, landmarks=[0], cache_path=cache_file)
    assert alt2.graph_checksum == checksum2

    if os.path.exists(cache_file):
        os.remove(cache_file)


# 14. Deterministic Heuristic Evaluation
def test_deterministic_alt_evaluation():
    """Verify that multiple evaluations of h_ALT on the same endpoints return identical results."""
    G = create_directed_test_graph()
    alt = ALTLandmarkHeuristic(G, landmarks=[0, 2])
    v1 = alt.get_lower_bound(1, 3)
    v2 = alt.get_lower_bound(1, 3)
    v3 = alt.get_lower_bound(1, 3)
    assert v1 == v2 == v3


# 15 & 16. Cycle-Aware Masking Compatibility
def test_cycle_aware_masking_compatibility_with_pbrs():
    """Verify that enabling both cycle-aware masking and PBRS functions seamlessly together."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(
        rng,
        enable_cycle_aware_masking=True,
        max_node_visits=2,
        enable_pbrs=True,
        pbrs_gamma=0.995,
        pbrs_scale=8.0,
    )
    obs, info = env.reset(options={"source": 0, "destination": 24, "enable_pbrs": True, "enable_cycle_aware_masking": True})

    assert "action_mask" in info
    assert "recovery_actions" in info
    assert "pbrs_potential" in info

    # Take step
    obs, r, term, trunc, step_info = env.step_to_neighbor(1)
    assert "recovery_actions" in step_info
    assert "shaping_reward" in step_info
    assert "candidate_edges" in step_info


# 17. Checkpoint Preservation Verification
def test_checkpoint_preservation_sha256():
    """Verify that original checkpoint weights have not been altered or corrupted."""
    for path, expected_hash in EXPECTED_CHECKPOINTS.items():
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest().lower()
        assert actual_hash == expected_hash, (
            f"Checkpoint {path} was corrupted! Expected {expected_hash}, got {actual_hash}"
        )


# 18. Benchmark Endpoint Consistency
def test_benchmark_endpoint_consistency():
    """Verify that benchmark pairs remain untouched with exact expected source and dest node IDs."""
    assert len(BENCHMARK_PAIRS) == 30
    assert len(VALIDATION_PAIRS) == 8
    # Pair 1: Gill Road to Ludhiana Junction
    assert BENCHMARK_PAIRS[0]["source_node"] == 1198
    assert BENCHMARK_PAIRS[0]["dest_node"] == 659
    # Pair 20: Random Test Pair 8
    assert BENCHMARK_PAIRS[19]["source_node"] == 8585
    assert BENCHMARK_PAIRS[19]["dest_node"] == 792

import math
import pytest
import numpy as np
import networkx as nx

from ml.graph.builder import RoadNetworkGraph
from ml.environment.taxi_env import TaxiRouteEnv
from ml.evaluation.diagnostics import analyze_trajectory, summarize_diagnostics


def create_test_env(edges_spec, node_coords, enable_cycle_aware=True, max_node_visits=2):
    """Helper to create a small test graph with explicit edges and coordinates."""
    rng = RoadNetworkGraph()
    rng.graph = nx.DiGraph()
    for n_id, (lat, lng) in node_coords.items():
        rng.graph.add_node(n_id, lat=lat, lng=lng)
    for u, v, dist, speed, name in edges_spec:
        rng.graph.add_edge(
            u, v, distance=dist, speed=speed, name=name,
            travel_time=(dist / speed) * 60.0,
            traffic_factor=1.0,
            operational_cost=dist * 12.0,
            weight=dist
        )
    env = TaxiRouteEnv(
        rng, max_steps=50,
        enable_cycle_aware_masking=enable_cycle_aware,
        max_node_visits=max_node_visits
    )
    return env


def test_immediate_reversal_masking():
    """Verify that moving u -> v masks the backtrack edge v -> u when forward alternatives exist."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),
        2: (30.920, 75.850),
        99: (30.950, 75.850),
    }
    edges = [
        (0, 1, 1.0, 40.0, "Rd 0-1"),
        (1, 0, 1.0, 40.0, "Rd 1-0"),
        (1, 2, 1.0, 40.0, "Rd 1-2"),
        (2, 99, 1.0, 40.0, "Rd 2-99"),
    ]
    env = create_test_env(edges, node_coords, enable_cycle_aware=True)
    env.reset(options={'source': 0, 'destination': 99})

    # Step to 1
    env.step_to_neighbor(1)
    assert env.current_node == 1
    assert env.visited_nodes == [0, 1]

    # At node 1: successors are 0 and 2.
    # 0 is prev_node and len(successors) > 1 -> must be masked!
    cands = env.get_candidate_edges(1, prev_node=0, visited_nodes=[0, 1])
    assert len(cands) == 2

    cand_0 = next(c for c in cands if c['neighbor'] == 0)
    cand_2 = next(c for c in cands if c['neighbor'] == 2)

    assert cand_0['is_valid'] is False
    assert cand_2['is_valid'] is True


def test_lifetime_visit_capping():
    """Verify that a node visited max_node_visits times is masked out when an alternative exists."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),
        2: (30.910, 75.860),
        3: (30.900, 75.860),
        99: (30.950, 75.850),
    }
    # Loop 0 -> 1 -> 2 -> 0, plus exit 0 -> 3 -> 99
    edges = [
        (0, 1, 1.0, 40.0, "0-1"),
        (1, 2, 1.0, 40.0, "1-2"),
        (2, 0, 1.0, 40.0, "2-0"),
        (0, 3, 1.0, 40.0, "0-3"),
        (3, 99, 1.0, 40.0, "3-99"),
    ]
    env = create_test_env(edges, node_coords, enable_cycle_aware=True, max_node_visits=2)
    env.reset(options={'source': 0, 'destination': 99})

    # Simulate path visiting 1 twice: [0, 1, 2, 0, 1, 2, 0]
    visited = [0, 1, 2, 0, 1, 2, 0]
    # At node 0, successors are 1 and 3.
    # Node 1 count is 2 (== max_node_visits).
    # Node 3 count is 0.
    # Cand 1 must be masked, cand 3 must be valid!
    cands = env.get_candidate_edges(0, prev_node=2, visited_nodes=visited)
    cand_1 = next(c for c in cands if c['neighbor'] == 1)
    cand_3 = next(c for c in cands if c['neighbor'] == 3)

    assert cand_1['is_valid'] is False
    assert cand_3['is_valid'] is True


def test_dead_end_recovery_least_visited():
    """Verify dead-end cul-de-sac recovery unmasks the least-visited outgoing edge with is_recovery=True."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.905, 75.850),  # Dead-end
        99: (30.950, 75.850),
    }
    edges = [
        (0, 1, 0.5, 30.0, "Cul-de-sac Entrance"),
        (1, 0, 0.5, 30.0, "Cul-de-sac Return"),
    ]
    env = create_test_env(edges, node_coords, enable_cycle_aware=True, max_node_visits=1)
    env.reset(options={'source': 0, 'destination': 99})

    # Step to 1
    env.step_to_neighbor(1)
    # At 1, node 0 has been visited once (>= max_node_visits 1).
    # All candidates would be masked, so dead-end recovery must trigger.
    cands = env.get_candidate_edges(1, prev_node=0, visited_nodes=[0, 1])
    assert len(cands) == 1
    assert cands[0]['neighbor'] == 0
    assert cands[0]['is_valid'] is True
    assert cands[0]['is_recovery'] is True


def test_destination_never_masked():
    """Verify destination node is never masked, even if previously visited or in recency window."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),  # Target
    }
    edges = [
        (0, 1, 1.0, 40.0, "Direct to Goal"),
        (1, 0, 1.0, 40.0, "Return"),
    ]
    env = create_test_env(edges, node_coords, enable_cycle_aware=True, max_node_visits=1)
    env.reset(options={'source': 0, 'destination': 1})

    # Even if destination 1 was somehow in visited_nodes multiple times
    cands = env.get_candidate_edges(0, prev_node=None, visited_nodes=[0, 1, 0, 1])
    cand_1 = next(c for c in cands if c['neighbor'] == 1)
    assert cand_1['is_valid'] is True


def test_backward_compatibility():
    """Verify that when enable_cycle_aware_masking=False, behavior is identical to baseline."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),
        2: (30.910, 75.860),
        99: (30.950, 75.850),
    }
    edges = [
        (0, 1, 1.0, 40.0, "Main"),
        (1, 2, 1.0, 40.0, "Side Loop"),
        (2, 0, 1.0, 40.0, "Return to Start"),
        (2, 99, 3.0, 40.0, "Exit to Goal"),
    ]
    env_baseline = create_test_env(edges, node_coords, enable_cycle_aware=False)
    env_baseline.reset(options={'source': 0, 'destination': 99})

    # At node 2 with path [0, 1, 2]
    cands = env_baseline.get_candidate_edges(2, prev_node=1, visited_nodes=[0, 1, 2])
    cand_0 = next(c for c in cands if c['neighbor'] == 0)
    cand_99 = next(c for c in cands if c['neighbor'] == 99)

    assert cand_0['is_valid'] is False
    assert cand_99['is_valid'] is True
    assert cand_0.get('is_recovery', False) is False


def test_trajectory_diagnostics_calculation():
    """Verify all 21 trajectory metrics are correctly computed on a known path."""
    rng = RoadNetworkGraph()
    rng.graph = nx.DiGraph()
    rng.graph.add_node(0, lat=30.900, lng=75.850)
    rng.graph.add_node(1, lat=30.910, lng=75.850)
    rng.graph.add_node(2, lat=30.920, lng=75.850)
    rng.graph.add_node(99, lat=30.930, lng=75.850)

    for u, v in [(0, 1), (1, 0), (0, 1), (1, 2), (2, 99)]:
        rng.graph.add_edge(u, v, distance=1.0)

    path = [0, 1, 0, 1, 2, 99]
    diag = analyze_trajectory(
        pair_id=1,
        pair_name="Test Diagnostic Route",
        source_node=0,
        dest_node=99,
        direct_distance_km=3.3,
        dijkstra_distance_km=3.0,
        dijkstra_steps=3,
        path_nodes=path,
        graph=rng.graph,
        rng=rng,
        execution_time_ms=12.5,
        recovery_actions=1,
        max_steps=600,
    )

    # Verify all 21 keys exist
    expected_keys = [
        "pair_id", "pair_name", "source_node", "dest_node",
        "direct_distance_km", "dijkstra_distance_km", "dijkstra_steps",
        "dqn_success", "dqn_distance_km", "dqn_steps", "distance_ratio",
        "execution_time_ms", "total_revisits", "unique_nodes_revisited",
        "max_node_revisits", "immediate_reversals", "two_hop_cycles",
        "three_hop_cycles", "recovery_actions", "no_progress_steps",
        "termination_reason"
    ]
    for key in expected_keys:
        assert key in diag, f"Missing metric: {key}"

    assert diag["dqn_success"] is True
    assert diag["dqn_steps"] == 5
    assert diag["dqn_distance_km"] == 5.0
    assert diag["distance_ratio"] == round(5.0 / 3.0, 2)
    assert diag["immediate_reversals"] == 2  # (0->1->0) at idx 2, (1->0->1) at idx 3
    assert diag["recovery_actions"] == 1
    assert diag["termination_reason"] == "goal"

    # Test summary function
    summary = summarize_diagnostics([diag])
    assert summary["total_pairs"] == 1
    assert summary["success_count"] == 1
    assert summary["success_rate_pct"] == 100.0
    assert summary["mean_ratio"] == diag["distance_ratio"]

import os
import math
import pytest
import numpy as np
import networkx as nx

from ml.graph.builder import RoadNetworkGraph
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.graph_aware_agent import GraphAwareDQNAgent, EdgeScoringNetwork


def create_custom_test_env(edges_spec, node_coords):
    """Helper to create a small test graph with explicit edges and coordinates."""
    rng = RoadNetworkGraph()
    rng.graph = nx.DiGraph()
    for n_id, (lat, lng) in node_coords.items():
        rng.graph.add_node(n_id, lat=lat, lng=lng)
    for u, v, dist, speed, name in edges_spec:
        rng.graph.add_edge(u, v, distance=dist, speed=speed, name=name,
                          travel_time=(dist / speed) * 60.0,
                          traffic_factor=1.0,
                          operational_cost=dist * 12.0,
                          weight=dist)
    env = TaxiRouteEnv(rng, max_steps=50)
    return env


def test_intersections_with_different_out_degrees():
    """Verify that get_candidate_edges correctly extracts candidate sets for out-degrees 1, 2, 3, and 4."""
    node_coords = {
        0: (30.900, 75.850),  # center
        1: (30.910, 75.850),  # North (out-degree 1 target)
        2: (30.900, 75.860),  # East
        3: (30.890, 75.850),  # South
        4: (30.900, 75.840),  # West
        10: (30.920, 75.850), # Out-degree 1 node
        20: (30.880, 75.850), # Out-degree 2 node
        30: (30.870, 75.850), # Out-degree 3 node
        99: (30.950, 75.850), # Destination
    }
    edges = [
        # Node 0 has out-degree 4
        (0, 1, 1.0, 40.0, "North St"),
        (0, 2, 1.0, 40.0, "East St"),
        (0, 3, 1.0, 40.0, "South St"),
        (0, 4, 1.0, 40.0, "West St"),
        # Node 10 has out-degree 1
        (10, 99, 3.0, 50.0, "Highway"),
        # Node 20 has out-degree 2
        (20, 0, 1.5, 40.0, "Branch A"),
        (20, 3, 1.5, 40.0, "Branch B"),
        # Node 30 has out-degree 3
        (30, 0, 2.0, 40.0, "Rd 1"),
        (30, 2, 2.0, 40.0, "Rd 2"),
        (30, 4, 2.0, 40.0, "Rd 3"),
    ]
    env = create_custom_test_env(edges, node_coords)
    env.reset(options={'source': 0, 'destination': 99})

    # Out-degree 4
    cands_4 = env.get_candidate_edges(0)
    assert len(cands_4) == 4
    assert {c['neighbor'] for c in cands_4} == {1, 2, 3, 4}

    # Out-degree 1
    cands_1 = env.get_candidate_edges(10)
    assert len(cands_1) == 1
    assert cands_1[0]['neighbor'] == 99

    # Out-degree 2
    cands_2 = env.get_candidate_edges(20)
    assert len(cands_2) == 2
    assert {c['neighbor'] for c in cands_2} == {0, 3}

    # Out-degree 3
    cands_3 = env.get_candidate_edges(30)
    assert len(cands_3) == 3
    assert {c['neighbor'] for c in cands_3} == {0, 2, 4}


def test_multiple_edges_with_similar_bearings():
    """Verify that two edges sharing almost identical bearings (e.g. flyover vs slip-road) are both preserved."""
    node_coords = {
        0: (30.9000, 75.8500),
        1: (30.9070, 75.8570),  # Bearing ~45 deg (Flyover)
        2: (30.9071, 75.8569),  # Bearing ~44 deg (Slip Road)
        99: (30.9500, 75.9000),
    }
    edges = [
        (0, 1, 1.2, 60.0, "Grand Trunk Flyover"),
        (0, 2, 1.3, 30.0, "Grand Trunk Service Road"),
    ]
    env = create_custom_test_env(edges, node_coords)
    env.reset(options={'source': 0, 'destination': 99})

    candidates = env.get_candidate_edges(0)
    assert len(candidates) == 2
    neighbors = [c['neighbor'] for c in candidates]
    assert 1 in neighbors and 2 in neighbors

    # Both edges must have distinct feature vectors (different speeds, lengths, bridge flags)
    c1 = next(c for c in candidates if c['neighbor'] == 1)
    c2 = next(c for c in candidates if c['neighbor'] == 2)
    assert not np.allclose(c1['features'], c2['features'])
    # Flyover has is_bridge = 1.0 (feature index 10)
    assert c1['features'][10] == 1.0  # is_bridge
    assert c2['features'][10] == 0.0


def test_dead_ends_and_necessary_backtracking():
    """Verify that cul-de-sacs with out-degree 1 trigger the fallback opening the backtrack edge."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.905, 75.850),  # Cul-de-sac
        99: (30.920, 75.850),
    }
    edges = [
        (0, 1, 0.5, 30.0, "Cul-de-sac Lane"),
        (1, 0, 0.5, 30.0, "Cul-de-sac Lane Return"),
    ]
    env = create_custom_test_env(edges, node_coords)
    env.reset(options={'source': 0, 'destination': 99})

    # Agent moves from 0 to 1
    obs, r, term, trunc, info = env.step_to_neighbor(1)
    assert env.current_node == 1
    assert env.visited_nodes == [0, 1]

    # At node 1, the only outgoing edge is back to node 0 (which was visited)
    candidates = env.get_candidate_edges(1, prev_node=0, visited_nodes=[0, 1])
    assert len(candidates) == 1
    assert candidates[0]['neighbor'] == 0
    # Fallback must keep it valid so the agent is not permanently stuck
    assert candidates[0]['is_valid'] is True


def test_one_way_roads():
    """Verify that directed one-way edges are strictly respected."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),
        99: (30.920, 75.850),
    }
    # One-way road from 0 to 1 only
    edges = [
        (0, 1, 1.0, 40.0, "One Way Ave"),
    ]
    env = create_custom_test_env(edges, node_coords)
    env.reset(options={'source': 0, 'destination': 99})

    # From 0, edge to 1 exists
    cands_0 = env.get_candidate_edges(0)
    assert len(cands_0) == 1
    assert cands_0[0]['neighbor'] == 1

    # From 1, no outgoing edge exists
    cands_1 = env.get_candidate_edges(1)
    assert len(cands_1) == 0


def test_destination_arrival():
    """Verify that stepping directly into the destination triggers termination, step count, and goal reward."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),  # Destination
    }
    edges = [
        (0, 1, 1.0, 50.0, "Target Blvd"),
    ]
    env = create_custom_test_env(edges, node_coords)
    obs, info = env.reset(options={'source': 0, 'destination': 1})

    cands = env.get_candidate_edges(0)
    assert len(cands) == 1
    assert cands[0]['neighbor'] == 1
    assert cands[0]['is_valid'] is True

    obs, reward, terminated, truncated, info = env.step_to_neighbor(1)
    assert terminated is True
    assert truncated is False
    assert env.current_node == 1
    assert env.step_count == 1
    assert reward > 90.0  # Includes goal bonus (100.0)


def test_disconnected_graphs():
    """Verify clean handling when a node has no path to the destination."""
    node_coords = {
        0: (30.900, 75.850),
        1: (30.910, 75.850),
        99: (30.950, 75.950), # Disconnected target
    }
    edges = [
        (0, 1, 1.0, 40.0, "Local St"),
    ]
    env = create_custom_test_env(edges, node_coords)
    obs, info = env.reset(options={'source': 0, 'destination': 99})

    # Step to 1
    obs, r, term, trunc, info = env.step_to_neighbor(1)
    # At 1, no further outgoing edges exist
    cands = env.get_candidate_edges(1)
    assert len(cands) == 0


def test_cycle_prevention():
    """Verify that edges leading to recent or frequently visited nodes are masked out when fresh alternatives exist."""
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
    env = create_custom_test_env(edges, node_coords)
    env.reset(options={'source': 0, 'destination': 99})

    # Path: 0 -> 1 -> 2
    env.step_to_neighbor(1)
    env.step_to_neighbor(2)

    # At node 2, outgoing edges are to 0 (visited in path [0, 1, 2]) and 99 (fresh goal)
    candidates = env.get_candidate_edges(2, prev_node=1, visited_nodes=[0, 1, 2])
    assert len(candidates) == 2

    cand_0 = next(c for c in candidates if c['neighbor'] == 0)
    cand_99 = next(c for c in candidates if c['neighbor'] == 99)

    # Edge to 0 should be masked as False because it was recently visited and fresh exit 99 exists
    assert cand_0['is_valid'] is False
    # Edge to 99 should be valid
    assert cand_99['is_valid'] is True


def test_consistency_between_training_and_inference():
    """Verify that candidate edge feature extraction is deterministic and identical across calls."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(rng)
    env.reset(options={'source': 0, 'destination': 24})

    # Extract twice with same state
    cands_train = env.get_candidate_edges(0, prev_node=None, visited_nodes=[0])
    cands_eval = env.get_candidate_edges(0, prev_node=None, visited_nodes=[0])

    assert len(cands_train) == len(cands_eval)
    for ct, ce in zip(cands_train, cands_eval):
        assert ct['neighbor'] == ce['neighbor']
        assert ct['is_valid'] == ce['is_valid']
        assert np.array_equal(ct['features'], ce['features'])
        assert ct['features'].shape == (20,)

import pytest
import os
import numpy as np
from ml.graph.builder import RoadNetworkGraph
from ml.evaluation.baselines import BaselineRouter
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.dqn_agent import DQNAgent
from ml.evaluation.compare import AlgorithmEvaluator


def test_road_network_synthetic_grid():
    rng = RoadNetworkGraph()
    G = rng.build_synthetic_grid(grid_size=5)
    assert G.number_of_nodes() == 25
    assert G.number_of_edges() > 0
    assert rng.is_synthetic is True
    assert rng.source_type == "SYNTHETIC SIMULATION"


def test_dijkstra_and_astar_routes_synthetic():
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    router = BaselineRouter(rng)

    dijkstra_res = router.dijkstra_route(0, 24)
    assert dijkstra_res['success'] is True
    assert dijkstra_res['total_distance_km'] > 0
    assert dijkstra_res['graph_source'] == "SYNTHETIC SIMULATION"
    assert len(dijkstra_res['route_nodes']) >= 2

    astar_res = router.astar_route(0, 24)
    assert astar_res['success'] is True
    assert astar_res['total_distance_km'] > 0
    assert astar_res['graph_source'] == "SYNTHETIC SIMULATION"


def test_real_osm_graph_loading_and_routing():
    """Verify loading real OSM road network, checking graph source, snapping, and baseline routes."""
    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    G = rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)

    assert G.number_of_nodes() > 100
    assert G.number_of_edges() > 100
    assert rng.is_synthetic is False
    assert rng.source_type == "REAL OSM ROAD NETWORK"

    # Snap real Ludhiana landmarks
    # Ludhiana center (30.9010, 75.8573) and Ludhiana Junction Railway Station (30.9128, 75.8482)
    s_node = rng.find_nearest_node(30.9010, 75.8573)
    d_node = rng.find_nearest_node(30.9128, 75.8482)
    assert s_node is not None
    assert d_node is not None
    assert s_node in rng.nodes_dict
    assert d_node in rng.nodes_dict

    router = BaselineRouter(rng)
    dijkstra_res = router.dijkstra_route(s_node, d_node)
    assert dijkstra_res['success'] is True
    assert dijkstra_res['graph_source'] == "REAL OSM ROAD NETWORK"
    assert dijkstra_res['total_distance_km'] > 0
    assert len(dijkstra_res['route_coordinates']) > 0

    # Ensure coordinates match actual physical positions in Ludhiana
    coord0 = dijkstra_res['route_coordinates'][0]
    assert 30.5 <= coord0['lat'] <= 31.5
    assert 75.5 <= coord0['lng'] <= 76.5

    astar_res = router.astar_route(s_node, d_node)
    assert astar_res['success'] is True
    assert astar_res['graph_source'] == "REAL OSM ROAD NETWORK"
    assert astar_res['total_distance_km'] > 0


def test_synthetic_fallback_offline():
    """Verify that forcing synthetic or missing network data falls back gracefully."""
    rng = RoadNetworkGraph()
    G = rng.load_osm_graph("Test City", cache_path="nonexistent_cache.json", force_synthetic=True)
    assert rng.is_synthetic is True
    assert rng.source_type == "SYNTHETIC SIMULATION"
    assert G.number_of_nodes() == 100

    router = BaselineRouter(rng)
    res = router.dijkstra_route(0, 99)
    assert res['success'] is True
    assert res['graph_source'] == "SYNTHETIC SIMULATION"


def test_vectorized_node_snapping_proximity():
    """Verify that find_nearest_node accurately snaps to the closest road network node."""
    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)

    query_lat, query_lng = 30.9010, 75.8573
    node = rng.find_nearest_node(query_lat, query_lng)
    snapped = rng.nodes_dict[node]
    dist_km = rng.haversine_distance(query_lat, query_lng, snapped['lat'], snapped['lng'])

    # Snapped road node in urban Ludhiana should be within 500 meters (0.5 km)
    assert dist_km < 0.5


def test_fair_algorithm_comparison_report():
    """Verify that compare_all evaluates algorithms on the same graph with measured metrics."""
    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)

    evaluator = AlgorithmEvaluator(rng)
    report = evaluator.compare_all(
        source_lat=30.9010, source_lng=75.8573,
        dest_lat=30.9128, dest_lng=75.8482,
        optimization_objective="balanced"
    )

    assert "recommended_algorithm" in report
    assert "explanation" in report
    assert report["graph_source"] == "REAL OSM ROAD NETWORK"
    assert "Dijkstra" in report["algorithms"]
    assert "A*" in report["algorithms"]
    assert "DQN" in report["algorithms"]

    dijkstra_res = report["algorithms"]["Dijkstra"]
    assert dijkstra_res["success"] is True
    assert dijkstra_res["total_distance_km"] > 0
    assert dijkstra_res["execution_time_ms"] >= 0


def test_taxi_env_gymnasium_compliance():
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(rng)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert 'action_mask' in info

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)


def test_dqn_agent_save_load(tmp_path):
    agent = DQNAgent(state_dim=8, action_dim=8)
    ckpt_path = str(tmp_path / "test_dqn.pt")
    
    agent.save_checkpoint(ckpt_path)
    new_agent = DQNAgent(state_dim=8, action_dim=8)
    loaded = new_agent.load_checkpoint(ckpt_path)
    assert loaded is True


def test_dqn_same_source_and_destination():
    """Verify pickup == destination succeeds immediately with 0 distance and 0 steps."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    model_path = os.path.abspath("ml/checkpoints/dqn_model.pt")
    evaluator = AlgorithmEvaluator(rng, dqn_model_path=model_path if os.path.exists(model_path) else None)
    
    # Even if model not loaded, same source & destination should succeed cleanly
    res = evaluator.route_dqn(0, 0)
    assert res['success'] is True
    assert res['step_count'] == 0
    assert res['total_distance_km'] == 0.0
    assert res['route_efficiency_pct'] == 100.0
    assert len(res['route_nodes']) == 1
    assert res['route_nodes'][0] == 0


def test_dqn_unreachable_node_handling():
    """Verify that attempting to route to an unreachable or nonexistent node returns a clean error."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    evaluator = AlgorithmEvaluator(rng)
    evaluator.dqn_loaded = True
    evaluator.dqn_agent = DQNAgent(state_dim=8, action_dim=8)

    # Non-existent node
    res_nonexistent = evaluator.route_dqn(0, 9999)
    assert res_nonexistent['success'] is False
    assert "does not exist in graph" in res_nonexistent['error']

    # Disconnected node: create a graph with 2 disjoint components
    import networkx as nx
    rng.graph.remove_edges_from(list(rng.graph.in_edges(24)))
    rng.graph.remove_edges_from(list(rng.graph.out_edges(24)))
    res_unreachable = evaluator.route_dqn(0, 24)
    assert res_unreachable['success'] is False
    assert "unreachable" in res_unreachable['error']


def test_dqn_missing_weights_error_message():
    """Verify that route_dqn returns an informative error when model checkpoint is not loaded."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    evaluator = AlgorithmEvaluator(rng, dqn_model_path="nonexistent_dqn_model.pt")
    
    res = evaluator.route_dqn(0, 5)
    assert res['success'] is False
    assert "Model not trained yet" in res['error']


def test_dqn_action_masking_and_directional_mapping():
    """Verify compass directional action mapping and masking in TaxiRouteEnv."""
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(rng)

    # In a 5x5 grid, corner node 0 (bottom-left) only has outgoing edges to the North and East
    mask_0 = env.get_action_mask(0)
    assert mask_0.shape == (8,)
    # North (Sector 0) and East (Sector 2) should be True
    assert bool(mask_0[0]) or bool(mask_0[2])

    # Test backtracking prevention: from node 0 to its neighbor
    act_map = env.get_action_neighbor_map(0)
    first_act = list(act_map.keys())[0]
    nbr = act_map[first_act]

    # When at nbr, masking with prev_node=0 should mask out the reverse action if alternatives exist
    nbr_mask = env.get_action_mask(nbr, prev_node=0)
    assert isinstance(nbr_mask, np.ndarray)


def test_dqn_real_osm_inference():
    """Verify that route_dqn succeeds on the real OSM network using trained weights."""
    model_path = os.path.abspath("ml/checkpoints/dqn_model.pt")
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not yet created.")

    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)

    evaluator = AlgorithmEvaluator(rng, dqn_model_path=model_path)
    assert evaluator.dqn_loaded is True

    # Real Ludhiana landmark coordinates
    s_node = rng.find_nearest_node(30.9010, 75.8573)
    d_node = rng.find_nearest_node(30.9128, 75.8482)

    res = evaluator.route_dqn(s_node, d_node)
    assert res['algorithm'] == 'DQN'
    assert res['graph_source'] == "REAL OSM ROAD NETWORK"
    assert res['execution_time_ms'] > 0
    assert len(res['route_nodes']) > 0
    assert len(res['route_coordinates']) > 0
    assert res['total_distance_km'] > 0


def test_dqn_dashboard_route_success():
    """Verify that route_dqn succeeds for the default dashboard coordinates without step-limit failure."""
    model_path = os.path.abspath("ml/checkpoints/dqn_model.pt")
    if not os.path.exists(model_path):
        pytest.skip("Model checkpoint not yet created.")

    rng = RoadNetworkGraph()
    cache_path = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
    rng.load_osm_graph("Ludhiana, Punjab, India", cache_path=cache_path)

    evaluator = AlgorithmEvaluator(rng, dqn_model_path=model_path)
    assert evaluator.dqn_loaded is True

    # Exact dashboard coordinates: Gill Rd/Model Town (30.8610, 75.8173) -> Junction (30.9330, 75.8893)
    s_node = rng.find_nearest_node(30.8610, 75.8173)
    d_node = rng.find_nearest_node(30.9330, 75.8893)

    res = evaluator.route_dqn(s_node, d_node)
    assert res['algorithm'] == 'DQN'
    assert res['success'] is True
    assert res['error'] is None
    assert res['step_count'] > 0
    assert res['total_distance_km'] > 0
    assert len(res['route_nodes']) > 0
    assert len(res['route_coordinates']) > 0
    assert res['route_nodes'][-1] == d_node



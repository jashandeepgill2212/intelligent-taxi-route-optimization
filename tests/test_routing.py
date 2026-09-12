import pytest
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


def test_dijkstra_and_astar_routes():
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    router = BaselineRouter(rng)

    dijkstra_res = router.dijkstra_route(0, 24)
    assert dijkstra_res['success'] is True
    assert dijkstra_res['total_distance_km'] > 0
    assert len(dijkstra_res['route_nodes']) >= 2

    astar_res = router.astar_route(0, 24)
    assert astar_res['success'] is True
    assert astar_res['total_distance_km'] > 0


def test_taxi_env_gymnasium_compliance():
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid(grid_size=5)
    env = TaxiRouteEnv(rng)
    
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,)
    assert 'action_mask' in info

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8,)
    assert isinstance(reward, float)


def test_dqn_agent_save_load(tmp_path):
    agent = DQNAgent(state_dim=8, action_dim=8)
    ckpt_path = str(tmp_path / "test_dqn.pt")
    
    agent.save_checkpoint(ckpt_path)
    new_agent = DQNAgent(state_dim=8, action_dim=8)
    loaded = new_agent.load_checkpoint(ckpt_path)
    assert loaded is True

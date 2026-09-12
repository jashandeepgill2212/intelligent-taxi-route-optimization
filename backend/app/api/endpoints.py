from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from backend.app.schemas.route_schema import (
    RouteRequest, RouteResponse, ComparisonResponse,
    TrainingStartRequest, GraphStatusResponse
)
from backend.app.services.routing_service import RoutingService

router = APIRouter()
service = RoutingService()


@router.get("/health", summary="API Health Check")
def health_check():
    return {
        "status": "healthy",
        "service": "TaxiRoute RL Backend API",
        "version": "1.0.0",
        "team": ["Jashandeep Singh (72510438)", "Harshpreet Singh Malhi (72520287)"]
    }


@router.get("/graph/status", response_model=GraphStatusResponse, summary="Get Road Network Graph Status")
def get_graph_status():
    return service.get_graph_status()


@router.post("/route/dijkstra", response_model=RouteResponse, summary="Calculate Route using Dijkstra Algorithm")
def get_dijkstra_route(req: RouteRequest):
    service.set_traffic_level(req.traffic_level)
    s_node = service.rng.find_nearest_node(req.source.lat, req.source.lng)
    d_node = service.rng.find_nearest_node(req.destination.lat, req.destination.lng)
    res = service.baseline_router.dijkstra_route(s_node, d_node)
    return res


@router.post("/route/astar", response_model=RouteResponse, summary="Calculate Route using A* Search Algorithm")
def get_astar_route(req: RouteRequest):
    service.set_traffic_level(req.traffic_level)
    s_node = service.rng.find_nearest_node(req.source.lat, req.source.lng)
    d_node = service.rng.find_nearest_node(req.destination.lat, req.destination.lng)
    res = service.baseline_router.astar_route(s_node, d_node)
    return res


@router.post("/route/rl", response_model=RouteResponse, summary="Calculate Route using Reinforcement Learning (DQN)")
def get_rl_route(req: RouteRequest):
    service.set_traffic_level(req.traffic_level)
    s_node = service.rng.find_nearest_node(req.source.lat, req.source.lng)
    d_node = service.rng.find_nearest_node(req.destination.lat, req.destination.lng)
    res = service.evaluator.route_dqn(s_node, d_node)
    return res


@router.post("/route/compare", response_model=ComparisonResponse, summary="Compare All Algorithms (Dijkstra vs A* vs DQN)")
def compare_routes(req: RouteRequest):
    service.set_traffic_level(req.traffic_level)
    res = service.evaluator.compare_all(
        source_lat=req.source.lat,
        source_lng=req.source.lng,
        dest_lat=req.destination.lat,
        dest_lng=req.destination.lng,
        optimization_objective=req.optimization_objective
    )
    return res


@router.post("/training/start", summary="Trigger DQN Agent Training")
def start_training(req: TrainingStartRequest):
    success, msg = service.start_background_training(episodes=req.episodes, batch_size=req.batch_size)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg, "episodes": req.episodes}


@router.get("/training/status", summary="Get DQN Training Progress & Metrics")
def get_training_status():
    return service.get_training_status()


@router.get("/metrics", summary="Get Global System & Evaluation Metrics")
def get_global_metrics():
    graph_st = service.get_graph_status()
    train_st = service.get_training_status()
    return {
        "graph": graph_st,
        "dqn_model_trained": train_st.get("model_checkpoint_exists", False),
        "latest_training_metrics": train_st.get("metrics", {}),
        "available_algorithms": ["Dijkstra", "A*", "DQN"]
    }

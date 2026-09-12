from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class LocationPoint(BaseModel):
    lat: float = Field(..., example=30.9010, description="Latitude coordinate")
    lng: float = Field(..., example=75.8573, description="Longitude coordinate")


class RouteRequest(BaseModel):
    source: LocationPoint
    destination: LocationPoint
    traffic_level: str = Field("MEDIUM", description="Traffic condition: LOW, MEDIUM, or HIGH")
    optimization_objective: str = Field("balanced", description="Objective: distance, time, cost, balanced")


class RouteCoordinate(BaseModel):
    lat: float
    lng: float
    node_id: int


class RouteResponse(BaseModel):
    algorithm: str
    success: bool
    error: Optional[str] = None
    route_nodes: List[int] = []
    route_coordinates: List[RouteCoordinate] = []
    total_distance_km: float = 0.0
    total_time_min: float = 0.0
    total_cost: float = 0.0
    operational_cost: float = 0.0
    step_count: int = 0
    route_efficiency_pct: float = 0.0
    execution_time_ms: float = 0.0


class ComparisonResponse(BaseModel):
    source: Dict[str, Any]
    destination: Dict[str, Any]
    optimization_objective: str
    recommended_algorithm: str
    explanation: str
    algorithms: Dict[str, RouteResponse]


class TrainingStartRequest(BaseModel):
    episodes: int = Field(150, ge=10, le=2000, description="Number of training episodes")
    batch_size: int = Field(32, ge=8, le=256, description="Batch size for experience replay")


class GraphStatusResponse(BaseModel):
    city_name: str
    is_synthetic: bool
    num_nodes: int
    num_edges: int
    current_traffic_level: str

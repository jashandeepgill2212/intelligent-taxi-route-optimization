export interface LocationPoint {
  lat: number;
  lng: number;
}

export interface LocationSearchResult {
  name: string;
  display_name: string;
  latitude: number;
  longitude: number;
}

export interface RouteCoordinate {
  lat: number;
  lng: number;
  node_id: number;
}

export interface RouteResponse {
  algorithm: string;
  success: boolean;
  error?: string;
  graph_source?: string;
  route_nodes: number[];
  route_coordinates: RouteCoordinate[];
  total_distance_km: number;
  total_time_min: number;
  total_cost: number;
  operational_cost: number;
  step_count: number;
  route_efficiency_pct: number;
  execution_time_ms: number;
}

export interface ComparisonResponse {
  source: { lat: number; lng: number; node_id: number };
  destination: { lat: number; lng: number; node_id: number };
  graph_source?: string;
  optimization_objective: string;
  recommended_algorithm: string;
  explanation: string;
  algorithms: Record<string, RouteResponse>;
}

export interface GraphStatusResponse {
  city_name: string;
  is_synthetic: boolean;
  source_type?: string;
  num_nodes: number;
  num_edges: number;
  current_traffic_level: string;
}

export interface TrainingMetrics {
  episodes: number[];
  rewards: number[];
  lengths: number[];
  successes: number[];
  losses: number[];
  epsilons: number[];
  training_time_sec: number;
  final_success_rate: number;
}

export interface TrainingStatusResponse {
  status: string;
  is_training: boolean;
  progress: { episodes_total: number; current_episode: number; message: string };
  model_checkpoint_exists: boolean;
  metrics: TrainingMetrics;
}

import {
  LocationPoint,
  RouteResponse,
  ComparisonResponse,
  GraphStatusResponse,
  TrainingStatusResponse
} from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`);
  return res.json();
}

export async function fetchGraphStatus(): Promise<GraphStatusResponse> {
  const res = await fetch(`${API_BASE_URL}/graph/status`);
  return res.json();
}

export async function compareRoutes(
  source: LocationPoint,
  destination: LocationPoint,
  trafficLevel: string = 'MEDIUM',
  optimizationObjective: string = 'balanced'
): Promise<ComparisonResponse> {
  const res = await fetch(`${API_BASE_URL}/route/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source,
      destination,
      traffic_level: trafficLevel,
      optimization_objective: optimizationObjective
    })
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.statusText}`);
  }
  return res.json();
}

export async function startDqnTraining(episodes: number = 150, batchSize: number = 32) {
  const res = await fetch(`${API_BASE_URL}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ episodes, batch_size: batchSize })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Failed to start training' }));
    throw new Error(errorData.detail || 'Training trigger failed');
  }
  return res.json();
}

export async function fetchTrainingStatus(): Promise<TrainingStatusResponse> {
  const res = await fetch(`${API_BASE_URL}/training/status`);
  return res.json();
}

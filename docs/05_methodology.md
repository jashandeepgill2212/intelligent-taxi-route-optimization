# Chapter 5: Methodology

## 5.1 System Workflow & Pipeline Architecture
The system follows a modular 6-phase pipeline:
1. **Data Preprocessing**: Standardizes taxi trip columns, filters invalid coordinates, computes speeds, and calculates traffic congestion indices.
2. **Road Network Graph Building**: Constructs directed graph representation using OpenStreetMap / OSMnx data with fallback synthetic grid generation.
3. **Environment Setup**: Initializes Gymnasium-compliant `TaxiRouteEnv` with normalized states and action masks.
4. **Agent Training**: Trains PyTorch Deep Q-Network (DQN) using Experience Replay and Target Networks.
5. **Baselines Execution**: Calculates exact Dijkstra and heuristic A* routes on identical source-destination pairs.
6. **Web Dashboard & API**: Serves FastAPI endpoints and renders interactive Leaflet map overlays with quantitative benchmarks.

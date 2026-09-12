# Chapter 6: System Architecture

## 6.1 Architectural Topology
```text
                 ┌──────────────────────┐
                 │ Historical Taxi Data │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Data Preprocessing   │
                 └──────────┬───────────┘
                            ↓
                 ┌──────────────────────┐
                 │ Road Network Graph   │
                 └──────────┬───────────┘
                            ↓
              ┌─────────────────────────────┐
              │ Reinforcement Learning Env  │
              └─────────────┬───────────────┘
                            ↓
                    ┌──────────────┐
                    │ DQN Agent    │
                    └──────┬───────┘
                           ↓
                  ┌─────────────────┐
                  │ Route Optimizer  │
                  └────────┬────────┘
                           ↓
              ┌──────────────────────────┐
              │ FastAPI Backend          │
              └────────────┬─────────────┘
                           ↓
              ┌──────────────────────────┐
              │ Next.js Dashboard        │
              └──────────────────────────┘
```

## 6.2 Component Interactions
- **ML Core (`ml/`)**: Encapsulates data processing, graph builder, Gymnasium environment, PyTorch DQN agent, Dijkstra/A* baselines, and evaluation module.
- **FastAPI Backend (`backend/`)**: Serves RESTful endpoints, manages graph state, background DQN training threads, and Pydantic validation.
- **Next.js Frontend (`frontend/`)**: Renders interactive Leaflet map overlays, comparison matrices, and Recharts analytics.

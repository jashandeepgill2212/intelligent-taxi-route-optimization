# Chapter 3: Project Scope

## 3.1 In-Scope Capabilities
- Synthetic & OpenStreetMap (OSMnx) road network graph building.
- Data preprocessing pipeline for historical taxi trip datasets with outlier filtering and temporal feature extraction.
- Gymnasium RL environment with discrete action masking.
- PyTorch DQN model training, serialization (.pt checkpoints), and inference.
- Dijkstra and A* classical routing implementations with standardized outputs.
- FastAPI REST backend with endpoints for routing, comparison, training, and graph metrics.
- Next.js 14 interactive map dashboard with route polylines, comparison tables, and training charts.

## 3.2 Out-of-Scope / Exclusions
- Live real-time GPS hardware tracking of physical vehicles.
- Multi-agent fleet management (coordinating 10,000 taxis simultaneously).
- Commercial payment gateway processing for taxi fares.

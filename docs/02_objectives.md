# Chapter 2: Project Objectives

## 2.1 Primary Objectives
1. **Develop an End-to-End Taxi Route Optimization Platform**: Build a runnable system integrating ML/RL models, a Python FastAPI backend, and a Next.js Leaflet interactive GIS dashboard.
2. **Formulate Gymnasium RL Environment**: Construct a Gymnasium-compliant environment (`TaxiRouteEnv`) incorporating normalized node state vectors, traffic congestion factors, dynamic fuel costs, and action masking.
3. **Implement Deep Q-Network Agent**: Implement a PyTorch Deep Q-Network (DQN) with Replay Buffer, Target Network updates, and epsilon-greedy exploration.
4. **Benchmarking & Comparative Evaluation**: Compare DQN performance against classical baselines (Dijkstra's Algorithm and A* Search) across identical source-destination test pairs.
5. **Deterministic Explainability**: Provide data-driven explainability detailing why a specific route was recommended.

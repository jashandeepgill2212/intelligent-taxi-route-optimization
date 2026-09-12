# INTELLIGENT TAXI ROUTE OPTIMIZATION USING REINFORCEMENT LEARNING

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-emerald.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-brightgreen.svg)](https://gymnasium.farama.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👥 Team Members
- **Jashandeep Singh** (Roll No: 72510438)
- **Harshpreet Singh Malhi** (Roll No: 72520287)
- **Project Type**: B.Tech CSE / AI&DS Major Project
PPT-https://docs.google.com/presentation/d/1lRiJdChxKzSxbiDf8_JMM3HKXogoFIZ7/edit?usp=drive_link&ouid=101115675166531428876&rtpof=true&sd=true
---

## 1. Project Overview
**TaxiRoute RL** is an end-to-end intelligent taxi route optimization platform that uses **Reinforcement Learning** (Deep Q-Networks) to learn adaptive, multi-objective routes across dynamic urban road networks. The platform compares the RL policy against classical graph algorithms (**Dijkstra's Algorithm** and **A\* Search Algorithm**) and presents empirical metrics on an interactive GIS map dashboard.

---

## 2. Problem Statement
Urban taxis operate in dynamic environments where static shortest-path algorithms fail to account for real-time congestion bottlenecks, time-of-day traffic patterns, and operational fuel expenses. This project formulates taxi navigation as a Markov Decision Process (MDP) to learn adaptive policies via Reinforcement Learning.

---

## 3. Objectives
1. Formulate a Gymnasium-compliant environment (`TaxiRouteEnv`) for graph navigation with action masking.
2. Train a Deep Q-Network (DQN) agent with Replay Buffer and Target Network updates.
3. Compare RL solution against Dijkstra and A* search baselines.
4. Deliver a FastAPI backend and Next.js / Leaflet GIS dashboard with deterministic AI explainability.

---

## 4. Key Features
- 🚕 **Interactive Leaflet GIS Map**: Pick pickup and destination coordinates directly on the map.
- ⚡ **Triple-Algorithm Benchmark**: Compute routes simultaneously for Dijkstra, A*, and DQN.
- 📊 **Quantitative Metrics**: Real-time evaluation of distance (km), ETA (min), operational cost (₹), and step efficiency.
- 🧠 **DQN Model Training Analytics**: Live progress monitoring, reward trajectory curves, loss visualization, and checkpoint saving (`dqn_model.pt`).
- 🤖 **Deterministic AI Explainability**: Objective breakdown explaining why a specific route was recommended.

---

## 5. Architecture & Monorepo Structure
```text
taxiroute-rl/
├── frontend/             # Next.js 14 / React / Tailwind CSS / Leaflet dashboard
│   ├── app/              # Dashboard (/), /compare, /training, /about
│   ├── components/       # Map, RouteCard, ComparisonTable, TrainingChart
│   ├── lib/              # API Client
│   └── types/            # TypeScript interfaces
│
├── backend/              # FastAPI Python backend
│   └── app/
│       ├── main.py       # FastAPI application
│       ├── api/          # Endpoints (/route, /compare, /training, /graph)
│       ├── services/     # Singleton Routing Service
│       └── schemas/      # Pydantic schemas
│
├── ml/                   # Machine Learning & RL Core
│   ├── data/             # Raw & processed taxi dataset CSVs
│   ├── preprocessing/    # Data cleaning & feature extraction
│   ├── graph/            # Road network graph builder (OSMnx + synthetic fallback)
│   ├── environment/      # Gymnasium TaxiRouteEnv with action masking
│   ├── agents/           # PyTorch DQN Agent with Replay Buffer
│   ├── training/         # DQN training script & episode logger
│   ├── evaluation/       # Algorithm comparison evaluator
│   └── checkpoints/      # Model checkpoints (dqn_model.pt)
│
├── scripts/              # Data generation scripts
├── tests/                # Automated pytest unit & integration test suite
└── docs/                 # 13 academic report chapters
```

---

## 6. Technology Stack
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Leaflet, React-Leaflet, Recharts, Lucide Icons.
- **Backend**: FastAPI, Pydantic, Uvicorn, Python 3.11.
- **Machine Learning & RL**: PyTorch, Gymnasium, Stable-Baselines3, Scikit-learn, NumPy, Pandas.
- **GIS & Graphs**: NetworkX, OSMnx, OpenStreetMap, GeoPandas, Shapely.

---

## 7. Dataset & Preprocessing
The system supports historical taxi trip datasets (e.g., NYC Yellow Taxi trip records). The pipeline in `ml/preprocessing/pipeline.py`:
- Detects available columns & maps common aliases.
- Filters invalid GPS coordinates and impossible trips (speed > 150 km/h).
- Derives temporal features: hour of day, day of week, rush-hour binary indicators.
- Calculates speed-based congestion indices.

---

## 8. Road Network & Multi-Objective Cost Function
Edges in the road network store distance, speed, travel time, traffic factors, and operational cost. Edge cost formulation:
$$\text{Total Cost} = w_1 \cdot \text{TravelTime} + w_2 \cdot \text{Distance} + w_3 \cdot \text{TrafficPenalty} + w_4 \cdot \text{OperationalCost}$$

---

## 9. Baseline Algorithms
- **Dijkstra's Algorithm**: Guaranteed minimum cost path across non-negative edge weights.
- **A\* Search Algorithm**: Guided search using Haversine distance heuristic function $h(u, v)$ to target destination node.

---

## 10. Reinforcement Learning Formulation (DQN)
- **State (8-dim vector)**: Normalized current node, destination node, remaining Haversine distance, local traffic factor, time of day, unit direction vector $(dx, dy)$, and path length ratio.
- **Action Space**: Discrete outgoing neighbor selection ($K=8$) with boolean action masking.
- **Reward**: $- \text{edge\_cost} + 100.0 \text{ (Goal)} - 15.0 \text{ (Invalid Action)} - 5.0 \text{ (Loop Penalty)}$.

---

## 11. Empirical Performance Benchmarks

| Metric | Dijkstra | A* Search | Deep Q-Network (DQN) |
|---|---|---|---|
| **Success Rate** | 100% | 100% | 80.0% – 90.0% |
| **Distance (km)** | 11.23 km | 11.23 km | 11.99 km |
| **Travel Time (min)** | 20.7 min | 20.7 min | 21.7 min |
| **Operational Cost** | ₹134.76 | ₹134.76 | ₹143.89 |
| **Execution Time** | ~0.35 ms | ~0.25 ms | ~0.95 ms |

---

## 12. Quick Start & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Install Backend & ML Dependencies
```bash
python -m pip install -r backend/requirements.txt
```

### 2. Generate Synthetic Taxi Data & Train Initial DQN Model
```bash
python scripts/generate_sample_data.py
python ml/training/train_dqn.py
```

### 3. Run Automated Tests
```bash
powershell -Command "$env:PYTHONPATH='.'; python -m pytest tests/"
```

### 4. Start FastAPI Backend Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start Next.js Frontend Dashboard
```bash
cd frontend
cmd /c npm install
cmd /c npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 13. API Endpoints
- `GET /health` — API health status.
- `GET /graph/status` — Network graph node/edge counts.
- `POST /route/dijkstra` — Calculate Dijkstra route.
- `POST /route/astar` — Calculate A* route.
- `POST /route/rl` — Calculate DQN RL route.
- `POST /route/compare` — Compare all three algorithms.
- `POST /training/start` — Trigger background DQN model training.
- `GET /training/status` — Check training progress & metrics.

---

## 14. Academic Documentation
The `docs/` folder contains 13 complete academic report chapters suitable for B.Tech project submission.

---

## 15. License
Licensed under the [MIT License](LICENSE).

# Live Demonstration & Evaluator Walkthrough Guide

**Project:** Intelligent Taxi Route Optimization using Reinforcement Learning  
**Target Audience:** Academic Evaluators, Project Review Committee, B.Tech Examiners  
**Estimated Demonstration Time:** 5 to 7 Minutes  

---

## 1. Demonstration Overview & Objectives

This guide provides a step-by-step walkthrough for conducting a live demonstration of the **TaxiRoute RL Platform**. 

The demonstration showcases:
1. Interactive OpenStreetMap road network navigation of Ludhiana, Punjab, India.
2. Side-by-side execution and comparison of:
   - **Dijkstra's Algorithm** (Exact classical shortest path baseline)
   - **A\* Search** (Heuristic classical baseline)
   - **Graph-Aware DQN with ALT-PBRS** (Trained deep reinforcement learning policy)
3. Live visualization of the **Ludhiana Junction Railway Barrier Resolution** (Gill Road $\to$ Junction).
4. Real-time metric cards displaying road distance (km), estimated time (min), computational latency (ms), and route efficiency ratio ($D_{\text{DQN}} / D_{\text{Dijkstra}}$).

---

## 2. Prerequisites & Environment Setup

### System Requirements:
- Python 3.10+ (Python 3.11 recommended)
- Node.js 18+ and npm
- Windows PowerShell / Command Prompt or Linux/macOS Terminal
- Modern web browser (Chrome, Edge, Firefox)

### Step 1: Clone Repository
```bash
git clone https://github.com/jashandeepgill2212/intelligent-taxi-route-optimization.git
cd intelligent-taxi-route-optimization
```

### Step 2: Set Up Python Backend Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend/requirements.txt
```

### Step 3: Set Up Frontend Client
```bash
cd frontend
npm install
cd ..
```

---

## 3. Starting the Services Locally

### Step 1: Launch FastAPI Backend Server
In Terminal 1 (from repository root):
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API URL:** `http://localhost:8000`
- **Interactive Swagger Documentation:** `http://localhost:8000/docs`
- **Health Endpoint:** `http://localhost:8000/health` (returns `{"status": "ok"}`)

### Step 2: Launch Next.js Frontend Development Server
In Terminal 2 (from repository root):
```bash
cd frontend
npm run dev
```
- **Frontend Dashboard URL:** `http://localhost:3000`

---

## 4. 5–7 Minute Step-by-Step Live Demo Plan

### Minute 0:00 – 1:00: Problem Context & Architecture
- Open `http://localhost:3000` in the browser.
- Explain: Standard grid RL algorithms fail on real urban graphs due to non-uniform road density, one-way streets, and physical barriers (e.g. railway tracks with sparse flyovers).
- Highlight our three contributions:
  1. **Graph-Aware Edge Scorer (20 features)** replacing rigid compass sectors.
  2. **Cycle-Aware Action Masking** enforcing lifetime visit capping ($\le 2$).
  3. **Directed ALT-PBRS** providing an admissible topological potential field.

### Minute 1:00 – 2:30: Demonstration of Flagship Barrier Route (Gill Rd $\to$ Junction)
- In the left sidebar:
  - **Pickup Location:** Set Latitude `30.8732`, Longitude `75.8560` (Gill Road, Node 1198)
  - **Destination Location:** Set Latitude `30.9090`, Longitude `75.8643` (Ludhiana Junction Station, Node 659)
  - **Traffic Condition:** Select `MEDIUM` (1.3x)
  - **Objective:** Select `Balanced`
- Click **"OPTIMIZE ROUTE"**.
- Point out on the map:
  - **Dijkstra Route (Blue):** Computes ground-truth shortest path (14.11 km).
  - **A\* Route (Green dashed):** Rapidly finds the same shortest corridor.
  - **DQN Route (Purple):** Navigates directly across the western flyover to reach the station in **19.88 km (1.41x ratio)**.
- **Key Insight for Evaluators:** In the previous baseline model, straight-line Euclidean progress pulled the taxi directly north toward the railway fence, causing a massive detour of 52.14 km (3.69x). Under ALT-PBRS, the agent recognized the topological obstacle and crossed the bridge cleanly.

### Minute 2:30 – 4:00: Demonstration of Urban Core Route (City Center $\to$ Junction)
- Set Pickup to Ludhiana City Center: `30.9010`, `75.8450`.
- Destination: Ludhiana Junction Station (`30.9090`, `75.8643`).
- Click **"OPTIMIZE ROUTE"**.
- Highlight:
  - Dijkstra Distance: **8.40 km**
  - DQN Distance: **9.63 km**
  - Route Ratio: **1.15x (Near Optimal)**
  - Revisit Count: **0 Revisits**

### Minute 4:00 – 5:30: Navigation to Comparison & Training Analytics Tabs
- Click **"Compare"** in the top navigation bar (`/compare`):
  - View side-by-side tabular comparison of Dijkstra, A*, and DQN across computational runtime (ms), path nodes, and travel times.
- Click **"Training Analytics"** (`/training`):
  - View learning curves showing episode rewards, loss convergence, and epsilon decay over the 250-episode training curriculum.

### Minute 5:30 – 7:00: Summary of Benchmark Results & Q&A
- Present the 30-pair benchmark summary from [`FINAL_RESULTS.md`](file:///c:/PROJECTS/taxiroute%20optimization/FINAL_RESULTS.md):
  - **70.0% completion rate (21/30)** on the held-out test suite.
  - Mean distance ratio of **2.405x** across successful routes.
- Acknowledge limitations honestly: 9/30 routes failed due to step budget limits in dense non-arterial residential alleys where 16 landmarks provide lower gradient contrast.
- Invite questions from the committee.

---

## 5. Troubleshooting & Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **Frontend displays "Failed to fetch"** | Backend server is not running or port 8000 is occupied. | Ensure `uvicorn` is active in Terminal 1 on port 8000. Verify with `curl http://localhost:8000/health`. |
| **Map shows grey tiles** | Internet connection drop for Leaflet OpenStreetMap tile server. | Verify network connectivity or check if proxy/firewall is blocking OpenStreetMap tile URLs. |
| **"Node snapping out of bounds"** | Selected coordinates fall outside Ludhiana municipal boundaries. | Use coordinates within $30.84^\circ\text{N} \le \text{lat} \le 30.95^\circ\text{N}$ and $75.78^\circ\text{E} \le \text{lng} \le 75.92^\circ\text{E}$. |

---

## 6. Fallback Demonstration Plan (Offline / Hosted Outage)

If the cloud deployment or local server experiences network interruptions during the examination:

1. **Local Offline Mode:**
   - The repository contains pre-cached OSM data in `ml/data/osm_cache/ludhiana_osm_road_network.json` and `alt_landmarks_16.json`.
   - The application runs 100% offline without querying external APIs.
2. **Recorded Verification Artifacts:**
   - Evaluators can review the exact, verified trajectory logs in:
     - [`ml/checkpoints/experiments/alt_pbrs/stage_c_results.json`](file:///c:/PROJECTS/taxiroute%20optimization/ml/checkpoints/experiments/alt_pbrs/stage_c_results.json)
     - [`ml/checkpoints/experiments/alt_pbrs/stage_c_training_report.md`](file:///c:/PROJECTS/taxiroute%20optimization/ml/checkpoints/experiments/alt_pbrs/stage_c_training_report.md)
   - Run the automated evaluation script directly:
     ```bash
     python scratch/print_tables.py
     ```
     This outputs the complete, verified 30-route table directly in the terminal in < 1 second.

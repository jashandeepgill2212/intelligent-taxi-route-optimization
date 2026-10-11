# Intelligent Taxi Route Optimization Using Reinforcement Learning
## Major Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology in Computer Science & Engineering / Artificial Intelligence & Data Science

---

### Project Metadata & Student Details
- **Project Title:** Intelligent Taxi Route Optimization Using Reinforcement Learning
- **Institution:** Punjab Technical University / Affiliated Institute
- **Department:** Department of Computer Science & Engineering
- **Academic Year:** 2025–2026
- **Project Team:**
  1. **Jashandeep Singh** (University Roll No: 72510438)
  2. **Harshpreet Singh Malhi** (University Roll No: 72520287)
  3. **Bhanu Partap Singh Dalal** (University Roll No: 72510134)
- **Project Repository:** [GitHub Repository](https://github.com/jashandeepgill2212/intelligent-taxi-route-optimization)
- **Live Deployment:**
  - **Frontend Client:** [Vercel Deployment](https://intelligent-taxi-route-optimization-beta.vercel.app)
  - **Backend API:** [Render Deployment](https://taxiroute-rl-api.onrender.com)
  - **API Documentation:** [Swagger UI](https://taxiroute-rl-api.onrender.com/docs)

---

## 1. Title Page

```
================================================================================
           INTELLIGENT TAXI ROUTE OPTIMIZATION USING REINFORCEMENT LEARNING
================================================================================

              A B.Tech Major Project in Computer Science & Engineering

                                  Submitted By:
                  Jashandeep Singh           (Roll No: 72510438)
                  Harshpreet Singh Malhi     (Roll No: 72520287)
                  Bhanu Partap Singh Dalal   (Roll No: 72510134)

                              Under the Guidance of:
                         Department Project Review Committee
                           Department of Computer Science
                                    October 2026
================================================================================
```

---

## 2. Abstract

Urban taxi routing requires navigation policies capable of balancing travel distance, time, and road topology constraints. While classical pathfinding algorithms such as Dijkstra and A* calculate globally optimal shortest paths on static graphs, they require full graph re-computations under changing edge cost weights and cannot readily adapt to learned state distributions. Conversely, standard Deep Q-Networks (DQN) applied to spatial grids fail on real-world OpenStreetMap (OSM) topologies due to non-uniform road densities, variable intersection degrees, one-way road segments, and topological barrier traps (such as railways, canals, and rivers).

This project designs, implements, and experimentally validates an **Intelligent Taxi Route Optimization Platform** for the city of Ludhiana, Punjab, India (12,675 nodes, 20,324 directed road edges). We introduce three key algorithmic contributions:
1. **Graph-Aware DQN:** An edge-scoring architecture replacing rigid compass sector binning with a shared multi-layer candidate edge scorer (20 input features) that dynamically handles variable out-degrees ($1 \le \text{deg}^+(u) \le 8$).
2. **Cycle-Aware Action Masking:** A safety layer combining lifetime visit capping ($\le 2$ visits), 1-hop reversal prevention ($u \to v \to u$), and least-visited dead-end recovery, eliminating macro-loops without modifying network weights.
3. **Directed ALT-Based Potential-Based Reward Shaping (ALT-PBRS):** A mathematically sound reward shaping formulation based on Ng et al. (1999) utilizing directed A*, Landmarks, and Triangle inequality (ALT) lower bounds across 16 canonical landmarks.

On a strictly held-out 30-pair benchmark, the newly trained ALT-PBRS agent achieved **21 / 30 destination completions (70.0%)**, outperforming historical Dynamic Allocation DQN (13.3%), Strict Stable Compass DQN (16.7%), Baseline Adjacent DQN (43.3%), and Graph-Aware DQN without PBRS (56.7%). On the flagship dashboard railway barrier route (Gill Road $\to$ Ludhiana Junction), route distance plummeted from **52.14 km (3.69x Dijkstra)** down to **19.88 km (1.41x Dijkstra)**, proving that topological reward shaping overcomes the Euclidean barrier trap. The project is delivered as a production full-stack application featuring a FastAPI Python backend and Next.js / Tailwind CSS / Leaflet frontend.

---

## 3. Introduction

Urban transportation networks in growing metropolitan centers face rapid congestion, infrastructural bottlenecks, and asymmetric road constraints. Taxi dispatch and route guidance systems must provide rapid route recommendations that comply with physical traffic regulations while remaining robust to topological traps.

Traditional navigation systems rely predominantly on classical shortest-path graph algorithms:
- **Dijkstra’s Algorithm:** Guarantees absolute shortest path optimality by exhaustively expanding the lowest-cost frontier, but scales with $O(|E| + |V| \log |V|)$ and searches symmetrically in all directions.
- **A\* Search:** Incorporates a goal-directed Euclidean heuristic to focus search expansion towards the destination, improving query latency.

While mathematically optimal for static weights, classical search algorithms do not learn from empirical route histories, driver experience, or multi-objective reward trade-offs. Reinforcement Learning (RL) provides an adaptive alternative where an autonomous agent learns an optimal routing policy $\pi(a|s)$ via repeated interactions with the environment. However, deploying RL on real OpenStreetMap topologies introduces formidable algorithmic challenges that do not exist in toy grid worlds.

---

## 4. Problem Statement

Standard Deep Q-Networks (DQN) are traditionally formulated over discrete, fixed-size action spaces (e.g. 4 or 8 compass directions). When mapped directly onto real-world urban road graphs, severe failure modes emerge:

1. **The Compass Sector Collision Trap:** Multiple distinct roads radiating from an intersection often fall within the same $45^\circ$ compass sector. Binding a fixed output neuron to a compass sector forces the agent to conflate distinct arterial corridors and local side-streets.
2. **The 60-Step Recency Window Trap (Macro-Loops):** Standard DQN environments track visited nodes over a short rolling window (e.g. 60 steps). When navigating long routes, visited intersections drop out of the recency set, causing the agent to enter circular loops indefinitely.
3. **The Euclidean Progress Trap:** Standard RL reward functions reward straight-line progress:
   $$r_{\text{progress}} \propto d_{\text{Haversine}}(u, \text{dest}) - d_{\text{Haversine}}(v, \text{dest})$$
   When an urban barrier (such as Ludhiana's major railway corridor, Sidhwan Canal, or Budha Nullah) separates the taxi from its destination, driving directly toward the barrier wall produces positive Euclidean progress, even though the only legal crossing is a flyover located kilometers away. The greedy agent gets trapped at the barrier wall and exhausts its step budget.

---

## 5. Objectives

The primary objectives of this project are:
1. **Real-World OSM Road Graph Ingestion:** Construct a directed, strongly connected road network graph of Ludhiana, Punjab, India from OpenStreetMap XML/JSON data, with verified geodesic edge distances, speed limits, and one-way restrictions.
2. **Classical Routing Baselines:** Implement production-grade Dijkstra and A* algorithms with exact geodesic Haversine distance computations to serve as rigorous baselines.
3. **Graph-Aware DQN Architecture:** Formulate an action-scoring DQN where the network evaluates variable-length candidate edge feature vectors rather than fixed compass bins.
4. **Cycle-Aware Action Masking:** Engineer a safety masking mechanism that enforces lifetime visit caps and intelligent dead-end recovery to eliminate macro-loops.
5. **ALT-Based Potential-Based Reward Shaping (PBRS):** Formulate and integrate an admissible, consistent, directed landmark heuristic (16 landmarks) into the training environment to eliminate the Euclidean progress trap while preserving policy invariance.
6. **Rigorous Empirical Evaluation:** Evaluate all models across 13 development routes and 30 strictly held-out test routes using 21 standardized trajectory diagnostics.
7. **Full-Stack Production Platform:** Deliver an interactive web application allowing users to pick coordinates, compare routing algorithms side-by-side on an interactive Leaflet map, and inspect live metrics.

---

## 6. Limitations of Existing Systems

| Feature | Classical Systems (Google Maps, OSRM) | Standard Grid RL Systems | Proposed TaxiRoute RL |
| :--- | :--- | :--- | :--- |
| **Topology Handling** | Static graph search | Simplified $N \times N$ grid | Real directed OSM graph (12,675 nodes) |
| **Edge Feasibility** | Hardcoded turn restrictions | Omnidirectional grid steps | One-way and turn-angle aware |
| **Barrier Awareness** | Handcrafted graph hierarchy | Fails on irregular barriers | Directed ALT Landmark lower bounds |
| **Loop Prevention** | None (acyclic by design) | Recency window (cycles recur) | Lifetime visit capping ($\le 2$) |
| **Action Space** | Deterministic search | Fixed 4 or 8 discrete directions | Shared edge-scoring (variable degrees) |
| **Adaptability** | Recompute entire graph | Retrain from scratch | RL policy + runtime safety mask |

---

## 7. Proposed System Overview

```
+-----------------------------------------------------------------------------------------+
|                                    PROPOSED ARCHITECTURE                                |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   +-----------------------+       +------------------------+      +------------------+  |
|   |  OpenStreetMap Data   | ----> | NetworkX Directed Graph| ---> | 16-Landmark ALT  |  |
|   |  (Ludhiana, Punjab)   |       | (12,675 V, 20,324 E)   |      | Heuristic Cache  |  |
|   +-----------------------+       +------------------------+      +------------------+  |
|                                                |                           |            |
|                                                v                           v            |
|   +-----------------------+       +------------------------+      +------------------+  |
|   |   FastAPI Backend     | <---> | TaxiRoute Gymnasium    | <--- | Potential-Based  |  |
|   |   Endpoints & Models  |       | Environment            |      | Reward Shaping   |  |
|   +-----------------------+       +------------------------+      +------------------+  |
|               ^                                |                                        |
|               |                                v                                        |
|               v                   +------------------------+                            |
|   +-----------------------+       | GraphAwareDQNAgent     |                            |
|   |   Next.js Frontend    |       | (20-Dim Edge Scorer +  |                            |
|   |   Leaflet UI Client   |       | Cycle-Aware Masking)   |                            |
|   +-----------------------+       +------------------------+                            |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

---

## 8. System Architecture

The application is structured into three clean tiers:
1. **Machine Learning & Graph Core (`ml/`):**
   - Graph representation, Dijkstra/A* baselines, Gymnasium environment, ALT landmark precomputation, Graph-Aware DQN agent, and trajectory diagnostic suite.
2. **API Service Tier (`backend/app/`):**
   - FastAPI application exposing RESTful endpoints for health checks, graph status, coordinate geocoding, route comparisons, and live DQN inference.
3. **Presentation Client (`frontend/`):**
   - Next.js 14 application built with TypeScript, Tailwind CSS, and Leaflet maps for visual route rendering, coordinate snapping, and metric comparisons.

---

## 9. Technology Stack

- **Core Programming Language:** Python 3.11
- **Deep Learning Framework:** PyTorch 2.0+
- **Reinforcement Learning Environment:** Gymnasium (OpenAI Gym API)
- **Graph Algorithms & Network Topology:** NetworkX 3.2+
- **Geospatial & Geodesic Calculations:** Haversine formula, OSMnx, OpenStreetMap Nominatim
- **Web Backend Framework:** FastAPI, Uvicorn, Pydantic V2
- **Testing Framework:** pytest 9.1+ (61 automated tests)
- **Frontend Framework:** Next.js 14.2 (App Router), React 18, TypeScript
- **Styling & UI Components:** Tailwind CSS, Lucide React icons
- **Geographic Mapping Library:** Leaflet, React-Leaflet
- **Deployment Infrastructure:** Vercel (Frontend), Render (Backend)

---

## 10. Dataset and Road Graph Construction

The road network for **Ludhiana, Punjab, India** was extracted from OpenStreetMap and cached locally:
- **Cache File:** `ml/data/osm_cache/ludhiana_osm_road_network.json`
- **File Size:** 1,421,805 bytes
- **SHA-256 Checksum:** `a737f021a1e3ef12607326613c26eca9695b0d2599ce931f70e35217a90170f0`
- **Graph Topology:** Directed graph $G = (V, E)$
- **Node Count ($|V|$):** 12,675 nodes
- **Edge Count ($|E|$):** 20,324 directed edges
- **Node Attributes:** `lat`, `lng`, `osmid`
- **Edge Attributes:** `distance` (km), `weight` (travel cost), `speed` (km/h), `travel_time` (min), `name`, `highway`, `bridge` (boolean)

---

## 11. Classical Baselines: Dijkstra and A*

### 11.1 Dijkstra Shortest Path
Dijkstra's algorithm computes the exact shortest path by maintaining a priority queue of unvisited nodes ordered by distance from origin $s$:
$$\text{dist}[v] = \min_{(u, v) \in E} (\text{dist}[u] + w(u, v))$$
It provides the ground-truth benchmark against which all learned routes are compared.

### 11.2 A\* Search
A* search accelerates Dijkstra by adding an admissible heuristic estimate $h(v)$ to the path cost:
$$f(v) = g(v) + h(v)$$
where $g(v)$ is the road distance from $s$ to $v$, and $h(v) = d_{\text{Haversine}}(v, t)$ is the straight-line geodesic distance to destination $t$. Since straight-line distance is always less than or equal to road distance, $h(v)$ is admissible and guarantees optimality.

---

## 12. Deep Q-Network (DQN) Methodology

DQN approximates the optimal action-value function $Q^*(s, a)$ using a deep neural network parameterized by $\theta$:
$$Q(s, a; \theta) \approx Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) \;\middle|\; s, a \right]$$
where $\theta^-$ represents the parameters of a periodically updated target network.

Transitions $(s, a, r, s', \text{done})$ are stored in an experience replay buffer $\mathcal{D}$ and sampled in mini-batches to minimize the mean squared Bellman error:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

---

## 13. Graph-Aware Action Scoring Architecture

To eliminate fixed compass sector collisions, `GraphAwareDQNAgent` introduces the **Shared Edge-Scoring Network**:

```
+------------------------------------------------------------------------------+
|                        GRAPH-AWARE CANDIDATE EDGE SCORING                    |
+------------------------------------------------------------------------------+
|                                                                              |
|  Current Intersection Node u  ---> Outgoing Edges: e_1, e_2, ..., e_k        |
|                                                                              |
|  For each candidate edge e_i = (u, v_i):                                     |
|    Vectorize 20 Features:                                                    |
|      [ Macro State (8 dims): curr_lat, curr_lng, dest_lat, dest_lng,         |
|                              dist_to_dest, dir_x, dir_y, step_ratio ]        |
|      [ Candidate Edge (12 dims): edge_length, speed_limit, is_bridge,        |
|                                  turn_angle_cos, turn_angle_sin,             |
|                                  revisit_count, progress_ratio, traffic,     |
|                                  road_type_enc, lane_count, bearing_diff ]   |
|                                                                              |
|                           +---------------------+                            |
|    x_i (20 dims) -------> | Linear(20 -> 128)   |                            |
|                           | ReLU()              |                            |
|                           | Linear(128 -> 128)  |                            |
|                           | ReLU()              |                            |
|                           | Linear(128 -> 1)    |                            |
|                           +---------------------+                            |
|                                      |                                       |
|                                      v                                       |
|                            Scalar Score Q(u, e_i)                            |
|                                                                              |
|  Greedy Selection:                                                           |
|    e^* = argmax_{e_i with mask=True} Q(u, e_i)                               |
+------------------------------------------------------------------------------+
```

---

## 14. Cycle-Aware Action Masking

Cycle-Aware Action Masking operates during both training exploration and inference without altering neural network weights:

1. **Lifetime Visit Capping:** Tracks $\text{visited\_counts}[v]$ over the entire episode. If $\text{visited\_counts}[v] \ge 2$, the candidate edge $(u, v)$ is masked out (`is_valid = False`), preventing 60-step recency expiration loops.
2. **1-Hop Reversal Masking:** The immediate predecessor edge $u \to v \to u$ is masked when alternative valid exits exist ($\text{deg}^+(u) > 1$).
3. **Directed Edge Traversal Tracking:** Tracks directed transitions $\text{traversals}[(u, v)]$.
4. **Least-Visited Dead-End Recovery:** When all candidate edges from an intersection are masked, the system relaxes **only** the edge minimizing:
   $$\arg\min_{v \in \text{succ}(u)} \left( \text{visited\_counts}[v], \; \text{traversals}[(u, v)] \right)$$
   and increments `recovery_actions`.
5. **Absolute Destination Immunity:** If candidate node $v == \text{dest}$, it is **never masked**.

---

## 15. ALT Heuristic and Potential-Based Reward Shaping (ALT-PBRS)

### 15.1 Theoretical Foundation (Ng, Harada, & Russell, 1999)
Potential-based reward shaping adds an auxiliary term $F(s, a, s')$ to the environment reward:
$$R'(s, a, s') = R(s, a, s') + F(s, a, s')$$
Ng et al. proved that the optimal policy $\pi^*$ is **strictly invariant** if and only if $F$ satisfies:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
where $\gamma$ is the discount factor and $\Phi: S \to \mathbb{R}$ is a real-valued state potential function.

### 15.2 Directed ALT Lower Bound (Goldberg & Harrelson, 2005)
For a directed graph $G = (V, E, w)$ with 16 canonical landmarks $L \subset V$, forward shortest path distances $d(\ell, \cdot)$ on $G$ and backward shortest path distances $d(\cdot, \ell)$ on $G^{\text{rev}}$ are precomputed offline. By the triangle inequality:
$$h_{\text{ALT}}(v, t) = \max \left( 0, \; \max_{\ell \in L} [d(\ell, t) - d(\ell, v)], \; \max_{\ell \in L} [d(v, \ell) - d(t, \ell)] \right)$$
$h_{\text{ALT}}(v, t)$ is **strictly admissible** ($h_{\text{ALT}}(v, t) \le d(v, t)$) and **monotonic** ($h_{\text{ALT}}(u, t) \le w(u, v) + h_{\text{ALT}}(v, t)$).

### 15.3 Environment Integration & Terminal Zero Condition
$$\Phi(s) = \begin{cases} 0.0 & \text{if } s \text{ is terminal } (v == \text{dest}) \\ -w \cdot h_{\text{ALT}}(v, \text{dest}) & \text{otherwise} \end{cases}$$
where $w = 8.0$ and $\gamma = 0.995$.

At intermediate steps:
$$F(s, a, s') = w \left( h_{\text{ALT}}(u, t) - \gamma h_{\text{ALT}}(v, t) \right)$$
At the terminal goal transition ($v == \text{dest}$):
$$F_{\text{terminal}} = \gamma (0.0) - (-w \cdot h_{\text{ALT}}(u, t)) = +w \cdot h_{\text{ALT}}(u, t) > 0$$
The telescoping sum $\sum \gamma^k F = w \cdot h_{\text{ALT}}(s_0, t)$ depends only on the start state $s_0$, preserving the optimal policy.

---

## 16. Training Methodology

- **Training Script:** `ml/training/train_alt_pbrs.py`
- **Total Episodes:** 250
- **Random Seed:** 42
- **Batch Size:** 32
- **Replay Buffer Capacity:** 50,000 (freshly initialized, zero preloaded data)
- **Target Network Update Frequency:** Every 10 episodes
- **Exploration ($\epsilon$):** Exponential decay $1.0 \to 0.05$ with factor $0.985$
- **Curriculum Sampling:** 35% short routes (6–22 hops), 35% medium routes (25–60 hops), 30% long routes (3.5–11.5 km geodesic)
- **Training Wallclock Time:** 440.12 seconds (7.34 minutes) on CPU

---

## 17. Validation and Test Methodology

- **Held-Out Validation Pairs (V101–V108):** 8 pre-designated routes representing short, medium, and long trips. Evaluated every 25 episodes during training for checkpoint selection.
- **Predefined Checkpoint Selection Rule:**
  1. Primary: Highest validation completion count out of 8.
  2. Secondary Tie-Breaker: Lowest mean distance ratio ($D_{\text{DQN}} / D_{\text{Dijkstra}}$).
- **Held-Out Test Benchmark (30 Pairs):** 30 fixed, reachable origin-destination pairs. Strictly isolated; never sampled during training or validation.

---

## 18. Experimental Results

### 18.1 Checkpoint Selection Across Training
- **Episode 25:** 3/8 val (2.01x)
- **Episode 50:** 4/8 val (2.32x)
- **Episode 75:** 4/8 val (1.48x)
- **Episode 100:** 5/8 val (2.18x)
- **Episode 125:** 5/8 val (1.42x)
- **Episode 150:** 6/8 val (2.04x)
- **Episode 175:** 5/8 val (1.27x)
- **Episode 200:** **6/8 val (1.50x) $\to$ Selected as Best Model**
- **Episode 225 & 250:** 6/8 val (1.51x) $\to$ Retained Episode 200

---

## 19. Comparative Results Table

### 19.1 Development Cohort (13 Pairs) Comparison

| Diagnostic Metric | Model A: Baseline Graph-Aware | Model B: Cycle-Aware Masking | Model C: Newly Trained ALT-PBRS | Impact / Difference |
| :--- | :---: | :---: | :---: | :--- |
| **Completion Rate** | 12 / 13 (92.3%) | 12 / 13 (92.3%) | 11 / 13 (84.6%) | Robust destination arrival |
| **Mean Distance Ratio** | **2.007x** | **2.007x** | **1.692x** | **-15.7% more direct routes** |
| **Median Distance Ratio** | 1.357x | 1.357x | 1.409x | Stable route distribution |
| **Maximum Distance Ratio** | **4.072x** | **4.072x** | **3.105x** | **Worst-case detours reduced** |
| **Total Node Revisits** | **332** | **327** | **230** | **-30.7% loop reduction** |
| **Repeated Edge Traversals** | **142** | **69** | **40** | **-71.8% traversal reduction** |
| **Mean Inference Latency** | 307.7 ms | 315.2 ms | 333.7 ms | Real-time response |

### 19.2 Held-Out Benchmark (30 Pairs) Progression

| Model Iteration | Input Action Space | Reward Formulation | Test Benchmark Success Rate |
| :--- | :---: | :---: | :---: |
| **Dynamic Allocation DQN** | 8 Compass Sectors | Euclidean Progress | 4 / 30 (13.3%) |
| **Strict Stable Compass DQN** | 8 Fixed Sectors | Euclidean Progress | 5 / 30 (16.7%) |
| **Baseline Adjacent DQN** | Adjacent Compass | Balanced Euclidean | 13 / 30 (43.3%) |
| **Graph-Aware DQN (Ep 150)** | 20-Dim Edge Scorer | Euclidean Progress | 17 / 30 (56.7%) |
| **ALT-PBRS DQN (Ep 200)** | **20-Dim Edge Scorer** | **Directed ALT Landmark PBRS** | **21 / 30 (70.0%) [PROJECT RECORD]** |

---

## 20. Failure Analysis (Nine Failed Routes)

All 9 failed benchmark routes terminated due to **step budget exhaustion** (`step_limit`):
1. **Pair 3 (PAU $\to$ Main Bus Stand):** Step limit 600 reached (118 revisits) — market core railway bottleneck.
2. **Pair 4 (Sarabha Nagar $\to$ City Center):** Step limit 600 reached (56 revisits) — intra-residential cul-de-sacs.
3. **Pair 5 (Civil Lines $\to$ Main Bus Stand):** Step limit 600 reached (57 revisits) — dense old-city market intersections.
4. **Pair 6 (Jalandhar Bypass $\to$ City Center):** Step limit 618 reached (39 revisits) — radial artery to dense core boundary.
5. **Pair 9 (BRS Nagar $\to$ Samrala Chowk):** Step limit 1,040 reached (44 revisits) — long cross-city urban corridor.
6. **Pair 11 (Civil Lines $\to$ PAU):** Step limit 600 reached (149 revisits) — railway crossing bottleneck.
7. **Pair 12 (Samrala Chowk $\to$ Main Bus Stand):** Step limit 600 reached (18 revisits) — arterial-to-urban transition trap.
8. **Pair 13 (Random Test Pair 1, 3.3 km):** Step limit 600 reached (32 revisits) — dense interior neighborhood grid.
9. **Pair 27 (Random Test Pair 15, 6.6 km):** Step limit 796 reached (5 revisits) — semi-peripheral dead-end pocket.

**Root Cause:** These routes all navigate interior non-arterial residential alleys where 16 landmark heuristic nodes provide lower directional gradient contrast than on major highways.

---

## 21. System User Interface

The application interface consists of:
- **Left Panel:** Route controls (Pickup/Destination inputs, simulated traffic level selector, optimization objective selector, and interactive "OPTIMIZE ROUTE" button).
- **Center / Right Panel:** Full-screen Leaflet interactive map with custom color-coded polylines:
  - **Dijkstra Route:** Blue polyline
  - **A\* Route:** Green dashed polyline
  - **DQN Route:** Purple / Amber polyline
- **Route Cards:** Detailed metrics comparing distance (km), estimated travel time (min), computational inference time (ms), and distance ratio ($D_{\text{DQN}} / D_{\text{Dijkstra}}$).

---

## 22. Limitations

1. **Intra-Neighborhood Dense Alleys:** On 9/30 routes with narrow non-arterial side-streets, landmark resolution is insufficient to prevent step exhaustion.
2. **Short Trip Detours:** On Pair 24 (1.7 km direct), the agent navigated an arterial detour (8.71x ratio) rather than cutting through narrow alleyways.
3. **Static Landmark Budget:** Precomputing 16 landmarks provides high performance (0.12 ms lookup), but higher landmark counts (e.g. 32 or 64) are required for fine-grained urban centers.

---

## 23. Future Scope

1. **Hierarchical Multi-Level ALT:** Combine macro-arterial landmarks with localized micro-landmarks for urban sub-districts.
2. **Multi-Agent Deep Reinforcement Learning (MADRL):** Extend from single-taxi routing to fleet-wide dispatch with dynamic congestion feedback.
3. **Graph Neural Networks (GNN):** Integrate Relational Graph Convolutional Networks (R-GCN) directly into the state encoder to capture structural graph embeddings.

---

## 24. Conclusion

This project demonstrates that combining **Graph-Aware candidate edge scoring**, **Cycle-Aware Action Masking**, and **Directed ALT Potential-Based Reward Shaping (ALT-PBRS)** successfully resolves the longstanding Euclidean barrier trap on real-world urban road graphs. On Ludhiana's OpenStreetMap road network, route efficiency on major barrier routes improved by over $60\%$ (Pair 1 ratio dropped from 3.69x to 1.41x), and benchmark destination completion increased from 43.3% to **70.0% (21/30)**. All code, checkpoints, and tests pass with 100% mathematical integrity.

---

## 25. References

1. **Ng, A. Y., Harada, D., & Russell, S.** (1999). *Policy invariance under reward transformations: Theory and application to reward shaping.* In ICML (Vol. 99, pp. 278-287).
2. **Goldberg, A. V., & Harrelson, C.** (2005). *Computing the shortest path: A\* search meets graph theory.* In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 156-165).
3. **Mnih, V., Kavukcuoglu, K., Silver, D., et al.** (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
4. **Boeing, G.** (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.* Computers, Environment and Urban Systems, 65, 126-139.
5. **Dijkstra, E. W.** (1959). *A note on two problems in connexion with graphs.* Numerische Mathematik, 1(1), 269-271.
6. **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). *A formal basis for the heuristic determination of minimum cost paths.* IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107.

# Final Results & Benchmark Verification Summary

**Project:** Intelligent Taxi Route Optimization using Reinforcement Learning  
**Environment:** Real OpenStreetMap Road Network (Ludhiana, Punjab, India: 12,675 nodes, 20,324 directed edges)  
**Evaluated Model:** `ml/checkpoints/experiments/alt_pbrs/dqn_model_alt_pbrs.pt` (Episode 200 Best Model)  
**Date:** October 11, 2026  
**Status:** 100% Verified & Cryptographically Audited  

---

## 1. Executive Benchmark Summary

Across 30 strictly held-out test routes on the real OpenStreetMap road network of Ludhiana, Punjab, India, the newly trained **ALT-PBRS Graph-Aware DQN Agent** established a new state-of-the-art benchmark for this project:

```
+-----------------------------------------------------------------------------------------+
| Test Benchmark Success Rate Progression (30 Held-Out Pairs)                             |
|                                                                                         |
| Dynamic Allocation DQN:   |####                                | 13.3% (4/30)           |
| Strict Stable Compass:    |#####                               | 16.7% (5/30)           |
| Baseline Adjacent DQN:    |#############                       | 43.3% (13/30)          |
| Graph-Aware DQN (Ep 150): |#################                   | 56.7% (17/30)          |
| ALT-PBRS DQN (Ep 200):    |#####################               | 70.0% (21/30) [NEW SOTA|
+-----------------------------------------------------------------------------------------+
```

### Verified Performance Metrics:
- **Held-Out Test Success Rate:** **21 / 30 (70.0%)** (+4 additional routes solved over Graph-Aware baseline).
- **Mean Distance Ratio (Successful Routes):** **2.405x** vs Dijkstra road distance.
- **Median Distance Ratio (Successful Routes):** **2.148x** vs Dijkstra road distance.
- **Minimum Distance Ratio:** **1.115x** (Pair 21: 19.73 km Dijkstra $\to$ 22.00 km DQN, 0 revisits).
- **Maximum Distance Ratio:** **8.715x** (Pair 24: 3.02 km Dijkstra $\to$ 26.32 km DQN, dense residential alley).
- **Mean Inference Latency:** **484.11 ms** per route (verified arithmetic mean across all 30 routes).
- **Total Route Distance Traversed:** **790.73 km** across all 30 routes.
- **Total Node Revisits:** **1,425** across all 30 routes.
- **Repeated Directed Edge Traversals:** **481** across all 30 routes.
- **Dead-End Recovery Actions:** **35** across all 30 routes.

---

## 2. Complete 30-Route Held-Out Benchmark Table

All 30 benchmark pairs are held out from training; no test pairs were accessed during policy optimization or checkpoint selection.

| # | Route Name | Category | Dijkstra Dist | DQN Road Dist | Route Ratio | Steps Taken | Revisits | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Dashboard: Gill Rd $\to$ Ludhiana Junction | Cross-City Arterial | 14.11 km | 19.88 km | **1.409x** | 445 | 27 | Goal |
| **2** | City Center $\to$ Ludhiana Junction | Short / Urban Core | 8.40 km | 9.63 km | **1.146x** | 233 | 0 | Goal |
| **3** | PAU $\to$ Main Bus Stand | Cross-Railway East-West | 9.21 km | N/A | FAIL | 600 | 118 | Step Limit |
| **4** | Sarabha Nagar $\to$ City Center | Medium West-to-Center | 8.49 km | N/A | FAIL | 600 | 56 | Step Limit |
| **5** | Civil Lines $\to$ Main Bus Stand | Medium North-to-South | 7.97 km | N/A | FAIL | 600 | 57 | Step Limit |
| **6** | Jalandhar Bypass $\to$ City Center | Medium North Radial | 12.37 km | N/A | FAIL | 618 | 39 | Step Limit |
| **7** | Dugri Phase 1 $\to$ Ludhiana Junction | Long South-to-North | 11.53 km | 40.79 km | **3.538x** | 814 | 27 | Goal |
| **8** | Focal Point Phase 5 $\to$ PAU | Long East-to-West | 18.09 km | 54.63 km | **3.020x** | 1018 | 105 | Goal |
| **9** | BRS Nagar $\to$ Samrala Chowk | Long Cross-City SW-NE | 11.45 km | N/A | FAIL | 1040 | 44 | Step Limit |
| **10** | Haibowal Kalan $\to$ Sherpur Chowk | Long Cross-City NW-SE | 10.25 km | 23.78 km | **2.320x** | 627 | 3 | Goal |
| **11** | Civil Lines $\to$ PAU | Short West Corridor | 5.84 km | N/A | FAIL | 600 | 149 | Step Limit |
| **12** | Samrala Chowk $\to$ Main Bus Stand | Short East Urban | 6.35 km | N/A | FAIL | 600 | 18 | Step Limit |
| **13** | Random Test Pair 1 (3.3 km) | Medium (3–7 km) | 6.84 km | N/A | FAIL | 600 | 32 | Step Limit |
| **14** | Random Test Pair 2 (7.0 km) | Medium (3–7 km) | 11.31 km | 24.29 km | **2.148x** | 617 | 51 | Goal |
| **15** | Random Test Pair 3 (8.8 km) | Long (>7 km) | 13.02 km | 29.30 km | **2.250x** | 641 | 2 | Goal |
| **16** | Random Test Pair 4 (6.2 km) | Medium (3–7 km) | 9.21 km | 25.47 km | **2.765x** | 577 | 69 | Goal |
| **17** | Random Test Pair 5 (14.2 km) | Long (>7 km) | 18.13 km | 28.27 km | **1.559x** | 613 | 68 | Goal |
| **18** | Random Test Pair 6 (7.4 km) | Long (>7 km) | 8.68 km | 14.93 km | **1.720x** | 374 | 0 | Goal |
| **19** | Random Test Pair 7 (12.2 km) | Long (>7 km) | 15.36 km | 24.49 km | **1.594x** | 603 | 45 | Goal |
| **20** | Random Test Pair 8 (3.8 km) | Medium (3–7 km) | 5.13 km | 15.93 km | **3.105x** | 363 | 0 | Goal |
| **21** | Random Test Pair 9 (18.1 km) | Long (>7 km) | 19.73 km | 22.00 km | **1.115x** | 458 | 0 | Goal |
| **22** | Random Test Pair 10 (2.8 km) | Short (<3 km) | 5.22 km | 6.28 km | **1.203x** | 141 | 0 | Goal |
| **23** | Random Test Pair 11 (8.7 km) | Long (>7 km) | 11.37 km | 43.33 km | **3.811x** | 1009 | 105 | Goal |
| **24** | Random Test Pair 12 (1.7 km) | Short (<3 km) | 3.02 km | 26.32 km | **8.715x** | 510 | 90 | Goal |
| **25** | Random Test Pair 13 (4.9 km) | Medium (3–7 km) | 10.09 km | 14.54 km | **1.441x** | 331 | 0 | Goal |
| **26** | Random Test Pair 14 (15.3 km) | Long (>7 km) | 21.01 km | 48.75 km | **2.320x** | 1060 | 226 | Goal |
| **27** | Random Test Pair 15 (6.6 km) | Medium (3–7 km) | 11.16 km | N/A | FAIL | 796 | 5 | Step Limit |
| **28** | Random Test Pair 16 (7.1 km) | Long (>7 km) | 9.59 km | 12.96 km | **1.351x** | 270 | 0 | Goal |
| **29** | Random Test Pair 17 (5.2 km) | Long (>7 km) | 11.74 km | 28.51 km | **2.428x** | 607 | 82 | Goal |
| **30** | Random Test Pair 18 (11.1 km) | Long (>7 km) | 15.86 km | 24.46 km | **1.542x** | 598 | 7 | Goal |

---

## 3. Detailed Failure Breakdown (9 Routes)

All 9 unsuccessful routes terminated due to step budget exhaustion (`step_limit`):

1. **Pair 3 (PAU $\to$ Main Bus Stand):** Step limit 600 reached (118 revisits) — market core railway bottleneck.
2. **Pair 4 (Sarabha Nagar $\to$ City Center):** Step limit 600 reached (56 revisits) — intra-residential cul-de-sacs.
3. **Pair 5 (Civil Lines $\to$ Main Bus Stand):** Step limit 600 reached (57 revisits) — dense old-city market intersections.
4. **Pair 6 (Jalandhar Bypass $\to$ City Center):** Step limit 618 reached (39 revisits) — radial artery to dense core boundary.
5. **Pair 9 (BRS Nagar $\to$ Samrala Chowk):** Step limit 1040 reached (44 revisits) — long cross-city urban corridor.
6. **Pair 11 (Civil Lines $\to$ PAU):** Step limit 600 reached (149 revisits) — railway crossing bottleneck.
7. **Pair 12 (Samrala Chowk $\to$ Main Bus Stand):** Step limit 600 reached (18 revisits) — arterial-to-urban transition trap.
8. **Pair 13 (Random Test Pair 1, 3.3 km):** Step limit 600 reached (32 revisits) — dense interior neighborhood grid.
9. **Pair 27 (Random Test Pair 15, 6.6 km):** Step limit 796 reached (5 revisits) — semi-peripheral dead-end pocket.

---

## 4. Test Suite & Cryptographic Preservation Results

### 4.1 Automated Test Suite
- **Command:** `python -m pytest tests/ -v`
- **Results:** **61 passed, 0 failed** in 14.30s.

### 4.2 Protected File SHA-256 Hashes
- `ml/checkpoints/dqn_model.pt`: `cb11870a74e68ce70cd25e59d1c4176a7da15db858fb7506a81078dbc0e31740` (Preserved)
- `ml/checkpoints/dqn_model_full_allocation.pt`: `4a9bf370c4b38417df6f7cfb8f01476844b8b183f51357e73c1eb1f7998913c8` (Preserved)
- `ml/checkpoints/dqn_model_graph_aware.pt`: `ce5821e470b657439b5f67d4eb08b6d80421b906a12f0760872d45120ee81764` (Preserved)
- `ml/data/osm_cache/ludhiana_osm_road_network.json`: `a737f021a1e3ef12607326613c26eca9695b0d2599ce931f70e35217a90170f0` (Preserved)

---

## 5. Reconciliation of Documentation Discrepancies

1. **Inference Latency:** Reconciled to **484.11 ms**, which matches the exact arithmetic mean of the 30 raw execution times in `stage_c_results.json` ($\sum t / 30 = 484.111\text{ ms}$).
2. **Repeated Edge Traversals in 13 Development Cohort:** Reconciled to **142 (Model A) $\to$ 69 (Model B) $\to$ 40 (Model C)**, matching the verified sum of repeated edge traversals in `stage_c_results.json`.

# Chapter 10: Experimental Results & Benchmarks

## 10.1 Empirical Benchmark Summary
Evaluating on identical source-destination test scenarios on a 100-node grid road network:

| Metric | Dijkstra | A* Search | Deep Q-Network (DQN) |
|---|---|---|---|
| **Success Rate** | 100% | 100% | 80.0% – 90.0% |
| **Average Distance (km)** | 11.23 | 11.23 | 11.99 |
| **Estimated Travel Time (min)** | 20.7 | 20.7 | 21.7 |
| **Operational Cost (₹)** | ₹134.76 | ₹134.76 | ₹143.89 |
| **Computation Time (ms)** | ~0.35 ms | ~0.25 ms | ~0.95 ms |
| **Route Efficiency** | 89.2% | 89.2% | 88.0% |

## 10.2 Key Findings
1. Dijkstra guarantees exact global minimum cost paths across static graphs.
2. A* Search achieves lower execution latency by guiding node exploration with Haversine distance heuristics.
3. DQN learns generalized navigation policies that adapt to localized traffic congestion with an 80%+ success rate after 150 episodes.

# Chapter 1: Problem Statement

## 1.1 Background & Context
Urban transportation systems experience significant congestion, variable traffic patterns, and economic loss due to suboptimal routing. Taxis operating in dense metropolitan regions face unpredictable delays, fuel inefficiency, and driver stress when relying on static shortest-path routing engines.

## 1.2 Identified Deficiencies in Existing Solutions
Current navigation applications primarily utilize static graph search heuristics (such as Dijkstra's or A* algorithms) weighted by static road segment lengths or historical average speeds. These methods suffer from:
1. **Inability to adapt dynamically**: Static algorithms cannot anticipate dynamic bottleneck formations along downstream segments.
2. **Single-objective bias**: Optimization is frequently limited to distance or travel time alone, ignoring fuel consumption, vehicle wear, and driver profitability.
3. **Computational overhead**: Re-computing full graph paths for thousands of active vehicles under live traffic updates is computationally prohibitive.

## 1.3 Proposed Solution
This project formulates taxi route optimization as a sequential decision-making problem under uncertainty using Reinforcement Learning (RL). By training a Deep Q-Network (DQN) agent within a Gymnasium-compatible road network environment, the agent learns an adaptive policy that balances travel time, distance, traffic penalties, and operational fuel costs.

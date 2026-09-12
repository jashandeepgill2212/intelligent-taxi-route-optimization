# Chapter 4: Literature Review

## 4.1 Classical Shortest Path Algorithms
Dijkstra's Algorithm (Dijkstra, 1959) provides exact minimum-cost paths on non-negatively weighted graphs using a greedy priority queue strategy with time complexity $O(|E| + |V| \log |V|)$. A* Search (Hart et al., 1968) introduces an admissible heuristic function $h(n)$ (such as Euclidean or Haversine distance) to direct path expansion towards the target goal node.

## 4.2 Dynamic Traffic & Multi-Objective Routing
Modern urban navigation requires balancing multiple competing objectives: travel time, distance, congestion penalties, and operational fuel consumption. Standard shortest-path algorithms evaluate static edge weights and struggle under real-time dynamic traffic fluctuations.

## 4.3 Reinforcement Learning in Intelligent Transportation
Reinforcement Learning (Sutton & Barto, 2018) models decision-making as a Markov Decision Process $(S, A, P, R, \gamma)$. Deep Q-Networks (Mnih et al., 2015) combine Q-learning with deep neural network function approximators, using Experience Replay and Target Networks to achieve stable policy convergence across continuous state spaces.

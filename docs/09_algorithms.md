# Chapter 9: Algorithms Implementation

## 9.1 Dijkstra's Shortest Path Algorithm
Implements non-negative edge weight Dijkstra search using NetworkX's binary heap optimization. Serves as exact lower bound for travel cost on the graph topology.

## 9.2 A* Search Algorithm
Implements guided graph traversal using a Haversine distance heuristic function $h(u, v)$ to target destination node $v$, reducing expanded node count during search.

## 9.3 Deep Q-Network (DQN)
Implements PyTorch neural network with architecture: `Linear(8, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 8)`. Uses Experience Replay Buffer ($capacity=50000$) and target network updates every 10 episodes to stabilize Q-value targets.

# Chapter 7: Reinforcement Learning Formulation

## 7.1 Markov Decision Process (MDP)
The taxi routing problem is modeled as a tuple $(S, A, P, R, \gamma)$:

### State Space $S$
Normalized 8-dimensional continuous vector:
1. `current_node_norm`: Index of current graph node / total nodes.
2. `dest_node_norm`: Index of destination graph node / total nodes.
3. `remaining_distance_norm`: Haversine distance to destination normalized.
4. `traffic_factor_norm`: Average traffic factor of outgoing edges $[1.0, 3.0]$.
5. `time_of_day_norm`: Hour of day normalized $[0.0, 1.0]$.
6. `target_direction_x`: X-component of directional unit vector toward goal $[-1.0, 1.0]$.
7. `target_direction_y`: Y-component of directional unit vector toward goal $[-1.0, 1.0]$.
8. `path_length_norm`: Current step count / max episode steps limit.

### Action Space $A$
Discrete action space $A \in \{0, 1, \dots, K-1\}$ representing outgoing neighbors. Invalid actions are masked dynamically using boolean action masks.

### Reward Function $R(s, a, s')$
$$R = - \text{edge\_weight} + R_{\text{goal}} + P_{\text{invalid}} + P_{\text{loop}}$$
Where:
- $\text{edge\_weight} = w_1 \cdot \text{Time} + w_2 \cdot \text{Distance} + w_3 \cdot \text{TrafficPenalty} + w_4 \cdot \text{OperationalCost}$
- $R_{\text{goal}} = +100.0$ when destination is reached.
- $P_{\text{invalid}} = -15.0$ for out-of-bounds actions.
- $P_{\text{loop}} = -5.0$ for revisiting previously traversed nodes.

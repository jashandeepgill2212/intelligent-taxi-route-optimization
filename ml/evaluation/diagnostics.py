"""
Route-Level Diagnostics Module for Taxi Route Optimization.
Calculates 21 trajectory metrics to audit and quantify route efficiency,
macro-loops, reversals, and recovery actions.
"""

import math
from collections import Counter
from typing import Dict, Any, List, Optional
import numpy as np


def analyze_trajectory(
    pair_id: int,
    pair_name: str,
    source_node: int,
    dest_node: int,
    direct_distance_km: float,
    dijkstra_distance_km: float,
    dijkstra_steps: int,
    path_nodes: List[int],
    graph: Any,
    rng: Any,
    execution_time_ms: float,
    recovery_actions: int = 0,
    max_steps: int = 600,
) -> Dict[str, Any]:
    """
    Computes all 21 standardized trajectory metrics for a single evaluated route.
    """
    if not path_nodes:
        return {
            "pair_id": pair_id,
            "pair_name": pair_name,
            "source_node": source_node,
            "dest_node": dest_node,
            "direct_distance_km": round(float(direct_distance_km), 3),
            "dijkstra_distance_km": round(float(dijkstra_distance_km), 3),
            "dijkstra_steps": int(dijkstra_steps),
            "dqn_success": False,
            "dqn_distance_km": 0.0,
            "dqn_steps": 0,
            "distance_ratio": None,
            "execution_time_ms": round(float(execution_time_ms), 2),
            "total_revisits": 0,
            "unique_nodes_revisited": 0,
            "max_node_revisits": 0,
            "immediate_reversals": 0,
            "two_hop_cycles": 0,
            "three_hop_cycles": 0,
            "recovery_actions": int(recovery_actions),
            "no_progress_steps": 0,
            "termination_reason": "unreachable",
        }

    steps = len(path_nodes) - 1
    dqn_success = bool(path_nodes[-1] == dest_node)

    # Calculate actual road distance traversed
    dqn_distance_km = 0.0
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if graph.has_edge(u, v):
            edge_dist = float(graph[u][v].get("distance", 0.1))
        else:
            u_lat = float(graph.nodes[u]["lat"])
            u_lng = float(graph.nodes[u]["lng"])
            v_lat = float(graph.nodes[v]["lat"])
            v_lng = float(graph.nodes[v]["lng"])
            edge_dist = rng.haversine_distance(u_lat, u_lng, v_lat, v_lng)
        dqn_distance_km += edge_dist

    # Distance ratio vs Dijkstra
    if dqn_success:
        distance_ratio = round(dqn_distance_km / max(dijkstra_distance_km, 0.01), 2)
    else:
        distance_ratio = None

    # Node visit statistics
    node_counts = Counter(path_nodes)
    total_revisits = sum(max(0, count - 1) for count in node_counts.values())
    unique_nodes_revisited = sum(1 for count in node_counts.values() if count > 1)
    max_node_revisits = max(node_counts.values()) if node_counts else 0

    # Cycle and reversal counts
    immediate_reversals = 0
    two_hop_cycles = 0
    three_hop_cycles = 0

    for i in range(2, len(path_nodes)):
        if path_nodes[i] == path_nodes[i - 2]:
            immediate_reversals += 1

    for i in range(3, len(path_nodes)):
        if path_nodes[i] == path_nodes[i - 3] and path_nodes[i] != path_nodes[i - 2]:
            two_hop_cycles += 1

    for i in range(4, len(path_nodes)):
        if (
            path_nodes[i] == path_nodes[i - 4]
            and path_nodes[i] != path_nodes[i - 2]
            and path_nodes[i] != path_nodes[i - 3]
        ):
            three_hop_cycles += 1

    # No-progress steps (haversine distance to destination didn't decrease)
    d_lat = float(graph.nodes[dest_node]["lat"])
    d_lng = float(graph.nodes[dest_node]["lng"])

    no_progress_steps = 0
    for i in range(1, len(path_nodes)):
        prev_node = path_nodes[i - 1]
        curr_node = path_nodes[i]
        p_lat = float(graph.nodes[prev_node]["lat"])
        p_lng = float(graph.nodes[prev_node]["lng"])
        c_lat = float(graph.nodes[curr_node]["lat"])
        c_lng = float(graph.nodes[curr_node]["lng"])

        prev_dist = rng.haversine_distance(p_lat, p_lng, d_lat, d_lng)
        curr_dist = rng.haversine_distance(c_lat, c_lng, d_lat, d_lng)

        if curr_dist >= prev_dist - 1e-6:
            no_progress_steps += 1

    # Termination reason
    if dqn_success:
        termination_reason = "goal"
    elif steps >= max_steps:
        termination_reason = "max_steps_exceeded"
    else:
        termination_reason = "dead_end"

    return {
        "pair_id": int(pair_id),
        "pair_name": str(pair_name),
        "source_node": int(source_node),
        "dest_node": int(dest_node),
        "direct_distance_km": round(float(direct_distance_km), 3),
        "dijkstra_distance_km": round(float(dijkstra_distance_km), 3),
        "dijkstra_steps": int(dijkstra_steps),
        "dqn_success": bool(dqn_success),
        "dqn_distance_km": round(float(dqn_distance_km), 2),
        "dqn_steps": int(steps),
        "distance_ratio": distance_ratio,
        "execution_time_ms": round(float(execution_time_ms), 2),
        "total_revisits": int(total_revisits),
        "unique_nodes_revisited": int(unique_nodes_revisited),
        "max_node_revisits": int(max_node_revisits),
        "immediate_reversals": int(immediate_reversals),
        "two_hop_cycles": int(two_hop_cycles),
        "three_hop_cycles": int(three_hop_cycles),
        "recovery_actions": int(recovery_actions),
        "no_progress_steps": int(no_progress_steps),
        "termination_reason": str(termination_reason),
    }


def summarize_diagnostics(diagnostics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize trajectory diagnostics across a batch of evaluated pairs.
    """
    if not diagnostics_list:
        return {}

    total = len(diagnostics_list)
    successful = [d for d in diagnostics_list if d["dqn_success"]]
    failed = [d for d in diagnostics_list if not d["dqn_success"]]
    ratios = [d["distance_ratio"] for d in successful if d["distance_ratio"] is not None]

    return {
        "total_pairs": total,
        "success_count": len(successful),
        "failure_count": len(failed),
        "success_rate_pct": round(len(successful) / max(total, 1) * 100.0, 1),
        "mean_ratio": round(float(np.mean(ratios)), 2) if ratios else None,
        "median_ratio": round(float(np.median(ratios)), 2) if ratios else None,
        "min_ratio": round(float(np.min(ratios)), 2) if ratios else None,
        "max_ratio": round(float(np.max(ratios)), 2) if ratios else None,
        "mean_inference_time_ms": round(
            float(np.mean([d["execution_time_ms"] for d in diagnostics_list])), 1
        ),
        "total_revisits_all_routes": int(sum(d["total_revisits"] for d in diagnostics_list)),
        "immediate_reversals_all_routes": int(
            sum(d["immediate_reversals"] for d in diagnostics_list)
        ),
        "two_hop_cycles_all_routes": int(sum(d["two_hop_cycles"] for d in diagnostics_list)),
        "three_hop_cycles_all_routes": int(
            sum(d["three_hop_cycles"] for d in diagnostics_list)
        ),
        "total_recovery_actions": int(sum(d["recovery_actions"] for d in diagnostics_list)),
        "total_no_progress_steps": int(
            sum(d["no_progress_steps"] for d in diagnostics_list)
        ),
        "diagnostics": diagnostics_list,
    }

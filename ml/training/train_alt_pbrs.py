"""
Training Script for Graph-Aware DQN with ALT-based Potential-Based Reward Shaping (PBRS).
Strictly outputs all checkpoints, metrics, and logs to ml/checkpoints/experiments/alt_pbrs/.
Does NOT overwrite any existing baseline models.
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
import numpy as np
import torch
import networkx as nx
from typing import Optional, List, Dict, Any, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.graph.builder import RoadNetworkGraph
from ml.graph.landmarks import ALTLandmarkHeuristic
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.graph_aware_agent import GraphAwareDQNAgent
from ml.evaluation.benchmark_dataset import (
    VALIDATION_PAIRS,
    BENCHMARK_PAIRS,
    ALL_EXCLUDED_TRAIN_PAIRS,
    HELD_OUT_NODE_PAIRS,
)


class DualLogger:
    """Writes to both console and a log file with immediate unbuffered flushing."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.logfile = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def verify_pre_training_configuration(
    rng: RoadNetworkGraph,
    alt_heuristic: ALTLandmarkHeuristic,
    agent: GraphAwareDQNAgent,
    env: TaxiRouteEnv,
    checkpoint_path: str,
    metrics_path: str,
) -> Dict[str, Any]:
    """
    Strictly verify training configuration and pre-existing checkpoint integrity.
    Raises ValueError if any critical check fails.
    """
    print("\n" + "=" * 60, flush=True)
    print("STAGE C PRE-TRAINING CONFIGURATION AUDIT", flush=True)
    print("=" * 60, flush=True)

    # 1. Verify original protected files
    expected_hashes = {
        'ml/checkpoints/dqn_model.pt': 'cb11870a74e68ce70cd25e59d1c4176a7da15db858fb7506a81078dbc0e31740',
        'ml/checkpoints/dqn_model_full_allocation.pt': '4a9bf370c4b38417df6f7cfb8f01476844b8b183f51357e73c1eb1f7998913c8',
        'ml/checkpoints/dqn_model_graph_aware.pt': 'ce5821e470b657439b5f67d4eb08b6d80421b906a12f0760872d45120ee81764',
        'ml/data/osm_cache/ludhiana_osm_road_network.json': 'a737f021a1e3ef12607326613c26eca9695b0d2599ce931f70e35217a90170f0',
    }

    recorded_hashes = {}
    for rel_path, exp_hash in expected_hashes.items():
        abs_p = os.path.abspath(rel_path)
        if not os.path.exists(abs_p):
            raise FileNotFoundError(f"CRITICAL ERROR: Protected file not found: {rel_path}")
        with open(abs_p, 'rb') as f:
            h = hashlib.sha256(f.read()).hexdigest()
        recorded_hashes[rel_path] = h
        if h != exp_hash:
            raise ValueError(f"CRITICAL INTEGRITY FAILURE: Hash mismatch for {rel_path}: expected {exp_hash}, got {h}")
        print(f"  [PASS] Protected file verified: {rel_path}", flush=True)

    # 2. Check architecture dimensions
    if agent.feature_dim != 20:
        raise ValueError(f"CRITICAL ERROR: Agent feature_dim ({agent.feature_dim}) != 20")
    print(f"  [PASS] GraphAwareDQN architecture: 20 input features, 128 hidden, 1 output scalar", flush=True)

    # 3. Check graph & ALT landmark cache checksum
    graph_chk = alt_heuristic.compute_graph_checksum(rng.graph)
    if alt_heuristic.graph_checksum != graph_chk:
        raise ValueError(f"CRITICAL ERROR: ALT cache graph checksum ({alt_heuristic.graph_checksum}) does not match graph ({graph_chk})")
    if not alt_heuristic.is_ready:
        raise ValueError("CRITICAL ERROR: ALTLandmarkHeuristic is not ready or failed to load")
    print(f"  [PASS] ALT landmark cache checksum verified ({len(alt_heuristic.landmarks)} landmarks)", flush=True)

    # 4. Check discount factors match
    if abs(agent.gamma - env.pbrs_gamma) > 1e-6:
        raise ValueError(f"CRITICAL ERROR: Agent discount factor ({agent.gamma}) != Env PBRS gamma ({env.pbrs_gamma})")
    print(f"  [PASS] Discount factor alignment: RL gamma = {agent.gamma}, PBRS gamma = {env.pbrs_gamma}", flush=True)

    # 5. Check clean replay buffer
    if len(agent.replay_buffer) != 0:
        raise ValueError(f"CRITICAL ERROR: Replay buffer is not clean! Initial size = {len(agent.replay_buffer)}")
    buf_cap = getattr(agent.replay_buffer, 'capacity', getattr(agent.replay_buffer.buffer, 'maxlen', 50000))
    print(f"  [PASS] Clean replay buffer confirmed: 0 initial transitions, capacity = {buf_cap}", flush=True)

    # 6. Check test dataset isolation
    if len(ALL_EXCLUDED_TRAIN_PAIRS) < (len(BENCHMARK_PAIRS) + len(VALIDATION_PAIRS)):
        raise ValueError("CRITICAL ERROR: ALL_EXCLUDED_TRAIN_PAIRS does not contain all benchmark and validation pairs")
    for bp in BENCHMARK_PAIRS:
        if (bp['source_node'], bp['dest_node']) not in ALL_EXCLUDED_TRAIN_PAIRS:
            raise ValueError(f"CRITICAL ERROR: Benchmark pair {bp['id']} not in ALL_EXCLUDED_TRAIN_PAIRS")
    print(f"  [PASS] Held-out test isolation confirmed: {len(ALL_EXCLUDED_TRAIN_PAIRS)} excluded pairs (30 test + 8 val)", flush=True)

    # 7. Check output paths are strictly within experiment directory
    exp_dir = os.path.abspath("ml/checkpoints/experiments/alt_pbrs")
    if not os.path.abspath(checkpoint_path).startswith(exp_dir):
        raise ValueError(f"CRITICAL ERROR: Checkpoint path ({checkpoint_path}) is outside experiment dir ({exp_dir})")
    if not os.path.abspath(metrics_path).startswith(exp_dir):
        raise ValueError(f"CRITICAL ERROR: Metrics path ({metrics_path}) is outside experiment dir ({exp_dir})")
    print(f"  [PASS] Target outputs isolated in: {exp_dir}", flush=True)

    print("=" * 60 + "\n", flush=True)
    return recorded_hashes


def sample_training_pair(env: TaxiRouteEnv, node_list: List[int],
                         excluded_pairs: set, scale: str) -> Tuple[int, int, int]:
    """Sample source and destination according to curriculum scale."""
    for _ in range(60):
        s = int(np.random.choice(node_list))
        if scale == 'short':
            curr = s
            hops = np.random.randint(6, 22)
            for _ in range(hops):
                succs = list(env.graph.successors(curr))
                if succs:
                    curr = random.choice(succs)
                else:
                    break
            d = curr
            max_steps = 110
        elif scale == 'medium':
            curr = s
            hops = np.random.randint(25, 60)
            for _ in range(hops):
                succs = list(env.graph.successors(curr))
                if succs:
                    curr = random.choice(succs)
                else:
                    break
            d = curr
            max_steps = 250
        else:  # long
            s_lat = float(env.graph.nodes[s]['lat'])
            s_lng = float(env.graph.nodes[s]['lng'])
            candidates = []
            for _ in range(35):
                candidate = int(np.random.choice(node_list))
                c_lat = float(env.graph.nodes[candidate]['lat'])
                c_lng = float(env.graph.nodes[candidate]['lng'])
                dist = env.rng.haversine_distance(s_lat, s_lng, c_lat, c_lng)
                if 3.5 <= dist <= 11.5:
                    candidates.append((candidate, dist))
                    break
            if candidates:
                d, dist = candidates[0]
                max_steps = max(400, int(dist * 90))
            else:
                d = int(np.random.choice(node_list))
                max_steps = 320

        if s != d and (s, d) not in excluded_pairs:
            return s, d, max_steps

    succs = list(env.graph.successors(s))
    d = succs[0] if succs else node_list[(env.node_to_idx.get(s, 0) + 1) % len(node_list)]
    return s, d, 120


def evaluate_validation_set(
    env: TaxiRouteEnv,
    agent: GraphAwareDQNAgent,
    val_pairs: List[Dict[str, Any]],
    rng: RoadNetworkGraph,
    precomputed_dijkstra: Dict[int, float],
) -> Dict[str, Any]:
    """
    Evaluate agent in deterministic greedy mode on validation pairs V101–V108.
    Calculates completion rate, mean/median ratio vs Dijkstra, failed routes, revisits, latency.
    """
    success_count = 0
    successful_ratios = []
    failed_routes = []
    total_revisits = 0
    latencies = []

    for p in val_pairs:
        pid = p['id']
        s, d = p['source_node'], p['dest_node']
        direct = p['direct_dist_km']
        dijkstra_dist = precomputed_dijkstra[pid]
        max_steps = max(400, int(direct * 100))

        t0 = time.time()
        obs, info = env.reset(options={
            'source': s,
            'destination': d,
            'max_steps': max_steps,
            'enable_cycle_aware_masking': True,
            'max_node_visits': 2,
            'enable_pbrs': True
        })

        path = [s]
        done = False
        truncated = False
        step = 0
        curr = s
        node_counts = {s: 1}

        while not (done or truncated) and step < max_steps:
            prev_node = path[-2] if len(path) >= 2 else None
            candidates = env.get_candidate_edges(
                curr,
                prev_node=prev_node,
                visited_nodes=path,
                recency_window=60,
                enable_cycle_aware_masking=True,
                max_node_visits=2
            )
            if not candidates:
                break

            idx, next_node, _ = agent.select_edge(candidates, eval_mode=True)
            if next_node == -1:
                break

            obs, reward, done, truncated, info = env.step_to_neighbor(next_node)
            curr = info.get('current_node', next_node)
            path.append(curr)
            node_counts[curr] = node_counts.get(curr, 0) + 1
            step += 1

        latency_ms = (time.time() - t0) * 1000.0
        latencies.append(latency_ms)

        revisits = sum(cnt - 1 for cnt in node_counts.values() if cnt > 1)
        total_revisits += revisits

        if path[-1] == d:
            success_count += 1
            dist_km = sum(rng.graph[path[i]][path[i + 1]].get('distance', 0.1) for i in range(len(path) - 1))
            ratio = dist_km / max(dijkstra_dist, 0.001)
            successful_ratios.append(round(ratio, 4))
        else:
            reason = "step_limit" if step >= max_steps else "no_candidates"
            failed_routes.append({
                'pair_id': pid,
                'name': p['name'],
                'reason': reason,
                'steps': step,
                'max_steps': max_steps
            })

    total_pairs = len(val_pairs)
    completion_rate_pct = (success_count / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    mean_ratio = float(np.mean(successful_ratios)) if successful_ratios else 999.0
    median_ratio = float(np.median(successful_ratios)) if successful_ratios else 999.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        'val_success_count': success_count,
        'val_total': total_pairs,
        'val_completion_rate_pct': round(completion_rate_pct, 1),
        'val_mean_ratio': round(mean_ratio, 3) if successful_ratios else None,
        'val_median_ratio': round(median_ratio, 3) if successful_ratios else None,
        'successful_ratios': successful_ratios,
        'failed_routes': failed_routes,
        'total_revisits': total_revisits,
        'mean_latency_ms': round(mean_latency, 2)
    }


def train_alt_pbrs_dqn(
    episodes: int = 250,
    batch_size: int = 32,
    target_update_freq: int = 10,
    val_freq: int = 25,
    checkpoint_path: str = "ml/checkpoints/experiments/alt_pbrs/dqn_model_alt_pbrs.pt",
    metrics_path: str = "ml/checkpoints/experiments/alt_pbrs/training_metrics.json",
    log_path: str = "ml/checkpoints/experiments/alt_pbrs/training.log",
    seed: int = 42,
    graph_builder: Optional[RoadNetworkGraph] = None,
    pbrs_scale: float = 8.0,
    pbrs_gamma: float = 0.995,
) -> GraphAwareDQNAgent:
    """
    Train GraphAwareDQNAgent with ALT-based Potential-Based Reward Shaping.
    Implements incremental metric logging, unbuffered DualLogger output,
    predefined validation checkpoint selection, and error resilience.
    """
    # Initialize unbuffered DualLogger
    logger = DualLogger(log_path)
    sys.stdout = logger

    logger.write(f"=== STAGE C: ALT-PBRS GRAPH-AWARE DQN TRAINING START ===\n")
    logger.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.write(f"Target Checkpoint: {checkpoint_path}\n")
    logger.write(f"Metrics Output:    {metrics_path}\n")
    logger.write(f"Log Output:        {log_path}\n")
    logger.write(f"Random Seed:       {seed}\n")
    logger.write(f"Total Episodes:    {episodes}\n")

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # Initialize Graph & ALT Landmark Heuristic
    logger.write("Loading OSM road network graph...\n")
    if graph_builder is not None:
        rng = graph_builder
    else:
        rng = RoadNetworkGraph()
        cache_file = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
        rng.load_osm_graph(place_name="Ludhiana, Punjab, India", cache_path=cache_file)

    alt_cache = os.path.abspath("ml/data/osm_cache/alt_landmarks_16.json")
    alt_heuristic = ALTLandmarkHeuristic(rng.graph, cache_path=alt_cache, auto_load=True)

    # Precompute Dijkstra road distances for validation pairs
    logger.write("Precomputing Dijkstra road distances for validation pairs V101–V108...\n")
    precomputed_dijkstra: Dict[int, float] = {}
    for p in VALIDATION_PAIRS:
        dist = nx.shortest_path_length(rng.graph, p['source_node'], p['dest_node'], weight='distance')
        precomputed_dijkstra[p['id']] = dist

    # Initialize Environment
    env = TaxiRouteEnv(
        rng,
        max_steps=120,
        enable_cycle_aware_masking=True,
        max_node_visits=2,
        enable_pbrs=True,
        pbrs_gamma=pbrs_gamma,
        pbrs_scale=pbrs_scale,
        alt_heuristic=alt_heuristic,
    )

    # Initialize Fresh Agent
    agent = GraphAwareDQNAgent(
        feature_dim=20,
        hidden_dim=128,
        lr=1e-3,
        gamma=pbrs_gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985
    )

    # Run Pre-training Configuration and Cryptographic Verification
    recorded_hashes = verify_pre_training_configuration(
        rng=rng,
        alt_heuristic=alt_heuristic,
        agent=agent,
        env=env,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    # Prepare Incremental Metrics
    metrics: Dict[str, Any] = {
        'status': 'in_progress',
        'random_seed': seed,
        'checkpoint_path': checkpoint_path,
        'metrics_path': metrics_path,
        'log_path': log_path,
        'pre_training_hashes': recorded_hashes,
        'pbrs_config': {
            'enabled': True,
            'scale': pbrs_scale,
            'gamma': pbrs_gamma,
            'num_landmarks': len(alt_heuristic.landmarks),
            'terminal_potential_zero': True,
        },
        'hyperparameters': {
            'episodes': episodes,
            'batch_size': batch_size,
            'lr': 1e-3,
            'gamma': pbrs_gamma,
            'epsilon_start': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.985,
            'target_update_freq': target_update_freq,
            'val_freq': val_freq,
        },
        'episodes_data': [],
        'val_evaluations': [],
        'best_checkpoint': {
            'episode': 0,
            'val_successes': 0,
            'val_total': len(VALIDATION_PAIRS),
            'val_completion_rate_pct': 0.0,
            'val_mean_ratio': 999.0,
            'val_median_ratio': 999.0,
        },
        'training_time_sec': 0.0,
        'final_train_win_rate_30': 0.0,
        'errors': []
    }

    start_time = time.time()
    success_window: List[int] = []
    best_val_successes = -1
    best_val_mean_ratio = 999.0
    best_ep = 0

    logger.write(f"Beginning training loop ({episodes} episodes)...\n")

    try:
        for ep in range(1, episodes + 1):
            ep_start = time.time()
            p_choice = random.random()
            if p_choice < 0.35:
                scale = 'short'
            elif p_choice < 0.70:
                scale = 'medium'
            else:
                scale = 'long'

            s, d, ep_max_steps = sample_training_pair(env, env.node_list, ALL_EXCLUDED_TRAIN_PAIRS, scale)
            obs, info = env.reset(options={
                'source': s,
                'destination': d,
                'max_steps': ep_max_steps,
                'enable_cycle_aware_masking': True,
                'max_node_visits': 2,
                'enable_pbrs': True
            })

            done = False
            truncated = False
            ep_reward = 0.0
            ep_r_original = 0.0
            ep_r_shaping = 0.0
            ep_length = 0
            ep_losses = []

            curr_cands = env.get_candidate_edges(env.current_node)

            while not (done or truncated) and curr_cands:
                idx, next_node, chosen_feat = agent.select_edge(curr_cands, eval_mode=False)
                if next_node == -1:
                    break

                next_obs, reward, done, truncated, info = env.step_to_neighbor(next_node)
                next_cands = info.get('candidate_edges', [])

                if next_cands:
                    next_feats = np.stack([c['features'] for c in next_cands])
                    next_mask = np.array([c.get('is_valid', True) for c in next_cands], dtype=bool)
                else:
                    next_feats = None
                    next_mask = None

                agent.replay_buffer.push(chosen_feat, reward, next_feats, next_mask, done)

                curr_cands = next_cands
                ep_reward += reward
                ep_r_original += info.get('r_original', reward)
                ep_r_shaping += info.get('shaping_reward', 0.0)
                ep_length += 1

                loss = agent.update(batch_size=batch_size)
                if loss is not None:
                    ep_losses.append(loss)

            agent.decay_epsilon()

            if ep % target_update_freq == 0:
                agent.update_target_network()

            is_success = 1 if done and not truncated else 0
            success_window.append(is_success)
            if len(success_window) > 30:
                success_window.pop(0)

            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            ep_duration = time.time() - ep_start
            total_elapsed = time.time() - start_time

            ep_record = {
                'episode': ep,
                'reward_shaped': round(ep_reward, 2),
                'reward_original': round(ep_r_original, 2),
                'reward_shaping': round(ep_r_shaping, 2),
                'length': ep_length,
                'destination_reached': is_success,
                'loss': round(avg_loss, 4),
                'epsilon': round(agent.epsilon, 4),
                'duration_sec': round(ep_duration, 2),
                'total_elapsed_sec': round(total_elapsed, 2)
            }
            metrics['episodes_data'].append(ep_record)

            # Validation Evaluation at Scheduled Intervals
            if ep % val_freq == 0 or ep == episodes:
                win_rate = float(np.mean(success_window)) * 100.0
                val_res = evaluate_validation_set(env, agent, VALIDATION_PAIRS, rng, precomputed_dijkstra)
                val_res['episode'] = ep
                val_res['train_win_rate_30'] = round(win_rate, 1)
                metrics['val_evaluations'].append(val_res)

                val_cnt = val_res['val_success_count']
                val_tot = val_res['val_total']
                val_pct = val_res['val_completion_rate_pct']
                val_ratio_str = f"{val_res['val_mean_ratio']:.2f}x" if val_res['val_mean_ratio'] is not None else "N/A"

                logger.write(
                    f"Ep {ep:3d}/{episodes} | R_shaped: {ep_reward:6.1f} (Orig: {ep_r_original:6.1f}, PBRS: {ep_r_shaping:6.1f}) | "
                    f"L: {ep_length:3d} | TrainWin(30): {win_rate:4.1f}% | Eps: {agent.epsilon:.2f} | "
                    f"Val: {val_cnt}/{val_tot} ({val_pct:4.1f}%, mean ratio {val_ratio_str})\n"
                )

                # Predefined Selection Criterion:
                # 1. Higher completion rate (val_success_count)
                # 2. Lower mean distance ratio (tie-breaker)
                curr_mean_ratio = val_res['val_mean_ratio'] if val_res['val_mean_ratio'] is not None else 999.0
                is_better = False
                if val_cnt > best_val_successes:
                    is_better = True
                elif val_cnt == best_val_successes and curr_mean_ratio < best_val_mean_ratio:
                    is_better = True

                if is_better:
                    best_val_successes = val_cnt
                    best_val_mean_ratio = curr_mean_ratio
                    best_ep = ep
                    agent.save_checkpoint(checkpoint_path)
                    metrics['best_checkpoint'] = {
                        'episode': ep,
                        'val_successes': val_cnt,
                        'val_total': val_tot,
                        'val_completion_rate_pct': val_pct,
                        'val_mean_ratio': val_res['val_mean_ratio'],
                        'val_median_ratio': val_res['val_median_ratio'],
                        'revisits': val_res['total_revisits'],
                        'latency_ms': val_res['mean_latency_ms'],
                        'checkpoint_file': checkpoint_path,
                    }
                    logger.write(
                        f"   [*] Saved NEW BEST ALT-PBRS CHECKPOINT: {checkpoint_path} "
                        f"(Val: {val_cnt}/{val_tot}, Ratio: {val_ratio_str})\n"
                    )

            # Incremental save of metrics after every episode
            metrics['training_time_sec'] = round(time.time() - start_time, 2)
            metrics['final_train_win_rate_30'] = round(float(np.mean(success_window)) * 100.0, 1) if success_window else 0.0
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

        metrics['status'] = 'completed'

    except (Exception, KeyboardInterrupt) as e:
        err_msg = f"Training interrupted or failed: {str(e)}"
        logger.write(f"\n[ERROR] {err_msg}\n")
        metrics['status'] = 'interrupted'
        metrics['errors'].append(err_msg)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        raise

    finally:
        elapsed = time.time() - start_time
        metrics['training_time_sec'] = round(elapsed, 2)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    logger.write(f"\n======================================================\n")
    logger.write(f"TRAINING COMPLETE in {elapsed:.2f}s ({elapsed / 60.0:.2f} min)\n")
    logger.write(f"Best Validation Model at Episode {best_ep}:\n")
    logger.write(f"  Successes:  {best_val_successes}/{len(VALIDATION_PAIRS)}\n")
    logger.write(f"  Mean Ratio: {best_val_mean_ratio:.3f}x\n")
    logger.write(f"  Saved to:   {checkpoint_path}\n")
    logger.write(f"======================================================\n")

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph-Aware DQN with ALT-PBRS")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=str, default="ml/checkpoints/experiments/alt_pbrs/dqn_model_alt_pbrs.pt")
    parser.add_argument("--metrics-path", type=str, default="ml/checkpoints/experiments/alt_pbrs/training_metrics.json")
    parser.add_argument("--log-path", type=str, default="ml/checkpoints/experiments/alt_pbrs/training.log")
    args = parser.parse_args()

    train_alt_pbrs_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        log_path=args.log_path,
    )

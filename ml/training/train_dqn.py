import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.graph.builder import RoadNetworkGraph
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.dqn_agent import DQNAgent
from ml.evaluation.benchmark_dataset import VALIDATION_PAIRS, ALL_EXCLUDED_TRAIN_PAIRS


def sample_training_pair(env: TaxiRouteEnv, node_list: List[int],
                         excluded_pairs: set, scale: str) -> Tuple[int, int, int]:
    """
    Sample source and destination nodes according to curriculum scale.
    Guarantees (s, d) is not in excluded_pairs and s != d.
    """
    for _ in range(60):  # Retry attempts to find non-excluded pair
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
        else:  # 'long' / cross-city
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

    # Fallback to random outgoing neighbor
    succs = list(env.graph.successors(s))
    d = succs[0] if succs else node_list[(env.node_to_idx.get(s, 0) + 1) % len(node_list)]
    return s, d, 120


def evaluate_validation_set(env: TaxiRouteEnv, agent: DQNAgent,
                            val_pairs: List[Dict[str, Any]],
                            rng: RoadNetworkGraph) -> Tuple[float, float, int]:
    """
    Evaluate agent on disjoint validation set in greedy mode.
    Returns: (val_success_rate, avg_distance_ratio_on_success, success_count)
    """
    success_count = 0
    ratios = []

    for p in val_pairs:
        s, d = p["source_node"], p["dest_node"]
        direct = p.get("direct_dist_km", 5.0)
        max_steps = max(500, int(direct * 110))
        obs, info = env.reset(options={"source": s, "destination": d, "max_steps": max_steps})
        path = [s]
        done = False
        truncated = False
        step = 0
        current_node = s

        while not (done or truncated) and step < max_steps:
            prev_node = path[-2] if len(path) >= 2 else None
            mask = env.get_action_mask(current_node, prev_node=prev_node, visited_nodes=path, recency_window=60)
            if not np.any(mask):
                mask = env.get_action_mask(current_node)
            action = agent.select_action(obs, mask=mask, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            current_node = info.get("current_node", current_node)
            path.append(current_node)
            step += 1

        if path[-1] == d:
            success_count += 1
            # Calculate actual road distance
            dist_km = sum(rng.graph[path[i]][path[i + 1]].get('distance', 0.1) for i in range(len(path) - 1))
            ratio = dist_km / max(direct, 0.01)
            ratios.append(ratio)

    success_rate = (success_count / len(val_pairs)) * 100.0 if val_pairs else 0.0
    avg_ratio = float(np.mean(ratios)) if ratios else 999.0
    return success_rate, avg_ratio, success_count


def train_dqn(episodes: int = 250, batch_size: int = 32,
              target_update_freq: int = 10,
              val_freq: int = 25,
              checkpoint_path: str = "ml/checkpoints/dqn_model.pt",
              metrics_path: str = "ml/checkpoints/training_metrics.json",
              seed: Optional[int] = 42,
              graph_builder: Optional[RoadNetworkGraph] = None) -> DQNAgent:
    """
    Train DQN agent on TaxiRouteEnv with episode logging, multi-scale curriculum,
    and validation-guided model checkpointing (strictly excluding held-out benchmark pairs).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set to: {seed}", flush=True)

    print("Initializing Road Network Graph & Gymnasium Environment...", flush=True)
    if graph_builder is not None:
        rng = graph_builder
    else:
        rng = RoadNetworkGraph()
        cache_file = os.path.abspath("ml/data/osm_cache/ludhiana_osm_road_network.json")
        force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() in ("true", "1")
        if os.path.exists(cache_file) and not force_synthetic:
            try:
                rng.load_osm_graph(place_name="Ludhiana, Punjab, India", cache_path=cache_file)
            except Exception as e:
                print(f"Warning: Failed to load OSM cache ({e}), falling back to synthetic grid.")
                rng.build_synthetic_grid()
        else:
            rng.build_synthetic_grid()

    env = TaxiRouteEnv(rng, max_steps=120)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.995,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.985
    )

    metrics = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'successes': [],
        'losses': [],
        'epsilons': [],
        'val_evaluations': [],
        'best_val_successes': 0,
        'best_val_ratio': 999.0,
        'best_episode': 0,
        'training_time_sec': 0.0,
        'final_success_rate': 0.0
    }

    start_time = time.time()
    success_window = []

    best_val_successes = -1
    best_val_ratio = 999.0
    best_ep = 0

    print(f"Starting DQN Training for {episodes} episodes on {getattr(rng, 'source_type', 'GRAPH')}...", flush=True)
    print(f"Total nodes: {env.num_nodes} | Excluded test pairs: {len(ALL_EXCLUDED_TRAIN_PAIRS)} | Validation pairs: {len(VALIDATION_PAIRS)}", flush=True)

    for ep in range(1, episodes + 1):
        # Multi-scale curriculum selection
        p_choice = random.random()
        if p_choice < 0.35:
            scale = 'short'
        elif p_choice < 0.70:
            scale = 'medium'
        else:
            scale = 'long'

        s, d, ep_max_steps = sample_training_pair(env, env.node_list, ALL_EXCLUDED_TRAIN_PAIRS, scale)
        obs, info = env.reset(options={'source': s, 'destination': d, 'max_steps': ep_max_steps})

        done = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0
        ep_losses = []

        while not (done or truncated):
            mask = info.get('action_mask')
            if mask is None or not np.any(mask):
                mask = env.get_action_mask(env.current_node)

            action = agent.select_action(obs, mask=mask, eval_mode=False)

            next_obs, reward, done, truncated, info = env.step(action)
            # Push only real goal termination (not step truncation) to preserve future value
            agent.replay_buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
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

        metrics['episodes'].append(ep)
        metrics['rewards'].append(round(ep_reward, 2))
        metrics['lengths'].append(ep_length)
        metrics['successes'].append(is_success)
        metrics['losses'].append(round(avg_loss, 4))
        metrics['epsilons'].append(round(agent.epsilon, 4))

        # Periodic logging & validation
        if ep % val_freq == 0 or ep == episodes:
            win_rate = np.mean(success_window) * 100.0
            val_pct, val_ratio, val_succ_cnt = evaluate_validation_set(env, agent, VALIDATION_PAIRS, rng)

            val_log = {
                'episode': ep,
                'val_success_count': val_succ_cnt,
                'val_total': len(VALIDATION_PAIRS),
                'val_success_pct': val_pct,
                'val_avg_ratio': round(val_ratio, 2)
            }
            metrics['val_evaluations'].append(val_log)

            print(f"Ep {ep:3d}/{episodes} | R: {ep_reward:6.1f} | L: {ep_length:3d} | "
                  f"TrainWin(30): {win_rate:4.1f}% | Eps: {agent.epsilon:.2f} | "
                  f"Val: {val_succ_cnt}/{len(VALIDATION_PAIRS)} ({val_pct:4.1f}%, ratio {val_ratio:4.2f}x)", flush=True)

            # Check if this checkpoint is the best validated model
            is_better = False
            if val_succ_cnt > best_val_successes:
                is_better = True
            elif val_succ_cnt == best_val_successes and val_ratio < best_val_ratio:
                is_better = True

            if is_better:
                best_val_successes = val_succ_cnt
                best_val_ratio = val_ratio
                best_ep = ep
                agent.save_checkpoint(checkpoint_path)
                print(f"   [*] Saved NEW BEST CHECKPOINT to {checkpoint_path} (Val: {val_succ_cnt}/{len(VALIDATION_PAIRS)})", flush=True)

    elapsed = time.time() - start_time
    metrics['training_time_sec'] = round(elapsed, 2)
    metrics['final_success_rate'] = round(float(np.mean(success_window[-30:])) * 100.0, 1)
    metrics['best_val_successes'] = best_val_successes
    metrics['best_val_ratio'] = round(best_val_ratio, 2)
    metrics['best_episode'] = best_ep

    print(f"\nTraining completed in {elapsed:.2f}s!", flush=True)
    print(f"Best validation model achieved at Episode {best_ep}: {best_val_successes}/{len(VALIDATION_PAIRS)} ({best_val_successes/len(VALIDATION_PAIRS)*100:.1f}%), ratio {best_val_ratio:.2f}x", flush=True)

    # Reload best validated checkpoint into returned agent
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(f"Loaded best checkpoint from: {checkpoint_path}", flush=True)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to: {metrics_path}", flush=True)

    return agent


class DualLogger:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN agent on road network")
    parser.add_argument("--episodes", type=int, default=250, help="Number of training episodes")
    parser.add_argument("--checkpoint", type=str, default="ml/checkpoints/dqn_model.pt", help="Path to save best checkpoint")
    parser.add_argument("--metrics", type=str, default="ml/checkpoints/training_metrics.json", help="Path to save metrics json")
    parser.add_argument("--log-file", type=str, default=None, help="Path to write unbuffered log output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--val-freq", type=int, default=25, help="Validation frequency in episodes")
    args = parser.parse_args()

    if args.log_file:
        logger = DualLogger(args.log_file)
        sys.stdout = logger
        sys.stderr = logger

    train_dqn(
        episodes=args.episodes,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        seed=args.seed,
        val_freq=args.val_freq
    )

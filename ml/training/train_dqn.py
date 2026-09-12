import os
import sys
import json
import time
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.graph.builder import RoadNetworkGraph
from ml.environment.taxi_env import TaxiRouteEnv
from ml.agents.dqn_agent import DQNAgent


def train_dqn(episodes: int = 150, batch_size: int = 32,
              target_update_freq: int = 10,
              checkpoint_path: str = "ml/checkpoints/dqn_model.pt",
              metrics_path: str = "ml/checkpoints/training_metrics.json") -> DQNAgent:
    """
    Train DQN agent on TaxiRouteEnv with episode logging and metric tracking.
    Supports development fast training (e.g., 150 episodes) and full training.
    """
    print(f"Initializing Road Network Graph & Gymnasium Environment...")
    rng = RoadNetworkGraph()
    rng.build_synthetic_grid()
    env = TaxiRouteEnv(rng, max_steps=50)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.98
    )

    metrics = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'successes': [],
        'losses': [],
        'epsilons': [],
        'training_time_sec': 0.0,
        'final_success_rate': 0.0
    }

    start_time = time.time()
    success_window = []

    print(f"Starting DQN Training for {episodes} episodes...")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0
        ep_losses = []

        while not (done or truncated):
            mask = info.get('action_mask')
            action = agent.select_action(obs, mask=mask, eval_mode=False)

            next_obs, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.push(obs, action, reward, next_obs, done or truncated)

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

        if ep % 25 == 0 or ep == episodes:
            win_rate = np.mean(success_window) * 100.0
            print(f"Episode {ep}/{episodes} | Reward: {ep_reward:.1f} | Length: {ep_length} | "
                  f"WinRate (last 30): {win_rate:.1f}% | Epsilon: {agent.epsilon:.2f}")

    elapsed = time.time() - start_time
    metrics['training_time_sec'] = round(elapsed, 2)
    metrics['final_success_rate'] = round(float(np.mean(success_window[-30:])) * 100.0, 1)

    print(f"Training finished in {elapsed:.2f}s! Final 30-ep success rate: {metrics['final_success_rate']}%")

    agent.save_checkpoint(checkpoint_path)
    print(f"Saved trained DQN checkpoint to: {checkpoint_path}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to: {metrics_path}")

    return agent


if __name__ == '__main__':
    train_dqn(episodes=150)

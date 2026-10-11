import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Any, Optional


class EdgeScoringNetwork(nn.Module):
    """
    Shared Edge-Scoring Network for Graph-Aware Reinforcement Learning.
    Takes a 20-dimensional feature vector [8 macro state + 12 candidate edge features]
    and computes a scalar Q-value score for the specific road branch.
    """

    def __init__(self, feature_dim: int = 20, hidden_dim: int = 128):
        super(EdgeScoringNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphAwareReplayBuffer:
    """Experience Replay Buffer for Edge-Conditioned Q-Learning."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, chosen_feat: np.ndarray, reward: float,
             next_candidate_feats: Optional[np.ndarray],
             next_candidate_mask: Optional[np.ndarray],
             done: bool):
        self.buffer.append((chosen_feat, reward, next_candidate_feats, next_candidate_mask, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class GraphAwareDQNAgent:
    """
    Graph-Aware DQN Agent.
    Evaluates variable candidate outgoing edge sets via a shared edge-scoring network.
    Eliminates fixed compass sector binning and sector collisions completely.
    """

    def __init__(self, feature_dim: int = 20, hidden_dim: int = 128,
                 lr: float = 1e-3, gamma: float = 0.995,
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.985, device: str = "cpu"):
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.q_network = EdgeScoringNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_network = EdgeScoringNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = GraphAwareReplayBuffer(capacity=50000)

    def select_edge(self, candidates: List[Dict[str, Any]], eval_mode: bool = False) -> Tuple[int, int, np.ndarray]:
        """
        Score candidate outgoing edges and select the optimal edge.
        Returns: (chosen_index, chosen_neighbor_node, chosen_features)
        """
        if not candidates:
            return -1, -1, np.zeros(self.feature_dim, dtype=np.float32)

        valid_indices = [i for i, c in enumerate(candidates) if c.get('is_valid', True)]
        if not valid_indices:
            valid_indices = list(range(len(candidates)))

        # Epsilon-greedy exploration
        if not eval_mode and random.random() < self.epsilon:
            idx = random.choice(valid_indices)
            return idx, candidates[idx]['neighbor'], candidates[idx]['features']

        # Greedy inference: score all candidates in a single batch forward pass
        feats = np.stack([c['features'] for c in candidates])
        with torch.no_grad():
            feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
            scores = self.q_network(feats_t).squeeze(-1).cpu().numpy()

        # Mask invalid candidates to -infinity
        for i in range(len(candidates)):
            if not candidates[i].get('is_valid', True):
                scores[i] = -np.inf

        best_idx = int(np.argmax(scores))
        return best_idx, candidates[best_idx]['neighbor'], candidates[best_idx]['features']

    def update(self, batch_size: int = 32) -> Optional[float]:
        """Perform one step of gradient descent on the edge-scoring network."""
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)

        chosen_feats = []
        targets = []

        with torch.no_grad():
            for chosen_feat, reward, next_feats, next_mask, done in batch:
                chosen_feats.append(chosen_feat)
                if done or next_feats is None or len(next_feats) == 0:
                    y = reward
                else:
                    next_t = torch.tensor(next_feats, dtype=torch.float32, device=self.device)
                    q_next = self.target_network(next_t).squeeze(-1).cpu().numpy()
                    if next_mask is not None:
                        for idx, v in enumerate(next_mask):
                            if not v:
                                q_next[idx] = -np.inf
                    max_next_q = float(np.max(q_next)) if np.any(next_mask) else 0.0
                    y = reward + self.gamma * max_next_q
                targets.append(y)

        chosen_t = torch.tensor(np.array(chosen_feats), dtype=torch.float32, device=self.device)
        targets_t = torch.tensor(np.array(targets), dtype=torch.float32, device=self.device).unsqueeze(1)

        curr_q = self.q_network(chosen_t)
        loss = self.criterion(curr_q, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Hard update target network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'feature_dim': self.feature_dim
        }, filepath)

    def load_checkpoint(self, filepath: str) -> bool:
        """Load model checkpoint."""
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        return True

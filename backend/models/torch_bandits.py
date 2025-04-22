import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class DoublyRobustBandit:
    def __init__(
        self,
        context_dim: int,
        n_arms: int = 5,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Doubly Robust model with multiple arms
        
        Args:
            context_dim: Dimension of context vectors
            n_arms: Number of arms (default 5)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            device: Device to run the model on
        """
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.device = device
        
        # Initialize network with n_arms outputs
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], n_arms)  # Output one score per arm
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Track number of pulls for each arm
        self.arm_pulls = {i: 0 for i in range(n_arms)}
        
        # Performance tracking
        self.cumulative_reward = 0
        self.n_trials = 0

    def get_ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """
        Calculate scores for all arms given a context
        """
        # Convert context to tensor
        context_tensor = torch.FloatTensor(context).to(self.device)
        
        # Get predictions for each arm
        with torch.no_grad():
            logits = self.network(context_tensor)
            probs = torch.sigmoid(logits)
            scores = probs.cpu().numpy().flatten()
            
        return scores

    def recommend_k(self, context: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Get top-k recommendations based on model scores
        """
        scores = self.get_ucb_scores(context)
        
        # Get top k arms
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

    def update(self, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update model parameters based on observed reward
        """
        # Convert to tensors
        context_tensor = torch.FloatTensor(context).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.network(context_tensor)
        
        # Create target tensor with zeros for all arms except chosen one
        target = torch.zeros(self.n_arms).to(self.device)
        target[chosen_arm] = reward
        
        # Compute Doubly Robust loss
        direct_loss = F.binary_cross_entropy_with_logits(logits, target)
        ips_loss = (reward_tensor - torch.sigmoid(logits[chosen_arm]))  # Using uniform propensities
        loss = direct_loss + ips_loss
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.arm_pulls[chosen_arm] += 1
        self.cumulative_reward += reward
        self.n_trials += 1

    def get_arm_weights(self, arm_idx: int) -> np.ndarray:
        """
        Get current weights for a specific arm
        """
        # Return the last layer weights for visualization
        return self.network[-1].weight.detach().cpu().numpy().flatten()

    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        """
        return {
            'cumulative_reward': self.cumulative_reward,
            'n_trials': self.n_trials,
            'average_reward': self.cumulative_reward / max(1, self.n_trials),
            'arm_pulls': self.arm_pulls.copy()
        }

    def reset(self) -> None:
        """Reset the model to initial state"""
        self.__init__(
            self.context_dim,
            self.n_arms
        )

    def save_model(self, path: str) -> None:
        """Save model state"""
        save_dict = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'arm_pulls': self.arm_pulls,
            'cumulative_reward': self.cumulative_reward,
            'n_trials': self.n_trials
        }
        torch.save(save_dict, path)

    def load_model(self, path: str) -> None:
        """Load model state"""
        if os.path.exists(path):
            save_dict = torch.load(path)
            self.network.load_state_dict(save_dict['network_state'])
            self.optimizer.load_state_dict(save_dict['optimizer_state'])
            self.arm_pulls = save_dict['arm_pulls']
            self.cumulative_reward = save_dict['cumulative_reward']
            self.n_trials = save_dict['n_trials']

class SlateRankingBandit:
    def __init__(
        self,
        context_dim: int,
        n_arms: int = 5,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Slate Ranking model with multiple arms
        """
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.device = device
        
        # Initialize networks
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], n_arms)  # Output one score per arm
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Track metrics
        self.arm_pulls = {i: 0 for i in range(n_arms)}
        self.cumulative_reward = 0
        self.n_trials = 0

    def get_scores(self, context: np.ndarray) -> np.ndarray:
        """
        Calculate scores for all arms given a context
        """
        context_tensor = torch.FloatTensor(context).to(self.device)
        
        with torch.no_grad():
            scores = self.encoder(context_tensor)
            scores = torch.sigmoid(scores)  # Convert to probabilities
            scores = scores.cpu().numpy().flatten()
            
        return scores

    def recommend_k(self, context: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Get top-k recommendations based on model scores
        """
        scores = self.get_scores(context)
        
        # Get top k arms
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

    def update(self, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update model parameters based on observed reward
        """
        context_tensor = torch.FloatTensor(context).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        scores = self.encoder(context_tensor)
        probs = torch.sigmoid(scores)
        
        # Create target tensor with zeros for all arms except chosen one
        target = torch.zeros(self.n_arms).to(self.device)
        target[chosen_arm] = reward
        
        # Compute ranking loss
        loss = F.binary_cross_entropy(probs, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.arm_pulls[chosen_arm] += 1
        self.cumulative_reward += reward
        self.n_trials += 1

    def get_arm_weights(self, arm_idx: int) -> np.ndarray:
        """
        Get current weights for a specific arm
        """
        # Return the last layer weights for visualization
        return self.encoder[-1].weight.detach().cpu().numpy()[arm_idx]

    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        """
        return {
            'cumulative_reward': self.cumulative_reward,
            'n_trials': self.n_trials,
            'average_reward': self.cumulative_reward / max(1, self.n_trials),
            'arm_pulls': self.arm_pulls.copy()
        }

    def reset(self) -> None:
        """Reset the model to initial state"""
        self.__init__(
            self.context_dim,
            self.n_arms
        )

    def save_model(self, path: str) -> None:
        """Save model state"""
        save_dict = {
            'encoder_state': self.encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'arm_pulls': self.arm_pulls,
            'cumulative_reward': self.cumulative_reward,
            'n_trials': self.n_trials
        }
        torch.save(save_dict, path)

    def load_model(self, path: str) -> None:
        """Load model state"""
        if os.path.exists(path):
            save_dict = torch.load(path)
            self.encoder.load_state_dict(save_dict['encoder_state'])
            self.optimizer.load_state_dict(save_dict['optimizer_state'])
            self.arm_pulls = save_dict['arm_pulls']
            self.cumulative_reward = save_dict['cumulative_reward']
            self.n_trials = save_dict['n_trials']
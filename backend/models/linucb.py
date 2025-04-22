import numpy as np
from typing import List, Dict, Tuple

class LinUCBModel:
    def __init__(
        self,
        context_dim: int,
        n_arms: int = 5,
        alpha: float = 1.0,
        lambda_reg: float = 1.0
    ):
        """
        Initialize LinUCB model with multiple arms
        
        Args:
            context_dim: Dimension of context vectors
            n_arms: Number of arms (default 5)
            alpha: Exploration parameter
            lambda_reg: Ridge regression regularization parameter
        """
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize model parameters for each arm
        self.A = {i: np.eye(context_dim) * lambda_reg for i in range(n_arms)}
        self.b = {i: np.zeros(context_dim) for i in range(n_arms)}
        self.theta = {i: np.zeros(context_dim) for i in range(n_arms)}
        
        # Track number of pulls for each arm
        self.arm_pulls = {i: 0 for i in range(n_arms)}
        
        # Performance tracking
        self.cumulative_reward = 0
        self.n_trials = 0

    def _update_theta(self, arm_idx: int) -> None:
        """
        Update theta parameter for given arm using current A and b
        """
        try:
            self.theta[arm_idx] = np.linalg.solve(self.A[arm_idx], self.b[arm_idx])
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            self.theta[arm_idx] = np.dot(
                np.linalg.pinv(self.A[arm_idx]), 
                self.b[arm_idx]
            )

    def get_ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """
        Calculate UCB scores for all arms given a context
        
        Args:
            context: Context vector (context_dim,)
            
        Returns:
            np.ndarray: UCB scores for each arm
        """
        ucb_scores = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Ensure theta is up to date
            self._update_theta(arm)
            
            # Calculate prediction and confidence bound
            pred = context.dot(self.theta[arm])
            
            # Calculate standard deviation (exploration term)
            A_inv = np.linalg.inv(self.A[arm])
            std = np.sqrt(context.dot(A_inv).dot(context))
            
            # UCB score
            ucb_scores[arm] = pred + self.alpha * std
            
        return ucb_scores

    def recommend_k(self, context: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Get top-k recommendations based on UCB scores
        
        Args:
            context: Context vector
            k: Number of recommendations to return
        
        Returns:
            Tuple[List[int], List[float]]: (Top-k arm indices, Their UCB scores)
        """
        ucb_scores = self.get_ucb_scores(context)
        
        # Get top k arms
        top_k_indices = np.argsort(ucb_scores)[-k:][::-1]
        top_k_scores = ucb_scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()

    def update(self, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update model parameters based on observed reward
        
        Args:
            context: Context vector for the interaction
            chosen_arm: Index of the chosen arm
            reward: Observed reward (0 or 1)
        """
        # Update matrices for chosen arm
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context
        
        # Update pull count and performance metrics
        self.arm_pulls[chosen_arm] += 1
        self.cumulative_reward += reward
        self.n_trials += 1
        
        # Update theta for the affected arm
        self._update_theta(chosen_arm)

    def get_arm_weights(self, arm_idx: int) -> np.ndarray:
        """
        Get current weights (theta) for a specific arm
        
        Args:
            arm_idx: Index of the arm
            
        Returns:
            np.ndarray: Current theta weights for the arm
        """
        return self.theta[arm_idx].copy()

    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dict containing performance metrics
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
            self.n_arms,
            self.alpha,
            self.lambda_reg
        )
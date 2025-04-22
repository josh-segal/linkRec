import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import multivariate_normal

class ThompsonSamplingModel:
    def __init__(
        self,
        context_dim: int,
        n_arms: int = 5,
        lambda_prior: float = 1.0,
        nu_prior: float = 1.0,
        alpha_prior: float = 1.0
    ):
        """
        Initialize Thompson Sampling model with multiple arms
        
        Args:
            context_dim: Dimension of context vectors
            n_arms: Number of arms (default 5)
            lambda_prior: Prior precision of the mean
            nu_prior: Prior sample size (degrees of freedom)
            alpha_prior: Prior confidence (variance scaling)
        """
        self.context_dim = context_dim
        self.n_arms = n_arms
        
        # Prior parameters
        self.lambda_prior = lambda_prior
        self.nu_prior = nu_prior
        self.alpha_prior = alpha_prior
        
        # Initialize model parameters for each arm
        self.B = {i: np.eye(context_dim) * lambda_prior for i in range(n_arms)}  # Precision matrix
        self.mu = {i: np.zeros(context_dim) for i in range(n_arms)}  # Mean vector
        self.f = {i: np.zeros(context_dim) for i in range(n_arms)}  # Weighted sum of contexts
        self.v = {i: nu_prior for i in range(n_arms)}  # Posterior degrees of freedom
        
        # Track number of pulls for each arm
        self.arm_pulls = {i: 0 for i in range(n_arms)}
        
        # Performance tracking
        self.cumulative_reward = 0
        self.n_trials = 0

    def _sample_theta(self, arm_idx: int) -> np.ndarray:
        """
        Sample parameters from the posterior distribution for an arm
        
        Args:
            arm_idx: Index of the arm
            
        Returns:
            np.ndarray: Sampled parameters
        """
        try:
            # Calculate posterior covariance
            cov = np.linalg.inv(self.B[arm_idx])
            # Sample from multivariate Gaussian
            sample = multivariate_normal.rvs(
                mean=self.mu[arm_idx],
                cov=cov / self.v[arm_idx]  # Scale by posterior degrees of freedom
            )
            return sample
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to prior mean if sampling fails
            return self.mu[arm_idx].copy()

    def get_samples(self, context: np.ndarray) -> np.ndarray:
        """
        Get reward estimates for all arms through sampling
        
        Args:
            context: Context vector (context_dim,)
            
        Returns:
            np.ndarray: Sampled rewards for each arm
        """
        rewards = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            # Sample parameters and compute reward estimate
            theta = self._sample_theta(arm)
            rewards[arm] = context.dot(theta)
            
        return rewards

    def recommend_k(self, context: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Get top-k recommendations based on Thompson sampling
        
        Args:
            context: Context vector
            k: Number of recommendations to return
        
        Returns:
            Tuple[List[int], List[float]]: (Top-k arm indices, Their sampled rewards)
        """
        sampled_rewards = self.get_samples(context)
        
        # Get top k arms
        top_k_indices = np.argsort(sampled_rewards)[-k:][::-1]
        top_k_rewards = sampled_rewards[top_k_indices]
        
        return top_k_indices.tolist(), top_k_rewards.tolist()

    def update(self, context: np.ndarray, chosen_arm: int, reward: float) -> None:
        """
        Update model parameters based on observed reward
        
        Args:
            context: Context vector for the interaction
            chosen_arm: Index of the chosen arm
            reward: Observed reward (0 or 1)
        """
        # Update precision matrix
        self.B[chosen_arm] += np.outer(context, context)
        
        # Update weighted sum of contexts
        self.f[chosen_arm] += reward * context
        
        # Update posterior mean
        try:
            B_inv = np.linalg.inv(self.B[chosen_arm])
            self.mu[chosen_arm] = B_inv.dot(self.f[chosen_arm])
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            B_inv = np.linalg.pinv(self.B[chosen_arm])
            self.mu[chosen_arm] = B_inv.dot(self.f[chosen_arm])
        
        # Update degrees of freedom
        self.v[chosen_arm] += 1
        
        # Update pull count and performance metrics
        self.arm_pulls[chosen_arm] += 1
        self.cumulative_reward += reward
        self.n_trials += 1

    def get_arm_weights(self, arm_idx: int) -> np.ndarray:
        """
        Get current mean weights for a specific arm
        
        Args:
            arm_idx: Index of the arm
            
        Returns:
            np.ndarray: Current mean weights for the arm
        """
        return self.mu[arm_idx].copy()

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
            self.lambda_prior,
            self.nu_prior,
            self.alpha_prior
        )
"""
Multi-Agent Behavioral Cloning for Two-Player Zero-Sum Markov Games

This module implements behavioral cloning for learning policies from expert demonstrations
in two-player zero-sum Markov games. The algorithm samples expert trajectories and uses
count-based maximum likelihood estimation to learn policies that mimic expert behavior.
"""

import numpy as np
from tqdm import tqdm

from utils import calc_exploitability_true

class MultiAgentBehaviorCloning:
    """
    Multi-Agent Behavioral Cloning for Two-Player Zero-Sum Markov Games
    
    This class implements behavioral cloning for learning policies from expert demonstrations
    in two-player zero-sum Markov games.
    
    Attributes:
        expert1 (np.ndarray): Expert policy for player 1, shape (S, A1)
        expert2 (np.ndarray): Expert policy for player 2, shape (S, A2)
        total_samples (int): Total number of state-action pairs to collect
        P (np.ndarray): Transition probabilities, shape (S, A1, A2, S')
        initial_state_dist (np.ndarray): Initial state distribution, shape (S,)
        true_r (np.ndarray): True reward matrix, shape (S, A1, A2)
        gamma (float): Discount factor for value computation
        S (int): Number of states in the environment
    
    Example:
        >>> expert_policies = (expert_policy_1, expert_policy_2)
        >>> bc = MultiAgentBehaviorCloning(
        ...     expert_policies=expert_policies,
        ...     total_samples=1000,
        ...     transition=transition_matrix,
        ...     initial_state_dist=initial_dist,
        ...     payoff_matrix=rewards,
        ...     gamma=0.9
        ... )
        >>> learned_policy_1, learned_policy_2, iters, exploits = bc.train()
    """
    def __init__(
            self,
            expert_policies: tuple[np.ndarray, np.ndarray],  # shape: (S, A)
            total_samples: int,
            transition: np.ndarray,  # (S, A1, A2, S')
            initial_state_dist: np.ndarray,  # (S,)
            payoff_matrix: np.ndarray,  # state->matrix
            gamma: float
        ):
        """
        Initialize the Multi-Agent Behavioral Cloning algorithm.
        
        Args:
            expert_policies: Tuple of expert policies (policy1, policy2)
                - policy1: np.ndarray of shape (S, A1) - Player 1's expert policy π₁(a|s)
                - policy2: np.ndarray of shape (S, A2) - Player 2's expert policy π₂(a|s)
            total_samples: Total number of state-action pairs to collect from expert trajectories
            transition: Transition probability tensor P(s'|s,a₁,a₂) of shape (S, A1, A2, S')
            initial_state_dist: Initial state distribution μ₀(s) of shape (S,)
            payoff_matrix: Reward tensor r(s,a₁,a₂) of shape (S, A1, A2)
            gamma: Discount factor gamma ∈ [0,1) for value computation
            
        Note:
            - Expert policies should be valid probability distributions (rows sum to 1)
            - total_samples determines the amount of expert data available for learning
            - The algorithm uses geometric episode lengths with parameter (1-gamma)
        """
        self.expert1, self.expert2 = expert_policies
        self.total_samples = total_samples
        self.P = transition
        self.initial_state_dist = initial_state_dist
        self.true_r = payoff_matrix
        self.gamma = gamma
        
        # Extract environment dimensions from expert policies
        # initialize estimated policies uniformly
        self.S, A1 = self.expert1.shape
        _, A2 = self.expert2.shape
            
    def generate_expert_data(self):
        """
        Sample expert trajectories under expert policies using geometric episode lengths.
        
        This method generates training data by sampling trajectories from the expert policies.
        Each trajectory starts from the initial state distribution and continues for a 
        geometrically distributed number of steps (with parameter 1-gamma).
        
        The sampling process:
        1. Sample initial state s₀ from initial_state_dist
        2. Sample episode length T ~ Geometric(1-gamma)
        3. For each timestep t = 0, 1, ..., T-1:
           - Sample actions a₁ᵗ ~ expert1[sᵗ], a₂ᵗ ~ expert2[sᵗ]
           - Record state-action pairs (sᵗ, a₁ᵗ) and (sᵗ, a₂ᵗ)
           - Sample next state sᵗ⁺¹ ~ P(·|sᵗ, a₁ᵗ, a₂ᵗ)
        4. Repeat until total_samples state-action pairs are collected
        
        Returns:
            tuple: (data1, data2) where:
                - data1: List of (state, action) pairs for player 1
                - data2: List of (state, action) pairs for player 2
                Each list contains exactly total_samples tuples.
                
        Note:
            The geometric episode length models the discounted future importance
            and ensures that trajectories have finite expected length.
        """
        data1 = []  # list of (s,a)
        data2 = []
        samples = 0
        while samples < self.total_samples:
            s = np.random.choice(self.initial_state_dist.shape[0], p=self.initial_state_dist)
            random_length = np.random.geometric(1-self.gamma)
            for _ in range(random_length):
                a1 = np.random.choice(self.expert1.shape[1], p=self.expert1[s])
                a2 = np.random.choice(self.expert2.shape[1], p=self.expert2[s])
                data1.append((s, a1))
                data2.append((s, a2))
                # sample next state
                prob = self.P[s, a1, a2]
                s = np.random.choice(self.initial_state_dist.shape[0], p=prob)
            samples += random_length
            data1 = data1[:self.total_samples]
            data2 = data2[:self.total_samples]
        
        return data1, data2
    
    def _compute_counts(self, t, data1, data2):
        """
        Compute state-action visit counts from expert data up to time step t.
        
        
        Args:
            t (int): Time step up to which to count (exclusive upper bound)
            data1 (list): Expert data for player 1, list of (state, action) tuples
            data2 (list): Expert data for player 2, list of (state, action) tuples
            
        Returns:
            tuple: (counts1, counts2) where:
                - counts1: np.ndarray of shape (S, A1) with visit counts for player 1
                - counts2: np.ndarray of shape (S, A2) with visit counts for player 2
                
        """
        # counts up to time t
        S, A1 = self.expert1.shape
        _, A2 = self.expert2.shape
        counts1 = np.zeros((S,A1), dtype=int)
        counts2 = np.zeros((S,A2), dtype=int)
        for (s,a) in data1[:t]:
            counts1[s,a] +=1
        for (s,a) in data2[:t]:
            counts2[s,a] +=1
        return counts1, counts2
    
    def _counts_to_policy(self, counts):
        """
        Convert visit counts to probability policy using maximum likelihood estimation.
        
        
        Args:
            counts (np.ndarray): Visit counts of shape (S, A) where counts[s,a] 
                               is the number of times action a was taken in state s
                               
        Returns:
            np.ndarray: Probability policy of shape (S, A) where policy[s,a] is
                       the probability of taking action a in state s
                       
        """
        S,A = counts.shape
        pol = np.zeros_like(counts, dtype=float)
        for s in range(S):
            total = counts[s].sum()
            if total>0:
                pol[s] = counts[s]/total
            else:
                pol[s] = 1.0/A
        return pol

   
    def train(self, eval_interval=10, grid_game=None):
        """
        Train policies using behavioral cloning on expert demonstrations.
        
        This method implements the core behavioral cloning algorithm:
        1. Generate expert demonstration data
        2. Incrementally update state-action counts
        3. Periodically evaluate learned policies using exploitability
        4. Return final learned policies and training statistics
        
        Args:
            eval_interval (int, optional): How often to evaluate policies (default: 10)
                                         Evaluation happens every eval_interval samples
            grid_game (optional): Grid game environment (unused, kept for compatibility)
            
        Returns:
            tuple: (final_policy_1, final_policy_2, iterations, exploitability) where:
                - final_policy_1: Learned policy for player 1, shape (S, A1)
                - final_policy_2: Learned policy for player 2, shape (S, A2)  
                - iterations: List of iteration numbers when evaluations occurred
                - exploitability: List of exploitability values at each evaluation
                
        """
        iterations = []
        exploitability = []
        data1, data2 = self.generate_expert_data()

        # 1. Initialize counts *before* the loop
        S, A1 = self.expert1.shape
        _, A2 = self.expert2.shape
        counts1 = np.zeros((S, A1), dtype=int)
        counts2 = np.zeros((S, A2), dtype=int)

        # 2. Loop through the data ONCE
        for it, ((s1, a1), (s2, a2)) in enumerate(tqdm(zip(data1, data2), total=self.total_samples)):
            # 3. Increment counts for the current sample
            counts1[s1, a1] += 1
            counts2[s2, a2] += 1

            # 4. Evaluate periodically using the up-to-date counts
            if it % eval_interval == 0:
                policy_1 = self._counts_to_policy(counts1.copy())
                policy_2 = self._counts_to_policy(counts2.copy())
                exp = calc_exploitability_true(policy_1, policy_2, self.true_r, self.P, self.initial_state_dist, self.gamma)
                iterations.append(it)
                exploitability.append(exp)

        # Final policies are based on all data
        final_policy_1 = self._counts_to_policy(counts1)
        final_policy_2 = self._counts_to_policy(counts2)

        return final_policy_1, final_policy_2, iterations, exploitability

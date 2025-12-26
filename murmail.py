"""
MURMAIL: Maximum Uncertainty Response Multi-Agent Imitation Learning

This module implements the MURMAIL algorithm for multi-agent imitation learning in 
zero-sum Markov games. MURMAIL addresses the challenge of learning Nash equilibrium
policies from expert demonstrations by using uncertainty-based inner loops and
mirror descent policy updates.

"""

import numpy as np
from tqdm import tqdm
from utils import calc_exploitability_true

class MaxUncertaintyResponseImitationLearning:
    """
    Maximum Uncertainty Response Multi-Agent Imitation Learning (MURMAIL) Algorithm.
    
    This class implements Algorithm 2 from the MURMAIL paper, which combines multi-agent
    imitation learning with maximum-uncertainty inner response loops. The algorithm
    learns Nash equilibrium policies in zero-sum games from expert demonstrations.
    
    Core Algorithm:
    For each iteration k:
    1. Inner Loop (Player μ): Build induced MDP and solve with uncertainty-maximizing RL
    2. Inner Loop (Player ν): Build induced MDP and solve with uncertainty-maximizing RL  
    3. Sampling: Draw (state, action) pairs from learned occupancy measures
    4. Policy Update: Apply mirror descent based on expert policy agreement
    5. Evaluation: Compute exploitability of average policies
    
    The algorithm uses synthetic reward functions based on expert policy agreement:
    R[s] = E[𝟙{A₁=A₂} - 2π(A₁|s) + Σₐπ(a|s)²]
    
    Attributes:
        K (int): Number of outer loop iterations
        P (np.ndarray): Transition probabilities, shape (S, A1, A2, S')
        muE, nuE (np.ndarray): Expert policies for players μ and ν
        algo: Inner loop RL algorithm (e.g., UCBVI)
        eta (float): Learning rate for mirror descent updates
        gamma (float): Discount factor
        eval_freq (int): Frequency of exploitability evaluation
        true_r (np.ndarray): True reward function, shape (S, A1, A2)
        init_dist (np.ndarray): Initial state distribution
        rollout_length (int): Maximum episode length for inner loop
        expert_samples (int): Number of expert action samples per iteration
        game_params (dict): Parameters for inner loop algorithm
        S, A1, A2 (int): Number of states and actions for each player
    """

    def __init__(
        self,
        num_iterations: int,
        transitions: np.ndarray,           # P[s,a1,a2,s']
        expert_policies: tuple[np.ndarray, np.ndarray],
        innerloop_algo,
        learning_rate: float,
        gamma: float,
        eval_freq: int,
        true_rewards: np.ndarray,  # r[s,a1,a2]
        initial_dist :np.ndarray,
        rollout_length: int = 1000,
        expert_samples: int = 1,
        game_params = None,
    ):
        """
        Initialize MURMAIL algorithm with game parameters and hyperparameters.
        
        Args:
            num_iterations (int): Number of outer loop iterations K
            transitions (np.ndarray): Transition probabilities P(s'|s,a₁,a₂), shape (S,A1,A2,S')
            expert_policies (tuple): Tuple of expert policies (μₑ, νₑ) for players 1 and 2
                                   Each policy has shape (S, A) where A is player's action space
            innerloop_algo: RL algorithm for inner loop (e.g., UCBVI instance)
                          Must implement run_algo(P, R, game_params, gamma) method
            learning_rate (float): Step size η for mirror descent updates
            gamma (float): Discount factor γ ∈ [0,1) for value computation
            eval_freq (int): Frequency of exploitability evaluation (every eval_freq iterations)
            true_rewards (np.ndarray): True reward function r(s,a₁,a₂), shape (S,A1,A2)
            initial_dist (np.ndarray): Initial state distribution μ₀(s), shape (S,)
            rollout_length (int, optional): Episode length for inner loop RL (default: 1000)
            expert_samples (int, optional): Number of expert samples per iteration (default: 1)
            game_params (dict, optional): Additional parameters for inner loop algorithm
            
        """
        self.K       = num_iterations
        self.P       = transitions
        self.muE, self.nuE = expert_policies
        self.algo    = innerloop_algo
        self.eta     = learning_rate
        self.gamma   = gamma
        self.eval_freq = eval_freq
        self.true_r  = true_rewards
        self.init_dist = initial_dist
        self.rollout_length = rollout_length
        self.expert_samples = expert_samples
        self.game_params = game_params

        self.S, self.A1, self.A2, _ = transitions.shape

    def run(self, batch_size = 1):
        """
        Execute the MURMAIL algorithm for multi-agent imitation learning.
        
        This method implements the complete MURMAIL algorithm, alternating between
        inner loop RL optimization and mirror descent policy updates. 
        
        Args:
            batch_size (int, optional): Number of samples per policy update (default: 1)
                                      - batch_size=1: Single sample updates (original algorithm)
                                      - batch_size>1: Batch updates for improved stability
                                      
        Returns:
            tuple: (queries, exploit, avg_policy_mu, avg_policy_nu) where:
                - queries: List of iteration numbers when evaluations occurred
                - exploit: List of exploitability values at evaluation points
                - avg_policy_mu: Final average policy for player μ, shape (S, A1)
                - avg_policy_nu: Final average policy for player ν, shape (S, A2)
                
    
        """
        # 1) initialize μ₀, ν₀ uniformly
        policy_mu = np.ones((self.S, self.A1)) / self.A1
        policy_nu = np.ones((self.S, self.A2)) / self.A2
        
        # Initialize average policies
        avg_policy_mu = policy_mu.copy()
        avg_policy_nu = policy_nu.copy()

        queries, exploit = [], []
        for k in tqdm(range(self.K // batch_size), desc="MURMAIL iterations"):
            # ---- inner solve for μ-player: build induced MDP (s, b) -> s'
            P_mu = np.einsum('sabt,sb->sat', self.P, policy_mu)
            samples_mu = self._sample_expert(self.muE, N=self.expert_samples)
            R_mu = self._state_reward(samples_mu, policy_mu)
            _, yk = self.algo.run_algo(P_mu, R_mu, self.game_params, gamma=self.gamma)

            # ---- inner solve for ν-player: swap roles a↔b in P
            P_nu = np.einsum('sabt,sb->sat', self.P, policy_nu)
            samples_nu = self._sample_expert(self.nuE, N=self.expert_samples)
            R_nu = self._state_reward(samples_nu, policy_nu)
            _, zk = self.algo.run_algo(P_nu, R_nu, self.game_params, gamma=self.gamma)

            # ---- sample (s,a) from occupancy measure
            if batch_size == 1:
                s_mu, a_mu = self._exact_sample(policy_mu, yk, expert_policy=self.muE)
                s_nu, a_nu = self._exact_sample(zk, policy_nu, expert_policy=self.nuE)
                # ---- exponentiated‐gradient (mirror descent) steps
                policy_mu = self._exp_grad_update(policy_mu, s_mu, a_mu)
                policy_nu = self._exp_grad_update(policy_nu, s_nu, a_nu)
            else:
                s_mu, a_mu = self._batch_sample(policy_mu, yk, expert_policy=self.muE, batch_size=batch_size)
                s_nu, a_nu = self._batch_sample(zk, policy_nu, expert_policy=self.nuE, batch_size=batch_size)
                policy_mu = self._batch_exp_grad_update(policy_mu, s_mu, a_mu, batch_size)
                policy_nu = self._batch_exp_grad_update(policy_nu, s_nu, a_nu, batch_size)
            
            # ---- update average policies
            avg_policy_mu = (k * avg_policy_mu + policy_mu) / (k + 1)
            avg_policy_nu = (k * avg_policy_nu + policy_nu) / (k + 1)

            # ---- logging
            if k % self.eval_freq == 0:
                queries.append(k)
                # Evaluate exploitability using the average policies
                exploit.append(calc_exploitability_true(avg_policy_mu, avg_policy_nu, self.true_r, self.P, self.init_dist, self.gamma))

        return queries, exploit, avg_policy_mu, avg_policy_nu    
    

    def _sample_expert(self, expert_pi: np.ndarray, N: int) -> np.ndarray:
        """
        Sample expert action pairs for reward function construction.
        
        This method generates multiple independent samples of expert actions for each
        state, which are used to construct the synthetic reward function based on
        expert policy agreement. The sampling is done independently for each state
        and each sample.
        
        The samples are used in the reward function:
        R[s] = E[𝟙{A₁=A₂} - 2π(A₁|s) + Σₐπ(a|s)²]
        where the expectation is over the expert action samples.

        Parameters
        ----------
        expert_pi : np.ndarray, shape (S, A)
            Expert policy π_E(a|s) providing action probabilities for each state.
        N : int
            Number of independent action pairs to sample per state.

        Returns
        -------
        np.ndarray, shape (N, 2, S)
            Array of expert action samples where:
            - draws[i, 0, s] is the first action A₁^{(i)} sampled for state s
            - draws[i, 1, s] is the second action A₂^{(i)} sampled for state s
            Both actions are sampled independently from expert_pi[s].
            
        Note:
            - Each action pair (A₁, A₂) is sampled independently from the expert policy
            - Higher N provides better estimates but increases computational cost
            - Used to construct agreement-based reward functions for inner loop RL
            - The same expert policy is used for both action samples in each pair
        """
        S, A = expert_pi.shape
        draws = np.zeros((N, 2, S), dtype=int)

        for i in range(N):
            # first and second sample for this i
            for k in (0, 1):
                # sample one action per state
                # [ np.random.choice(A, p=expert_pi[s]) for s in range(S) ] → shape (S,)
                draws[i, k, :] = [
                    np.random.choice(A, p=expert_pi[s])
                    for s in range(S)
                ]

        return draws

    def _state_reward(self,
                  draws: np.ndarray,
                  policy: np.ndarray) -> np.ndarray:
        """
        Construct synthetic reward function for RL inner loop.

        Parameters
        ----------
        draws : np.ndarray, shape (N, 2, S)
            Expert action samples from _sample_expert method where:
            - N is the number of independent samples
            - draws[i, 0, s] and draws[i, 1, s] are expert actions for state s in sample i
        policy : np.ndarray, shape (S, A)
            Current learned policy π(a|s) for the player being updated.

        Returns
        -------
        np.ndarray, shape (S,)
            Synthetic reward function R[s] for each state, averaged over expert samples.
            Higher rewards encourage states where learned policy agrees with expert.

        """
        # number of samples and number of states
        N, _, S = draws.shape

        # draws[:, 0, :] is A1^{(i)} for i=1..N, shape (N, S)
        A1 = draws[:, 0, :]
        A2 = draws[:, 1, :]

        # indicator array: shape (N, S)
        indicators = (A1 == A2).astype(float)

        # for each sample i and state s, π(A1^{(i)}|s):
        # policy[s, A1[i, s]] → we need to index smartly
        # we'll build an array shape (N, S) of π(A1|s)
        # by first repeating the state indices
        states = np.arange(S)[None, :].repeat(N, axis=0)  # shape (N, S)
        mu_A1 = policy[states, A1]                        # shape (N, S)

        # constant term: ∑_a π(a|s)^2, shape (S,)
        norm2 = np.sum(policy**2, axis=1)[None, :]

        # per-sample estimate: shape (N, S)
        per_sample = indicators - 2 * mu_A1 + norm2

        # average over N samples: shape (S,)
        return per_sample.mean(axis=0)

    def _compute_state_visitation(self, pi_primary: np.ndarray, pi_opponent: np.ndarray) -> np.ndarray:
        """
        Compute exact discounted state visitation distribution for joint policy.
        
        
        Args:
            pi_primary (np.ndarray): Policy for the primary player, shape (S, A_primary)
            pi_opponent (np.ndarray): Policy for the opponent player, shape (S, A_opponent)
            
        Returns:
            np.ndarray: State visitation distribution d^π(s), shape (S,)
                       Normalized to sum to 1, represents probability distribution over states.
        """
        # 1) Build joint transition P_pi[s,s']
        #    We assume pi_primary is π_μ and pi_opponent is π_ν,
        #    but if swap=True, we swap roles when contracting.
        P = self.P
        
        # standard: sum over a,b
        weights = pi_primary[:, :, None] * pi_opponent[:, None, :]
        P_joint = (P * weights[:, :, :, None]).sum(axis=(1,2))  # (S,S')

        # 2) Solve (I - gamma P_pi)^{-1} 1
        identity_matrix = np.eye(self.S)  # Avoid ambiguous variable name
        inv = np.linalg.inv(identity_matrix - self.gamma * P_joint)         # (S,S)
        ones = np.ones(self.S)
        d_tilde = (1 - self.gamma) * (inv @ ones)             # (S,)

        # 3) Normalize to sum to 1
        return d_tilde / d_tilde.sum()
    
    def _batch_sample(self, pi_primary: np.ndarray, pi_opponent: np.ndarray, expert_policy: np.ndarray, batch_size: int):
        """
        Draw a batch of (s,a) pairs from the discounted state visitation.
        """
        # 1) Compute discounted state-visitation distribution d(s)
        d = self._compute_state_visitation(pi_primary, pi_opponent)

        # 2) Sample a batch of states from d(s)
        s_batch = np.random.choice(self.S, size=batch_size, p=d)

        # 3) For each sampled state, sample an action from the expert policy
        A = expert_policy.shape[1]
        a_batch = np.array([
            np.random.choice(A, p=expert_policy[s])
            for s in s_batch
        ])

        return s_batch, a_batch

    def _batch_exp_grad_update(self, policy: np.ndarray, s_batch: np.ndarray, a_batch: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Performs a mirror-descent update using an average gradient from a batch of (s,a) samples.
        This version uses an explicit loop to compute the "raw sum" of gradients.
        """
        # 1. Initialize a tensor to accumulate the sum of gradients.
        #    This will hold Σ [π(·|s_i) - e_{a_i}] for each state s.
        grad_sum = np.zeros_like(policy)

        # 2. Iterate through each sample in the batch to compute the raw sum.
        for s, a in zip(s_batch, a_batch):
            # For each sample (s_i, a_i), the gradient is π(·|s_i) - e_{a_i}.
            # We add this gradient vector to the accumulator for state s.
            
            # Add the policy part: +π(·|s)
            grad_sum[s, :] += policy[s, :]
            
            # Add the one-hot part: -e_{a}
            grad_sum[s, a] -= 1

        # 3. Calculate the average gradient over the batch.
        avg_grad = grad_sum / batch_size

        # 4. Apply the exponentiated gradient update.
        #    The update is only non-trivial for states that were in the batch,
        #    as avg_grad is zero for all other states.
        new_policy = policy * np.exp(-self.eta * avg_grad)

        # 5. Re-normalize the distributions only for the states that were updated.
        updated_states = np.unique(s_batch)
        for s in updated_states:
            new_policy[s, :] /= new_policy[s, :].sum()

        return new_policy

    def _exact_sample(self, pi_primary: np.ndarray, pi_opponent: np.ndarray, expert_policy: np.ndarray):
        """
        Sample state-action pair from exact discounted visitation distribution.
    
        Args:
            pi_primary (np.ndarray): Current policy for primary player, shape (S, A_primary)
            pi_opponent (np.ndarray): Current policy for opponent player, shape (S, A_opponent)  
            expert_policy (np.ndarray): Expert policy for action sampling, shape (S, A_expert)
            
        Returns:
            tuple: (s, a) where:
                - s (int): Sampled state index
                - a (int): Sampled action index from expert policy
                
        """
        d = self._compute_state_visitation(pi_primary, pi_opponent)

        # sample state
        s = np.random.choice(self.S, p=d)

        # Sample expert action for the given state
        a = np.random.choice(expert_policy.shape[1], p=expert_policy[s])
        
        return s, a

    def _exp_grad_update(self, policy: np.ndarray, s: int, a: int) -> np.ndarray:
        """
        Apply mirror descent update using exponentiated gradient method.
        
        Args:
            policy (np.ndarray): Current policy π_k, shape (S, A)
            s (int): Sampled state index where update is applied
            a (int): Sampled action index from expert policy
            
        Returns:
            np.ndarray: Updated policy π_{k+1}, shape (S, A)
                       Only state s is modified, other states remain unchanged.
                       
        """
        new_policy = policy.copy()
        # Following the gradient formula in the algorithm
        g = policy[s, a] - (1 if a == a else 0)  # Indicator is 1 when a equals the sampled action
        # Apply the exponentiated gradient update
        new_policy[s, a] = policy[s, a] * np.exp(-self.eta * g)
        # Re-normalize
        new_policy[s] /= new_policy[s].sum()
        return new_policy

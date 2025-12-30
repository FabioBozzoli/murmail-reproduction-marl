"""
MURMAIL: Maximum Uncertainty Response Multi-Agent Imitation Learning
VERSIONE CORRETTA - FIX EVAL_FREQ

🔧 FIX APPLICATO: Condizione di valutazione corretta
   Problema: k % eval_freq con eval_freq=10000 e k∈[1,200] non si verifica mai
   Soluzione: Converti eval_freq da queries a iterazioni
"""

import numpy as np
from tqdm import tqdm
from utils import calc_exploitability_true

class MaxUncertaintyResponseImitationLearning:
    """
    MURMAIL Algorithm 2 - VERSIONE CORRETTA
    """

    def __init__(
        self,
        num_iterations: int,
        transitions: np.ndarray,
        expert_policies: tuple[np.ndarray, np.ndarray],
        innerloop_algo,
        learning_rate: float,
        gamma: float,
        eval_freq: int,
        true_rewards: np.ndarray,
        initial_dist: np.ndarray,
        rollout_length: int = 1000,
        expert_samples: int = 1,
        game_params=None,
    ):
        self.K = num_iterations
        self.P = transitions
        self.muE, self.nuE = expert_policies
        self.algo = innerloop_algo
        self.eta = learning_rate
        self.gamma = gamma
        self.eval_freq = eval_freq  # In queries (es. 10000)
        self.true_r = true_rewards
        self.init_dist = initial_dist
        self.rollout_length = rollout_length
        self.expert_samples = expert_samples
        self.game_params = game_params

        self.S, self.A1, self.A2, _ = transitions.shape
        
        print(f"📊 MURMAIL Initialized:")
        print(f"   States: {self.S}, Actions μ: {self.A1}, ν: {self.A2}")
        print(f"   LR: {self.eta}, Gamma: {self.gamma}, Expert samples: {self.expert_samples}")

    def run(self, batch_size=1):
        """Execute MURMAIL algorithm (Algorithm 2 from paper)."""
        # Initialize uniform policies
        policy_mu = np.ones((self.S, self.A1), dtype=np.float64) / self.A1
        policy_nu = np.ones((self.S, self.A2), dtype=np.float64) / self.A2
        
        avg_policy_mu = policy_mu.copy()
        avg_policy_nu = policy_nu.copy()

        queries, exploit = [], []
        
        # Initial gap
        ex_init = calc_exploitability_true(
            avg_policy_mu, avg_policy_nu, self.true_r, 
            self.P, self.init_dist, self.gamma
        )
        queries.append(0)
        exploit.append(ex_init)
        print(f"🎯 Initial Gap: {ex_init:.6f}")
        
        num_iters = self.K // batch_size
        
        # *** FIX CRITICO: Converti eval_freq da queries a iterazioni ***
        eval_every_k = max(1, self.eval_freq // batch_size)
        print(f"📊 Training: {num_iters} iters, eval ogni {eval_every_k} iters (={self.eval_freq} queries)")
        
        for k in tqdm(range(1, num_iters + 1), desc="MURMAIL"):
            # ========== PLAYER μ UPDATE ==========
            P_mu = np.einsum('sabt,sa->sbt', self.P, policy_mu)
            samples_mu = self._sample_expert(self.muE, N=self.expert_samples)
            R_mu = self._state_reward(samples_mu, policy_mu)
            _, yk = self.algo.run_algo(P_mu, R_mu, self.game_params, gamma=self.gamma)

            # ========== PLAYER ν UPDATE ==========
            P_nu = np.einsum('sabt,sb->sat', self.P, policy_nu)
            samples_nu = self._sample_expert(self.nuE, N=self.expert_samples)
            R_nu = self._state_reward(samples_nu, policy_nu)
            _, zk = self.algo.run_algo(P_nu, R_nu, self.game_params, gamma=self.gamma)

            # ========== POLICY UPDATES ==========
            s_mu, a_mu = self._batch_sample(policy_mu, yk, self.muE, batch_size)
            s_nu, a_nu = self._batch_sample(zk, policy_nu, self.nuE, batch_size)
            
            policy_mu = self._batch_exp_grad_update(policy_mu, s_mu, a_mu, batch_size)
            policy_nu = self._batch_exp_grad_update(policy_nu, s_nu, a_nu, batch_size)
            
            policy_mu = self._project_simplex(policy_mu)
            policy_nu = self._project_simplex(policy_nu)
            
            # ========== AVERAGING ==========
            weight = 1.0 / k
            avg_policy_mu = (1 - weight) * avg_policy_mu + weight * policy_mu
            avg_policy_nu = (1 - weight) * avg_policy_nu + weight * policy_nu

            # ========== EVALUATION (FIX APPLICATO QUI) ==========
            if k % eval_every_k == 0:  # ✅ ORA CORRETTO!
                ex = calc_exploitability_true(
                    avg_policy_mu, avg_policy_nu, self.true_r,
                    self.P, self.init_dist, self.gamma
                )
                current_queries = k * batch_size
                queries.append(current_queries)
                exploit.append(ex)
                
                improvement = exploit[-2] - ex if len(exploit) > 1 else 0
                direction = "↓" if improvement > 0 else "↑"
                print(f" Iter {k:5d} | Queries {current_queries:7d} | Gap: {ex:.6f} {direction} | Δ: {improvement:+.6f}")

        return queries, exploit, avg_policy_mu, avg_policy_nu

    def _sample_expert(self, expert_pi: np.ndarray, N: int) -> np.ndarray:
        """Sample expert action pairs for reward construction (Lemma G.7)."""
        S, A = expert_pi.shape
        draws = np.zeros((N, 2, S), dtype=np.int32)

        for i in range(N):
            for k in (0, 1):
                draws[i, k, :] = [
                    np.random.choice(A, p=expert_pi[s])
                    for s in range(S)
                ]
        return draws

    def _state_reward(self, draws: np.ndarray, policy: np.ndarray) -> np.ndarray:
        """Construct synthetic reward (Lemma G.7 from paper)."""
        N, _, S = draws.shape
        A1 = draws[:, 0, :]
        A2 = draws[:, 1, :]

        indicators = (A1 == A2).astype(np.float64)
        states = np.arange(S)[None, :].repeat(N, axis=0)
        mu_A1 = policy[states, A1]
        norm2 = np.sum(policy**2, axis=1)[None, :]

        per_sample = indicators - 2 * mu_A1 + norm2
        return per_sample.mean(axis=0)

    def _compute_state_visitation(self, pi_primary: np.ndarray, pi_opponent: np.ndarray) -> np.ndarray:
        """Compute discounted state visitation distribution d^π(s)."""
        weights = pi_primary[:, :, None] * pi_opponent[:, None, :]
        P_joint = (self.P * weights[:, :, :, None]).sum(axis=(1, 2))

        identity_matrix = np.eye(self.S)
        inv = np.linalg.inv(identity_matrix - self.gamma * P_joint)
        ones = np.ones(self.S)
        d_tilde = (1 - self.gamma) * (inv @ ones)

        return d_tilde / d_tilde.sum()

    def _batch_sample(self, pi_primary: np.ndarray, pi_opponent: np.ndarray, 
                     expert_policy: np.ndarray, batch_size: int):
        """Sample batch of (state, action) pairs."""
        d = self._compute_state_visitation(pi_primary, pi_opponent)
        s_batch = np.random.choice(self.S, size=batch_size, p=d)
        
        A = expert_policy.shape[1]
        a_batch = np.array([
            np.random.choice(A, p=expert_policy[s])
            for s in s_batch
        ])

        return s_batch, a_batch

    def _batch_exp_grad_update(self, policy: np.ndarray, s_batch: np.ndarray, 
                               a_batch: np.ndarray, batch_size: int) -> np.ndarray:
        """Mirror descent update using exponential weights."""
        grad_sum = np.zeros_like(policy)

        for s, a in zip(s_batch, a_batch):
            grad_sum[s, :] += policy[s, :]
            grad_sum[s, a] -= 1.0

        avg_grad = grad_sum / batch_size
        new_policy = policy * np.exp(-self.eta * avg_grad)

        updated_states = np.unique(s_batch)
        for s in updated_states:
            new_policy[s, :] /= new_policy[s, :].sum()

        return new_policy

    def _project_simplex(self, policy: np.ndarray) -> np.ndarray:
        """Project policy onto probability simplex (safety check)."""
        policy = np.maximum(policy, 0.0)
        row_sums = policy.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0
        return policy / row_sums
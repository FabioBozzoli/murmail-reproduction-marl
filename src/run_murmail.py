"""
run_murmail_FIXED.py - MURMAIL con W&B logging completo
VERSIONE CORRETTA con eval_freq fix
"""

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from numba import jit, prange
import wandb

from murmail import MaxUncertaintyResponseImitationLearning
from utils import calc_exploitability_true

# ============= WANDB SETUP =============
wandb.init(
    entity="307972-unimore",
    project="MARL",
    config={
        # Hyperparameters
        "discretization_bins": 6,
        "gamma": 0.9,
        "learning_rate": 500.0,
        "num_iterations": 200000,
        "eval_freq": 10000,
        "rollout_length": 50,
        "expert_samples": 20,
        "batch_size": 1000,
    },
    tags=["murmail", "speaker-listener", "expert-imitation"],
    notes="MURMAIL training on Speaker-Listener with discretization bins=6"
)

config = wandb.config

# ============= LOAD =============
print("📂 Loading data...")

pi_s = np.load('expert_policy_speaker_bins6.npy')
pi_l = np.load('expert_policy_listener_bins6.npy')
init_dist = np.load('expert_initial_dist_bins6.npy')

try:
    P = np.load('P_bins6.npy').astype(np.float64)
    R = np.load('R_bins6.npy').astype(np.float64)
    print("✓ Loaded pre-built P, R")
except FileNotFoundError:
    print("❌ P_bins6.npy not found!")
    wandb.finish(exit_code=1)
    exit(1)

S = P.shape[0]
A_SPEAKER = P.shape[1]
A_LISTENER = P.shape[2]

print(f"✓ P: {P.shape}, R: {R.shape}")
print(f"✓ States: {S}, Actions: μ={A_SPEAKER}, ν={A_LISTENER}")

wandb.config.update({
    "num_states": S,
    "num_actions_speaker": A_SPEAKER,
    "num_actions_listener": A_LISTENER,
})

# ============= FAST VI =============
@jit(nopython=True, cache=True, parallel=True)
def _fast_vi(P_ats, R, gamma, tol=1e-8, max_iter=200):
    A, S, _ = P_ats.shape
    V = np.zeros(S, dtype=np.float64)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((A, S), dtype=np.float64)
        
        for a in prange(A):
            for s in range(S):
                Q[a, s] = R[s]
                for sp in range(S):
                    Q[a, s] += gamma * P_ats[a, s, sp] * V_old[sp]
        
        for s in prange(S):
            max_val = Q[0, s]
            for a in range(1, A):
                if Q[a, s] > max_val:
                    max_val = Q[a, s]
            V[s] = max_val
        
        max_diff = 0.0
        for s in range(S):
            diff = abs(V[s] - V_old[s])
            if diff > max_diff:
                max_diff = diff
        
        if max_diff < tol:
            break
    
    pi = np.zeros((S, A), dtype=np.float64)
    for s in range(S):
        best_a = 0
        best_q = Q[0, s]
        for a in range(1, A):
            if Q[a, s] > best_q:
                best_q = Q[a, s]
                best_a = a
        pi[s, best_a] = 1.0
    
    return V, pi

class FastVISolver:
    def run_algo(self, P, R_in, params, gamma):
        P_ats = np.ascontiguousarray(np.transpose(P, (1, 0, 2)))
        R = R_in.mean(axis=1) if R_in.ndim > 1 else R_in
        return _fast_vi(P_ats, R, gamma)

# ============= VALIDATE =============
print("\n🔍 Validation...")
gap_expert = calc_exploitability_true(pi_s, pi_l, R, P, init_dist, 0.9)
print(f"   Expert gap: {gap_expert:.6f}")

mu_unif = np.ones((S, A_SPEAKER), dtype=np.float64) / A_SPEAKER
nu_unif = np.ones((S, A_LISTENER), dtype=np.float64) / A_LISTENER
gap_uniform = calc_exploitability_true(mu_unif, nu_unif, R, P, init_dist, 0.9)
print(f"   Uniform gap: {gap_uniform:.6f}")
print(f"   Ratio: {gap_uniform/gap_expert:.1f}x")

# Log baselines
wandb.log({
    "baseline/expert_gap": gap_expert,
    "baseline/uniform_gap": gap_uniform,
    "baseline/improvement_potential": gap_uniform - gap_expert,
    "baseline/ratio": gap_uniform / gap_expert,
}, step=0)

wandb.run.summary["expert_gap"] = gap_expert
wandb.run.summary["uniform_gap"] = gap_uniform

if gap_expert > 0.1:
    print("\n⚠️  WARNING: Expert gap alto!")
    wandb.alert(
        title="High Expert Gap",
        text=f"Expert exploitability is {gap_expert:.6f} (>0.1)",
        level=wandb.AlertLevel.WARN
    )
    response = input("   Continuare? (y/n): ")
    if response.lower() != 'y':
        wandb.finish(exit_code=1)
        exit(0)

# ============= MURMAIL WITH WANDB LOGGING =============
print("\n🚀 MURMAIL...")

game_params = {'num_states': S, 'num_actions': max(A_SPEAKER, A_LISTENER)}

class MURMAILWithWandb(MaxUncertaintyResponseImitationLearning):
    """
    Extended MURMAIL with W&B logging
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.logged_iterations = set()  # Track logged iters to avoid duplicates
    
    def run(self, batch_size=1):
        """Override run to add detailed W&B logging"""
        self.start_time = time.time()
        
        # Initialize policies
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
        
        # Log initial state
        self._log_to_wandb(
            iteration=0,
            queries=0,
            gap=ex_init,
            policy_mu=avg_policy_mu,
            policy_nu=avg_policy_nu
        )
        
        num_iters = self.K // batch_size
        eval_every_k = max(1, self.eval_freq // batch_size)
        
        print(f"📊 Training: {num_iters} iters, eval ogni {eval_every_k} iters")
        
        from tqdm import tqdm
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

            # ========== EVALUATION ==========
            if k % eval_every_k == 0:
                ex = calc_exploitability_true(
                    avg_policy_mu, avg_policy_nu, self.true_r,
                    self.P, self.init_dist, self.gamma
                )
                current_queries = k * batch_size
                queries.append(current_queries)
                exploit.append(ex)
                
                # Log to W&B
                self._log_to_wandb(
                    iteration=k,
                    queries=current_queries,
                    gap=ex,
                    policy_mu=avg_policy_mu,
                    policy_nu=avg_policy_nu
                )
                
                improvement = exploit[-2] - ex if len(exploit) > 1 else 0
                direction = "↓" if improvement > 0 else "↑"
                print(f" Iter {k:5d} | Queries {current_queries:7d} | Gap: {ex:.6f} {direction} | Δ: {improvement:+.6f}")

        return queries, exploit, avg_policy_mu, avg_policy_nu
    
    def _log_to_wandb(self, iteration, queries, gap, policy_mu, policy_nu):
        """Log detailed metrics to W&B"""
        
        # Avoid duplicate logging
        if iteration in self.logged_iterations:
            return
        self.logged_iterations.add(iteration)
        
        elapsed = time.time() - self.start_time
        
        # Compute improvement metrics
        improvement_from_uniform = gap_uniform - gap
        improvement_percentage = (improvement_from_uniform / (gap_uniform - gap_expert)) * 100
        remaining_gap = gap - gap_expert
        
        # Policy entropy
        def policy_entropy(pi):
            pi_safe = np.clip(pi, 1e-10, 1.0)
            H = -np.sum(pi_safe * np.log(pi_safe), axis=1)
            return np.mean(H)
        
        entropy_mu = policy_entropy(policy_mu)
        entropy_nu = policy_entropy(policy_nu)
        
        # Policy sparsity (determinism)
        def policy_sparsity(pi):
            max_probs = np.max(pi, axis=1)
            return np.mean(max_probs > 0.99)
        
        sparsity_mu = policy_sparsity(policy_mu)
        sparsity_nu = policy_sparsity(policy_nu)
        
        # Policy L2 norm (concentration measure)
        l2_mu = np.mean(np.sum(policy_mu**2, axis=1))
        l2_nu = np.mean(np.sum(policy_nu**2, axis=1))
        
        # Log everything to W&B
        log_dict = {
            # Core metrics
            "iteration": iteration,
            "queries": queries,
            "gap/current": gap,
            "gap/improvement_from_uniform": improvement_from_uniform,
            "gap/remaining_to_expert": remaining_gap,
            "gap/percentage_of_possible": improvement_percentage,
            "gap/normalized": (gap - gap_expert) / (gap_uniform - gap_expert) if gap_uniform > gap_expert else 0,
            
            # Policy characteristics - Speaker (μ)
            "policy/entropy_speaker": entropy_mu,
            "policy/sparsity_speaker": sparsity_mu,
            "policy/l2_speaker": l2_mu,
            
            # Policy characteristics - Listener (ν)
            "policy/entropy_listener": entropy_nu,
            "policy/sparsity_listener": sparsity_nu,
            "policy/l2_listener": l2_nu,
            
            # Efficiency metrics
            "efficiency/queries_per_iteration": queries / max(iteration, 1),
            "efficiency/gap_reduction_per_1k_queries": (gap_uniform - gap) / (queries / 1000 + 1),
            "time/elapsed_seconds": elapsed,
            "time/elapsed_minutes": elapsed / 60,
            "time/queries_per_second": queries / max(elapsed, 1),
            "time/iterations_per_second": iteration / max(elapsed, 1),
        }
        
        wandb.log(log_dict, step=iteration)
        
        # Update summary for best values
        if gap < wandb.run.summary.get("best_gap", float('inf')):
            wandb.run.summary.update({
                "best_gap": gap,
                "best_gap_iteration": iteration,
                "best_gap_queries": queries,
                "best_improvement_pct": improvement_percentage,
                "best_gap_remaining": remaining_gap,
            })

# Initialize MURMAIL with W&B logging
murmail = MURMAILWithWandb(
    num_iterations=config.num_iterations,
    transitions=P,
    expert_policies=(pi_s, pi_l),
    innerloop_algo=FastVISolver(),
    learning_rate=config.learning_rate,
    gamma=config.gamma,
    eval_freq=config.eval_freq,
    true_rewards=R,
    initial_dist=init_dist,
    rollout_length=config.rollout_length,
    expert_samples=config.expert_samples,
    game_params=game_params
)

# Run training
print("\n" + "="*70)
print("Starting MURMAIL training with W&B logging...")
print("="*70)

start = time.time()
queries, exploit, p_s, p_l = murmail.run(batch_size=config.batch_size)
elapsed = time.time() - start

# ============= FINAL RESULTS =============
print(f"\n✅ Done in {elapsed/60:.1f} min")
print(f"\n📊 FINAL RESULTS:")
print(f"   Initial:  {exploit[0]:.6f}")
print(f"   Final:    {exploit[-1]:.6f}")
print(f"   Improve:  {exploit[0] - exploit[-1]:.6f}")

# Log final summary
final_improvement_pct = ((gap_uniform - exploit[-1]) / (gap_uniform - gap_expert)) * 100

wandb.run.summary.update({
    "final/gap": exploit[-1],
    "final/queries": queries[-1],
    "final/improvement_from_uniform": gap_uniform - exploit[-1],
    "final/improvement_percentage": final_improvement_pct,
    "final/elapsed_minutes": elapsed / 60,
    "final/queries_per_second": queries[-1] / elapsed,
    "final/total_iterations": len(queries) - 1,
})

# ============= VISUALIZATION =============
print("\n📊 Creating visualizations...")

# Main convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.semilogy(queries, exploit, 'o-', linewidth=2, markersize=4, label='MURMAIL', color='blue')
ax1.axhline(y=gap_expert, color='green', linestyle='--', label=f'Expert ({gap_expert:.4f})', linewidth=2)
ax1.axhline(y=gap_uniform, color='red', linestyle='--', label=f'Uniform ({gap_uniform:.4f})', linewidth=2, alpha=0.5)
ax1.set_xlabel('Expert Queries', fontsize=12)
ax1.set_ylabel('Exploitability Gap (log scale)', fontsize=12)
ax1.set_title('MURMAIL Convergence - Log Scale', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(queries, exploit, 'o-', linewidth=2, markersize=4, label='MURMAIL', color='blue')
ax2.axhline(y=gap_expert, color='green', linestyle='--', label=f'Expert ({gap_expert:.4f})', linewidth=2)
ax2.axhline(y=gap_uniform, color='red', linestyle='--', label=f'Uniform ({gap_uniform:.4f})', linewidth=2, alpha=0.5)
ax2.fill_between(queries, gap_expert, gap_uniform, alpha=0.1, color='gray', label='Improvement region')
ax2.set_xlabel('Expert Queries', fontsize=12)
ax2.set_ylabel('Exploitability Gap', fontsize=12)
ax2.set_title('MURMAIL Convergence - Linear Scale', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = 'murmail_convergence_bins6.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {plot_path}")

wandb.log({"plots/convergence": wandb.Image(plot_path)})

# Policy determinism histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

speaker_max_probs = np.max(p_s, axis=1)
axes[0].hist(speaker_max_probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='Deterministic threshold (0.99)')
axes[0].set_xlabel('Max Probability per State', fontsize=12)
axes[0].set_ylabel('Number of States', fontsize=12)
pct_det_speaker = np.mean(speaker_max_probs > 0.99) * 100
axes[0].set_title(f'Speaker Policy Determinism\n{pct_det_speaker:.1f}% deterministic states', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

listener_max_probs = np.max(p_l, axis=1)
axes[1].hist(listener_max_probs, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(x=0.99, color='red', linestyle='--', linewidth=2, label='Deterministic threshold (0.99)')
axes[1].set_xlabel('Max Probability per State', fontsize=12)
axes[1].set_ylabel('Number of States', fontsize=12)
pct_det_listener = np.mean(listener_max_probs > 0.99) * 100
axes[1].set_title(f'Listener Policy Determinism\n{pct_det_listener:.1f}% deterministic states', 
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
policy_plot_path = 'murmail_policies_bins6.png'
plt.savefig(policy_plot_path, dpi=150, bbox_inches='tight')
print(f"   ✓ Saved: {policy_plot_path}")

wandb.log({"plots/policy_determinism": wandb.Image(policy_plot_path)})

# ============= SAVE ARTIFACTS =============
print("\n📦 Saving W&B artifacts...")

# Save learned policies
np.save('murmail_policy_speaker_final.npy', p_s)
np.save('murmail_policy_listener_final.npy', p_l)

# Create artifact
artifact = wandb.Artifact(
    name=f'murmail-policies-{wandb.run.id}',
    type='model',
    description=f'MURMAIL learned policies | Gap: {exploit[-1]:.6f} | Queries: {queries[-1]}'
)

# Add files
artifact.add_file('expert_policy_speaker_bins6.npy')
artifact.add_file('expert_policy_listener_bins6.npy')
artifact.add_file('murmail_policy_speaker_final.npy')
artifact.add_file('murmail_policy_listener_final.npy')

# Log artifact
wandb.log_artifact(artifact)
print(f"   ✓ Artifact logged: {artifact.name}")

# ============= FINISH =============
print("\n" + "="*70)
print("🎉 Training Complete!")
print("="*70)
print(f"📊 View results at: {wandb.run.get_url()}")
print(f"📊 Project page: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print("="*70)

wandb.finish()
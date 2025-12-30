"""
run_murmail_FIXED.py - Carica P e R dal pickle originale
"""

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from numba import jit, prange

from murmail import MaxUncertaintyResponseImitationLearning
from utils import calc_exploitability_true

# ============= LOAD =============
print("📂 Loading data...")

pi_s = np.load('expert_policy_speaker_bins6.npy')
pi_l = np.load('expert_policy_listener_bins6.npy')
init_dist = np.load('expert_initial_dist_bins6.npy')

# *** LOAD P, R DIRETTAMENTE DAL GENERATE SCRIPT ***
# Opzione 1: Se hai salvato P, R
try:
    P = np.load('P_bins6.npy').astype(np.float64)
    R = np.load('R_bins6.npy').astype(np.float64)
    print("✓ Loaded pre-built P, R")
except FileNotFoundError:
    print("❌ P_bins6.npy not found!")
    print("   Devi rigenerare con generate_expert_ROBUST.py modificato")
    print("   O usa il workaround sotto")
    exit(1)

S = P.shape[0]
A_SPEAKER = P.shape[1]
A_LISTENER = P.shape[2]

print(f"✓ P: {P.shape}, R: {R.shape}")
print(f"✓ States: {S}, Actions: μ={A_SPEAKER}, ν={A_LISTENER}")

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

if gap_expert > 0.1:
    print("\n⚠️  WARNING: Expert gap alto! Verifica generazione.")
    response = input("   Continuare comunque? (y/n): ")
    if response.lower() != 'y':
        exit(0)

# ============= MURMAIL =============
print("\n🚀 MURMAIL...")

game_params = {'num_states': S, 'num_actions': max(A_SPEAKER, A_LISTENER)}

murmail = MaxUncertaintyResponseImitationLearning(
    num_iterations=200000,
    transitions=P,
    expert_policies=(pi_s, pi_l),
    innerloop_algo=FastVISolver(),
    learning_rate=500.0,
    gamma=0.9,
    eval_freq=10000,
    true_rewards=R,
    initial_dist=init_dist,
    rollout_length=50,
    expert_samples=20,
    game_params=game_params
)

start = time.time()
queries, exploit, p_s, p_l = murmail.run(batch_size=1000)
elapsed = time.time() - start

print(f"\n✅ Done in {elapsed/60:.1f} min")
print(f"\n📊 RESULTS:")
print(f"   Initial:  {exploit[0]:.6f}")
print(f"   Final:    {exploit[-1]:.6f}")
print(f"   Improve:  {exploit[0] - exploit[-1]:.6f}")

# ============= PLOT =============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.semilogy(queries, exploit, 'o-', linewidth=2, markersize=4)
ax1.axhline(y=gap_expert, color='green', linestyle='--', label=f'Expert ({gap_expert:.4f})')
ax1.set_xlabel('Queries')
ax1.set_ylabel('Gap (log)')
ax1.set_title('MURMAIL Convergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(queries, exploit, 'o-', linewidth=2, markersize=4)
ax2.axhline(y=gap_expert, color='green', linestyle='--', label=f'Expert ({gap_expert:.4f})')
ax2.set_xlabel('Queries')
ax2.set_ylabel('Gap')
ax2.set_title('Linear Scale')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('murmail_convergence_bins6.png', dpi=150)
print("\n📊 Plot saved")
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Assicuriamoci che la cartella esista
os.makedirs('results/plots', exist_ok=True)

# Parametro di conversione (MPE ha max 25 step per episodio)
STEPS_PER_EPISODE = 25

# ============= LOAD RESULTS =============
print("📂 Loading results...")

# 1. Joint-DQN
try:
    with open('joint_dqn_results.pkl', 'rb') as f:
        dqn = pickle.load(f)
    # Filter valid DQN gaps
    valid_dqn = [(ep, g) for ep, g in zip(dqn['eval_episodes'], dqn['eval_gaps']) if g is not None]
    eps_dqn, gaps_dqn = zip(*valid_dqn) if valid_dqn else ([], [])
    
    # CONVERSIONE EPISODI -> SAMPLES (STEPS)
    samples_dqn = [e * STEPS_PER_EPISODE for e in eps_dqn]
    
    print(f"✓ Joint-DQN loaded ({len(samples_dqn)} points, max samples: {max(samples_dqn) if samples_dqn else 0})")
except FileNotFoundError:
    print("⚠️ joint_dqn_results.pkl not found, skipping DQN.")
    eps_dqn, gaps_dqn, samples_dqn = [], [], []

# 2. Joint-SAC
try:
    with open('joint_sac_results.pkl', 'rb') as f:
        sac = pickle.load(f)
    # Filter valid SAC gaps
    valid_sac = [(ep, g) for ep, g in zip(sac['eval_episodes'], sac['eval_gaps']) if g is not None]
    eps_sac, gaps_sac = zip(*valid_sac) if valid_sac else ([], [])
    
    # CONVERSIONE EPISODI -> SAMPLES (STEPS)
    samples_sac = [e * STEPS_PER_EPISODE for e in eps_sac]
    
    print(f"✓ Joint-SAC loaded ({len(samples_sac)} points, max samples: {max(samples_sac) if samples_sac else 0})")
except FileNotFoundError:
    print("⚠️ joint_sac_results.pkl not found, skipping SAC.")
    eps_sac, gaps_sac, samples_sac = [], [], []

# 3. MURMAIL
try:
    queries_murmail = np.load('murmail_queries.npy')
    gaps_murmail = np.load('murmail_exploit.npy')
    print(f"✓ MURMAIL loaded ({len(queries_murmail)} points, max queries: {max(queries_murmail) if len(queries_murmail)>0 else 0})")
except FileNotFoundError:
    print("⚠️ murmail files not found, skipping.")
    queries_murmail, gaps_murmail = [], []

# Baselines (Hardcoded or loaded)
expert_gap = 0.013
uniform_gap = 0.240

# ============= PLOT 1: CONVERGENCE =============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Log Scale ---
if len(samples_dqn) > 0:
    ax1.semilogy(samples_dqn, gaps_dqn, 'o-', label='Joint-DQN', color='orange', alpha=0.6, markersize=4)
if len(samples_sac) > 0:
    ax1.semilogy(samples_sac, gaps_sac, '^-', label='Joint-SAC', color='purple', alpha=0.8, markersize=5)
if len(queries_murmail) > 0:
    ax1.semilogy(queries_murmail, gaps_murmail, 's-', label='MURMAIL', color='blue', alpha=0.7, markersize=4)

ax1.axhline(expert_gap, color='green', linestyle='--', label='Expert (Nash)', linewidth=2)
ax1.axhline(uniform_gap, color='red', linestyle='--', label='Uniform', linewidth=2, alpha=0.5)

ax1.set_xlabel('Training Samples (Step/Queries)', fontsize=12)
ax1.set_ylabel('Exploitability Gap (log)', fontsize=12)
ax1.set_title('Convergence Speed (Log Scale)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which="both")

# --- Linear Scale ---
if len(samples_dqn) > 0:
    ax2.plot(samples_dqn, gaps_dqn, 'o-', label='Joint-DQN', color='orange', alpha=0.6, markersize=4)
if len(samples_sac) > 0:
    ax2.plot(samples_sac, gaps_sac, '^-', label='Joint-SAC', color='purple', alpha=0.8, markersize=5)
if len(queries_murmail) > 0:
    ax2.plot(queries_murmail, gaps_murmail, 's-', label='MURMAIL', color='blue', alpha=0.7, markersize=4)

ax2.axhline(expert_gap, color='green', linestyle='--', linewidth=2)
ax2.axhline(uniform_gap, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Evidenzia l'area di miglioramento
max_x = max(
    max(samples_dqn) if samples_dqn else 0, 
    max(samples_sac) if samples_sac else 0,
    max(queries_murmail) if list(queries_murmail) else 0
)
if max_x > 0:
    ax2.fill_between([0, max_x], expert_gap, uniform_gap, alpha=0.05, color='gray')

ax2.set_xlabel('Training Samples (Step/Queries)', fontsize=12)
ax2.set_ylabel('Exploitability Gap', fontsize=12)
ax2.set_title('Convergence Stability (Linear Scale)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/convergence_comparison_sac.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/plots/convergence_comparison_sac.png")
plt.close()

# ============= PLOT 2: FINAL COMPARISON BAR CHART =============
# Prep Data
methods = ['Expert\n(Nash)', 'Uniform']
gaps = [expert_gap, uniform_gap]
colors = ['green', 'gray']

# MURMAIL
if len(gaps_murmail) > 0:
    methods.insert(1, 'MURMAIL')
    gaps.insert(1, gaps_murmail[-1])
    colors.insert(1, 'blue')

# SAC (Generalmente più stabile, prendiamo il finale)
if len(gaps_sac) > 0:
    methods.insert(2, 'Joint-SAC')
    gaps.insert(2, gaps_sac[-1])
    colors.insert(2, 'purple')

# DQN (Spesso instabile, mostriamo Best e Final se differiscono molto, qui semplifichiamo al Best)
if len(gaps_dqn) > 0:
    methods.insert(3, 'Joint-DQN\n(Best)')
    gaps.insert(3, min(gaps_dqn))
    colors.insert(3, 'orange')

plt.figure(figsize=(11, 6))
bars = plt.bar(methods, gaps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

plt.ylabel('Exploitability Gap (Lower is Better)', fontsize=12)
plt.title('Final Performance Comparison: Imitation vs RL', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add values on bars
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{gap:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/plots/final_comparison_sac.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/plots/final_comparison_sac.png")
plt.close()

# ============= SUMMARY TABLE =============
print("\n" + "="*70)
print(f"{'METHOD':<20} | {'FINAL GAP':<10} | {'BEST GAP':<10} | {'IMPROVEMENT vs UNIFORM'}")
print("-" * 70)

print(f"{'Expert (Nash)':<20} | {expert_gap:.4f}     | {expert_gap:.4f}     | 100% (Target)")

if len(gaps_murmail) > 0:
    imp = (uniform_gap - gaps_murmail[-1]) / (uniform_gap - expert_gap) * 100
    print(f"{'MURMAIL':<20} | {gaps_murmail[-1]:.4f}     | {min(gaps_murmail):.4f}     | {imp:.1f}%")

if len(gaps_sac) > 0:
    imp = (uniform_gap - gaps_sac[-1]) / (uniform_gap - expert_gap) * 100
    print(f"{'Joint-SAC':<20} | {gaps_sac[-1]:.4f}     | {min(gaps_sac):.4f}     | {imp:.1f}%")

if len(gaps_dqn) > 0:
    imp = (uniform_gap - min(gaps_dqn)) / (uniform_gap - expert_gap) * 100
    print(f"{'Joint-DQN':<20} | {gaps_dqn[-1]:.4f}     | {min(gaps_dqn):.4f}     | {imp:.1f}% (using best)")

print(f"{'Uniform':<20} | {uniform_gap:.4f}     | {uniform_gap:.4f}     | 0.0%")
print("="*70)
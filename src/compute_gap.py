"""
compare_all_gaps.py - UNIVERSAL BENCHMARK

Confronta tutti gli algoritmi (MAPPO, DQN, MURMAIL, EXPERT) 
sullo stesso terreno di gioco (Dinamiche 6 Bins).
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import os
import time
from utils import calc_exploitability_true

# ============= CONFIG =============
GAMMA = 0.9
BINS = 6

# Nomi dei file attesi (modifica se i tuoi sono diversi)
FILES = {
    "Expert (Nash)": {
        "s": f"expert_policy_speaker_bins{BINS}.npy",
        "l": f"expert_policy_listener_bins{BINS}.npy"
    },
    "MAPPO": {
        "s": f"mappo_final_converged_probs_speaker.npy", # O quello che hai esportato
        "l": f"mappo_final_converged_probs_listener.npy"
    },
    "Joint-DQN": {
        "s": "dqn_policy_speaker.npy",
        "l": "dqn_policy_listener.npy"
    },
    "MuRMAIL": {
        "s": "murmail_policy_speaker.npy",
        "l": "murmail_policy_listener.npy"
    }
}

DYNAMICS_FILES = {
    "P": f"P_bins{BINS}.npy",
    "R": f"R_bins{BINS}.npy",
    "init": f"expert_initial_dist_bins{BINS}.npy"
}

# ============= LOAD DYNAMICS =============
print("="*70)
print(f"📊 UNIVERSAL BENCHMARK (BINS={BINS})")
print("="*70)

print("\n📂 Loading Environment Dynamics...")
try:
    P = np.load(DYNAMICS_FILES["P"]).astype(np.float64)
    R = np.load(DYNAMICS_FILES["R"]).astype(np.float64)
    init_dist = np.load(DYNAMICS_FILES["init"]).astype(np.float64)
    print(f"✓ Loaded P: {P.shape}, R: {R.shape}")
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Dynamics file not found: {e}")
    print(f"   Run 'generate_expert_FINAL.py' first to create P/R matrices for bins={BINS}.")
    exit(1)

# ============= HELPER FUNCTION =============
def get_uniform_policy(num_states, num_actions):
    return np.ones((num_states, num_actions)) / num_actions

# ============= MAIN LOOP =============
results = {}

print("\n🚀 Starting Evaluation Loop...")

# 1. Evaluate UNIFORM Baseline first
print(f"\n🔹 Evaluating: Random Uniform")
try:
    s_dim, a_s_dim, a_l_dim, _ = P.shape
    pi_s_unif = get_uniform_policy(s_dim, a_s_dim)
    pi_l_unif = get_uniform_policy(s_dim, a_l_dim)
    gap = calc_exploitability_true(pi_s_unif, pi_l_unif, R, P, init_dist, GAMMA)
    results["Uniform"] = gap
    print(f"   Gap: {gap:.6f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# 2. Evaluate all other algorithms
for name, files in FILES.items():
    print(f"\n🔹 Evaluating: {name}")
    
    # Check if files exist
    if not os.path.exists(files["s"]) or not os.path.exists(files["l"]):
        print(f"   ⚠️  Skipping: Policy files not found ({files['s']})")
        continue
        
    try:
        # Load
        pi_s = np.load(files["s"])
        pi_l = np.load(files["l"])
        
        # Validation shapes
        if pi_s.shape[0] != P.shape[0]:
            print(f"   ⚠️  Shape Mismatch! Env States: {P.shape[0]}, Policy States: {pi_s.shape[0]}")
            print("       (Did you mix 6-bin policies with 10-bin dynamics?)")
            continue

        # Compute Gap
        start = time.time()
        gap = calc_exploitability_true(pi_s, pi_l, R, P, init_dist, GAMMA)
        elapsed = time.time() - start
        
        results[name] = gap
        print(f"   Gap: {gap:.6f} (Computed in {elapsed:.1f}s)")
        
    except Exception as e:
        print(f"   ❌ Error computing gap: {e}")

# ============= RANKING & PLOT =============
print("\n" + "="*70)
print("🏆 FINAL RANKING (Lower is Better)")
print("="*70)

# Sort by gap (ascending)
sorted_res = sorted(results.items(), key=lambda x: x[1])

expert_gap = results.get("Expert (Nash)", 0.0)
uniform_gap = results.get("Uniform", 1.0)
improvement_range = uniform_gap - expert_gap

stats = []

for rank, (name, gap) in enumerate(sorted_res, 1):
    # Calculate % improvement over random
    if improvement_range > 0:
        score = 100 * (uniform_gap - gap) / improvement_range
    else:
        score = 0.0
        
    print(f"{rank}. {name:20s} Gap: {gap:.6f} | Score: {score:5.1f}% (0=Random, 100=Expert)")
    
    stats.append({
        "name": name,
        "gap": gap,
        "score": score
    })

# ============= SAVE JSON =============
with open(f"benchmark_results_bins{BINS}.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\n💾 Saved detailed stats to benchmark_results_bins{BINS}.json")

# ============= PLOT =============
names = [x["name"] for x in stats]
gaps = [x["gap"] for x in stats]
colors = ['green' if 'Expert' in n else 'grey' if 'Uniform' in n else 'skyblue' for n in names]

plt.figure(figsize=(10, 6))
bars = plt.bar(names, gaps, color=colors, edgecolor='black', alpha=0.7)

plt.ylabel('Exploitability Gap (Lower is Better)')
plt.title(f'Multi-Agent Algorithm Benchmark (Bins={BINS})')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Labels
for bar, gap in zip(bars, gaps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{gap:.4f}', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f"benchmark_plot_bins{BINS}.png", dpi=150)
print(f"📊 Saved comparison plot to benchmark_plot_bins{BINS}.png")
print("✅ DONE.")
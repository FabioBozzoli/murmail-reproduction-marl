"""
fix_expert_distribution.py - CALCOLA d^π CORRETTAMENTE

Usa policy esistenti + P esistenti per calcolare vera state visitation.
"""

import numpy as np

print("="*70)
print("🔧 FIX EXPERT DISTRIBUTION")
print("="*70)

# Load
print("\n📂 Loading files...")
pi_s = np.load('expert_policy_speaker_bins6.npy')
pi_l = np.load('expert_policy_listener_bins6.npy')
P = np.load('P_bins6.npy').astype(np.float64)

S = P.shape[0]
gamma = 0.9

print(f"✓ States: {S}")
print(f"✓ Policies: μ={pi_s.shape}, ν={pi_l.shape}")

# Compute d^π corretta
print("\n🔧 Computing d^π = (1-γ)μ₀(I - γP^π)^(-1)...")

# Joint policy
P_joint = np.zeros((S, S), dtype=np.float64)

for s in range(S):
    for a1 in range(pi_s.shape[1]):
        for a2 in range(pi_l.shape[1]):
            weight = pi_s[s, a1] * pi_l[s, a2]
            P_joint[s, :] += P[s, a1, a2, :] * weight

print(f"✓ P_joint computed")

# d^π = (1-γ) * (I - γP^π)^(-1) @ μ₀
mu_0 = np.ones(S) / S  # Uniform initial
I = np.eye(S)

print("   Inverting (I - γP^π)...")
inv_matrix = np.linalg.inv(I - gamma * P_joint)

d_pi = (1 - gamma) * (inv_matrix @ mu_0)

print(f"✓ d^π computed")
print(f"   Sum: {d_pi.sum():.6f} (should be 1.0)")
print(f"   Max: {d_pi.max():.6f}")
print(f"   Min: {d_pi.min():.6f}")
print(f"   Nonzero: {(d_pi > 1e-6).sum()}/{S}")

# Validate
print("\n🔍 Validating...")
from utils import calc_exploitability_true

R = np.load('R_bins6.npy').astype(np.float64)

gap_with_d_pi = calc_exploitability_true(pi_s, pi_l, R, P, d_pi, gamma)
print(f"   Gap con d^π corretta: {gap_with_d_pi:.6f}")

# Save
print("\n💾 Saving...")
np.save('expert_initial_dist_bins6.npy', d_pi)
print("✓ Overwritten expert_initial_dist_bins6.npy")

print("\n" + "="*70)
if gap_with_d_pi < 0.1:
    print("✅ SUCCESSO! Gap expert < 0.1")
    print(f"   Ora puoi usare run_murmail_FIXED.py")
else:
    print(f"⚠️  Gap ancora alto: {gap_with_d_pi:.6f}")
    print("   Problema probabilmente nelle policy, non nella distribuzione")
print("="*70)
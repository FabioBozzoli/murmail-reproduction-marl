"""
generate_expert_FINAL.py - VERSIONE CORRETTA DEFINITIVA

Fix applicati:
1. Più Nash iterations (2000 invece di 1200)
2. Più VI iterations (150 invece di 120)
3. Validazione con ENTRAMBE le distribuzioni
4. Salva P e R direttamente
"""

import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import time

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true

# ============= CONFIG =============
NUM_EPISODES_EXPLORATION = 20000
NASH_ITERATIONS = 2000          # ⬆️ Aumentato da 1200
VI_ITERATIONS = 150             # ⬆️ Aumentato da 120
GAMMA = 0.9
DISCRETIZATION_BINS = 6

OUTPUT_DATA_FILE = 'murmail_data_bins6.pkl'
OUTPUT_SPEAKER_POLICY = 'expert_policy_speaker_bins6.npy'
OUTPUT_LISTENER_POLICY = 'expert_policy_listener_bins6.npy'
OUTPUT_INITIAL_DIST = 'expert_initial_dist_bins6.npy'

A_SPEAKER = 3
A_LISTENER = 5
NUM_LISTENER_STATES = 27 * (DISCRETIZATION_BINS ** 2)
NUM_GOALS = 3
S_DIM = NUM_GOALS * NUM_LISTENER_STATES

print("="*70)
print("🎯 GENERAZIONE ESPERTO BINS=6 - VERSIONE FINALE CORRETTA")
print("="*70)
print(f"✓ Bins: {DISCRETIZATION_BINS}")
print(f"✓ Stati: {S_DIM}")
print(f"✓ Nash iters: {NASH_ITERATIONS}")
print(f"✓ VI iters: {VI_ITERATIONS}")
print(f"✓ Tempo stimato: 50-60 minuti")
print("="*70)


# ============= BUILD MATRICES =============
def build_matrices_robust(transitions, rewards, visits, S, A1, A2):
    """Costruisce matrici usando numpy puro."""
    print("   🔧 Allocating arrays...")
    P = np.zeros((S, A1, A2, S), dtype=np.float32)
    R = np.zeros((S, A1, A2), dtype=np.float32)
    
    print("   🔧 Filling rewards...")
    for s_joint, actions_dict in tqdm(rewards.items(), desc="   Rewards", leave=False):
        s_idx = get_joint_idx(*s_joint)
        if s_idx >= S:
            continue
        
        for a_joint, r_val in actions_dict.items():
            count = visits[s_joint][a_joint]
            if count > 0:
                R[s_idx, a_joint[0], a_joint[1]] = r_val / count
    
    print("   🔧 Filling transitions...")
    for s_joint, actions_dict in tqdm(transitions.items(), desc="   Trans", leave=False):
        s_idx = get_joint_idx(*s_joint)
        if s_idx >= S:
            continue
        
        for a_joint, next_dict in actions_dict.items():
            for ns_joint, t_count in next_dict.items():
                ns_idx = get_joint_idx(*ns_joint)
                if ns_idx < S:
                    P[s_idx, a_joint[0], a_joint[1], ns_idx] += t_count
    
    print("   🔧 Normalizing transitions...")
    for s in tqdm(range(S), desc="   Normalize", leave=False):
        for a1 in range(A1):
            for a2 in range(A2):
                row_sum = P[s, a1, a2, :].sum()
                if row_sum > 1e-9:
                    P[s, a1, a2, :] /= row_sum
                else:
                    P[s, a1, a2, :] = 1.0 / S
    
    return P, R


# ============= VALUE ITERATION =============
def value_iteration_robust(P_ind, R_ind, gamma, max_iter):
    """VI usando numpy puro."""
    S, A = R_ind.shape
    V = np.zeros(S, dtype=np.float64)
    
    for iteration in range(max_iter):
        V_old = V.copy()
        Q = np.zeros((S, A), dtype=np.float64)
        
        for a in range(A):
            Q[:, a] = R_ind[:, a] + gamma * np.dot(P_ind[a, :, :], V_old)
        
        V = np.max(Q, axis=1)
        
        if np.max(np.abs(V - V_old)) < 1e-6:
            break
    
    return V, Q


# ============= NASH SOLVER =============
class RobustNashSolver:
    def __init__(self, P, R, gamma, iterations):
        self.P = P.astype(np.float64)
        self.R = R.astype(np.float64)
        self.gamma = gamma
        self.iterations = iterations
        self.S, self.A1, self.A2, _ = P.shape

    def solve(self):
        print(f"   🧠 Nash Solver: {self.iterations} iterations")
        
        pi_1 = np.ones((self.S, self.A1), dtype=np.float64) / self.A1
        pi_2 = np.ones((self.S, self.A2), dtype=np.float64) / self.A2
        
        avg_pi_1 = pi_1.copy()
        avg_pi_2 = pi_2.copy()
        
        update_every = max(1, self.iterations // 10)
        start_time = time.time()
        
        for i in range(1, self.iterations + 1):
            br_1 = self._best_response(avg_pi_2, player=1)
            br_2 = self._best_response(avg_pi_1, player=2)
            
            alpha = 1.0 / i
            avg_pi_1 = (1 - alpha) * avg_pi_1 + alpha * br_1
            avg_pi_2 = (1 - alpha) * avg_pi_2 + alpha * br_2
            
            if i % update_every == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.iterations - i)
                print(f"      Iter {i}/{self.iterations} | {elapsed:.1f}s | ~{remaining:.1f}s left")
        
        print(f"   ✅ Nash done in {time.time() - start_time:.1f}s")
        return avg_pi_1, avg_pi_2

    def _best_response(self, opp_policy, player):
        """Best response usando numpy."""
        if player == 1:
            P_ind = np.zeros((self.A1, self.S, self.S), dtype=np.float64)
            R_ind = np.zeros((self.S, self.A1), dtype=np.float64)
            
            for a1 in range(self.A1):
                for s in range(self.S):
                    for a2 in range(self.A2):
                        weight = opp_policy[s, a2]
                        P_ind[a1, s, :] += self.P[s, a1, a2, :] * weight
                        R_ind[s, a1] += self.R[s, a1, a2] * weight
            
            n_actions = self.A1
        else:
            P_ind = np.zeros((self.A2, self.S, self.S), dtype=np.float64)
            R_ind = np.zeros((self.S, self.A2), dtype=np.float64)
            
            for a2 in range(self.A2):
                for s in range(self.S):
                    for a1 in range(self.A1):
                        weight = opp_policy[s, a1]
                        P_ind[a2, s, :] += self.P[s, a1, a2, :] * weight
                        R_ind[s, a2] -= self.R[s, a1, a2] * weight
            
            n_actions = self.A2
        
        V, Q = value_iteration_robust(P_ind, R_ind, self.gamma, VI_ITERATIONS)
        
        policy = np.zeros((self.S, n_actions), dtype=np.float64)
        best_actions = np.argmax(Q, axis=1)
        policy[np.arange(self.S), best_actions] = 1.0
        
        return policy


# ============= UTILS =============
def recursive_dict_conversion(d):
    if isinstance(d, defaultdict):
        return {k: recursive_dict_conversion(v) for k, v in d.items()}
    return d

def get_joint_idx(s_spk, s_lst):
    return s_spk * NUM_LISTENER_STATES + s_lst

def sanitize(policy):
    policy = np.asarray(policy, dtype=np.float64)
    policy = np.maximum(policy, 1e-10)
    return policy / policy.sum(axis=1, keepdims=True)

def normalize_rewards(R):
    r_min, r_max = R.min(), R.max()
    if r_max > r_min:
        print(f"      [{r_min:.4f}, {r_max:.4f}] → [0, 1]")
        return (R - r_min) / (r_max - r_min)
    return R


# ============= MAIN =============
def main():
    total_start = time.time()
    
    # ========== EXPLORATION ==========
    print(f"\nSTEP 1: EXPLORATION")
    print("="*70)
    
    step_start = time.time()
    
    raw_env = simple_speaker_listener_v4.parallel_env(
        continuous_actions=False, 
        render_mode=None,
        max_cycles=25
    )
    env = DiscretizedSpeakerListenerWrapper(raw_env, bins=DISCRETIZATION_BINS)
    
    transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    rewards = defaultdict(lambda: defaultdict(float))
    visits = defaultdict(lambda: defaultdict(int))

    for ep in tqdm(range(NUM_EPISODES_EXPLORATION), desc="Episodes"):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            s_spk, s_lst = obs["speaker_0"], obs["listener_0"]
            a_spk, a_lst = np.random.randint(0, A_SPEAKER), np.random.randint(0, A_LISTENER)
            
            actions = {"speaker_0": a_spk, "listener_0": a_lst}
            next_obs, rew_dict, term, trunc, _ = env.step(actions)
            
            s_joint = (s_spk, s_lst)
            a_joint = (a_spk, a_lst)
            ns_joint = (next_obs["speaker_0"], next_obs["listener_0"])
            
            transitions[s_joint][a_joint][ns_joint] += 1
            rewards[s_joint][a_joint] += rew_dict["speaker_0"]
            visits[s_joint][a_joint] += 1
            
            done = any(term.values()) or any(trunc.values())
            obs = next_obs
            steps += 1
    
    env.close()
    
    print(f"✅ Exploration: {time.time()-step_start:.1f}s")
    print(f"   Coverage: {len(transitions)}/{S_DIM} ({100*len(transitions)/S_DIM:.1f}%)")

    # ========== BUILD MATRICES ==========
    print(f"\nSTEP 2: BUILD MATRICES")
    print("="*70)
    
    step_start = time.time()
    P, R = build_matrices_robust(transitions, rewards, visits, S_DIM, A_SPEAKER, A_LISTENER)
    
    print(f"✅ Matrices: {time.time()-step_start:.1f}s")
    print(f"   P: {P.shape}, R: {R.shape}")
    
    R = normalize_rewards(R)

    # Save data
    norm_rewards_dict = defaultdict(lambda: defaultdict(float))
    for s_joint, actions_dict in rewards.items():
        s_idx = get_joint_idx(*s_joint)
        if s_idx < S_DIM:
            for a_joint in actions_dict.keys():
                norm_rewards_dict[s_joint][a_joint] = float(R[s_idx, a_joint[0], a_joint[1]])

    with open(OUTPUT_DATA_FILE, 'wb') as f:
        pickle.dump({
            'transitions': recursive_dict_conversion(transitions),
            'rewards': recursive_dict_conversion(norm_rewards_dict),
            'visits': recursive_dict_conversion(visits),
            'config': {'bins': DISCRETIZATION_BINS, 'num_states': S_DIM}
        }, f)

    # ========== NASH ==========
    print(f"\nSTEP 3: NASH EQUILIBRIUM")
    print("="*70)
    
    solver = RobustNashSolver(P, R, GAMMA, NASH_ITERATIONS)
    pi_s, pi_l = solver.solve()
    
    pi_s = sanitize((1 - 0.02) * pi_s + 0.02 / A_SPEAKER)
    pi_l = sanitize((1 - 0.02) * pi_l + 0.02 / A_LISTENER)

    # ========== VALIDATION (CRITICAL!) ==========
    print(f"\nSTEP 4: VALIDATION")
    print("="*70)
    
    init_dist_uniform = np.ones(S_DIM, dtype=np.float64) / S_DIM
    
    # Compute expert distribution
    expert_dist = np.zeros(S_DIM)
    for s_joint in transitions.keys():
        s_idx = get_joint_idx(*s_joint)
        if s_idx < S_DIM:
            expert_dist[s_idx] = len(transitions[s_joint])
    
    if expert_dist.sum() > 0:
        expert_dist = expert_dist / expert_dist.sum()
        expert_dist = 0.9 * expert_dist + 0.1 * init_dist_uniform
        expert_dist = expert_dist / expert_dist.sum()
    
    # *** CRITICAL: Test con ENTRAMBE le distribuzioni ***
    print("   🔍 Gap con distribuzione UNIFORME:")
    gap_expert_unif = calc_exploitability_true(pi_s, pi_l, R, P, init_dist_uniform, GAMMA)
    print(f"      Expert: {gap_expert_unif:.6f}")
    
    mu_unif = np.ones((S_DIM, A_SPEAKER), dtype=np.float64) / A_SPEAKER
    nu_unif = np.ones((S_DIM, A_LISTENER), dtype=np.float64) / A_LISTENER
    gap_uniform_unif = calc_exploitability_true(mu_unif, nu_unif, R, P, init_dist_uniform, GAMMA)
    print(f"      Uniform: {gap_uniform_unif:.6f}")
    print(f"      Ratio: {gap_uniform_unif/gap_expert_unif:.1f}x")
    
    print("\n   🔍 Gap con distribuzione EXPERT:")
    gap_expert_exp = calc_exploitability_true(pi_s, pi_l, R, P, expert_dist, GAMMA)
    print(f"      Expert: {gap_expert_exp:.6f}")
    
    gap_uniform_exp = calc_exploitability_true(mu_unif, nu_unif, R, P, expert_dist, GAMMA)
    print(f"      Uniform: {gap_uniform_exp:.6f}")
    print(f"      Ratio: {gap_uniform_exp/gap_expert_exp:.1f}x")

    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("📊 RISULTATI FINALI")
    print("="*70)
    
    if gap_expert_exp < 0.1 and gap_uniform_exp / gap_expert_exp > 5:
        print("✅ SUCCESSO!")
        print(f"   Expert gap (d_expert): {gap_expert_exp:.6f}")
        print(f"   Uniform gap (d_expert): {gap_uniform_exp:.6f}")
        print(f"   Ratio: {gap_uniform_exp/gap_expert_exp:.1f}x")
    else:
        print("⚠️  RISULTATI SUBOTTIMALI:")
        print(f"   Expert gap: {gap_expert_exp:.6f} (target: <0.1)")
        print(f"   Ratio: {gap_uniform_exp/gap_expert_exp:.1f}x (target: >5)")
        print("\n   Possibili cause:")
        print("   - Nash iterations insufficienti")
        print("   - VI iterations insufficienti")
        print("   - Gioco intrinsecamente difficile")

    # ========== SAVE =============
    print(f"\n💾 Saving...")
    np.save(OUTPUT_SPEAKER_POLICY, pi_s)
    np.save(OUTPUT_LISTENER_POLICY, pi_l)
    np.save(OUTPUT_INITIAL_DIST, expert_dist)
    
    # *** SALVA ANCHE P e R ***
    np.save('P_bins6.npy', P)
    np.save('R_bins6.npy', R)
    print(f"   ✓ Policies, distribution, P, R saved")
    
    total_time = time.time() - total_start
    
    print(f"\n✅ COMPLETATO IN {total_time/60:.1f} MINUTI")

if __name__ == "__main__":
    main()
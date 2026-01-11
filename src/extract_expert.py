"""
generate_expert_FINAL.py - VERSIONE CON WANDB LOGGING

Fix applicati:
1. Più Nash iterations (2000 invece di 1200)
2. Più VI iterations (150 invece di 120)
3. Validazione con ENTRAMBE le distribuzioni
4. Salva P e R direttamente
5. ✨ W&B logging completo
"""

import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import wandb  # ← NUOVO
import matplotlib.pyplot as plt  # ← NUOVO per plots

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true

# ============= CONFIG =============
NUM_EPISODES_EXPLORATION = 50000
NASH_ITERATIONS = 2000
VI_ITERATIONS = 150
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

# ============= WANDB SETUP =============
wandb.init(
    project="expert-nash-generation",
    name=f"expert-bins{DISCRETIZATION_BINS}-{time.strftime('%Y%m%d-%H%M%S')}",
    config={
        "discretization_bins": DISCRETIZATION_BINS,
        "num_states": S_DIM,
        "gamma": GAMMA,
        "nash_iterations": NASH_ITERATIONS,
        "vi_iterations": VI_ITERATIONS,
        "exploration_episodes": NUM_EPISODES_EXPLORATION,
        "num_actions_speaker": A_SPEAKER,
        "num_actions_listener": A_LISTENER,
    },
    tags=["expert-generation", "fictitious-play", "speaker-listener"],
    notes=f"Nash equilibrium generation with bins={DISCRETIZATION_BINS}"
)

config = wandb.config

print("="*70)
print("🎯 GENERAZIONE ESPERTO BINS=6 - VERSIONE CON WANDB")
print("="*70)
print(f"✓ Bins: {DISCRETIZATION_BINS}")
print(f"✓ Stati: {S_DIM}")
print(f"✓ Nash iters: {NASH_ITERATIONS}")
print(f"✓ VI iters: {VI_ITERATIONS}")
print(f"✓ Tempo stimato: 50-60 minuti")
print(f"📊 W&B Run: {wandb.run.get_url()}")
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


# ============= NASH SOLVER WITH WANDB =============
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
        
        # ← NOVO: Lista per tracking convergenza
        nash_iterations = []
        policy_changes_speaker = []
        policy_changes_listener = []
        
        for i in range(1, self.iterations + 1):
            # Save old policies per tracking change
            old_avg_pi_1 = avg_pi_1.copy()
            old_avg_pi_2 = avg_pi_2.copy()
            
            br_1 = self._best_response(avg_pi_2, player=1)
            br_2 = self._best_response(avg_pi_1, player=2)
            
            alpha = 1.0 / i
            avg_pi_1 = (1 - alpha) * avg_pi_1 + alpha * br_1
            avg_pi_2 = (1 - alpha) * avg_pi_2 + alpha * br_2
            
            # ← NOVO: Compute policy change
            change_pi_1 = np.mean(np.abs(avg_pi_1 - old_avg_pi_1))
            change_pi_2 = np.mean(np.abs(avg_pi_2 - old_avg_pi_2))
            
            if i % update_every == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.iterations - i)
                print(f"      Iter {i}/{self.iterations} | {elapsed:.1f}s | ~{remaining:.1f}s left")
                
                # ← NOVO: Log to W&B
                nash_iterations.append(i)
                policy_changes_speaker.append(change_pi_1)
                policy_changes_listener.append(change_pi_2)
                
                # Compute policy entropy (measure of determinism)
                def policy_entropy(pi):
                    pi_safe = np.clip(pi, 1e-10, 1.0)
                    H = -np.sum(pi_safe * np.log(pi_safe), axis=1)
                    return np.mean(H)
                
                entropy_1 = policy_entropy(avg_pi_1)
                entropy_2 = policy_entropy(avg_pi_2)
                
                # Compute sparsity (determinism)
                sparsity_1 = np.mean(np.max(avg_pi_1, axis=1) > 0.99)
                sparsity_2 = np.mean(np.max(avg_pi_2, axis=1) > 0.99)
                
                wandb.log({
                    "nash/iteration": i,
                    "nash/policy_change_speaker": change_pi_1,
                    "nash/policy_change_listener": change_pi_2,
                    "nash/entropy_speaker": entropy_1,
                    "nash/entropy_listener": entropy_2,
                    "nash/sparsity_speaker": sparsity_1,
                    "nash/sparsity_listener": sparsity_2,
                    "nash/elapsed_seconds": elapsed,
                    "nash/progress": i / self.iterations,
                }, step=i)
        
        total_time = time.time() - start_time
        print(f"   ✅ Nash done in {total_time:.1f}s")
        
        # ← NOVO: Log final Nash metrics
        wandb.run.summary.update({
            "nash/total_time_seconds": total_time,
            "nash/iterations_completed": self.iterations,
            "nash/final_entropy_speaker": policy_entropy(avg_pi_1),
            "nash/final_entropy_listener": policy_entropy(avg_pi_2),
        })
        
        # ← NOVO: Create convergence plot
        if len(nash_iterations) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.semilogy(nash_iterations, policy_changes_speaker, 'o-', label='Speaker', alpha=0.7)
            ax1.semilogy(nash_iterations, policy_changes_listener, 's-', label='Listener', alpha=0.7)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Policy Change (log scale)')
            ax1.set_title('Fictitious Play Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(nash_iterations, policy_changes_speaker, 'o-', label='Speaker', alpha=0.7)
            ax2.plot(nash_iterations, policy_changes_listener, 's-', label='Listener', alpha=0.7)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Policy Change')
            ax2.set_title('Linear Scale')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('nash_convergence.png', dpi=150, bbox_inches='tight')
            wandb.log({"plots/nash_convergence": wandb.Image('nash_convergence.png')})
            plt.close()
        
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
    
    # ← NOVO: Track exploration metrics
    episodes_completed = []
    states_visited = []
    unique_transitions = []

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
        
        # ← NOVO: Log exploration progress every 1000 episodes
        if (ep + 1) % 1000 == 0:
            episodes_completed.append(ep + 1)
            states_visited.append(len(transitions))
            unique_transitions.append(sum(len(actions) for actions in transitions.values()))
            
            coverage = len(transitions) / S_DIM
            
            wandb.log({
                "exploration/episodes": ep + 1,
                "exploration/states_visited": len(transitions),
                "exploration/coverage": coverage,
                "exploration/unique_transitions": unique_transitions[-1],
            }, step=ep + 1)
    
    env.close()
    
    exploration_time = time.time() - step_start
    coverage = len(transitions) / S_DIM
    
    print(f"✅ Exploration: {exploration_time:.1f}s")
    print(f"   Coverage: {len(transitions)}/{S_DIM} ({100*coverage:.1f}%)")
    
    # ← NOVO: Log exploration summary
    wandb.run.summary.update({
        "exploration/time_seconds": exploration_time,
        "exploration/final_coverage": coverage,
        "exploration/states_visited": len(transitions),
    })
    
    # ← NOVO: Plot exploration
    if len(episodes_completed) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(episodes_completed, states_visited, 'o-')
        ax1.axhline(y=S_DIM, color='r', linestyle='--', label=f'Total states ({S_DIM})')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Unique States Visited')
        ax1.set_title('Exploration Coverage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(episodes_completed, np.array(states_visited) / S_DIM, 'o-')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Coverage Fraction')
        ax2.set_title('State Space Coverage')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('exploration_coverage.png', dpi=150, bbox_inches='tight')
        wandb.log({"plots/exploration": wandb.Image('exploration_coverage.png')})
        plt.close()

    # ========== BUILD MATRICES ==========
    print(f"\nSTEP 2: BUILD MATRICES")
    print("="*70)
    
    step_start = time.time()
    P, R = build_matrices_robust(transitions, rewards, visits, S_DIM, A_SPEAKER, A_LISTENER)
    
    matrix_time = time.time() - step_start
    print(f"✅ Matrices: {matrix_time:.1f}s")
    print(f"   P: {P.shape}, R: {R.shape}")
    
    # ← NOVO: Log matrix statistics
    wandb.log({
        "matrices/build_time_seconds": matrix_time,
        "matrices/reward_min": float(R.min()),
        "matrices/reward_max": float(R.max()),
        "matrices/reward_mean": float(R.mean()),
        "matrices/reward_std": float(R.std()),
        "matrices/transition_sparsity": float(np.mean(P == 0)),
    })
    
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
    
    # Test con ENTRAMBE le distribuzioni
    print("   🔍 Gap con distribuzione UNIFORME:")
    gap_expert_unif = calc_exploitability_true(pi_s, pi_l, R, P, init_dist_uniform, GAMMA)
    print(f"      Expert: {gap_expert_unif:.6f}")
    
    mu_unif = np.ones((S_DIM, A_SPEAKER), dtype=np.float64) / A_SPEAKER
    nu_unif = np.ones((S_DIM, A_LISTENER), dtype=np.float64) / A_LISTENER
    gap_uniform_unif = calc_exploitability_true(mu_unif, nu_unif, R, P, init_dist_uniform, GAMMA)
    print(f"      Uniform: {gap_uniform_unif:.6f}")
    ratio_unif = gap_uniform_unif / gap_expert_unif
    print(f"      Ratio: {ratio_unif:.1f}x")
    
    print("\n   🔍 Gap con distribuzione EXPERT:")
    gap_expert_exp = calc_exploitability_true(pi_s, pi_l, R, P, expert_dist, GAMMA)
    print(f"      Expert: {gap_expert_exp:.6f}")
    
    gap_uniform_exp = calc_exploitability_true(mu_unif, nu_unif, R, P, expert_dist, GAMMA)
    print(f"      Uniform: {gap_uniform_exp:.6f}")
    ratio_exp = gap_uniform_exp / gap_expert_exp
    print(f"      Ratio: {ratio_exp:.1f}x")
    
    # ← NOVO: Log validation results
    wandb.log({
        "validation/expert_gap_uniform_dist": gap_expert_unif,
        "validation/uniform_gap_uniform_dist": gap_uniform_unif,
        "validation/ratio_uniform_dist": ratio_unif,
        "validation/expert_gap_expert_dist": gap_expert_exp,
        "validation/uniform_gap_expert_dist": gap_uniform_exp,
        "validation/ratio_expert_dist": ratio_exp,
    })

    # ========== RESULTS ==========
    print("\n" + "="*70)
    print("📊 RISULTATI FINALI")
    print("="*70)
    
    success = gap_expert_exp < 0.1 and ratio_exp > 5
    
    if success:
        print("✅ SUCCESSO!")
        print(f"   Expert gap (d_expert): {gap_expert_exp:.6f}")
        print(f"   Uniform gap (d_expert): {gap_uniform_exp:.6f}")
        print(f"   Ratio: {ratio_exp:.1f}x")
        
        wandb.run.summary["status"] = "success"
        wandb.alert(
            title="Expert Generation Success",
            text=f"Expert gap: {gap_expert_exp:.6f}, Ratio: {ratio_exp:.1f}x",
            level=wandb.AlertLevel.INFO
        )
    else:
        print("⚠️  RISULTATI SUBOTTIMALI:")
        print(f"   Expert gap: {gap_expert_exp:.6f} (target: <0.1)")
        print(f"   Ratio: {ratio_exp:.1f}x (target: >5)")
        print("\n   Possibili cause:")
        print("   - Nash iterations insufficienti")
        print("   - VI iterations insufficienti")
        print("   - Gioco intrinsecamente difficile")
        
        wandb.run.summary["status"] = "suboptimal"
        wandb.alert(
            title="Expert Generation Suboptimal",
            text=f"Expert gap: {gap_expert_exp:.6f} (target <0.1), Ratio: {ratio_exp:.1f}x (target >5)",
            level=wandb.AlertLevel.WARN
        )
    
    # ← NOVO: Summary metrics
    total_time = time.time() - total_start
    
    wandb.run.summary.update({
        "final/expert_gap": gap_expert_exp,
        "final/uniform_gap": gap_uniform_exp,
        "final/ratio": ratio_exp,
        "final/total_time_minutes": total_time / 60,
        "final/success": success,
    })

    # ========== SAVE =============
    print(f"\n💾 Saving...")
    np.save(OUTPUT_SPEAKER_POLICY, pi_s)
    np.save(OUTPUT_LISTENER_POLICY, pi_l)
    np.save(OUTPUT_INITIAL_DIST, expert_dist)
    np.save('P_bins6.npy', P)
    np.save('R_bins6.npy', R)
    print(f"   ✓ Policies, distribution, P, R saved")
    
    # ← NOVO: Save as W&B artifacts
    artifact = wandb.Artifact(
        name=f'expert-policies-bins{DISCRETIZATION_BINS}',
        type='model',
        description=f'Expert Nash policies | Gap: {gap_expert_exp:.6f} | Ratio: {ratio_exp:.1f}x'
    )
    
    artifact.add_file(OUTPUT_SPEAKER_POLICY)
    artifact.add_file(OUTPUT_LISTENER_POLICY)
    artifact.add_file(OUTPUT_INITIAL_DIST)
    artifact.add_file('P_bins6.npy')
    artifact.add_file('R_bins6.npy')
    artifact.add_file(OUTPUT_DATA_FILE)
    
    wandb.log_artifact(artifact)
    print(f"   ✓ Artifacts logged to W&B")
    
    # ← NOVO: Create final comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Expert\n(d_expert)', 'Uniform\n(d_expert)', 'Expert\n(d_uniform)', 'Uniform\n(d_uniform)']
    gaps = [gap_expert_exp, gap_uniform_exp, gap_expert_unif, gap_uniform_unif]
    colors = ['green', 'red', 'lightgreen', 'lightcoral']
    
    bars = ax.bar(methods, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Exploitability Gap', fontsize=12)
    ax.set_title('Expert vs Uniform Policy - Gap Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('expert_validation.png', dpi=150, bbox_inches='tight')
    wandb.log({"plots/validation": wandb.Image('expert_validation.png')})
    plt.close()
    
    print(f"\n✅ COMPLETATO IN {total_time/60:.1f} MINUTI")
    print(f"📊 View results at: {wandb.run.get_url()}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
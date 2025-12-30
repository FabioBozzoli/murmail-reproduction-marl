"""
Run Behavioral Cloning baseline on Speaker-Listener environment.
UPDATED: Fixes TypeError by summing transition counts correctly.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Assicurati che questi file siano presenti nella directory
from behavior_cloning import MultiAgentBehaviorCloning
from utils import calc_exploitability_true

def load_data_from_pickle(path='murmail_data_perfect.pkl'):
    """
    Carica il dataset generato da extract_expert.py e lo converte
    in matrici dense per l'algoritmo BC.
    """
    print(f"📂 Loading data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file '{path}' not found. Run extract_expert.py first!")

    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    d_trans = data['transitions'] 
    d_rewards = data['rewards']
    d_visits = data['visits']
    d_exp_s = data['expert_counts_speaker']
    d_exp_l = data['expert_counts_listener']
    
    # --- 1. DEFINIZIONE DIMENSIONI (CORRETTO) ---
    NUM_GOALS = 3
    # Listener: 5x5(Pos) * 3x3(Vel) * 3(Msg) = 675 stati
    # Questo deve coincidere con la logica del Wrapper!
    NUM_LISTENER_STATES = 675 
    
    S = NUM_GOALS * NUM_LISTENER_STATES # 3 * 675 = 2025 Stati Congiunti
    A1 = 3 # Speaker Actions
    A2 = 5 # Listener Actions
    
    print(f"  - Joint State Space: {S} (3 goals * {NUM_LISTENER_STATES} listener states)")
    print(f"  - Action Spaces: Speaker {A1}, Listener {A2}")

    # --- 2. INIZIALIZZAZIONE MATRICI DENSE ---
    P = np.zeros((S, A1, A2, S), dtype=np.float64)
    R = np.zeros((S, A1, A2), dtype=np.float64)
    pi_s = np.zeros((S, A1), dtype=np.float64)
    pi_l = np.zeros((S, A2), dtype=np.float64)
    
    def get_joint_idx(s_spk, s_lst):
        # s_lst può arrivare fino a 674, quindi moltiplichiamo s_spk per 675
        idx = s_spk * NUM_LISTENER_STATES + s_lst
        if idx >= S:
            raise IndexError(f"Calculated index {idx} exceeds size {S}. s_spk={s_spk}, s_lst={s_lst}")
        return idx

    # --- 3. RIEMPIMENTO TRANSIZIONI E REWARD ---
    print("  - Building Transition & Reward Tensors...")
    for s_joint, actions_dict in d_trans.items():
        s_idx = get_joint_idx(*s_joint)
        
        for a_joint, next_states_dict in actions_dict.items():
            a1, a2 = a_joint
            
            # Fix conteggio (dict vs int)
            if isinstance(d_visits[s_joint][a_joint], dict):
                 count = sum(d_visits[s_joint][a_joint].values())
            else:
                 count = d_visits[s_joint][a_joint]
            
            # Calcolo Reward Media
            if count > 0:
                R[s_idx, a1, a2] = d_rewards[s_joint][a_joint] / count
            
            # Calcolo Transizioni
            for ns_joint, trans_count in next_states_dict.items():
                ns_idx = get_joint_idx(*ns_joint)
                P[s_idx, a1, a2, ns_idx] = trans_count
            
            # Normalizzazione
            row_sum = np.sum(P[s_idx, a1, a2])
            if row_sum > 0:
                P[s_idx, a1, a2] /= row_sum

    # Padding per stati non visitati
    for s in range(S):
        for a1 in range(A1):
            for a2 in range(A2):
                if np.sum(P[s, a1, a2]) == 0:
                    P[s, a1, a2] = 1.0 / S

    # --- 4. COSTRUZIONE POLICY ESPERTE ---
    print("  - Constructing Expert Policies...")
    for s_idx in range(S):
        s_spk = s_idx // NUM_LISTENER_STATES
        s_lst = s_idx % NUM_LISTENER_STATES
        
        # Speaker
        if s_spk in d_exp_s:
            counts = d_exp_s[s_spk]
            if isinstance(counts, dict):
                 arr = np.zeros(A1)
                 for k, v in counts.items():
                     if k < A1: arr[k] = v
                 counts = arr
            
            counts = np.array(counts)
            if counts.sum() > 0:
                pi_s[s_idx] = counts / counts.sum()
            else:
                pi_s[s_idx] = 1.0 / A1
        else:
            pi_s[s_idx] = 1.0 / A1 
            
        # Listener
        if s_lst in d_exp_l:
            counts = d_exp_l[s_lst]
            if isinstance(counts, dict):
                 arr = np.zeros(A2)
                 for k, v in counts.items():
                     if k < A2: arr[k] = v
                 counts = arr
                 
            counts = np.array(counts)
            if counts.sum() > 0:
                pi_l[s_idx] = counts / counts.sum()
            else:
                pi_l[s_idx] = 1.0 / A2
        else:
            pi_l[s_idx] = 1.0 / A2

    initial_dist = np.ones(S) / S
    
    return pi_s, pi_l, P, R, initial_dist

def run_bc_experiment(total_samples=10000, gamma=0.9):
    """Run BC with varying dataset sizes."""
    print("📊 Running Behavioral Cloning Baseline...")
    
    expert_speaker, expert_listener, transitions, rewards, initial_dist = load_data_from_pickle()
    
    bc = MultiAgentBehaviorCloning(
        expert_policies=(expert_speaker, expert_listener),
        total_samples=total_samples,
        transition=transitions,
        payoff_matrix=rewards,
        gamma=gamma,
        initial_state_dist=initial_dist
    )
    
    print("🚀 Training started...")
    policy_speaker, policy_listener, iterations, exploitability = bc.train(
        eval_interval=100 
    )
    
    final_exploit = calc_exploitability_true(
        policy_speaker, policy_listener, 
        rewards, transitions, initial_dist, gamma
    )
    
    print(f"\n✅ BC Results:")
    print(f"  - Final exploitability: {final_exploit:.4f}")
    print(f"  - Dataset size: {total_samples}")
    
    return {
        'iterations': iterations,
        'exploitability': exploitability,
        'final_policies': (policy_speaker, policy_listener),
        'final_exploit': final_exploit
    }

def plot_bc_results(results, save_path='bc_results.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(results['iterations'], results['exploitability'], 
             label='BC', color='green', linewidth=2)
    plt.xlabel('Dataset Size', fontsize=12)
    plt.ylabel('Nash Gap (Exploitability)', fontsize=12)
    plt.title('Behavioral Cloning on Speaker-Listener', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Saved plot to {save_path}")

def main():
    try:
        results = run_bc_experiment(total_samples=10000, gamma=0.9)
        plot_bc_results(results)
        np.savez('bc_baseline_results.npz',
                 iterations=results['iterations'],
                 exploitability=results['exploitability'],
                 final_exploit=results['final_exploit'])
        print("\n💾 Saved results to bc_baseline_results.npz")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
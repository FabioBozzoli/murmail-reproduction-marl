import numpy as np
import pickle
import os
from numba import jit

# --- 1. FUNZIONE JIT (Compilata al volo in C) ---
@jit(nopython=True)
def _fill_dense_matrices(
    t_keys, t_vals,     
    r_keys, r_vals,     
    v_keys, v_vals,     
    num_states, num_actions
):
    # USA FLOAT64 PER LA MASSIMA PRECISIONE
    T = np.zeros((num_states, num_actions, num_actions, num_states), dtype=np.float64)
    R = np.zeros((num_states, num_actions, num_actions), dtype=np.float64)
    Visits = np.zeros((num_states, num_actions, num_actions), dtype=np.float64)

    # 1. Riempiamo Visits
    for i in range(len(v_vals)):
        s, a1, a2 = v_keys[i]
        Visits[s, a1, a2] = v_vals[i]

    # 2. Riempiamo Rewards
    for i in range(len(r_vals)):
        s, a1, a2 = r_keys[i]
        R[s, a1, a2] = r_vals[i]

    # 3. Riempiamo Transizioni
    for i in range(len(t_vals)):
        s, a1, a2, ns = t_keys[i]
        T[s, a1, a2, ns] = t_vals[i]

    # 4. Normalizzazione Robusta
    for s in range(num_states):
        for a1 in range(num_actions):
            for a2 in range(num_actions):
                count = Visits[s, a1, a2]
                
                if count > 0:
                    # Media Reward
                    R[s, a1, a2] /= count
                    # Normalizzazione Transizioni
                    T[s, a1, a2, :] /= count
                else:
                    # Stati non visitati: Uniforme
                    val = 1.0 / num_states
                    T[s, a1, a2, :] = val

                # --- PASSAGGIO EXTRA DI SICUREZZA ---
                # Corregge errori tipo 0.9999999 o 1.0000001
                row_sum = np.sum(T[s, a1, a2, :])
                if row_sum > 0:
                    T[s, a1, a2, :] /= row_sum
                else:
                    # Fallback estremo se la somma è ancora 0 (non dovrebbe succedere)
                    val = 1.0 / num_states
                    T[s, a1, a2, :] = val

    return T, R

# --- 2. FUNZIONE DI CARICAMENTO (Wrapper Python) ---
def load_expert_and_dynamics():
    """Load all necessary data using JIT for speed."""
    print("📂 Loading data (JIT Accelerated)...")
    
    if not os.path.exists('expert_policy_speaker.npy'):
        raise FileNotFoundError("Expert policies not found.")
        
    expert_speaker = np.load('expert_policy_speaker.npy')
    expert_listener = np.load('expert_policy_listener.npy')
    
    # Helper sanitizzazione (Numpy puro è già veloce qui)
    def sanitize(policy):
        sums = policy.sum(axis=1, keepdims=True)
        # Dove la somma è 0, metti uniforme
        mask = (sums == 0).flatten()
        policy[mask] = 1.0 / policy.shape[1]
        # Dove la somma è > 0, normalizza
        policy[~mask] /= sums[~mask]
        return policy

    expert_speaker = sanitize(expert_speaker)
    expert_listener = sanitize(expert_listener)
    
    num_states = expert_speaker.shape[0]
    num_actions = expert_speaker.shape[1]
    
    print(f"  - State Space: {num_states}, Action Space: {num_actions}")

    # Caricamento Pickle
    if not os.path.exists('dynamics_sparse.pkl'):
        raise FileNotFoundError("Dynamics file not found.")
        
    with open('dynamics_sparse.pkl', 'rb') as f:
        data = pickle.load(f)
        # Dizionari Python
        d_trans = data['transitions']
        d_rewards = data['rewards']
        d_visits = data['visits']
    
    print("  - Converting dictionaries to Numpy arrays for Numba...")
    
    # CONVERSIONE DIZIONARI -> NUMPY (Cruciale per Numba)
    # 1. Transitions
    if len(d_trans) > 0:
        t_keys = np.array(list(d_trans.keys()), dtype=np.int32)
        t_vals = np.array(list(d_trans.values()), dtype=np.float64)
    else:
        t_keys = np.zeros((0, 4), dtype=np.int32)
        t_vals = np.zeros((0,), dtype=np.float64)

    # 2. Rewards
    if len(d_rewards) > 0:
        r_keys = np.array(list(d_rewards.keys()), dtype=np.int32)
        r_vals = np.array(list(d_rewards.values()), dtype=np.float32)
    else:
        r_keys = np.zeros((0, 3), dtype=np.int32)
        r_vals = np.zeros((0,), dtype=np.float32)

    # 3. Visits
    if len(d_visits) > 0:
        v_keys = np.array(list(d_visits.keys()), dtype=np.int32)
        v_vals = np.array(list(d_visits.values()), dtype=np.float32)
    else:
        v_keys = np.zeros((0, 3), dtype=np.int32)
        v_vals = np.zeros((0,), dtype=np.float32)

    print("  - Reconstructing dense matrices (JIT)...")
    
    # CHIAMATA ALLA FUNZIONE COMPILATA
    transitions, rewards = _fill_dense_matrices(
        t_keys, t_vals, 
        r_keys, r_vals, 
        v_keys, v_vals, 
        num_states, num_actions
    )
    
    initial_dist = np.ones(num_states) / num_states
    
    return expert_speaker, expert_listener, transitions, rewards, initial_dist
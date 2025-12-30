import numpy as np
from numba import jit

# --- 1. CORE JIT OTTIMIZZATO (PLANNING DIRETTO) ---
@jit(nopython=True, cache=True)
def _run_ucbvi_fast_core(
    S, A, 
    num_episodes, horizon, gamma,
    P_true,      # (S, A, S) - Dinamiche reali
    R_true,      # (S, A)    - Reward sintetico
    start_dist   # (S,)      - Distribuzione iniziale (non usata nel planning puro)
):
    """
    Versione ottimizzata di UCBVI per Imitation Learning con dinamiche note.
    Invece di simulare episodi per stimare P_hat (lento e rumoroso),
    usiamo P_true direttamente per fare Value Iteration (veloce e preciso).
    
    Questo equivale a UCBVI nel limite di infiniti dati (Bonus -> 0).
    """
    
    # Inizializzazione Value Function
    V = np.zeros(S, dtype=np.float64)
    Q = np.zeros((S, A), dtype=np.float64)
    
    # Value Iteration
    # Eseguiamo un numero di iterazioni pari all'orizzonte o fino a convergenza
    # Usiamo un orizzonte esteso per garantire la propagazione del reward
    limit = max(horizon, 50) 
    
    for _ in range(limit):
        # Q(s,a) = R(s,a) + gamma * sum(P(s,a,ns) * V(ns))
        # Numba ottimizza questo loop matriciale automaticamente
        for s in range(S):
            for a in range(A):
                expected_val = 0.0
                for ns in range(S):
                    expected_val += P_true[s, a, ns] * V[ns]
                
                Q[s, a] = R_true[s, a] + gamma * expected_val
        
        # V(s) = max_a Q(s,a)
        for s in range(S):
            best_val = -1e10 # Valore molto basso
            for a in range(A):
                if Q[s, a] > best_val:
                    best_val = Q[s, a]
            V[s] = best_val

    # Estrazione Policy Deterministica Finale
    policy_final = np.zeros((S, A), dtype=np.float64)
    
    for s in range(S):
        best_a = 0
        best_val = -1e10
        for a in range(A):
            if Q[s, a] > best_val:
                best_val = Q[s, a]
                best_a = a
        policy_final[s, best_a] = 1.0

    return V, policy_final

# --- 2. PYTHON WRAPPER (INVARIATO) ---
class UCBVI:
    def __init__(self, num_episodes, horizon):
        self.num_episodes = num_episodes
        self.horizon = horizon

    def run_algo(
        self,
        transitions, 
        rewards,       
        mdp_params,
        initial_dist=None,
        gamma=0.99
    ):
        S = int(mdp_params['num_states'])
        A = int(mdp_params['num_actions'])

        # Gestione input (Robustezza)
        if rewards is None:
            R_in = np.zeros((S, A), dtype=np.float64)
        else:
            R_in = np.array(rewards, dtype=np.float64)
            if R_in.ndim == 1:
                R_in = np.tile(R_in.reshape(-1, 1), (1, A))
            elif R_in.ndim == 3:
                R_in = R_in.mean(axis=2)
            if R_in.shape != (S, A):
                # Fallback reshape
                temp = np.zeros((S, A), dtype=np.float64)
                flat = R_in.flatten()
                limit = min(len(flat), S*A)
                temp.flat[:limit] = flat[:limit]
                R_in = temp

        if transitions is None:
            P_in = np.zeros((S, A, S), dtype=np.float64)
        else:
            P_in = np.array(transitions, dtype=np.float64)
            if P_in.ndim == 4: # (S, A, A_opp, S)
                P_in = P_in.mean(axis=2) # Media sulle azioni avversarie
                # Rinormalizza
                sums = P_in.sum(axis=2, keepdims=True)
                sums[sums == 0] = 1.0
                P_in /= sums

        # Check dimensionale P
        if P_in.shape != (S, A, S):
             # Se la shape è sbagliata, prova a correggerla o inizializza uniforme
             P_in = np.ones((S, A, S), dtype=np.float64) / S

        if initial_dist is None:
            D_in = np.zeros(S, dtype=np.float64)
            D_in[0] = 1.0
        else:
            D_in = np.array(initial_dist, dtype=np.float64)

        # Contiguous array per Numba
        P_in = np.ascontiguousarray(P_in)
        R_in = np.ascontiguousarray(R_in)
        D_in = np.ascontiguousarray(D_in)

        # Esecuzione JIT
        V_final, policy_final = _run_ucbvi_fast_core(
            S, A,
            self.num_episodes,
            self.horizon,
            gamma,
            P_in,
            R_in,
            D_in
        )
        
        return V_final, policy_final
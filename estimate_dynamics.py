"""
Estimate transition dynamics P(s'|s,a1,a2) and rewards R(s,a1,a2) empirically.
FIXED VERSION: Corrected dimension handling for observation and action spaces.
"""

import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4
import supersuit as ss
from stable_baselines3.common.vec_env import VecNormalize
from collections import defaultdict
import pickle
from tqdm import tqdm
import os

class DynamicsEstimator:
    def __init__(self, stats_path, grid_size=5, n_samples_per_action=50):
        self.grid_size = grid_size
        self.n_samples_per_action = n_samples_per_action
        
        # 1. Check path
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")

        # 2. Create and Load Env
        self.env = self._create_env()
        self.env = VecNormalize.load(stats_path, self.env)
        self.env.training = False
        self.env.norm_reward = False
        
        # 3. Load Mapping
        if not os.path.exists('state_mapping.pkl'):
             raise FileNotFoundError("state_mapping.pkl not found. Run extract_expert.py first.")
             
        with open('state_mapping.pkl', 'rb') as f:
            self.state_mapping = pickle.load(f)
        
        # --- CORREZIONE DIMENSIONI (CRUCIALE) ---
        # Observation Space: È una tupla (Dim,), non (Batch, Dim)
        self.obs_dim = self.env.observation_space.shape[0]
        
        # Action Space: Gestione robusta Discrete vs Box
        if hasattr(self.env.action_space, 'n'):
            # Caso Discreto (Simple Speaker Listener usa Discrete(5) dopo il padding)
            self.num_actions = self.env.action_space.n
        else:
            # Caso Continuo/Box
            self.num_actions = self.env.action_space.shape[0]
            
        # Calcolo dimensione spazio stati discretizzato
        self.num_states = self.grid_size ** self.obs_dim
        
        print(f"🔧 Dynamics Config:")
        print(f"  - Obs Dim: {self.obs_dim}")
        print(f"  - Num Actions: {self.num_actions}")
        print(f"  - Est. Num States (Max): {self.num_states}")
        
    def _create_env(self):
        env = simple_speaker_listener_v4.parallel_env(render_mode=None, 
                                                      continuous_actions=False)
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, 
                                    base_class='stable_baselines3')
        return env
    
    def _discretize_observation(self, obs):
        """
        Versione SMART: Usa solo le prime 5-6 dimensioni significative.
        Taglia via il padding inutile per permettere grid_size=3.
        """
        obs = np.array(obs).flatten()
        
        # --- MODIFICA CHIAVE ---
        # Invece di usare tutto obs (11 dim), usiamo solo le prime 5.
        # Speaker: Goal (3 dim) + Silenzio (2 dim) = 5
        # Listener: Posizione relativa + Comm = ~5 significative
        USE_DIMS = 5 
        
        obs_relevant = obs[:USE_DIMS]
        # -----------------------

        obs_clipped = np.clip(obs_relevant, -3, 3)
        bins = np.linspace(-3, 3, self.grid_size + 1)
        
        discrete = np.digitize(obs_clipped, bins) - 1
        discrete = np.clip(discrete, 0, self.grid_size - 1)
        
        state_idx = 0
        for i, d in enumerate(discrete):
            state_idx += d * (self.grid_size ** i)
        
        return int(state_idx)
    
    def estimate_dynamics(self):
        # Nota: Usiamo dizionari sparsi per evitare memory overflow se num_states è enorme
        # Convertiremo in matrice densa solo alla fine se necessario
        
        # Per semplicità, qui manteniamo l'approccio denso ma fai attenzione alla RAM
        # Se crasha per memoria, riduci grid_size
        
        # Mapping inverso (Index -> State) per iterare solo sugli stati noti
        visited_states = list(self.state_mapping.keys())
        n_visited = len(visited_states)
        
        print(f"🔬 Estimating dynamics for {n_visited} visited states...")
        
        # Inizializziamo solo le matrici per gli stati visitati per risparmiare memoria?
        # Per ora manteniamo la struttura completa ma occhio a `self.num_states` che può essere 5^11!
        # Se 5^11 è troppo grande, dobbiamo usare un mapping compatto.
        
        # CHECK SAFETY:
        if self.num_states > 100000:
             print("⚠️ WARNING: State space is huge. Switching to Sparse Estimation not implemented here.")
             print("   Reducing grid_size implicitly or expecting MemoryError.")
        
        # Pre-allocate (potrebbe fallire se obs_dim è alto)
        # Usiamo dizionari per sicurezza
        transition_counts = defaultdict(int) # Key: (s, a1, a2, next_s)
        reward_sums = defaultdict(float)     # Key: (s, a1, a2)
        visit_counts = defaultdict(int)      # Key: (s, a1, a2)
        
        total_steps = n_visited * self.num_actions * self.num_actions * self.n_samples_per_action
        
        # Usiamo tqdm per progress bar
        pbar = tqdm(total=total_steps, desc="Sampling")
        
        for state_idx in visited_states:
            # Qui c'è il problema concettuale: Non possiamo settare lo stato!
            # obs_prototype = self.state_mapping[state_idx]
            
            for a_spk in range(self.num_actions):
                for a_lst in range(self.num_actions):
                    
                    for _ in range(self.n_samples_per_action):
                        # ⚠️ LIMITAZIONE: Reset casuale invece di set_state
                        # Questo resetta l'ambiente a uno stato random, NON a 'state_idx'
                        obs = self.env.reset()
                        
                        # Cerchiamo di capire in che stato siamo finiti davvero
                        # (Questo è l'unico modo onesto di farlo senza set_state)
                        real_start_obs_spk = obs[0]
                        real_start_obs_lst = obs[1] # Listener o Speaker? obs[0] è spk in simple_speaker
                        
                        # Discretizziamo lo stato di partenza REALE
                        real_start_state = self._discretize_observation(real_start_obs_spk)
                        
                        # Eseguiamo l'azione
                        actions = np.array([a_spk, a_lst])
                        obs_next, rewards, dones, infos = self.env.step(actions)
                        
                        # Discretizziamo stato finale
                        next_state = self._discretize_observation(obs_next[0])
                        
                        # Salviamo la transizione: Real_Start -> Action -> Next
                        # Nota: rewards è un array, prendiamo la somma o il primo
                        r = rewards[0] if isinstance(rewards, np.ndarray) else rewards
                        
                        transition_counts[(real_start_state, a_spk, a_lst, next_state)] += 1
                        reward_sums[(real_start_state, a_spk, a_lst)] += r
                        visit_counts[(real_start_state, a_spk, a_lst)] += 1
                        
                        pbar.update(1)
        
        pbar.close()
        
        print("✅ Estimation complete. Saving sparse data...")
        # Restituiamo i dizionari raw perché la matrice densa sarebbe troppo grande
        return transition_counts, reward_sums, visit_counts

def main():
    RUN_NAME = "ppo_jamming_normalized"
    STATS_PATH = f"models/{RUN_NAME}/vec_normalize.pkl"
    
    # Riduciamo i sample per il test
    estimator = DynamicsEstimator(
        stats_path=STATS_PATH,
        grid_size=3,  # Ridotto per evitare esplosione stati (3^5 = 243 è gestibile)
        n_samples_per_action=20 
    )
    
    trans_counts, rew_sums, visits = estimator.estimate_dynamics()
    
    # Salvataggio in formato Pickle (più sicuro per dati sparsi/dizionari)
    with open('dynamics_sparse.pkl', 'wb') as f:
        pickle.dump({
            'transitions': dict(trans_counts),
            'rewards': dict(rew_sums),
            'visits': dict(visits)
        }, f)
    
    print("\n💾 Saved dynamics to 'dynamics_sparse.pkl'")

if __name__ == "__main__":
    main()
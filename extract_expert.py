"""
Extract tabular expert policies from trained PPO model for MURMAIL.
FIXED VERSION: Correct handling of VecEnv dimensions and indexing.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from pettingzoo.mpe import simple_speaker_listener_v4
import supersuit as ss
from collections import defaultdict
import pickle
import os

class ExpertPolicyExtractor:
    def __init__(self, model_path, stats_path, grid_size=5, n_samples=1000):
        self.grid_size = grid_size
        self.n_samples = n_samples
        
        # 1. Carica ambiente
        self.env = self._create_env()
        
        # 2. Carica statistiche di normalizzazione
        # È CRUCIALE che questo path sia corretto
        if os.path.exists(stats_path):
            self.env = VecNormalize.load(stats_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        else:
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        # 3. Carica Modello
        self.model = PPO.load(model_path, env=self.env)
        
        # --- CORREZIONE DIMENSIONI ---
        # In un VecEnv creato da PettingZoo, action_space e observation_space 
        # si riferiscono già al SINGOLO agente. NON bisogna dividere per num_agents.
        
        # Action Space
        if hasattr(self.env.action_space, 'n'):
            self.num_actions = self.env.action_space.n
        else:
            # Fallback per spazi continui o complessi (non dovrebbe accadere qui)
            self.num_actions = self.env.action_space.shape[0]

        # Observation Space
        self.obs_dim = self.env.observation_space.shape[0]
        
        print(f"🔍 Dimensioni Rilevate (Corrected):")
        print(f"  - Obs Dim (per agente): {self.obs_dim}")
        print(f"  - Num Actions (per agente): {self.num_actions}")
        
    def _create_env(self):
        """Ricrea l'ambiente con lo stesso padding del training."""
        env = simple_speaker_listener_v4.parallel_env(render_mode=None, 
                                                      continuous_actions=False)
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        # num_cpus=0 per evitare problemi di multiprocessing
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
        
    def extract_policies(self):
        # Inizializza contatori con la dimensione corretta (self.num_actions)
        state_action_counts_speaker = defaultdict(lambda: np.zeros(self.num_actions))
        state_action_counts_listener = defaultdict(lambda: np.zeros(self.num_actions))
        state_obs_map = {} 
        
        print(f"🎯 Sampling {self.n_samples} trajectories...")
        
        # Reset iniziale
        obs = self.env.reset() 
        # obs shape: (2, obs_dim) -> Riga 0: Speaker, Riga 1: Listener
        
        for episode in range(self.n_samples):
            done = False
            steps = 0
            
            while not done and steps < 25:
                # --- ESTRAZIONE OSSERVAZIONI ---
                # Non serve slicing. L'indice 0 è l'agente 0.
                obs_speaker = obs[0]
                obs_listener = obs[1]
                
                # Discretizzazione
                s_speaker = self._discretize_observation(obs_speaker)
                s_listener = self._discretize_observation(obs_listener)
                
                # Mappatura inversa (per debug/visualizzazione futura)
                if s_speaker not in state_obs_map: state_obs_map[s_speaker] = obs_speaker
                if s_listener not in state_obs_map: state_obs_map[s_listener] = obs_listener
                
                # --- PREDIZIONE ---
                # Il modello restituisce azioni per entrambi gli agenti
                actions, _ = self.model.predict(obs, deterministic=False)
                
                a_speaker = int(actions[0])
                a_listener = int(actions[1])
                
                # --- REGISTRAZIONE ---
                # Controllo di sicurezza
                if a_speaker < self.num_actions:
                    state_action_counts_speaker[s_speaker][a_speaker] += 1
                
                if a_listener < self.num_actions:
                    state_action_counts_listener[s_listener][a_listener] += 1
                
                # --- STEP ---
                obs, rewards, dones, infos = self.env.step(actions)
                
                # Gestione fine episodio
                if isinstance(dones, np.ndarray):
                    done = dones.any()
                else:
                    done = dones
                
                steps += 1
                
            if (episode + 1) % 100 == 0:
                print(f"  Episode {episode + 1}/{self.n_samples}")
        
        # --- NORMALIZZAZIONE (Counts -> Probabilities) ---
        print("📊 Normalizing distributions...")
        
        # Stimiamo la dimensione massima dello spazio degli stati visitati
        # Usiamo un dizionario sparso o una matrice se gli stati sono pochi
        # Qui restituiamo dizionari per flessibilità, o convertiamo in array se necessario per MURMAIL
        
        # Per compatibilità col formato precedente, creiamo array densi ma solo per gli stati visitati?
        # MURMAIL di solito si aspetta matrici (S, A).
        # Calcoliamo max state index visitato per dimensionare la matrice
        max_state_idx = max(
            max(state_action_counts_speaker.keys(), default=0),
            max(state_action_counts_listener.keys(), default=0)
        )
        
        # +1 perché gli indici partono da 0
        policy_speaker = np.zeros((max_state_idx + 1, self.num_actions))
        policy_listener = np.zeros((max_state_idx + 1, self.num_actions))
        
        # Riempimento Speaker
        for s, counts in state_action_counts_speaker.items():
            total = counts.sum()
            if total > 0:
                policy_speaker[s] = counts / total
                
        # Riempimento Listener
        for s, counts in state_action_counts_listener.items():
            total = counts.sum()
            if total > 0:
                policy_listener[s] = counts / total
                
        return policy_speaker, policy_listener, state_obs_map

def main():
    RUN_NAME = "ppo_jamming_normalized"
    MODEL_PATH = f"models/{RUN_NAME}/final_model.zip"
    STATS_PATH = f"models/{RUN_NAME}/vec_normalize.pkl"
    
    # Parametri ridotti per test rapido, aumentali se vuoi più precisione
    GRID_SIZE = 3 
    N_SAMPLES = 5000
    
    print("🚀 Starting Extraction...")
    
    try:
        extractor = ExpertPolicyExtractor(MODEL_PATH, STATS_PATH, GRID_SIZE, N_SAMPLES)
        pi_s, pi_l, mapping = extractor.extract_policies()
        
        print(f"\n✅ Extraction Complete!")
        print(f"  Speaker Policy Shape: {pi_s.shape}")
        print(f"  Listener Policy Shape: {pi_l.shape}")
        
        np.save('expert_policy_speaker.npy', pi_s)
        np.save('expert_policy_listener.npy', pi_l)
        with open('state_mapping.pkl', 'wb') as f:
            pickle.dump(mapping, f)
            
        print("💾 Files saved successfully.")
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
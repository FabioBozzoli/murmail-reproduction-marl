"""
discrete_wrapper_IMPROVED.py - VERSIONE CON BINS CONFIGURABILI

Supporta bins variabili per creare ambienti più o meno difficili.
"""

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class DiscretizedSpeakerListenerWrapper(BaseParallelWrapper):
    def __init__(self, env, bins=5):
        super().__init__(env)
        self.bins = bins
        self.grid_size = bins  # Per compatibilità
        
        self.low = -1.5
        self.high = 1.5
        
        # Landmarks per l'esperto (angoli della griglia)
        max_idx = bins - 1
        self.landmarks = {
            0: (0, 0),
            1: (max_idx, 0),
            2: (max_idx, max_idx)
        }
        
        print(f"🎮 Wrapper inizializzato: bins={bins}, stati_listener≈{bins**2 * 3 * 3}")

    def _discretize_single_obs(self, obs):
        """Converte osservazione continua in intero."""
        # Speaker: Goal vector (dim 3)
        if obs.shape[0] == 3: 
            return int(np.argmax(obs))
        
        # Listener: [vel(2), rel_pos(2), comm(3), ...]
        else:
            vel = obs[0:2] 
            rel_pos = obs[2:4]
            comm = obs[4:7]
            
            # Discretizza Posizione (bins configurabile)
            pos_bins = np.linspace(self.low, self.high, self.bins + 1)
            p_idx = np.digitize(rel_pos, pos_bins) - 1
            p_idx = np.clip(p_idx, 0, self.bins - 1)
            
            # Discretizza Velocità (sempre 3 bins per semplicità)
            vel_bins = np.linspace(-1.0, 1.0, 4)
            v_idx = np.digitize(vel, vel_bins) - 1
            v_idx = np.clip(v_idx, 0, 2)
            
            # Messaggio (sempre 3)
            comm_idx = np.argmax(comm) if np.sum(comm) > 0 else 0
            
            # Hash: posizione + velocità + comunicazione
            state_idx = (p_idx[0]) + \
                        (p_idx[1] * self.bins) + \
                        (v_idx[0] * (self.bins**2)) + \
                        (v_idx[1] * (self.bins**2) * 3) + \
                        (comm_idx * (self.bins**2) * 3 * 3)
            return int(state_idx)

    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        new_obs = {agent: self._discretize_single_obs(obs) for agent, obs in obs_dict.items()}
        return new_obs, infos

    def step(self, actions):
        obs_dict, rews, terms, truncs, infos = self.env.step(actions)
        new_obs = {agent: self._discretize_single_obs(obs) for agent, obs in obs_dict.items()}
        return new_obs, rews, terms, truncs, infos
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)
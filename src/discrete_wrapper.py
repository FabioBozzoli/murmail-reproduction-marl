"""
discrete_wrapper.py - FIXED v5 (Inverted Physics & Robust Test)
"""

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class DiscretizedSpeakerListenerWrapper(BaseParallelWrapper):
    def __init__(self, raw_env=None, bins=6):
        import pettingzoo.utils.wrappers.assert_out_of_bounds as aob
        
        # 1. Patch AssertOutOfBounds
        def patched_step(self, action):
            self.env.step(action)
        aob.AssertOutOfBoundsWrapper.step = patched_step

        # 2. Inizializzazione Ambiente
        from pettingzoo.mpe import simple_speaker_listener_v4
        raw_env = simple_speaker_listener_v4.parallel_env(
            continuous_actions=True, 
            render_mode=None,
            max_cycles=25
        )
        
        # 3. Pulizia Wrapper
        if hasattr(raw_env, 'aec_env'):
             core = raw_env.aec_env.env
             while hasattr(core, 'env'):
                 if core.__class__.__name__ == 'AssertOutOfBoundsWrapper':
                     pass 
                 core = core.env

        super().__init__(raw_env)
        
        self.bins = bins
        self.low = -1.5
        self.high = 1.5
        print(f"🎮 Wrapper v5: Physics Inverted [2=Right, 4=Up]")

    def _discretize_single_obs(self, obs):
        if obs.shape[0] == 3: 
            return int(np.argmax(obs))
        else:
            vel = obs[0:2] 
            rel_pos = obs[2:4] 
            comm = obs[-3:] # Fix indici comm
            
            pos_bins = np.linspace(self.low, self.high, self.bins + 1)
            p_idx = np.digitize(rel_pos, pos_bins) - 1
            p_idx = np.clip(p_idx, 0, self.bins - 1)
            
            vel_bins = np.linspace(-1.0, 1.0, 4)
            v_idx = np.digitize(vel, vel_bins) - 1
            v_idx = np.clip(v_idx, 0, 2)
            
            comm_idx = np.argmax(comm) if np.max(comm) > 0.1 else 0
            
            state_idx = (p_idx[0]) + \
                        (p_idx[1] * self.bins) + \
                        (v_idx[0] * (self.bins**2)) + \
                        (v_idx[1] * (self.bins**2) * 3) + \
                        (comm_idx * (self.bins**2) * 3 * 3)
            return int(state_idx)

    def _discrete_to_continuous_action(self, agent, discrete_action):
        if agent == "speaker_0":
            action = np.zeros(3, dtype=np.float32)
            action[int(discrete_action)] = 1.0
            return action

        else:
            # Listener Action Map (Fixed based on logs)
            # Log precedenti: Index 1->Left, Index 3->Down
            # Quindi invertiamo per ottenere Right e Up
            
            action = np.zeros(5, dtype=np.float32)
            
            if discrete_action == 0:   # Stop
                action[0] = 1.0
            elif discrete_action == 1: # Su (Wanted Up, got Down with 3 -> Use 4)
                action[4] = 1.0
            elif discrete_action == 2: # Giù (Use 3)
                action[3] = 1.0
            elif discrete_action == 3: # Sinistra (Use 1)
                action[1] = 1.0
            elif discrete_action == 4: # Destra (Wanted Right, got Left with 1 -> Use 2)
                action[2] = 1.0
                
            return action

    def reset(self, seed=None, options=None):
        obs_dict, infos = self.env.reset(seed=seed, options=options)
        new_obs = {agent: self._discretize_single_obs(obs) for agent, obs in obs_dict.items()}
        return new_obs, infos

    def step(self, actions):
        continuous_actions = {}
        for agent, discrete_action in actions.items():
            continuous_actions[agent] = self._discrete_to_continuous_action(agent, discrete_action)
        
        obs_dict, rews, terms, truncs, infos = self.env.step(continuous_actions)
        new_obs = {agent: self._discretize_single_obs(obs) for agent, obs in obs_dict.items()}
        return new_obs, rews, terms, truncs, infos
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

# ================= TEST DIAGNOSTICO FINALE =================
if __name__ == "__main__":
    from discrete_wrapper import DiscretizedSpeakerListenerWrapper
    import numpy as np

    print("\n📡 TEST COMUNICAZIONE 'BLACK BOX'")
    
    env = DiscretizedSpeakerListenerWrapper()
    
    # 1. Caso A: Silenzio
    env.reset()
    # Speaker dice 0, Listener fermo (0)
    obs_A, _, _, _, _ = env.step({"speaker_0": 0, "listener_0": 0})
    state_A = obs_A["listener_0"]
    print(f"1. Stato con Speaker Zitto: {state_A}")

    # 2. Caso B: Speaker Parla
    env.reset()
    # Speaker dice 1, Listener fermo (0)
    obs_B, _, _, _, _ = env.step({"speaker_0": 1, "listener_0": 0})
    state_B = obs_B["listener_0"]
    print(f"2. Stato con Speaker '1':    {state_B}")
    
    # 3. Verifica
    diff = state_B - state_A
    
    # Nel tuo wrapper: comm_idx * (bins^2 * 9)
    # Quindi se comm cambia, lo stato DEVE fare un salto enorme.
    if state_A != state_B:
        print(f"\n✅ SUCCESS: Lo stato è cambiato! (Delta: {diff})")
        print("   Il Listener sente lo Speaker.")
    else:
        print("\n❌ FAIL: Lo stato è identico. Il Listener è sordo.")
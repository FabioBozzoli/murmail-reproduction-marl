import gymnasium as gym
from pettingzoo.mpe import simple_speaker_listener_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList 
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

# --- CONFIGURAZIONE ---
# NOTA: Abbiamo rimosso NUM_CPUS perché useremo un singolo processo 
# (più stabile con le versioni attuali delle librerie)
TOTAL_TIMESTEPS = 2000000  
LEARNING_RATE = 3e-4
START_ENTROPY = 0.60       
END_ENTROPY = 0.05       
PROJECT_NAME = "MARL_Fixed"
RUN_NAME = "ppo_jamming_normalized"

class SignalJammingWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rng = np.random.default_rng()

    def step(self, actions):
        # Dizionario per accumulare le penalità
        penalties = {agent: 0.0 for agent in self.env.possible_agents}
        
        if 'speaker_0' in actions:
            act = actions['speaker_0']
            # Se l'azione è illegale (Padding 3 o 4)
            if act >= 3:
                # 1. SOSTITUZIONE: Cambia l'azione in rumore (0, 1, 2)
                # Questo rompe la comunicazione
                actions['speaker_0'] = self.rng.integers(0, 3)
                
                # 2. PUNIZIONE: Assegna una penalità negativa forte
                # Questo insegna all'agente a NON PROVARCI NEMMENO
                penalties['speaker_0'] = -5.0 
                
                # (Opzionale) Puniamo anche il listener per scoraggiare la coppia? 
                # Per ora puniamo solo chi commette l'errore (lo speaker).

        # Esegui lo step nell'ambiente reale
        obs, rews, terms, truncs, infos = self.env.step(actions)
        
        # Applica la penalità al reward restituito dall'ambiente
        for agent, penalty in penalties.items():
            if agent in rews:
                rews[agent] += penalty
                
        return obs, rews, terms, truncs, infos

class EntropyCallback(BaseCallback):
    def __init__(self, start_ent: float, end_ent: float, total_timesteps: int, verbose=0):
        super(EntropyCallback, self).__init__(verbose)
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        current_ent = self.start_ent - (progress * (self.start_ent - self.end_ent))
        current_ent = max(self.end_ent, current_ent)
        self.model.ent_coef = current_ent
        return True

def make_env():
    env = simple_speaker_listener_v4.parallel_env(render_mode=None, 
                                                  continuous_actions=False)
    
    # PRIMA crei i canali 3 e 4
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    
    # POI applichi il wrapper che li controlla
    env = SignalJammingWrapper(env)  # <--- Qui entra in gioco la penalità
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 5. Concatenazione "Dummy" (Cruciale per compatibilità versioni!)
    # num_cpus=0 forza l'uso del processo principale (niente errori Async)
    # num_vec_envs=1 mantiene un singolo ambiente ma "pulisce" l'interfaccia per SB3
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs=1, 
        num_cpus=0, 
        base_class='stable_baselines3'
    )
    
    # 6. Monitor e Normalizzazione
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    return env

if __name__ == "__main__":
    # Assicuriamoci che le cartelle esistano
    os.makedirs(f"models/{RUN_NAME}", exist_ok=True)
    os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)
    
    wandb.init(
        entity="307972-unimore", 
        project=PROJECT_NAME,
        name=RUN_NAME,
        sync_tensorboard=True, 
        monitor_gym=True,
        config={
            "method": "Signal Jamming + VecNormalize",
            "learning_rate": LEARNING_RATE
        }
    )

    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=2048,         # Aumentato n_steps standard per compensare la mancanza di parallelelismo
        batch_size=256,       
        ent_coef=START_ENTROPY, 
        tensorboard_log=f"runs/{RUN_NAME}"
    )

    entropy_cb = EntropyCallback(START_ENTROPY, END_ENTROPY, TOTAL_TIMESTEPS)
    wandb_cb = WandbCallback(
        gradient_save_freq=10000, 
        model_save_path=f"models/{RUN_NAME}", 
        verbose=2
    )

    print(f"🚀 Inizio addestramento '{RUN_NAME}' (Single Process + Normalized)...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([entropy_cb, wandb_cb])
        )
    except KeyboardInterrupt:
        print("Interrotto dall'utente, salvataggio in corso...")

    # Salvataggio finale
    model.save(f"models/{RUN_NAME}/final_model")
    env.save(f"models/{RUN_NAME}/vec_normalize.pkl") # Salva le statistiche di normalizzazione!
    
    print("✅ Modello salvato.")
    env.close()
    wandb.finish()
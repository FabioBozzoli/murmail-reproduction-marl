from pettingzoo.mpe import simple_speaker_listener_v4
import supersuit as ss
from stable_baselines3 import PPO
import time

# Carica il modello addestrato
model = PPO.load("expert_ppo_speaker_listener")

# Crea l'ambiente per il rendering (1 sola istanza, render_mode="human")
env = simple_speaker_listener_v4.parallel_env(render_mode="human", continuous_actions=True)

# Applica ESATTAMENTE gli stessi wrapper del training (tranne la vettorizzazione parallela)
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)

# Loop di visualizzazione
print("🎥 Avvio visualizzazione... Premi Ctrl+C nel terminale per fermare.")

try:
    obs = env.reset()
    for _ in range(1000):
        # Chiedi al modello l'azione (deterministic=True per vedere la best performance)
        action, _states = model.predict(obs, deterministic=True)
        
        # Esegui lo step
        obs, rewards, dones, infos = env.step(action)
        
        # Rallenta un po' per farci capire cosa succede
        time.sleep(0.05)
        
        # Renderizza
        env.render()
        
except KeyboardInterrupt:
    print("Visualizzazione interrotta.")
finally:
    env.close()
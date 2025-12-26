import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pettingzoo.mpe import simple_speaker_listener_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
import os

# --- CONFIGURAZIONE ---
RUN_NAME = "ppo_jamming_normalized" # Assicurati che corrisponda alla cartella creata prima
MODEL_PATH = f"models/{RUN_NAME}/final_model.zip"
STATS_PATH = f"models/{RUN_NAME}/vec_normalize.pkl"
N_SAMPLES = 1000

def make_eval_env():
    """
    Ricrea l'ambiente con la stessa struttura del training.
    """
    env = simple_speaker_listener_v4.parallel_env(render_mode=None, continuous_actions=False)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Importante: num_cpus=0 per evitare errori su Mac/Linux recenti
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')
    return env

def evaluate_equilibrium(model, env, n_episodes=100):
    """
    Testa se deviazioni unilaterali riducono la reward
    (condizione necessaria per equilibrio di Nash)
    """
    baseline_rewards = []
    deviation_rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        
        # Baseline: entrambi gli agenti seguono la policy
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward.mean()
        
        baseline_rewards.append(ep_reward)
        
        # Deviazione: speaker usa azione random
        obs = env.reset()
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Devia solo lo speaker
            action[0] = env.action_space.sample()[0]
            obs, reward, done, _ = env.step(action)
            ep_reward += reward.mean()
        
        deviation_rewards.append(ep_reward)
    
    print(f"Baseline reward: {np.mean(baseline_rewards):.2f}")
    print(f"Deviation reward: {np.mean(deviation_rewards):.2f}")
    print(f"È equilibrio? {np.mean(baseline_rewards) > np.mean(deviation_rewards)}")

def main():
    print(f"📂 Cerco modello in: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STATS_PATH):
        print("❌ Errore: File del modello o statistiche di normalizzazione non trovati!")
        return

    # 1. Crea l'ambiente base
    env = make_eval_env()

    # 2. Carica le statistiche di Normalizzazione (CRUCIALE)
    # Questo applica la media/varianza imparata durante il training
    env = VecNormalize.load(STATS_PATH, env)
    
    # 3. Imposta modalità valutazione
    env.training = False     # Non aggiornare più le medie
    env.norm_reward = False  # Non ci interessa normalizzare i reward ora

    # 4. Carica il modello PPO
    model = PPO.load(MODEL_PATH, env=env)
    
    print(f"📊 Inizio analisi su {N_SAMPLES} campioni...")

    # Matrice: Righe=Target Reale (3), Colonne=Messaggio Inviato (5 incl. padding)
    confusion_matrix = np.zeros((3, 5))
    target_counts = {0: 0, 1: 0, 2: 0}

    # Reset iniziale
    obs = env.reset()

    for _ in range(N_SAMPLES):
        # In un ambiente vettorizzato PettingZoo convertito:
        # obs è un array con shape (num_agents, obs_dim) -> (2, 14 circa)
        # Indice 0 = Speaker, Indice 1 = Listener
        
        speaker_obs = obs[0] # Osservazione dello speaker
        
        # --- A. IDENTIFICA IL TARGET REALE ---
        # L'osservazione è normalizzata, quindi i valori non sono esattamente 0 o 1.
        # Ma il valore più alto tra i primi 3 corrisponde comunque al target.
        goal_vector = speaker_obs[:3]
        target_idx = np.argmax(goal_vector)
        target_counts[target_idx] += 1

        # --- B. CHIEDI AL MODELLO ---
        # model.predict restituisce le azioni per TUTTI gli agenti nell'env vettorizzato
        actions, _ = model.predict(obs, deterministic=True)
        
        # actions[0] è l'azione dello Speaker
        speaker_action = actions[0]
        
        # Gestione formato (numpy array 0-d o scalare)
        if isinstance(speaker_action, np.ndarray):
            message_idx = speaker_action.item()
        else:
            message_idx = int(speaker_action)

        # --- C. AGGIORNA MATRICE ---
        if 0 <= message_idx < 5:
            confusion_matrix[target_idx, int(message_idx)] += 1
            
        # Step dell'ambiente (per avere nuova osservazione)
        obs, _, _, _ = env.step(actions)

    env.close()
    
    print(f"\n✅ Analisi completata. Target visti: {target_counts}")
    
    # --- PLOTTING ---
    plot_confusion_matrix(confusion_matrix)

def plot_confusion_matrix(cm):
    print("🎨 Generazione grafico...")
    
    # Normalizza per righe (somma a 1 per ogni target)
    row_sums = cm.sum(axis=1)
    # Evita divisione per zero
    row_sums[row_sums == 0] = 1 
    normalized_matrix = cm / row_sums[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    
    # Etichette
    y_labels = ["Target Red (0)", "Target Green (1)", "Target Blue (2)"]
    x_labels = ["Msg 0", "Msg 1", "Msg 2", "Pad (3)", "Pad (4)"]
    
    sns.heatmap(
        normalized_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues",
        xticklabels=x_labels,
        yticklabels=y_labels,
        vmin=0, vmax=1
    )
    
    plt.title("Emergent Language Protocol\n(Normalized Env Evaluation)")
    plt.xlabel("Message Sent by Speaker")
    plt.ylabel("Real Goal Assigned")
    plt.tight_layout()
    
    filename = "language_matrix_normalized.png"
    plt.savefig(filename)
    print(f"📸 Grafico salvato come '{filename}'")
    plt.show()

if __name__ == "__main__":
    main()
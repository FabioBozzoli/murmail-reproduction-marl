import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
import os
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true 

# ================= CONFIGURAZIONE =================
BINS = 6
EPISODES = 500       # Numero di episodi per la media empirica
DEVICE = torch.device("cpu") # O "cuda"
GAMMA = 0.9

# Nomi file
FILES = {
    # DINAMICHE (Servono solo per il Gap)
    "P": f"P_bins{BINS}.npy",
    "R_norm": f"R_bins{BINS}.npy",       
    "Init": f"expert_initial_dist_bins{BINS}.npy",
    
    # POLICIES
    "Expert_S": f"expert_policy_speaker_bins{BINS}.npy",
    "Expert_L": f"expert_policy_listener_bins{BINS}.npy",
    "MURMAIL_S": "murmail_policy_speaker_final.npy",
    "MURMAIL_L": "murmail_policy_listener_final.npy",
    
    # CHECKPOINT MAPPO
    "MAPPO_Ckpt": "mappo_final_converged_checkpoint.pth" 
    # Se hai salvato i file numpy di MAPPO, puoi caricare anche quelli, 
    # ma il checkpoint è meglio per la valutazione empirica.
}

# ================= CLASSI DEEP RL =================
class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(obs_dim, hidden)
        self.policy = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
    def forward(self, obs):
        x = self.embed(obs)
        return Categorical(logits=self.policy(x))

def get_uniform_policy(n_states, n_actions):
    return np.ones((n_states, n_actions)) / n_actions

# ================= CARICAMENTO DATI =================
def load_data():
    print("📂 Caricamento Dati...")
    data = {}
    try:
        # Carichiamo solo ciò che serve per il Gap
        data["P"] = np.load(FILES["P"])
        data["R_norm"] = np.load(FILES["R_norm"]) 
        data["Init"] = np.load(FILES["Init"])
        
        # Policy Tabellari
        data["Pi_Exp_S"] = np.load(FILES["Expert_S"])
        data["Pi_Exp_L"] = np.load(FILES["Expert_L"])
        
        if os.path.exists(FILES["MURMAIL_S"]):
            data["Pi_Mur_S"] = np.load(FILES["MURMAIL_S"])
            data["Pi_Mur_L"] = np.load(FILES["MURMAIL_L"])
        else:
            print("⚠️ MuRMAIL non trovato, verrà saltato.")

        print("✅ Dati caricati.")
    except FileNotFoundError as e:
        print(f"❌ ERRORE: {e}")
        exit()
    return data

# ================= 1. CALCOLO NASH GAP (TEORICO) =================
def compute_gaps(data):
    print("\n🧮 Calcolo Nash Gaps (su Reward Normalizzato)...")
    results = {}
    
    # Lista algoritmi tabellari disponibili
    targets = [("Expert", data["Pi_Exp_S"], data["Pi_Exp_L"])]
    if "Pi_Mur_S" in data:
        targets.append(("MuRMAIL", data["Pi_Mur_S"], data["Pi_Mur_L"]))
    
    for name, pi_s, pi_l in targets:
        gap = calc_exploitability_true(pi_s, pi_l, data["R_norm"], data["P"], data["Init"], GAMMA)
        results[name] = gap
        print(f"   🔹 {name} Gap: {gap:.6f}")
    
    # Random Uniform Baseline
    s_dim = data["P"].shape[0]
    u_s = get_uniform_policy(s_dim, 3)
    u_l = get_uniform_policy(s_dim, 5)
    gap = calc_exploitability_true(u_s, u_l, data["R_norm"], data["P"], data["Init"], GAMMA)
    results["Random"] = gap
    print(f"   🔹 Random Gap: {gap:.6f}")
    
    # Nota: Il Gap di MAPPO è difficile da calcolare esattamente se non hai estratto 
    # la policy tabellare (probability matrix) dalla rete neurale. 
    # Se hai i file .npy di MAPPO, aggiungili qui. Altrimenti lo stimiamo o lo omettiamo nel plot teorico.
    results["MAPPO"] = np.nan # Placeholder se non abbiamo la matrice NxAxA
    
    return results

# ================= 2. CALCOLO REWARD (EMPIRICO UNIFICATO) =================
def eval_all_agents_on_env(data):
    print(f"\n🎮 Valutazione Empirica ({EPISODES} episodi) - Tutti sullo stesso Environment...")
    env = DiscretizedSpeakerListenerWrapper(bins=BINS)
    results = {}
    
    # --- A. AGENTI TABELLARI (Expert, MuRMAIL) ---
    tabular_agents = [("Expert", data["Pi_Exp_S"], data["Pi_Exp_L"])]
    if "Pi_Mur_S" in data:
        tabular_agents.append(("MuRMAIL", data["Pi_Mur_S"], data["Pi_Mur_L"]))
        
    for name, pi_s, pi_l in tabular_agents:
        r_log = []
        for _ in range(EPISODES):
            obs_dict, _ = env.reset()
            obs_s, obs_l = obs_dict["speaker_0"], obs_dict["listener_0"]
            ep_r = 0
            for _ in range(25):
                # Campioniamo dall'array di probabilità
                act_s = np.random.choice(len(pi_s[obs_s]), p=pi_s[obs_s])
                act_l = np.random.choice(len(pi_l[obs_l]), p=pi_l[obs_l])
                
                obs_dict, rews, terms, truncs, _ = env.step({"speaker_0": act_s, "listener_0": act_l})
                ep_r += rews["listener_0"] # RAW REWARD dall'env
                
                obs_s, obs_l = obs_dict["speaker_0"], obs_dict["listener_0"]
                if any(terms.values()) or any(truncs.values()): break
            r_log.append(ep_r)
        results[name] = np.mean(r_log)
        print(f"   🔹 {name}: {results[name]:.2f}")

    # --- B. AGENTE RANDOM ---
    r_log = []
    for _ in range(EPISODES):
        env.reset(); ep_r = 0
        for _ in range(25):
            _, r, t, _, _ = env.step({"speaker_0": np.random.randint(0,3), "listener_0": np.random.randint(0,5)})
            ep_r += r["listener_0"]
            if any(t.values()): break
        r_log.append(ep_r)
    results["Random"] = np.mean(r_log)
    print(f"   🔹 Random: {results['Random']:.2f}")

    # --- C. AGENTE MAPPO (Deep RL) ---
    # Qui usiamo os.path (Libreria OS)
    if os.path.exists(FILES["MAPPO_Ckpt"]):
        OBS_L_DIM = BINS**2 * 27 
        agent_s = DiscreteActor(3, 3).to(DEVICE)
        agent_l = DiscreteActor(OBS_L_DIM, 5).to(DEVICE)
        
        try:
            ckpt = torch.load(FILES["MAPPO_Ckpt"], map_location=DEVICE)
            agent_s.load_state_dict(ckpt['actor_s'])
            agent_l.load_state_dict(ckpt['actor_l'])
            
            r_log = []
            for _ in range(EPISODES):
                obs, _ = env.reset(); ep_r = 0
                for _ in range(25):
                    with torch.no_grad():
                        # ✅ FIX: Rinominate variabili per evitare conflitto con 'import os'
                        t_os = torch.tensor([obs["speaker_0"]], device=DEVICE)
                        t_ol = torch.tensor([obs["listener_0"]], device=DEVICE)
                        
                        as_ = agent_s(t_os).logits.argmax().item()
                        al_ = agent_l(t_ol).logits.argmax().item()
                        
                    obs, r, t, _, _ = env.step({"speaker_0": as_, "listener_0": al_})
                    ep_r += r["listener_0"]
                    if any(t.values()): break
                r_log.append(ep_r)
            results["MAPPO"] = np.mean(r_log)
            print(f"   🔹 MAPPO:  {results['MAPPO']:.2f}")
        except Exception as e:
            print(f"   ⚠️ MAPPO Load Error: {e}")
            results["MAPPO"] = np.nan
    else:
        results["MAPPO"] = np.nan

    return results
# ================= REPORT FINALE =================
def make_final_report(gaps, rewards):
    print("\n" + "="*60)
    print("📊 REPORT FINALE UNIFICATO")
    print("="*60)
    
    # Creiamo DataFrame
    df_data = {}
    for name in rewards.keys():
        df_data[name] = {
            "Gap (Lower Better)": gaps.get(name, np.nan),
            "Reward (Higher Better)": rewards.get(name, np.nan)
        }
    
    df = pd.DataFrame.from_dict(df_data, orient='index')
    print(df)
    df.to_csv("thesis_results.csv")
    
    # Plot
    plt.figure(figsize=(10, 7))
    colors = {"Expert": "green", "MAPPO": "orange", "Random": "grey", "MuRMAIL": "purple"}
    
    for name, row in df.iterrows():
        g, r = row["Gap (Lower Better)"], row["Reward (Higher Better)"]
        if np.isnan(g) or np.isnan(r): continue
        
        plt.scatter(g, r, s=200, c=colors.get(name, "blue"), label=name, edgecolors="black")
        plt.annotate(f"{name}\n{r:.1f}", (g, r), xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Nash Exploitability Gap")
    plt.ylabel("Average Episode Reward (Raw)")
    plt.title(f"Performance vs Robustness (Bins={BINS})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("thesis_plot_final.png")
    print("\n✅ Plot salvato come thesis_plot_final.png")

if __name__ == "__main__":
    d = load_data()
    gaps = compute_gaps(d)
    rewards = eval_all_agents_on_env(d) # Unica funzione per tutti!
    make_final_report(gaps, rewards)
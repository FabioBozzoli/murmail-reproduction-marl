import torch
import torch.nn as nn
import numpy as np
import os
from torch.distributions import Categorical

# ================= CONFIGURAZIONE =================
BINS = 6
HIDDEN_SIZE = 128
FILENAME = "mappo_final_converged_checkpoint.pth" 
BACKUP_FILENAME = "mappo_backup_fixed_checkpoint.pth"
DEVICE = torch.device("cpu")
# ==================================================

class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.embed = nn.Embedding(obs_dim, hidden)
        self.policy = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
    def forward(self, obs):
        x = self.embed(obs)
        return Categorical(logits=self.policy(x))

def smooth_policy(probs, epsilon=0.05):
    """
    Mescola la policy (95%) con un rumore uniforme (5%)
    per renderla robusta a piccole discrepanze nel modello.
    """
    n_actions = probs.shape[1]
    uniform = np.ones_like(probs) / n_actions
    return (1 - epsilon) * probs + epsilon * uniform

def fix_and_export():
    print("🔧 FIX EXPORT TOOL per MAPPO (con Smoothing)")
    
    # 1. Trova il file giusto
    if os.path.exists(FILENAME):
        checkpoint_path = FILENAME
    elif os.path.exists(BACKUP_FILENAME):
        print(f"⚠️ '{FILENAME}' non trovato. Uso backup '{BACKUP_FILENAME}'")
        checkpoint_path = BACKUP_FILENAME
    else:
        print(f"❌ ERRORE: Nessun checkpoint trovato. Controlla la cartella.")
        return

    n_states_local = (BINS**2) * 27
    n_states_global = 3 * n_states_local
    
    print(f"📂 Caricamento: {checkpoint_path}")
    
    # 2. Inizializza e Carica
    actor_s = DiscreteActor(3, 3, HIDDEN_SIZE).to(DEVICE)
    actor_l = DiscreteActor(n_states_local, 5, HIDDEN_SIZE).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    actor_s.load_state_dict(checkpoint['actor_s'])
    actor_l.load_state_dict(checkpoint['actor_l'])
    actor_s.eval(); actor_l.eval()

    # 3. SPEAKER EXPANSION
    print("🧠 Processing Speaker...")
    obs_s = torch.tensor([0, 1, 2], device=DEVICE)
    with torch.no_grad(): probs_s_small = actor_s(obs_s).probs.cpu().numpy()
    
    probs_s_expanded = np.zeros((n_states_global, 3))
    probs_s_expanded[0 : n_states_local] = probs_s_small[0]
    probs_s_expanded[n_states_local : 2*n_states_local] = probs_s_small[1]
    probs_s_expanded[2*n_states_local : ] = probs_s_small[2]

    # 4. LISTENER EXPANSION
    print("🧠 Processing Listener...")
    probs_l_list = []
    all_states = torch.arange(n_states_local, device=DEVICE)
    with torch.no_grad():
        for i in range(0, n_states_local, 1024):
            batch = all_states[i : i+1024]
            probs_l_list.append(actor_l(batch).probs.cpu().numpy())
    probs_l_local = np.concatenate(probs_l_list, axis=0)
    probs_l_expanded = np.tile(probs_l_local, (3, 1))

    # 5. SMOOTHING (Il trucco magico)
    print("🍦 Applying Smoothing (epsilon=0.05)...")
    probs_s_final = smooth_policy(probs_s_expanded)
    probs_l_final = smooth_policy(probs_l_expanded)

    # 6. SALVATAGGIO
    np.save("mappo_final_converged_probs_speaker.npy", probs_s_final)
    np.save("mappo_final_converged_probs_listener.npy", probs_l_final)

    print("\n✅ DONE! File salvati (Smoothed).")
    print(f"   Speaker Shape: {probs_s_final.shape}")
    print(f"   Listener Shape: {probs_l_final.shape}")

if __name__ == "__main__":
    fix_and_export()
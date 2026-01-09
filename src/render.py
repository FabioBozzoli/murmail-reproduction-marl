import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import pygame
from pettingzoo.mpe import simple_speaker_listener_v4
import pettingzoo.utils.wrappers.assert_out_of_bounds as aob

# ================= CONFIGURAZIONE =================
BINS = 6
CHECKPOINT = "mappo_final_fixed_checkpoint.pth" 
MAX_STEPS = 50
NUM_EPISODES = 10
DEVICE = torch.device("cpu")
GRID_BOUNDS = 1.5 # MPE World Size
# ==================================================

# Patch per evitare crash su azioni float
def patched_step(self, action): self.env.step(action)
aob.AssertOutOfBoundsWrapper.step = patched_step

# --- Modelli (Identici al training) ---
class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(obs_dim, hidden)
        self.policy = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
    def forward(self, obs):
        x = self.embed(obs)
        return Categorical(logits=self.policy(x))

# --- Discretizzazione ---
def discretize_obs(agent, obs):
    if agent == "speaker_0": return int(np.argmax(obs))
    vel, rel_pos, comm = obs[0:2], obs[2:4], obs[-3:]
    p_idx = np.clip(np.digitize(rel_pos, np.linspace(-1.5, 1.5, BINS+1))-1, 0, BINS-1)
    v_idx = np.clip(np.digitize(vel, np.linspace(-1.0, 1.0, 4))-1, 0, 2)
    c_idx = np.argmax(comm) if np.max(comm) > 0.1 else 0
    return int(p_idx[0] + p_idx[1]*BINS + v_idx[0]*(BINS**2) + v_idx[1]*(BINS**2)*3 + c_idx*(BINS**2)*9)

def get_action(agent, d_act):
    if agent == "speaker_0": a = np.zeros(3, dtype=np.float32); a[int(d_act)] = 1.0; return a
    # Mapping v5: 0:Stop, 1:Up, 2:Down, 3:Left, 4:Right
    a = np.zeros(5, dtype=np.float32); m = {0:0, 1:4, 2:3, 3:1, 4:2}
    a[m[d_act]] = 1.0
    return a

def get_world(env):
    if hasattr(env, 'world'): return env.world
    if hasattr(env, 'unwrapped'): return env.unwrapped.world if hasattr(env.unwrapped, 'world') else None
    if hasattr(env, 'aec_env'): return get_world(env.aec_env)
    if hasattr(env, 'env'): return get_world(env.env)
    return None

# =================================================
# 🎨 MOTORE GRAFICO COMPLETO
# =================================================
class UltimateRenderer:
    def __init__(self):
        pygame.font.init()
        self.font_big = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.scale = 350 # Zoom
        
        # Mappa Colori Target (Indice -> Nome, ColoreRGB)
        self.target_map = {
            0: ("RED", (255, 50, 50)),
            1: ("GREEN", (0, 200, 0)),
            2: ("BLUE", (50, 50, 255))
        }

    def to_px(self, pos, screen_size):
        w, h = screen_size
        x = int((pos[0] * self.scale) + w/2)
        y = int((-pos[1] * self.scale) + h/2)
        return (x, y)

    def process_events(self):
        """Mantiene viva la finestra su Mac"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
        return True

    def draw(self, world, goal_idx, reward, msg):
        screen = pygame.display.get_surface()
        if not screen or not world: return
        w, h = screen.get_size()

        # 1. DISEGNA GRIGLIA (Sfondo)
        grid_color = (200, 200, 200) # Grigio chiaro
        ticks = np.linspace(-GRID_BOUNDS, GRID_BOUNDS, BINS + 1)
        for t in ticks:
            # Verticali
            s = self.to_px((t, -GRID_BOUNDS), (w,h))
            e = self.to_px((t, GRID_BOUNDS), (w,h))
            pygame.draw.line(screen, grid_color, s, e, 1)
            # Orizzontali
            s = self.to_px((-GRID_BOUNDS, t), (w,h))
            e = self.to_px((GRID_BOUNDS, t), (w,h))
            pygame.draw.line(screen, grid_color, s, e, 1)

        # 2. LOGICA AGENTI
        listener = world.agents[0]
        target = world.landmarks[goal_idx]
        
        pos_l = self.to_px(listener.state.p_pos, (w, h))
        pos_t = self.to_px(target.state.p_pos, (w, h))

        # 3. LINEA GUIDA (Magenta Fluo)
        pygame.draw.line(screen, (255, 0, 255), pos_l, pos_t, 3)

        # 4. MARKER TARGET
        # Cerchio attorno al landmark giusto
        pygame.draw.circle(screen, (255, 0, 255), pos_t, 35, 3)
        
        # Etichetta TARGET sopra il landmark
        lbl, col = self.target_map.get(goal_idx, ("??", (0,0,0)))
        t_surf = self.font_big.render(f"TARGET: {lbl}", True, (0,0,0))
        t_rect = t_surf.get_rect(center=(pos_t[0], pos_t[1]-45))
        pygame.draw.rect(screen, (255,255,255), t_rect.inflate(10,5)) # Sfondo bianco
        pygame.draw.rect(screen, (0,0,0), t_rect.inflate(10,5), 2) # Bordo nero
        screen.blit(t_surf, t_rect)

        # 5. HUD (Info Panel)
        panel = pygame.Rect(10, 10, 250, 90)
        s = pygame.Surface((panel.w, panel.h))
        s.set_alpha(200); s.fill((0,0,0)) # Sfondo semitrasparente
        screen.blit(s, panel)

        # Testi
        screen.blit(self.font_big.render(f"Reward: {reward:.2f}", True, (255,255,0)), (20, 20))
        screen.blit(self.font_small.render(f"Heard Message: {msg}", True, (0,255,255)), (20, 55))
        screen.blit(self.font_small.render(f"Goal Index: {goal_idx}", True, (200,200,200)), (20, 75))

        pygame.display.flip()

# ================= MAIN =================
def main():
    actor_s = DiscreteActor(3, 3).to(DEVICE)
    actor_l = DiscreteActor((BINS**2)*27, 5).to(DEVICE)
    try:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        actor_s.load_state_dict(ckpt["actor_s"])
        actor_l.load_state_dict(ckpt["actor_l"])
        print("✅ Modello caricato.")
    except: 
        print(f"❌ Errore: File {CHECKPOINT} non trovato.")
        return

    # Init Env
    env = simple_speaker_listener_v4.parallel_env(render_mode="human", max_cycles=MAX_STEPS, continuous_actions=True)
    obs, _ = env.reset()
    
    # Warmup Pygame
    env.render()
    time.sleep(0.5)
    
    world = get_world(env)
    renderer = UltimateRenderer()

    for ep in range(NUM_EPISODES):
        obs_dict, _ = env.reset()
        time.sleep(0.5)
        goal_idx = int(np.argmax(obs_dict["speaker_0"]))
        
        print(f"\n--- EPISODIO {ep+1} ---")
        print(f"🎯 Target: {renderer.target_map[goal_idx][0]}")

        for step in range(MAX_STEPS):
            if not renderer.process_events(): break # Gestione chiusura finestra

            # Inferenza
            with torch.no_grad():
                s_act = actor_s(torch.tensor([discretize_obs("speaker_0", obs_dict["speaker_0"])])).probs.argmax().item()
                l_act = actor_l(torch.tensor([discretize_obs("listener_0", obs_dict["listener_0"])])).probs.argmax().item()
            
            # Step
            actions = {"speaker_0": get_action("speaker_0", s_act), "listener_0": get_action("listener_0", l_act)}
            obs_dict, rewards, terms, truncs, _ = env.step(actions)
            
            # Render
            renderer.draw(world, goal_idx, rewards["listener_0"], s_act)
            time.sleep(0.1)

            if any(terms.values()): break
        
        time.sleep(1.0)

    env.close()

if __name__ == "__main__":
    main()
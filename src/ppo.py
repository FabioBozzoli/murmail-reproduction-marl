import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb
import os

from discrete_wrapper import DiscretizedSpeakerListenerWrapper

# ==========================================
# 1. CONFIGURAZIONE (Perfetta per Benchmark)
# ==========================================
CONFIG = {
    "bins": 6,
    "rollout_len": 1024,
    "epochs": 4000,           
    "ppo_epochs": 10,
    "batch_size": 64,
    
    # --- CRUCIALE PER BENCHMARK ---
    "gamma": 0.9,             # Allineato a Expert/DQN
    "gae_lambda": 0.92,       
    
    "clip": 0.2,
    "lr_actor": 5e-4,
    "lr_critic": 1e-3,
    "hidden": 128,
    
    # Annealing
    "ent_start": 0.05,
    "ent_end": 0.001,
    "anneal_start_epoch": 2000, 
}

device = torch.device("cpu")
print("🐢 Force CPU (Faster on M1 for small nets)")

# ==========================================
# 2. MODELLI
# ==========================================
class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.embed = nn.Embedding(obs_dim, hidden)
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        x = self.embed(obs)
        logits = self.policy(x)
        return Categorical(logits=logits)

class CentralCritic(nn.Module):
    def __init__(self, obs_dim_s, obs_dim_l, hidden):
        super().__init__()
        self.embed_s = nn.Embedding(obs_dim_s, hidden)
        self.embed_l = nn.Embedding(obs_dim_l, hidden)
        self.value = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs_s, obs_l):
        es = self.embed_s(obs_s)
        el = self.embed_l(obs_l)
        x = torch.cat([es, el], dim=-1)
        return self.value(x).squeeze(-1)

# ==========================================
# 3. AGENTE
# ==========================================
class MAPPO:
    def __init__(self, obs_dim_s, obs_dim_l, cfg):
        self.actor_s = DiscreteActor(obs_dim_s, 3, cfg["hidden"]).to(device)
        self.actor_l = DiscreteActor(obs_dim_l, 5, cfg["hidden"]).to(device)
        self.critic = CentralCritic(obs_dim_s, obs_dim_l, cfg["hidden"]).to(device)

        self.opt_actor = optim.Adam(
            list(self.actor_s.parameters()) + list(self.actor_l.parameters()),
            lr=cfg["lr_actor"], eps=1e-5
        )
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"], eps=1e-5)

        self.gamma = cfg["gamma"]
        self.clip = cfg["clip"]
        self.gae_lambda = cfg["gae_lambda"]
        self.ppo_epochs = cfg["ppo_epochs"]
        self.mini_batch_size = cfg["batch_size"]
        self.current_ent_coef = cfg["ent_start"]

    def update_annealing(self, epoch, cfg):
        if epoch < cfg["anneal_start_epoch"]:
            self.current_ent_coef = cfg["ent_start"]
        else:
            total_decay_steps = cfg["epochs"] - cfg["anneal_start_epoch"]
            progress = (epoch - cfg["anneal_start_epoch"]) / total_decay_steps
            progress = min(1.0, max(0.0, progress))
            self.current_ent_coef = cfg["ent_start"] - progress * (cfg["ent_start"] - cfg["ent_end"])
        return self.current_ent_coef

# ==========================================
# 4. RACCOLTA DATI
# ==========================================
def collect_rollout(env, agent, T):
    buffer = {'obs_s': [], 'obs_l': [], 'a_s': [], 'a_l': [], 'logp_s': [], 'logp_l': [], 'rewards': [], 'dones': [], 'values': []}
    ep_rewards = []
    obs, _ = env.reset()
    ep_r = 0.0

    for _ in range(T):
        obs_s_t = torch.tensor(obs["speaker_0"], dtype=torch.long, device=device).unsqueeze(0)
        obs_l_t = torch.tensor(obs["listener_0"], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            dist_s = agent.actor_s(obs_s_t)
            dist_l = agent.actor_l(obs_l_t)
            a_s = dist_s.sample()
            a_l = dist_l.sample()
            logp_s = dist_s.log_prob(a_s)
            logp_l = dist_l.log_prob(a_l)
            val = agent.critic(obs_s_t, obs_l_t)

        next_obs, rewards, terms, truncs, _ = env.step({"speaker_0": a_s.item(), "listener_0": a_l.item()})
        done = any(terms.values()) or any(truncs.values())

        raw_r = rewards["listener_0"]
        scaled_r = raw_r * 0.1 
        ep_r += raw_r

        buffer['obs_s'].append(obs_s_t); buffer['obs_l'].append(obs_l_t)
        buffer['a_s'].append(a_s); buffer['a_l'].append(a_l)
        buffer['logp_s'].append(logp_s); buffer['logp_l'].append(logp_l)
        buffer['rewards'].append(scaled_r); buffer['dones'].append(float(done)); buffer['values'].append(val)

        obs = next_obs
        if done:
            ep_rewards.append(ep_r); obs, _ = env.reset(); ep_r = 0.0
            
    # GAE
    obs_s_t = torch.tensor(obs["speaker_0"], dtype=torch.long, device=device).unsqueeze(0)
    obs_l_t = torch.tensor(obs["listener_0"], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(): next_value = agent.critic(obs_s_t, obs_l_t).item()

    rewards = np.array(buffer['rewards']); dones = np.array(buffer['dones'])
    values = np.array([v.item() for v in buffer['values']] + [next_value])
    returns, gae = [], 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + agent.gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + agent.gamma * agent.gae_lambda * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])

    batch = {k: torch.cat(v) for k, v in buffer.items() if k not in ['rewards', 'dones', 'values']}
    batch['returns'] = torch.tensor(returns, dtype=torch.float32, device=device)
    batch['values'] = torch.cat(buffer['values'])
    batch['advantages'] = batch['returns'] - batch['values']
    return batch, np.mean(ep_rewards) if ep_rewards else 0.0

# ==========================================
# 5. UPDATE
# ==========================================
def update(agent, batch):
    adv = batch['advantages']
    batch['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    dataset_size = len(batch['returns']); indices = np.arange(dataset_size)
    total_a_loss, total_c_loss, total_entropy, updates = 0, 0, 0, 0

    for _ in range(agent.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, agent.mini_batch_size):
            end = start + agent.mini_batch_size; idx = indices[start:end]
            
            mb_obs_s = batch['obs_s'][idx]; mb_obs_l = batch['obs_l'][idx]
            mb_a_s = batch['a_s'][idx]; mb_a_l = batch['a_l'][idx]
            mb_old_logp_s = batch['logp_s'][idx]; mb_old_logp_l = batch['logp_l'][idx]
            mb_returns = batch['returns'][idx]; mb_adv = batch['advantages'][idx]
            
            dist_s = agent.actor_s(mb_obs_s); dist_l = agent.actor_l(mb_obs_l)
            values = agent.critic(mb_obs_s, mb_obs_l)
            
            ent_s = dist_s.entropy().mean()
            ent_l = dist_l.entropy().mean()
            entropy = 1.2 * ent_s + 1.0 * ent_l 
            
            logp_s = dist_s.log_prob(mb_a_s); ratio_s = torch.exp(logp_s - mb_old_logp_s)
            surr1_s = ratio_s * mb_adv; surr2_s = torch.clamp(ratio_s, 1-agent.clip, 1+agent.clip) * mb_adv
            loss_s = -torch.min(surr1_s, surr2_s).mean()
            
            logp_l = dist_l.log_prob(mb_a_l); ratio_l = torch.exp(logp_l - mb_old_logp_l)
            surr1_l = ratio_l * mb_adv; surr2_l = torch.clamp(ratio_l, 1-agent.clip, 1+agent.clip) * mb_adv
            loss_l = -torch.min(surr1_l, surr2_l).mean()
            
            actor_loss = loss_s + loss_l
            critic_loss = nn.HuberLoss(delta=10.0)(values, mb_returns)
            
            loss = actor_loss + 0.5 * critic_loss - agent.current_ent_coef * entropy
            
            agent.opt_actor.zero_grad(); agent.opt_critic.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_s.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(agent.actor_l.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
            agent.opt_actor.step(); agent.opt_critic.step()
            
            total_a_loss += actor_loss.item(); total_c_loss += critic_loss.item(); total_entropy += entropy.item(); updates += 1

    return total_a_loss/updates, total_c_loss/updates, total_entropy/updates

# ==========================================
# 6. EXPORT (RIABILITATO E CORRETTO)
# ==========================================
def export_policy(agent, filename="mappo_final_converged"):
    print(f"\n💾 Esportazione Policy per Benchmark: {filename}...")
    
    # Checkpoint standard
    torch.save({
        'actor_s': agent.actor_s.state_dict(),
        'actor_l': agent.actor_l.state_dict(),
        'critic': agent.critic.state_dict()
    }, f"{filename}_checkpoint.pth")

    # 1. SPEAKER POLICY (EXPANDED FOR NASH)
    # MAPPO ha policy (3,3). Nash vuole (Num_States, 3).
    obs_s = torch.tensor([0, 1, 2], device=device)
    with torch.no_grad(): 
        probs_s_small = agent.actor_s(obs_s).probs.cpu().numpy()

    n_states_total = CONFIG["bins"]**2 * 27
    states_per_goal = n_states_total // 3
    
    probs_s_expanded = np.zeros((n_states_total, 3))
    # Copia la policy del Goal 0 su tutti gli stati dove goal=0
    probs_s_expanded[0 : states_per_goal] = probs_s_small[0]
    # Copia la policy del Goal 1
    probs_s_expanded[states_per_goal : 2*states_per_goal] = probs_s_small[1]
    # Copia la policy del Goal 2
    probs_s_expanded[2*states_per_goal : ] = probs_s_small[2]

    # 2. LISTENER POLICY
    probs_l_list = []
    with torch.no_grad():
        for i in range(0, n_states_total, 1024):
            batch = torch.arange(i, min(i+1024, n_states_total), device=device)
            probs_l_list.append(agent.actor_l(batch).probs.cpu().numpy())
    probs_l = np.concatenate(probs_l_list, axis=0)

    # 3. SALVA NPY
    np.save(f"{filename}_probs_speaker.npy", probs_s_expanded)
    np.save(f"{filename}_probs_listener.npy", probs_l)
    
    print(f"✅ Export completato (Speaker Expanded: {probs_s_expanded.shape})")

# ==========================================
# 7. MAIN
# ==========================================
def train():
    wandb.init(project="mappo-speaker-listener", config=CONFIG, name="mappo-benchmark-ready")
    env = DiscretizedSpeakerListenerWrapper(bins=CONFIG["bins"])
    agent = MAPPO(3, CONFIG["bins"]**2 * 27, CONFIG)
    
    print("🚀 Starting MAPPO (Gamma 0.9, Fixed Params)...")

    try:
        for epoch in range(CONFIG["epochs"]):
            cur_ent = agent.update_annealing(epoch, CONFIG)
            batch, avg_ep_reward = collect_rollout(env, agent, CONFIG["rollout_len"])
            a_loss, c_loss, entropy = update(agent, batch)

            wandb.log({
                "epoch": epoch, "reward": avg_ep_reward,
                "loss_actor": a_loss, "loss_critic": c_loss, 
                "entropy_weighted": entropy * cur_ent,
                "ent_coef": cur_ent
            })

            if epoch % 10 == 0:
                print(f"Ep {epoch} | R: {avg_ep_reward:.2f} | Ent: {entropy:.3f}")
            
            # Backup periodico (che salva anche gli NPY)
            if epoch > 0 and epoch % 500 == 0:
                export_policy(agent, filename="mappo_backup_fixed")

    except KeyboardInterrupt:
        print("\n🛑 Interrotto.")
    finally:
        env.close(); wandb.finish()
        export_policy(agent, filename="mappo_final_converged")

if __name__ == "__main__":
    train()
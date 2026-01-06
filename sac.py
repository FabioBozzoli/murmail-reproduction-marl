"""
baseline_joint_sac.py - Joint-Action Soft Actor-Critic

APPROCCIO CORRETTO:
- Single actor: π(a_speaker, a_listener | s_speaker, s_listener)
- Single critic: Q(s_global, a_joint)
- Categorical distribution su 15 joint actions
- Entropy regularization
- Confronto DIRETTO con MURMAIL/Joint-DQN (stesso state/action space)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
from tqdm import tqdm
import time
import wandb
import matplotlib.pyplot as plt

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true

print("="*70)
print("🎯 JOINT-ACTION SAC (Centralizzato)")
print("="*70)

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES = 20000
BATCH_SIZE = 256
GAMMA = 0.9
TAU = 0.005
LR_ACTOR = 0.0003
LR_CRITIC = 0.0003
LR_ALPHA = 0.0003
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1000
EVAL_EVERY = 500
TARGET_ENTROPY_SCALE = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= WANDB SETUP =============
wandb.init(
    project="joint-sac-speaker-listener",
    name=f"joint-sac-bins{DISCRETIZATION_BINS}-{time.strftime('%Y%m%d-%H%M%S')}",
    config={
        "algorithm": "Joint-SAC",
        "discretization_bins": DISCRETIZATION_BINS,
        "num_episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "tau": TAU,
        "lr_actor": LR_ACTOR,
        "lr_critic": LR_CRITIC,
        "lr_alpha": LR_ALPHA,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "target_entropy_scale": TARGET_ENTROPY_SCALE,
        "device": str(device),
    },
    tags=["joint-sac", "baseline", "speaker-listener", "centralized"],
    notes="Joint-Action SAC with centralized policy (fair comparison with MURMAIL)"
)

print(f"✓ Algorithm: Joint-Action SAC (Centralized)")
print(f"✓ Policy: π(a_speaker, a_listener | s_global)")
print(f"✓ Critic: Q(s_global, a_joint)")
print(f"✓ Fair comparison with MURMAIL/Joint-DQN ✅")
print(f"📊 W&B Run: {wandb.run.get_url()}")
print("="*70)

# ============= ENVIRONMENT =============
raw_env = simple_speaker_listener_v4.parallel_env(
    continuous_actions=False,
    render_mode=None,
    max_cycles=25
)
env = DiscretizedSpeakerListenerWrapper(raw_env, bins=DISCRETIZATION_BINS)

NUM_LISTENER_STATES = 27 * (DISCRETIZATION_BINS ** 2)
S_SPEAKER = 3
A_SPEAKER = 3
A_LISTENER = 5
NUM_JOINT_STATES = S_SPEAKER * NUM_LISTENER_STATES
NUM_JOINT_ACTIONS = A_SPEAKER * A_LISTENER  # 15

print(f"✓ Joint states: {NUM_JOINT_STATES}")
print(f"✓ Joint actions: {NUM_JOINT_ACTIONS}")

wandb.config.update({
    "num_joint_states": NUM_JOINT_STATES,
    "num_joint_actions": NUM_JOINT_ACTIONS,
})

# ============= ENCODING =============
def encode_joint_state(s_speaker, s_listener):
    return s_speaker * NUM_LISTENER_STATES + s_listener

def decode_joint_action(joint_action):
    a_speaker = joint_action // A_LISTENER
    a_listener = joint_action % A_LISTENER
    return a_speaker, a_listener

def encode_joint_action(a_speaker, a_listener):
    return a_speaker * A_LISTENER + a_listener

# ============= NETWORKS =============
class JointActor(nn.Module):
    """
    Centralized stochastic policy
    π(a_speaker, a_listener | s_speaker, s_listener)
    """
    def __init__(self, num_states, num_actions, embed_dim=128, hidden=256):
        super().__init__()
        
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)  # Logits per 15 joint actions
        )
    
    def forward(self, state_indices):
        """Returns: logits for joint action distribution"""
        embedded = self.state_embedding(state_indices)
        logits = self.net(embedded)
        return logits
    
    def get_action_and_log_prob(self, state_indices):
        """Sample joint action and return log probability"""
        logits = self.forward(state_indices)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, state_indices, actions):
        """Compute log prob of given joint actions"""
        logits = self.forward(state_indices)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions)

class JointCritic(nn.Module):
    """
    Centralized Q-function (Double Q)
    Q(s_global, a_joint)
    """
    def __init__(self, num_states, num_actions, embed_dim=128, hidden=256):
        super().__init__()
        
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    
    def forward(self, state_indices):
        """Returns: (Q1, Q2) values for all joint actions"""
        embedded = self.state_embedding(state_indices)
        q1 = self.q1(embedded)
        q2 = self.q2(embedded)
        return q1, q2

# ============= REPLAY BUFFER =============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, joint_state, joint_action, reward, next_joint_state, done):
        self.buffer.append((joint_state, joint_action, reward, next_joint_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int64),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============= JOINT SAC AGENT =============
class JointSACAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Networks
        self.actor = JointActor(num_states, num_actions).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = JointCritic(num_states, num_actions).to(device)
        self.critic_target = JointCritic(num_states, num_actions).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # Temperature
        self.target_entropy = -TARGET_ENTROPY_SCALE * np.log(1.0 / num_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        print(f"✓ Target entropy: {self.target_entropy:.4f}")
        
        wandb.watch(self.actor, log="all", log_freq=1000)
        
        total_params = sum(p.numel() for p in self.actor.parameters()) + \
                      sum(p.numel() for p in self.critic.parameters())
        wandb.config.update({"total_parameters": total_params})
        print(f"✓ Total parameters: {total_params:,}")
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, joint_state, eval_mode=False):
        """Select joint action"""
        state_t = torch.LongTensor([joint_state]).to(device)
        
        if eval_mode:
            # Deterministic (greedy)
            with torch.no_grad():
                logits = self.actor(state_t)
                action = logits.argmax(dim=1).item()
        else:
            # Stochastic (sample from policy)
            with torch.no_grad():
                logits = self.actor(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample().item()
        
        return action
    
    def update(self):
        """Discrete SAC update - NUMERICALLY STABLE VERSION"""
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        states_t = torch.LongTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.LongTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        
        # ========== CRITIC UPDATE ==========
        with torch.no_grad():
            next_logits = self.actor(next_states_t)
            
            # ✅ FIX: Clamp logits per evitare overflow in softmax
            next_logits = torch.clamp(next_logits, min=-20, max=20)
            
            next_probs = F.softmax(next_logits, dim=1)
            next_log_probs = F.log_softmax(next_logits, dim=1)
            
            # ✅ FIX: Clamp probs per evitare log(0)
            next_probs = torch.clamp(next_probs, min=1e-8, max=1.0)
            next_log_probs = torch.clamp(next_log_probs, min=-20, max=0)
            
            next_q1_target, next_q2_target = self.critic_target(next_states_t)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # V(s') = E[Q(s',a) - α log π(a|s')]
            next_v_target = (next_probs * (next_q_target - self.alpha * next_log_probs)).sum(dim=1)
            
            # ✅ FIX: Clamp value target
            next_v_target = torch.clamp(next_v_target, min=-100, max=100)
            
            q_target = rewards_t + GAMMA * (1 - dones_t) * next_v_target
        
        q1, q2 = self.critic(states_t)
        q1_pred = q1.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q2_pred = q2.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        critic_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        
        # ✅ FIX: Check for NaN in loss
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print("⚠️  NaN/Inf detected in critic_loss, skipping update")
            return None
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # ✅ Più conservativo
        self.critic_optimizer.step()
        
        # ========== ACTOR UPDATE ==========
        logits = self.actor(states_t)
        
        # ✅ FIX: Clamp logits
        logits = torch.clamp(logits, min=-20, max=20)
        
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # ✅ FIX: Clamp probabilities
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.clamp(log_probs, min=-20, max=0)
        
        with torch.no_grad():
            q1, q2 = self.critic(states_t)
            q_values = torch.min(q1, q2)
            
            # ✅ FIX: Clamp Q-values
            q_values = torch.clamp(q_values, min=-100, max=100)
        
        # Actor loss
        inside_term = self.alpha.detach() * log_probs - q_values
        actor_loss = (probs * inside_term).sum(dim=1).mean()
        
        # ✅ FIX: Check for NaN
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            print("⚠️  NaN/Inf detected in actor_loss, skipping update")
            return None
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # ✅ Più conservativo
        self.actor_optimizer.step()
        
        # ========== TEMPERATURE UPDATE ==========
        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=1).mean()
            
            # ✅ FIX: Clamp entropy
            entropy = torch.clamp(entropy, min=0, max=10)
        
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy).detach()
        
        # ✅ FIX: Check for NaN
        if torch.isnan(alpha_loss) or torch.isinf(alpha_loss):
            print("⚠️  NaN/Inf detected in alpha_loss, skipping update")
            return None
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ✅ FIX: Clamp log_alpha per evitare alpha troppo grandi
        with torch.no_grad():
            self.log_alpha.data = torch.clamp(self.log_alpha.data, min=-5, max=5)
        
        # Soft update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'mean_entropy': entropy.item(),
        }

# ============= EVALUATION =============
def evaluate(agent, n_episodes=50):
    """Evaluate with deterministic policy"""
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            s_spk = obs["speaker_0"]
            s_lst = obs["listener_0"]
            
            joint_state = encode_joint_state(s_spk, s_lst)
            joint_action = agent.select_action(joint_state, eval_mode=True)
            
            a_spk, a_lst = decode_joint_action(joint_action)
            
            obs, rewards, terms, truncs, _ = env.step({
                "speaker_0": a_spk,
                "listener_0": a_lst
            })
            
            ep_reward += rewards["speaker_0"]
            done = any(terms.values()) or any(truncs.values())
        
        total_reward += ep_reward
    
    avg_reward = total_reward / n_episodes
    
    # Calculate exploitability (extract marginal policies correctly)
    gap = None
    try:
        # ✅ CORREZIONE: Dimensioni corrette
        # Joint state space = S_SPEAKER * NUM_LISTENER_STATES
        # Ma le policy marginalizzate hanno dimensioni:
        # - pi_speaker: (S_SPEAKER, A_SPEAKER) = (3, 3)
        # - pi_listener: (NUM_LISTENER_STATES, A_LISTENER) = (972, 5)
        
        # PERÒ per calc_exploitability_true servono policy joint:
        # - pi_speaker: (NUM_JOINT_STATES, A_SPEAKER) = (2916, 3)
        # - pi_listener: (NUM_JOINT_STATES, A_LISTENER) = (2916, 5)
        
        NUM_JOINT_STATES = S_SPEAKER * NUM_LISTENER_STATES
        
        pi_speaker_joint = np.zeros((NUM_JOINT_STATES, A_SPEAKER))
        pi_listener_joint = np.zeros((NUM_JOINT_STATES, A_LISTENER))
        
        # Estrai policy per OGNI joint state
        for joint_state in range(NUM_JOINT_STATES):
            state_t = torch.LongTensor([joint_state]).to(device)
            
            with torch.no_grad():
                logits = agent.actor(state_t)
                probs_joint = F.softmax(logits, dim=1).cpu().numpy()[0]  # (15,)
            
            # Marginalizza per speaker
            for a_spk in range(A_SPEAKER):
                prob_spk = 0.0
                for a_lst in range(A_LISTENER):
                    joint_action = encode_joint_action(a_spk, a_lst)
                    prob_spk += probs_joint[joint_action]
                pi_speaker_joint[joint_state, a_spk] = prob_spk
            
            # Marginalizza per listener
            for a_lst in range(A_LISTENER):
                prob_lst = 0.0
                for a_spk in range(A_SPEAKER):
                    joint_action = encode_joint_action(a_spk, a_lst)
                    prob_lst += probs_joint[joint_action]
                pi_listener_joint[joint_state, a_lst] = prob_lst
        
        # Normalizza (safety check)
        pi_speaker_joint = pi_speaker_joint / pi_speaker_joint.sum(axis=1, keepdims=True)
        pi_listener_joint = pi_listener_joint / pi_listener_joint.sum(axis=1, keepdims=True)
        
        # Calculate gap
        P = np.load('P_bins6.npy').astype(np.float64)
        R = np.load('R_bins6.npy').astype(np.float64)
        init_dist = np.load('expert_initial_dist_bins6.npy').astype(np.float64)
        
        gap = calc_exploitability_true(pi_speaker_joint, pi_listener_joint, R, P, init_dist, GAMMA)
        
    except Exception as e:
        print(f"Warning: Gap calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return avg_reward, gap
# ============= TRAINING =============
print("\n🚀 Training Joint-Action SAC...")

agent = JointSACAgent(NUM_JOINT_STATES, NUM_JOINT_ACTIONS)

# Load baselines
try:
    pi_s = np.load('expert_policy_speaker_bins6.npy')
    pi_l = np.load('expert_policy_listener_bins6.npy')
    P = np.load('P_bins6.npy').astype(np.float64)
    R = np.load('R_bins6.npy').astype(np.float64)
    init_dist = np.load('expert_initial_dist_bins6.npy')
    
    expert_gap = calc_exploitability_true(pi_s, pi_l, R, P, init_dist, GAMMA)
    
    mu_unif = np.ones((NUM_JOINT_STATES, A_SPEAKER)) / A_SPEAKER
    nu_unif = np.ones((NUM_JOINT_STATES, A_LISTENER)) / A_LISTENER
    uniform_gap = calc_exploitability_true(mu_unif, nu_unif, R, P, init_dist, GAMMA)
    
    wandb.log({"baseline/expert_gap": expert_gap, "baseline/uniform_gap": uniform_gap}, step=0)
    wandb.run.summary.update({"expert_gap": expert_gap, "uniform_gap": uniform_gap})
    
    print(f"✓ Baselines: Expert={expert_gap:.6f}, Uniform={uniform_gap:.6f}")
except Exception as e:
    print(f"Warning: Could not load baselines: {e}")
    expert_gap = None
    uniform_gap = None

# Training loop
eval_rewards = []
eval_gaps = []
eval_episodes = []
episode_rewards = []
training_start_time = time.time()

for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Episodes"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # 1. Recupero stato corrente (già codificato o da codificare)
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        joint_state = encode_joint_state(s_spk, s_lst)
        
        # 2. Selezione Azione
        joint_action = agent.select_action(joint_state, eval_mode=False)
        a_spk, a_lst = decode_joint_action(joint_action)
        
        # 3. Step Ambiente (UNA SOLA VOLTA)
        next_obs, rewards, terms, truncs, _ = env.step({
            "speaker_0": a_spk,
            "listener_0": a_lst
        })
        
        # 4. Calcolo Next Joint State (Fix applicato)
        ns_spk = next_obs["speaker_0"]
        ns_lst = next_obs["listener_0"]
        next_joint_state = encode_joint_state(ns_spk, ns_lst)
        
        # 5. Gestione Reward e Done
        done = any(terms.values()) or any(truncs.values())
        
        # Reward Scaling
        raw_reward = rewards["speaker_0"] + rewards["listener_0"]
        
        # 1. Normalizza [-100, 0] -> [0, 1]
        norm_reward = (raw_reward + 100.0) / 100.0
        norm_reward = np.clip(norm_reward, 0.0, 1.0)
        
        # 2. Applica Esponente (punisce la distanza media) e Scala (aiuta i gradienti)
        # Random (dist -50) -> 0.5^2 * 10 = 2.5
        # Perfect (dist 0)  -> 1.0^2 * 10 = 10.0
        # Delta = 7.5 (Molto forte!)
        reward = (norm_reward ** 2) * 10.0 
        
        episode_reward += raw_reward 
        
        agent.replay_buffer.push(joint_state, joint_action, reward, next_joint_state, done)
                
        # 7. Update Networks
        losses = agent.update()
        
        # 8. Aggiornamento osservazione per il prossimo giro
        obs = next_obs
        
    episode_rewards.append(episode_reward)
    
    # Log training
    if episode % 10 == 0 and losses is not None:
        wandb.log({
            "train/episode": episode,
            "train/reward": episode_reward,
            "train/reward_ma10": np.mean(episode_rewards[-10:]),
            "train/buffer_size": len(agent.replay_buffer),
            "train/critic_loss": losses['critic_loss'],
            "train/actor_loss": losses['actor_loss'],
            "train/alpha": losses['alpha'],
            "train/entropy": losses['mean_entropy'],
        }, step=episode)
    
    # Evaluation
    if episode % EVAL_EVERY == 0:
        avg_reward, gap = evaluate(agent)
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        eval_episodes.append(episode)
        
        elapsed = time.time() - training_start_time
        gap_str = f"{gap:.6f}" if gap else "N/A"
        
        print(f"\nEp {episode} | Reward: {avg_reward:.2f} | Gap: {gap_str}")
        
        log_dict = {"eval/episode": episode, "eval/reward": avg_reward}
        
        if gap is not None:
            log_dict["eval/gap"] = gap
            if expert_gap and uniform_gap:
                improvement_pct = ((uniform_gap - gap) / (uniform_gap - expert_gap)) * 100
                log_dict["eval/improvement_percentage"] = improvement_pct
        
        wandb.log(log_dict, step=episode)

env.close()

# Save results (rest of code similar to before)
print(f"\n✅ Training Complete")

wandb.finish()
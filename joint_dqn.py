"""
baseline_joint_dqn.py - Joint-Action Deep Q-Network con W&B logging

APPROCCIO CORRETTO per questo task:
- Stato globale discretizzato (s_speaker, s_listener)
- Q-network unica: Q(s_global, a_speaker, a_listener)
- No decomposizione (no QMIX bias)
- Action space: 3×5 = 15 joint actions
- ✨ W&B logging completo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from tqdm import tqdm
import time
import wandb  # ← NUOVO
import matplotlib.pyplot as plt  # ← NUOVO

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from true_exploitability import calc_true_exploitability_from_q_network

print("="*70)
print("🎯 JOINT-ACTION DQN with W&B Logging")
print("="*70)

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES = 20000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 0.0005
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_EPISODES = 15000
EVAL_EVERY = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= WANDB SETUP =============
wandb.init(
    project="joint-dqn-speaker-listener",
    name=f"joint-dqn-bins{DISCRETIZATION_BINS}-{time.strftime('%Y%m%d-%H%M%S')}",
    config={
        "discretization_bins": DISCRETIZATION_BINS,
        "num_episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LR,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "min_replay_size": MIN_REPLAY_SIZE,
        "target_update_freq": TARGET_UPDATE_FREQ,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_episodes": EPSILON_DECAY_EPISODES,
        "eval_frequency": EVAL_EVERY,
        "device": str(device),
    },
    tags=["joint-dqn", "baseline", "speaker-listener"],
    notes="Joint-Action DQN baseline with no factorization"
)

config = wandb.config

print(f"✓ Joint Q-network (no decomposition)")
print(f"✓ LR: {LR}, Gamma: {GAMMA}")
print(f"✓ Action space: 3×5 = 15 joint actions")
print(f"✓ Device: {device}")
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
NUM_JOINT_ACTIONS = A_SPEAKER * A_LISTENER  # 15

print(f"✓ States: Speaker={S_SPEAKER}, Listener={NUM_LISTENER_STATES}")
print(f"✓ Joint actions: {NUM_JOINT_ACTIONS}")

# Update W&B config
wandb.config.update({
    "num_speaker_states": S_SPEAKER,
    "num_listener_states": NUM_LISTENER_STATES,
    "num_joint_states": S_SPEAKER * NUM_LISTENER_STATES,
    "num_actions_speaker": A_SPEAKER,
    "num_actions_listener": A_LISTENER,
    "num_joint_actions": NUM_JOINT_ACTIONS,
})

# ============= JOINT STATE ENCODING =============
def encode_joint_state(s_speaker, s_listener):
    """Encode joint state as single index"""
    return s_speaker * NUM_LISTENER_STATES + s_listener

def decode_joint_action(joint_action):
    """joint_action ∈ [0, 14] → (a_speaker, a_listener)"""
    a_speaker = joint_action // A_LISTENER
    a_listener = joint_action % A_LISTENER
    return a_speaker, a_listener

def encode_joint_action(a_speaker, a_listener):
    """(a_speaker, a_listener) → joint_action"""
    return a_speaker * A_LISTENER + a_listener

# ============= Q-NETWORK =============
class JointQNetwork(nn.Module):
    """
    Q(s_global, a_joint)
    
    Input: joint state index
    Output: Q-values for all 15 joint actions
    """
    def __init__(self, num_states, num_actions, embed_dim=128, hidden=128):
        super().__init__()
        
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
    
    def forward(self, state_indices):
        """
        state_indices: (batch,) long tensor
        Returns: (batch, num_actions) Q-values
        """
        embedded = self.state_embedding(state_indices)
        return self.net(embedded)

# ============= REPLAY BUFFER =============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
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

# ============= DQN AGENT =============
class JointDQNAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Networks
        self.q_network = JointQNetwork(num_states, num_actions).to(device)
        self.target_network = JointQNetwork(num_states, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Exploration
        self.epsilon = EPSILON_START
        self.steps = 0
        
        # ← NOVO: Log model architecture to W&B
        wandb.watch(self.q_network, log="all", log_freq=1000)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })
        
        print(f"✓ Q-Network: {total_params:,} total params ({trainable_params:,} trainable)")
    
    def select_action(self, state, eval_mode=False):
        """
        state: joint state index
        Returns: joint action index
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        with torch.no_grad():
            state_t = torch.LongTensor([state]).to(device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
    
    def update(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # To tensors
        states_t = torch.LongTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.LongTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        q_values = self.q_network(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Select actions with online network
            next_q_values_online = self.q_network(next_states_t)
            next_actions = next_q_values_online.argmax(dim=1)
            
            # Evaluate with target network
            next_q_values_target = self.target_network(next_states_t)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # TD target
            targets = rewards_t + GAMMA * (1 - dones_t) * next_q_values
        
        # Loss
        loss = F.mse_loss(q_values, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_epsilon(self, episode):
        """Linear decay"""
        if episode < EPSILON_DECAY_EPISODES:
            self.epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * episode / EPSILON_DECAY_EPISODES
        else:
            self.epsilon = EPSILON_END

# ============= EVALUATION =============
def evaluate(agent, n_episodes=50):
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
    
    # Calculate TRUE Nash exploitability using LP
    gap = None
    try:
        P = np.load('P_bins6.npy').astype(np.float64)
        R = np.load('R_bins6.npy').astype(np.float64)
        init_dist = np.load('expert_initial_dist_bins6.npy').astype(np.float64)
        
        gap = calc_true_exploitability_from_q_network(
            agent.q_network, device, R, P, init_dist, GAMMA,
            NUM_JOINT_STATES, A_SPEAKER, A_LISTENER
        )
    except Exception as e:
        print(f"Warning: Gap calculation failed: {e}")
    
    return avg_reward, gap

# ============= TRAINING =============
print("\n🚀 Training Joint-Action DQN...")

NUM_JOINT_STATES = S_SPEAKER * NUM_LISTENER_STATES
agent = JointDQNAgent(NUM_JOINT_STATES, NUM_JOINT_ACTIONS)

# ← NOVO: Load baselines for comparison
try:
    expert_gap = wandb.run.summary.get("baseline/expert_gap")
    uniform_gap = wandb.run.summary.get("baseline/uniform_gap")
    
    if expert_gap is None or uniform_gap is None:
        # Load from file
        pi_s = np.load('expert_policy_speaker_bins6.npy')
        pi_l = np.load('expert_policy_listener_bins6.npy')
        P = np.load('P_bins6.npy').astype(np.float64)
        R = np.load('R_bins6.npy').astype(np.float64)
        init_dist = np.load('expert_initial_dist_bins6.npy')
        
        from utils import calc_exploitability_true
        expert_gap = calc_exploitability_true(pi_s, pi_l, R, P, init_dist, GAMMA)
        
        mu_unif = np.ones((NUM_JOINT_STATES, A_SPEAKER)) / A_SPEAKER
        nu_unif = np.ones((NUM_JOINT_STATES, A_LISTENER)) / A_LISTENER
        uniform_gap = calc_exploitability_true(mu_unif, nu_unif, R, P, init_dist, GAMMA)
        
        wandb.log({
            "baseline/expert_gap": expert_gap,
            "baseline/uniform_gap": uniform_gap,
        }, step=0)
        
        wandb.run.summary["expert_gap"] = expert_gap
        wandb.run.summary["uniform_gap"] = uniform_gap
        
        print(f"✓ Baselines: Expert={expert_gap:.6f}, Uniform={uniform_gap:.6f}")
except Exception as e:
    print(f"Warning: Could not load baselines: {e}")
    expert_gap = None
    uniform_gap = None

eval_rewards = []
eval_gaps = []
eval_episodes = []
all_losses = []

# ← NOVO: Training metrics
training_start_time = time.time()
episode_rewards = []
episode_lengths = []

for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Episodes"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    while not done:
        # Encode joint state
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        joint_state = encode_joint_state(s_spk, s_lst)
        
        # Select joint action
        joint_action = agent.select_action(joint_state)
        
        # Decode to individual actions
        a_spk, a_lst = decode_joint_action(joint_action)
        
        # Step
        next_obs, rewards, terms, truncs, _ = env.step({
            "speaker_0": a_spk,
            "listener_0": a_lst
        })
        
        done = any(terms.values()) or any(truncs.values())
        reward = rewards["speaker_0"] + rewards["listener_0"]
        episode_reward += reward
        episode_steps += 1
        
        # Encode next joint state
        next_joint_state = encode_joint_state(next_obs["speaker_0"], next_obs["listener_0"])
        
        # Store transition
        agent.replay_buffer.push(joint_state, joint_action, reward, next_joint_state, done)
        
        # Update
        loss = agent.update()
        if loss is not None:
            all_losses.append(loss)
        
        obs = next_obs
    
    # Update epsilon
    agent.update_epsilon(episode)
    
    # ← NOVO: Track episode metrics
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_steps)
    
    # ← NOVO: Log training metrics every episode
    if episode % 10 == 0:  # Log every 10 episodes to avoid overhead
        avg_reward_10 = np.mean(episode_rewards[-10:])
        avg_length_10 = np.mean(episode_lengths[-10:])
        avg_loss_100 = np.mean(all_losses[-100:]) if all_losses else 0
        
        wandb.log({
            "train/episode": episode,
            "train/reward": episode_reward,
            "train/reward_ma10": avg_reward_10,
            "train/episode_length": episode_steps,
            "train/episode_length_ma10": avg_length_10,
            "train/epsilon": agent.epsilon,
            "train/buffer_size": len(agent.replay_buffer),
            "train/loss": avg_loss_100,
            "train/steps_total": agent.steps,
        }, step=episode)
    
    # Evaluation
    if episode % EVAL_EVERY == 0:
        eval_start = time.time()
        avg_reward, gap = evaluate(agent)
        eval_time = time.time() - eval_start
        
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        eval_episodes.append(episode)
        
        # ← NOVO: Compute improvement metrics
        if gap is not None and expert_gap is not None and uniform_gap is not None:
            improvement_from_uniform = uniform_gap - gap
            improvement_pct = (improvement_from_uniform / (uniform_gap - expert_gap)) * 100
            remaining_gap = gap - expert_gap
            gap_normalized = (gap - expert_gap) / (uniform_gap - expert_gap)
        else:
            improvement_from_uniform = None
            improvement_pct = None
            remaining_gap = None
            gap_normalized = None
        
        gap_str = f"{gap:.6f}" if gap else "N/A"
        avg_loss = np.mean(all_losses[-100:]) if all_losses else 0
        
        elapsed = time.time() - training_start_time
        
        print(f"\nEp {episode} | Reward: {avg_reward:.2f} | Gap: {gap_str} | ε: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")
        if improvement_pct is not None:
            print(f"       Improvement: {improvement_pct:.1f}% of possible")
        
        # ← NOVO: Log evaluation metrics
        log_dict = {
            "eval/episode": episode,
            "eval/reward": avg_reward,
            "eval/eval_time_seconds": eval_time,
            "time/elapsed_minutes": elapsed / 60,
            "time/episodes_per_minute": episode / (elapsed / 60),
        }
        
        if gap is not None:
            log_dict["eval/gap"] = gap
            
            if expert_gap is not None and uniform_gap is not None:
                log_dict.update({
                    "eval/improvement_from_uniform": improvement_from_uniform,
                    "eval/improvement_percentage": improvement_pct,
                    "eval/remaining_to_expert": remaining_gap,
                    "eval/gap_normalized": gap_normalized,
                })
        
        wandb.log(log_dict, step=episode)
        
        # ← NOVO: Update best metrics
        if gap is not None:
            current_best = wandb.run.summary.get("best_gap", float('inf'))
            if gap < current_best:
                wandb.run.summary.update({
                    "best_gap": gap,
                    "best_gap_episode": episode,
                    "best_gap_reward": avg_reward,
                })
                if improvement_pct is not None:
                    wandb.run.summary["best_improvement_pct"] = improvement_pct

env.close()

# ============= FINAL RESULTS =============
total_time = time.time() - training_start_time

print(f"\n✅ Training Complete ({total_time/60:.1f} minutes)")

if len(eval_gaps) > 0 and any(g is not None for g in eval_gaps):
    valid_gaps = [g for g in eval_gaps if g is not None]
    final_gap = valid_gaps[-1] if valid_gaps else None
    
    if final_gap:
        print(f"   Final gap: {final_gap:.6f}")
        
        best_gap = min(valid_gaps)
        best_idx = eval_gaps.index(best_gap)
        best_ep = eval_episodes[best_idx]
        print(f"   Best gap: {best_gap:.6f} (episode {best_ep})")
        
        # ← NOVO: Log final summary
        wandb.run.summary.update({
            "final/gap": final_gap,
            "final/best_gap": best_gap,
            "final/best_gap_episode": best_ep,
            "final/total_time_minutes": total_time / 60,
            "final/total_episodes": NUM_EPISODES,
            "final/total_steps": agent.steps,
        })

# ============= SAVE MODEL =============
print("\n💾 Saving model and results...")

torch.save({
    'q_network': agent.q_network.state_dict(),
    'target_network': agent.target_network.state_dict(),
    'episode': NUM_EPISODES
}, 'joint_dqn_model.pth')

import pickle
results = {
    'eval_episodes': eval_episodes,
    'eval_rewards': eval_rewards,
    'eval_gaps': eval_gaps,
    'losses': all_losses,
    'episode_rewards': episode_rewards,
    'episode_lengths': episode_lengths,
}

with open('joint_dqn_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("   ✓ Model saved: joint_dqn_model.pth")
print("   ✓ Results saved: joint_dqn_results.pkl")

# ← NOVO: Save as W&B artifact
artifact = wandb.Artifact(
    name=f'joint-dqn-model-{wandb.run.id}',
    type='model',
    description=f'Joint-Action DQN model | Final gap: {final_gap:.6f}' if final_gap else 'Joint-Action DQN model'
)

artifact.add_file('joint_dqn_model.pth')
artifact.add_file('joint_dqn_results.pkl')

wandb.log_artifact(artifact)
print("   ✓ Artifact logged to W&B")

# ============= VISUALIZATION =============
print("\n📊 Creating visualizations...")

# Plot 1: Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Episode rewards
axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode reward')
if len(episode_rewards) >= 100:
    ma100 = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    axes[0, 0].plot(range(99, len(episode_rewards)), ma100, linewidth=2, label='MA(100)')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Training Rewards')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
if len(all_losses) > 0:
    axes[0, 1].plot(all_losses, alpha=0.3)
    if len(all_losses) >= 1000:
        ma1000 = np.convolve(all_losses, np.ones(1000)/1000, mode='valid')
        axes[0, 1].plot(range(999, len(all_losses)), ma1000, linewidth=2, color='red')
    axes[0, 1].set_xlabel('Update Step')
    axes[0, 1].set_ylabel('TD Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)

# Eval rewards
axes[1, 0].plot(eval_episodes, eval_rewards, 'o-', linewidth=2, markersize=4)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Average Reward')
axes[1, 0].set_title('Evaluation Rewards')
axes[1, 0].grid(True, alpha=0.3)

# Exploitability gap
valid_gaps_with_eps = [(ep, g) for ep, g in zip(eval_episodes, eval_gaps) if g is not None]
if len(valid_gaps_with_eps) > 0:
    eps, gaps = zip(*valid_gaps_with_eps)
    axes[1, 1].plot(eps, gaps, 'o-', linewidth=2, markersize=4, label='Joint-DQN')
    if expert_gap is not None:
        axes[1, 1].axhline(y=expert_gap, color='green', linestyle='--', linewidth=2, label=f'Expert ({expert_gap:.4f})')
    if uniform_gap is not None:
        axes[1, 1].axhline(y=uniform_gap, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Uniform ({uniform_gap:.4f})')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Exploitability Gap')
    axes[1, 1].set_title('Nash Exploitability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_dqn_training.png', dpi=150, bbox_inches='tight')
wandb.log({"plots/training_curves": wandb.Image('joint_dqn_training.png')})
plt.close()

print("   ✓ Plot saved: joint_dqn_training.png")

# Plot 2: Gap convergence (if available)
if len(valid_gaps_with_eps) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    eps, gaps = zip(*valid_gaps_with_eps)
    
    # Log scale
    ax1.semilogy(eps, gaps, 'o-', linewidth=2, markersize=4, label='Joint-DQN')
    if expert_gap is not None:
        ax1.axhline(y=expert_gap, color='green', linestyle='--', linewidth=2, label=f'Expert ({expert_gap:.4f})')
    if uniform_gap is not None:
        ax1.axhline(y=uniform_gap, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Uniform ({uniform_gap:.4f})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Exploitability Gap (log scale)', fontsize=12)
    ax1.set_title('Convergence - Log Scale', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Linear scale
    ax2.plot(eps, gaps, 'o-', linewidth=2, markersize=4, label='Joint-DQN')
    if expert_gap is not None:
        ax2.axhline(y=expert_gap, color='green', linestyle='--', linewidth=2, label=f'Expert ({expert_gap:.4f})')
    if uniform_gap is not None:
        ax2.axhline(y=uniform_gap, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Uniform ({uniform_gap:.4f})')
        if expert_gap is not None:
            ax2.fill_between(eps, expert_gap, uniform_gap, alpha=0.1, color='gray')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Exploitability Gap', fontsize=12)
    ax2.set_title('Convergence - Linear Scale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_dqn_convergence.png', dpi=150, bbox_inches='tight')
    wandb.log({"plots/convergence": wandb.Image('joint_dqn_convergence.png')})
    plt.close()
    
    print("   ✓ Plot saved: joint_dqn_convergence.png")

print("\n" + "="*70)
print("🎉 Training Complete!")
print("="*70)
print(f"📊 View results at: {wandb.run.get_url()}")
print(f"📊 Project page: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}")
print("="*70)

wandb.finish()
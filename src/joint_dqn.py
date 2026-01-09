"""
baseline_joint_dqn.py - Joint-Action Deep Q-Network con SOFTMAX exploration

MODIFICHE CHIAVE:
- Softmax exploration invece di epsilon-greedy
- Learning rate ridotto (0.0001)
- Temperature decay invece di epsilon decay
- Reward normalization [0, 1]
- Policy extraction con softmax (no greedy)
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
import wandb
import matplotlib.pyplot as plt

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper

print("="*70)
print("🎯 JOINT-ACTION DQN with SOFTMAX Exploration")
print("="*70)

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES = 20000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 0.0001  # ← RIDOTTO da 0.0005
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_FREQ = 500  # ← PIÙ FREQUENTE da 1000

# ✅ SOFTMAX PARAMETERS (invece di epsilon)
TEMPERATURE_START = 2.0
TEMPERATURE_END = 0.1
TEMPERATURE_DECAY_EPISODES = 10000  # ← PIÙ VELOCE da 15000

EVAL_EVERY = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============= WANDB SETUP =============
wandb.init(
    project="joint-dqn-speaker-listener",
    name=f"joint-dqn-softmax-bins{DISCRETIZATION_BINS}-{time.strftime('%Y%m%d-%H%M%S')}",
    config={
        "discretization_bins": DISCRETIZATION_BINS,
        "num_episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LR,
        "replay_buffer_size": REPLAY_BUFFER_SIZE,
        "min_replay_size": MIN_REPLAY_SIZE,
        "target_update_freq": TARGET_UPDATE_FREQ,
        "temperature_start": TEMPERATURE_START,
        "temperature_end": TEMPERATURE_END,
        "temperature_decay_episodes": TEMPERATURE_DECAY_EPISODES,
        "eval_frequency": EVAL_EVERY,
        "device": str(device),
        "exploration": "softmax",  # ← NEW
    },
    tags=["joint-dqn", "softmax", "speaker-listener"],
    notes="Joint-Action DQN with Softmax exploration and policy extraction"
)

config = wandb.config

print(f"✓ Joint Q-network (no decomposition)")
print(f"✓ LR: {LR}, Gamma: {GAMMA}")
print(f"✓ Exploration: Softmax (T: {TEMPERATURE_START} → {TEMPERATURE_END})")
print(f"✓ Device: {device}")
print(f"📊 W&B Run: {wandb.run.get_url()}")
print("="*70)

# ============= ENVIRONMENT =============
env = DiscretizedSpeakerListenerWrapper(bins=DISCRETIZATION_BINS)

NUM_LISTENER_STATES = 27 * (DISCRETIZATION_BINS ** 2)
S_SPEAKER = 3
A_SPEAKER = 3
A_LISTENER = 5
NUM_JOINT_ACTIONS = A_SPEAKER * A_LISTENER  # 15
NUM_JOINT_STATES = S_SPEAKER * NUM_LISTENER_STATES

print(f"✓ States: Speaker={S_SPEAKER}, Listener={NUM_LISTENER_STATES}")
print(f"✓ Joint states: {NUM_JOINT_STATES}")
print(f"✓ Joint actions: {NUM_JOINT_ACTIONS}")

wandb.config.update({
    "num_speaker_states": S_SPEAKER,
    "num_listener_states": NUM_LISTENER_STATES,
    "num_joint_states": NUM_JOINT_STATES,
    "num_actions_speaker": A_SPEAKER,
    "num_actions_listener": A_LISTENER,
    "num_joint_actions": NUM_JOINT_ACTIONS,
})

# ============= JOINT STATE ENCODING =============
def encode_joint_state(s_speaker, s_listener):
    return s_speaker * NUM_LISTENER_STATES + s_listener

def decode_joint_action(joint_action):
    a_speaker = joint_action // A_LISTENER
    a_listener = joint_action % A_LISTENER
    return a_speaker, a_listener

def encode_joint_action(a_speaker, a_listener):
    return a_speaker * A_LISTENER + a_listener

# ============= Q-NETWORK =============
class JointQNetwork(nn.Module):
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
        embedded = self.state_embedding(state_indices)
        return self.net(embedded)

# ============= REPLAY BUFFER =============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample with priority to good rewards"""
        
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            # ✅ Prioritize rewards > 0.3 (normalized scale)
            good = [t for t in self.buffer if t[2] > 0.3]
            rest = [t for t in self.buffer if t[2] <= 0.3]
            
            # 70% good, 30% rest
            n_good = min(int(batch_size * 0.7), len(good))
            n_rest = batch_size - n_good
            
            batch = []
            if good and len(good) >= n_good:
                batch.extend(random.sample(good, n_good))
            if rest and len(rest) >= n_rest:
                batch.extend(random.sample(rest, n_rest))
            
            # Fill if needed
            while len(batch) < batch_size and self.buffer:
                batch.append(random.choice(self.buffer))
        
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

# ============= DQN AGENT WITH SOFTMAX =============
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
        
        # ✅ TEMPERATURE instead of epsilon
        self.temperature = TEMPERATURE_START
        self.steps = 0
        
        wandb.watch(self.q_network, log=None)
        
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })
        
        print(f"✓ Q-Network: {total_params:,} total params ({trainable_params:,} trainable)")
    
    def select_action(self, state, eval_mode=False):
        """
        ✅ SOFTMAX exploration instead of epsilon-greedy
        
        state: joint state index
        Returns: joint action index
        """
        with torch.no_grad():
            state_t = torch.LongTensor([state]).to(device)
            q_values = self.q_network(state_t).cpu().numpy()[0]
            
            if eval_mode:
                # Evaluation: use softmax with low temperature
                temperature_eval = 0.1
                q_scaled = q_values / temperature_eval
            else:
                # Training: use current temperature
                q_scaled = q_values / self.temperature
            
            # Softmax
            q_scaled = q_scaled - q_scaled.max()  # Numerical stability
            probs = np.exp(q_scaled)
            probs = probs / probs.sum()
            
            # Sample from distribution
            action = np.random.choice(self.num_actions, p=probs)
            
            return action
    
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
            next_q_values_online = self.q_network(next_states_t)
            next_actions = next_q_values_online.argmax(dim=1)
            
            next_q_values_target = self.target_network(next_states_t)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
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
        #if self.steps % TARGET_UPDATE_FREQ == 0:
        #    self.target_network.load_state_dict(self.q_network.state_dict())
        TAU = 0.005  # Soft update rate
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        return loss.item()
    
    def update_temperature(self, episode):
        """✅ Temperature decay (like epsilon decay)"""
        if episode < TEMPERATURE_DECAY_EPISODES:
            self.temperature = TEMPERATURE_START - (TEMPERATURE_START - TEMPERATURE_END) * episode / TEMPERATURE_DECAY_EPISODES
        else:
            self.temperature = TEMPERATURE_END

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
    
    return avg_reward

# ============= TRAINING =============
print("\n🚀 Training Joint-Action DQN with Softmax...")

agent = JointDQNAgent(NUM_JOINT_STATES, NUM_JOINT_ACTIONS)

# Load baselines
try:
    from utils import calc_exploitability_true
    
    pi_s = np.load('expert_policy_speaker_bins6.npy')
    pi_l = np.load('expert_policy_listener_bins6.npy')
    P = np.load('P_bins6.npy').astype(np.float64)
    R = np.load('R_bins6.npy').astype(np.float64)
    init_dist = np.load('expert_initial_dist_bins6.npy')
    
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
eval_episodes = []
all_losses = []

training_start_time = time.time()
episode_rewards = []
episode_lengths = []

for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Episodes"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    
    while not done:
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        joint_state = encode_joint_state(s_spk, s_lst)
        
        joint_action = agent.select_action(joint_state)
        
        a_spk, a_lst = decode_joint_action(joint_action)
        
        next_obs, rewards, terms, truncs, _ = env.step({
            "speaker_0": a_spk,
            "listener_0": a_lst
        })
        
        done = any(terms.values()) or any(truncs.values())
        
        # ✅ REWARD NORMALIZATION [0, 1]
        raw_reward = rewards["speaker_0"]
        reward = ((raw_reward + 100.0) / 100.0) ** 0.7

        
        episode_reward += raw_reward  # Track original for logging
        episode_steps += 1
        
        next_joint_state = encode_joint_state(next_obs["speaker_0"], next_obs["listener_0"])
        
        agent.replay_buffer.push(joint_state, joint_action, reward, next_joint_state, done)
        
        loss = agent.update()
        if loss is not None:
            all_losses.append(loss)
        
        obs = next_obs
    
    # ✅ Update temperature
    agent.update_temperature(episode)
    
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_steps)
    
    # Log training metrics
    if episode % 50 == 0:
        avg_reward_10 = np.mean(episode_rewards[-10:])
        avg_length_10 = np.mean(episode_lengths[-10:])
        avg_loss_100 = np.mean(all_losses[-100:]) if all_losses else 0
        
        wandb.log({
            "train/episode": episode,
            "train/reward": episode_reward,
            "train/reward_ma10": avg_reward_10,
            "train/episode_length": episode_steps,
            "train/episode_length_ma10": avg_length_10,
            "train/temperature": agent.temperature,  # ← Instead of epsilon
            "train/buffer_size": len(agent.replay_buffer),
            "train/loss": avg_loss_100,
            "train/steps_total": agent.steps,
        }, step=episode)
    
    # Evaluation
    if episode % EVAL_EVERY == 0:
        eval_start = time.time()
        avg_reward = evaluate(agent)
        eval_time = time.time() - eval_start
        
        eval_rewards.append(avg_reward)
        eval_episodes.append(episode)
        
        avg_loss = np.mean(all_losses[-100:]) if all_losses else 0
        elapsed = time.time() - training_start_time
        
        print(f"\nEp {episode} | Reward: {avg_reward:.2f} | T: {agent.temperature:.3f} | Loss: {avg_loss:.4f}")
        
        wandb.log({
            "eval/episode": episode,
            "eval/reward": avg_reward,
            "eval/eval_time_seconds": eval_time,
            "time/elapsed_minutes": elapsed / 60,
            "time/episodes_per_minute": episode / (elapsed / 60),
        }, step=episode)

env.close()

# ============= EXTRACT SOFTMAX POLICIES =============
print("\n💾 Extracting SOFTMAX policies from Q-network...")

pi_speaker = np.zeros((NUM_JOINT_STATES, A_SPEAKER))
pi_listener = np.zeros((NUM_JOINT_STATES, A_LISTENER))

# ✅ Use softmax with low temperature for policy extraction
EXTRACTION_TEMPERATURE = 0.5

with torch.no_grad():
    batch_size_extract = 1000
    for start_idx in range(0, NUM_JOINT_STATES, batch_size_extract):
        end_idx = min(start_idx + batch_size_extract, NUM_JOINT_STATES)
        states_batch = torch.arange(start_idx, end_idx, dtype=torch.long).to(device)
        
        q_values = agent.q_network(states_batch).cpu().numpy()
        
        # Softmax for each state
        for i, s in enumerate(range(start_idx, end_idx)):
            q = q_values[i]
            q_scaled = q / EXTRACTION_TEMPERATURE
            q_scaled = q_scaled - q_scaled.max()
            probs = np.exp(q_scaled)
            probs = probs / probs.sum()
            
            # Marginalize
            for joint_action in range(NUM_JOINT_ACTIONS):
                a_spk, a_lst = decode_joint_action(joint_action)
                pi_speaker[s, a_spk] += probs[joint_action]
                pi_listener[s, a_lst] += probs[joint_action]

# Normalize
pi_speaker = pi_speaker / pi_speaker.sum(axis=1, keepdims=True)
pi_listener = pi_listener / pi_listener.sum(axis=1, keepdims=True)

np.save('dqn_policy_speaker.npy', pi_speaker)
np.save('dqn_policy_listener.npy', pi_listener)

print("   ✓ Policies saved: dqn_policy_speaker.npy, dqn_policy_listener.npy")
print("   ℹ️  Run 'python calculate_dqn_gap.py' to compute exploitability")

# ============= FINAL RESULTS =============
total_time = time.time() - training_start_time

print(f"\n✅ Training Complete ({total_time/60:.1f} minutes)")

wandb.run.summary.update({
    "final/total_time_minutes": total_time / 60,
    "final/total_episodes": NUM_EPISODES,
    "final/total_steps": agent.steps,
    "final/avg_reward_last_100": np.mean(episode_rewards[-100:]),
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
    'losses': all_losses,
    'episode_rewards': episode_rewards,
    'episode_lengths': episode_lengths,
}

with open('joint_dqn_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("   ✓ Model saved: joint_dqn_model.pth")
print("   ✓ Results saved: joint_dqn_results.pkl")

artifact = wandb.Artifact(
    name=f'joint-dqn-softmax-{wandb.run.id}',
    type='model',
    description='Joint-Action DQN with Softmax exploration'
)

artifact.add_file('joint_dqn_model.pth')
artifact.add_file('joint_dqn_results.pkl')
artifact.add_file('dqn_policy_speaker.npy')
artifact.add_file('dqn_policy_listener.npy')

wandb.log_artifact(artifact)
print("   ✓ Artifact logged to W&B")

# ============= VISUALIZATION =============
print("\n📊 Creating visualizations...")

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

# Temperature decay
temperature_curve = []
for ep in range(1, NUM_EPISODES + 1):
    if ep < TEMPERATURE_DECAY_EPISODES:
        temp = TEMPERATURE_START - (TEMPERATURE_START - TEMPERATURE_END) * ep / TEMPERATURE_DECAY_EPISODES
    else:
        temp = TEMPERATURE_END
    temperature_curve.append(temp)

axes[1, 1].plot(temperature_curve)
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Temperature')
axes[1, 1].set_title('Softmax Temperature Decay')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_dqn_training.png', dpi=150, bbox_inches='tight')
wandb.log({"plots/training_curves": wandb.Image('joint_dqn_training.png')})
plt.close()

print("   ✓ Plot saved: joint_dqn_training.png")

print("\n" + "="*70)
print("🎉 Training Complete!")
print("="*70)
print(f"📊 View results at: {wandb.run.get_url()}")
print(f"📝 Next: Run 'python calculate_dqn_gap.py' to compute Nash gap")
print("="*70)

wandb.finish()
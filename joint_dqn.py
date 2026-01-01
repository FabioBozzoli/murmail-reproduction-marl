"""
baseline_joint_dqn.py - Joint-Action Deep Q-Network

APPROCCIO CORRETTO per questo task:
- Stato globale discretizzato (s_speaker, s_listener)
- Q-network unica: Q(s_global, a_speaker, a_listener)
- No decomposizione (no QMIX bias)
- Action space: 3×5 = 15 joint actions

Più semplice, più efficace, nessun bias di decomposizione.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from tqdm import tqdm

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
# Use TRUE Nash exploitability with LP
from true_exploitability import calc_true_exploitability_from_q_network

print("="*70)
print("🎯 JOINT-ACTION DQN")
print("="*70)

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES = 20000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 0.0005
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000  # Steps (increased for stability)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_EPISODES = 15000
EVAL_EVERY = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ Joint Q-network (no decomposition)")
print(f"✓ LR: {LR}, Gamma: {GAMMA}")
print(f"✓ Action space: 3×5 = 15 joint actions")
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
        
        # Embedding per joint state
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        
        # Q-network (2 hidden layers, smaller)
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
            
            # Encode joint state
            joint_state = encode_joint_state(s_spk, s_lst)
            
            # Select joint action
            joint_action = agent.select_action(joint_state, eval_mode=True)
            
            # Decode to individual actions
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
        import traceback
        traceback.print_exc()
        gap = None
    
    return avg_reward, gap

# ============= TRAINING =============
print("\n🚀 Training Joint-Action DQN...")

NUM_JOINT_STATES = S_SPEAKER * NUM_LISTENER_STATES
agent = JointDQNAgent(NUM_JOINT_STATES, NUM_JOINT_ACTIONS)

eval_rewards = []
eval_gaps = []
eval_episodes = []
all_losses = []

for episode in tqdm(range(1, NUM_EPISODES + 1), desc="Episodes"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
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
        # Use both agents' rewards (cooperative task, should be same)
        reward = rewards["speaker_0"] + rewards["listener_0"]
        episode_reward += reward
        
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
    
    # Evaluation
    if episode % EVAL_EVERY == 0:
        avg_reward, gap = evaluate(agent)
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        eval_episodes.append(episode)
        
        gap_str = f"{gap:.6f}" if gap else "N/A"
        avg_loss = np.mean(all_losses[-100:]) if all_losses else 0
        
        print(f"\nEp {episode} | Reward: {avg_reward:.2f} | Gap: {gap_str} | ε: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}")

env.close()

print(f"\n✅ Training Complete")
if len(eval_gaps) > 0 and eval_gaps[-1]:
    print(f"   Final gap: {eval_gaps[-1]:.6f}")
    best_gap = min([g for g in eval_gaps if g])
    best_ep = eval_episodes[eval_gaps.index(best_gap)]
    print(f"   Best gap: {best_gap:.6f} (episode {best_ep})")

# Save
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
    'losses': all_losses
}

with open('joint_dqn_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("💾 Model and results saved")
print("="*70)
"""
baseline_dqn_curriculum.py - DQN con Curriculum Learning

Fix: Instead of random other agent, use partially trained policy.
Questo simula meglio un setting cooperativo.

Tempo: 40-50 minuti
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import time
from collections import deque
from tqdm import tqdm

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true

print("="*70)
print("🧠 DQN WITH CURRICULUM LEARNING")
print("="*70)

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES_PHASE1 = 5000   # Train speaker with expert listener
NUM_EPISODES_PHASE2 = 10000  # Train listener with trained speaker
NUM_EPISODES_PHASE3 = 5000   # Joint refinement
LEARNING_RATE = 0.001
GAMMA = 0.9
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9998       # Slower decay
TARGET_UPDATE = 500
EVAL_EVERY = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✓ Device: {device}")
print(f"✓ Phase 1: Train speaker ({NUM_EPISODES_PHASE1} eps)")
print(f"✓ Phase 2: Train listener ({NUM_EPISODES_PHASE2} eps)")
print(f"✓ Phase 3: Joint training ({NUM_EPISODES_PHASE3} eps)")
print(f"✓ Total: {NUM_EPISODES_PHASE1 + NUM_EPISODES_PHASE2 + NUM_EPISODES_PHASE3} episodes")
print(f"✓ Estimated time: 40-50 minutes")
print("="*70)

# ============= NEURAL NETWORK =============
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.net(x.float())

# ============= REPLAY BUFFER =============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============= DQN AGENT =============
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        
    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_oh = np.zeros(self.state_dim)
                state_oh[state] = 1.0
                state_tensor = torch.FloatTensor(state_oh).unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, batch_size, gamma):
        if len(self.buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        
        q_values = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states_t).max(1)[0]
            target_q_values = rewards_t + gamma * next_q_values * (1 - dones_t)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ============= LOAD EXPERT POLICIES =============
print("\n📂 Loading expert policies...")
try:
    pi_s_expert = np.load('expert_policy_speaker_bins6.npy')
    pi_l_expert = np.load('expert_policy_listener_bins6.npy')
    print("✓ Expert policies loaded")
    has_expert = True
except:
    print("⚠️  Expert policies not found, using heuristic")
    has_expert = False

def expert_action(state, policy):
    """Sample action from expert policy."""
    if policy is None:
        return np.random.randint(policy.shape[1])
    probs = policy[state]
    return np.random.choice(len(probs), p=probs)

# ============= ENVIRONMENT =============
print("\n🎮 Creating environment...")

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

# Joint state index
def get_joint_idx(s_spk, s_lst):
    return s_spk * NUM_LISTENER_STATES + s_lst

print(f"✓ Environment ready")

# ============= AGENTS =============
speaker_agent = DQNAgent(S_SPEAKER, A_SPEAKER, LEARNING_RATE)
listener_agent = DQNAgent(NUM_LISTENER_STATES, A_LISTENER, LEARNING_RATE)

# ============= EVALUATION =============
def evaluate_agents(speaker, listener, env, n_episodes=50):
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            s_spk = obs["speaker_0"]
            s_lst = obs["listener_0"]
            
            a_spk = speaker.select_action(s_spk, epsilon=0.0)
            a_lst = listener.select_action(s_lst, epsilon=0.0)
            
            actions = {"speaker_0": a_spk, "listener_0": a_lst}
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            episode_reward += rewards["speaker_0"]
            done = any(terms.values()) or any(truncs.values())
        
        total_reward += episode_reward
    
    avg_reward = total_reward / n_episodes
    
    # Compute gap
    S_joint = S_SPEAKER * NUM_LISTENER_STATES
    pi_speaker = np.zeros((S_SPEAKER, A_SPEAKER))
    pi_listener = np.zeros((NUM_LISTENER_STATES, A_LISTENER))
    
    for s in range(S_SPEAKER):
        a = speaker.select_action(s, epsilon=0.0)
        pi_speaker[s, a] = 1.0
    
    for s in range(NUM_LISTENER_STATES):
        a = listener.select_action(s, epsilon=0.0)
        pi_listener[s, a] = 1.0
    
    pi_speaker_joint = np.zeros((S_joint, A_SPEAKER))
    pi_listener_joint = np.zeros((S_joint, A_LISTENER))
    
    for s_spk in range(S_SPEAKER):
        for s_lst in range(NUM_LISTENER_STATES):
            s_j = s_spk * NUM_LISTENER_STATES + s_lst
            pi_speaker_joint[s_j] = pi_speaker[s_spk]
            pi_listener_joint[s_j] = pi_listener[s_lst]
    
    try:
        P = np.load('P_bins6.npy').astype(np.float64)
        R = np.load('R_bins6.npy').astype(np.float64)
        init_dist = np.load('expert_initial_dist_bins6.npy').astype(np.float64)
        gap = calc_exploitability_true(pi_speaker_joint, pi_listener_joint, R, P, init_dist, GAMMA)
    except:
        gap = None
    
    return avg_reward, gap

# ============= TRAINING =============
print("\n🚀 Training...")

eval_episodes = []
eval_gaps = []
eval_rewards = []
all_rewards = []

epsilon = EPSILON_START
episode_count = 0
start_time = time.time()

# ========== PHASE 1: Train Speaker with Expert Listener ==========
print("\n📍 PHASE 1: Training Speaker...")

for episode in tqdm(range(NUM_EPISODES_PHASE1), desc="Phase 1"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        
        # Speaker learns, listener uses expert
        a_spk = speaker_agent.select_action(s_spk, epsilon)
        
        if has_expert:
            s_j = get_joint_idx(s_spk, s_lst)
            a_lst = expert_action(s_j, pi_l_expert)
        else:
            # Simple heuristic: move towards goal
            a_lst = np.random.randint(A_LISTENER)
        
        actions = {"speaker_0": a_spk, "listener_0": a_lst}
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        
        ns_spk = next_obs["speaker_0"]
        done = any(terms.values()) or any(truncs.values())
        
        # Normalize reward to [0, 1]
        r_spk = (rewards["speaker_0"] + 100) / 100  # Rough normalization
        r_spk = np.clip(r_spk, 0, 1)
        
        s_spk_oh = np.zeros(S_SPEAKER)
        s_spk_oh[s_spk] = 1.0
        ns_spk_oh = np.zeros(S_SPEAKER)
        ns_spk_oh[ns_spk] = 1.0
        
        speaker_agent.buffer.push(s_spk_oh, a_spk, r_spk, ns_spk_oh, done)
        speaker_agent.update(BATCH_SIZE, GAMMA)
        
        obs = next_obs
        episode_reward += rewards["speaker_0"]
    
    all_rewards.append(episode_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_count += 1
    
    if episode % TARGET_UPDATE == 0:
        speaker_agent.update_target()
    
    if (episode + 1) % EVAL_EVERY == 0:
        # For phase 1, use expert listener
        print(f"\n   Episode {episode+1} | Avg reward: {np.mean(all_rewards[-100:]):.2f} | ε: {epsilon:.3f}")

print("✓ Speaker trained")

# ========== PHASE 2: Train Listener with Trained Speaker ==========
print("\n📍 PHASE 2: Training Listener...")

epsilon = EPSILON_START  # Reset epsilon

for episode in tqdm(range(NUM_EPISODES_PHASE2), desc="Phase 2"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        
        # Speaker uses learned policy, listener learns
        a_spk = speaker_agent.select_action(s_spk, epsilon=0.1)  # Small exploration
        a_lst = listener_agent.select_action(s_lst, epsilon)
        
        actions = {"speaker_0": a_spk, "listener_0": a_lst}
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        
        ns_lst = next_obs["listener_0"]
        done = any(terms.values()) or any(truncs.values())
        
        r_lst = (rewards["listener_0"] + 100) / 100
        r_lst = np.clip(r_lst, 0, 1)
        
        s_lst_oh = np.zeros(NUM_LISTENER_STATES)
        s_lst_oh[s_lst] = 1.0
        ns_lst_oh = np.zeros(NUM_LISTENER_STATES)
        ns_lst_oh[ns_lst] = 1.0
        
        listener_agent.buffer.push(s_lst_oh, a_lst, r_lst, ns_lst_oh, done)
        listener_agent.update(BATCH_SIZE, GAMMA)
        
        obs = next_obs
        episode_reward += rewards["speaker_0"]
    
    all_rewards.append(episode_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_count += 1
    
    if episode % TARGET_UPDATE == 0:
        listener_agent.update_target()
    
    if (episode + 1) % EVAL_EVERY == 0:
        avg_reward, gap = evaluate_agents(speaker_agent, listener_agent, env)
        eval_episodes.append(episode_count)
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        
        gap_str = f"{gap:.6f}" if gap is not None else "N/A"
        print(f"\n   Episode {episode_count} | Reward: {avg_reward:.3f} | Gap: {gap_str} | ε: {epsilon:.3f}")

print("✓ Listener trained")

# ========== PHASE 3: Joint Refinement ==========
print("\n📍 PHASE 3: Joint Training...")

epsilon = 0.1  # Low exploration

for episode in tqdm(range(NUM_EPISODES_PHASE3), desc="Phase 3"):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        
        a_spk = speaker_agent.select_action(s_spk, epsilon)
        a_lst = listener_agent.select_action(s_lst, epsilon)
        
        actions = {"speaker_0": a_spk, "listener_0": a_lst}
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        
        ns_spk = next_obs["speaker_0"]
        ns_lst = next_obs["listener_0"]
        done = any(terms.values()) or any(truncs.values())
        
        r_spk = (rewards["speaker_0"] + 100) / 100
        r_lst = (rewards["listener_0"] + 100) / 100
        r_spk = np.clip(r_spk, 0, 1)
        r_lst = np.clip(r_lst, 0, 1)
        
        s_spk_oh = np.zeros(S_SPEAKER)
        s_spk_oh[s_spk] = 1.0
        ns_spk_oh = np.zeros(S_SPEAKER)
        ns_spk_oh[ns_spk] = 1.0
        
        s_lst_oh = np.zeros(NUM_LISTENER_STATES)
        s_lst_oh[s_lst] = 1.0
        ns_lst_oh = np.zeros(NUM_LISTENER_STATES)
        ns_lst_oh[ns_lst] = 1.0
        
        speaker_agent.buffer.push(s_spk_oh, a_spk, r_spk, ns_spk_oh, done)
        listener_agent.buffer.push(s_lst_oh, a_lst, r_lst, ns_lst_oh, done)
        
        speaker_agent.update(BATCH_SIZE, GAMMA)
        listener_agent.update(BATCH_SIZE, GAMMA)
        
        obs = next_obs
        episode_reward += rewards["speaker_0"]
    
    all_rewards.append(episode_reward)
    epsilon = max(0.05, epsilon * 0.999)
    episode_count += 1
    
    if episode % TARGET_UPDATE == 0:
        speaker_agent.update_target()
        listener_agent.update_target()
    
    if (episode + 1) % EVAL_EVERY == 0:
        avg_reward, gap = evaluate_agents(speaker_agent, listener_agent, env)
        eval_episodes.append(episode_count)
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        
        gap_str = f"{gap:.6f}" if gap is not None else "N/A"
        print(f"\n   Episode {episode_count} | Reward: {avg_reward:.3f} | Gap: {gap_str} | ε: {epsilon:.3f}")

env.close()

total_time = time.time() - start_time

print(f"\n✅ Training Complete in {total_time/60:.1f} minutes")
if len(eval_rewards) > 0:
    print(f"   Final reward: {eval_rewards[-1]:.3f}")
if len(eval_gaps) > 0 and eval_gaps[-1] is not None:
    print(f"   Final gap: {eval_gaps[-1]:.6f}")

# ============= COMPARISON =============
print("\n📊 COMPARISON WITH MURMAIL:")

try:
    with open('murmail_results_bins6.pkl', 'rb') as f:
        murmail_results = pickle.load(f)
    
    dqn_samples = episode_count * 20
    
    print("\n   MURMAIL:")
    print(f"      Samples: {murmail_results['queries'][-1]}")
    print(f"      Final gap: {murmail_results['exploit'][-1]:.6f}")
    
    print("\n   DQN (Curriculum):")
    print(f"      Episodes: {episode_count}")
    print(f"      Samples: ~{dqn_samples}")
    print(f"      Final gap: {eval_gaps[-1]:.6f}")
    
    print(f"\n   Sample Efficiency:")
    print(f"      DQN uses {dqn_samples/murmail_results['queries'][-1]:.1f}x more samples")
    print(f"      MURMAIL gap is {eval_gaps[-1]/murmail_results['exploit'][-1]:.1f}x better")
    
except Exception as e:
    print(f"   Could not load MURMAIL: {e}")

# ============= PLOT =============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1
ax = axes[0, 0]
window = 100
if len(all_rewards) >= window:
    smoothed = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
    ax.plot(smoothed)
ax.axvline(x=NUM_EPISODES_PHASE1, color='r', linestyle='--', label='Phase 1→2')
ax.axvline(x=NUM_EPISODES_PHASE1+NUM_EPISODES_PHASE2, color='g', linestyle='--', label='Phase 2→3')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('DQN Learning (3 Phases)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2
ax = axes[0, 1]
if len(eval_gaps) > 0:
    ax.plot(eval_episodes, eval_gaps, 'o-', markersize=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Nash Gap')
    ax.set_title('DQN Convergence')
    ax.grid(True, alpha=0.3)

# Plot 3
ax = axes[1, 0]
try:
    dqn_samples_arr = np.array(eval_episodes) * 20
    ax.plot(dqn_samples_arr, eval_gaps, 'o-', label='DQN (Curriculum)', markersize=4)
    ax.plot(murmail_results['queries'], murmail_results['exploit'], 's-', label='MURMAIL', markersize=4)
    ax.axhline(y=murmail_results['config']['expert_gap'], color='red', linestyle='--', label='Expert')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Nash Gap')
    ax.set_title('DQN vs MURMAIL')
    ax.legend()
    ax.grid(True, alpha=0.3)
except:
    pass

# Plot 4
ax = axes[1, 1]
try:
    methods = ['Expert', 'MURMAIL', 'DQN\n(Curriculum)']
    gaps = [murmail_results['config']['expert_gap'], murmail_results['exploit'][-1], eval_gaps[-1]]
    colors = ['red', 'green', 'blue']
    bars = ax.bar(methods, gaps, color=colors, alpha=0.7)
    ax.set_ylabel('Nash Gap')
    ax.set_title('Final Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
except:
    pass

plt.tight_layout()
plt.savefig('dqn_curriculum_vs_murmail.png', dpi=150)
print("\n📊 Plot saved")

# Save
results = {
    'eval_episodes': eval_episodes,
    'eval_gaps': eval_gaps,
    'eval_rewards': eval_rewards,
    'all_rewards': all_rewards,
    'config': {
        'total_episodes': episode_count,
        'phase1': NUM_EPISODES_PHASE1,
        'phase2': NUM_EPISODES_PHASE2,
        'phase3': NUM_EPISODES_PHASE3
    }
}

with open('dqn_curriculum_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("💾 Results saved")
print("\n" + "="*70)
print("✅ DQN CURRICULUM COMPLETE!")
print("="*70)
"""
baseline_iql.py - Independent Q-Learning Baseline

Confronto con MURMAIL:
- Usa stesso ambiente (bins=6)
- Training indipendente speaker + listener
- Metriche: gap, reward, sample efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pickle
import time

from pettingzoo.mpe import simple_speaker_listener_v4
from discrete_wrapper import DiscretizedSpeakerListenerWrapper
from utils import calc_exploitability_true

# ============= CONFIG =============
DISCRETIZATION_BINS = 6
NUM_EPISODES = 50000        # ~1M samples (20 steps × 50k eps)
LEARNING_RATE = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995

EVAL_EVERY = 2000           # Episodes
TARGET_UPDATE = 500

print("="*70)
print("🤖 INDEPENDENT Q-LEARNING BASELINE")
print("="*70)
print(f"✓ Environment: Speaker-Listener, bins={DISCRETIZATION_BINS}")
print(f"✓ Episodes: {NUM_EPISODES}")
print(f"✓ Estimated time: 60-90 minutes")
print("="*70)

# ============= ENVIRONMENT =============
raw_env = simple_speaker_listener_v4.parallel_env(
    continuous_actions=False,
    render_mode=None,
    max_cycles=25
)
env = DiscretizedSpeakerListenerWrapper(raw_env, bins=DISCRETIZATION_BINS)

NUM_LISTENER_STATES = 27 * (DISCRETIZATION_BINS ** 2)
S_SPEAKER = 3  # goal states
S_LISTENER = NUM_LISTENER_STATES
A_SPEAKER = 3
A_LISTENER = 5

print(f"\n📊 State/Action Space:")
print(f"   Speaker: {S_SPEAKER} states, {A_SPEAKER} actions")
print(f"   Listener: {S_LISTENER} states, {A_LISTENER} actions")

# ============= Q-TABLES =============
# Separate Q-tables for each agent
Q_speaker = np.zeros((S_SPEAKER, A_SPEAKER))
Q_listener = np.zeros((S_LISTENER, A_LISTENER))

# Target networks for stability
Q_speaker_target = Q_speaker.copy()
Q_listener_target = Q_listener.copy()

# ============= EPSILON-GREEDY =============
def select_action(state, Q_table, epsilon, n_actions):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q_table[state])

# ============= Q-LEARNING UPDATE =============
def update_q_table(Q, Q_target, state, action, reward, next_state, done, lr, gamma):
    """Standard Q-learning update."""
    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q_target[next_state])
    
    Q[state, action] += lr * (target - Q[state, action])

# ============= EVALUATION =============
def evaluate_policy(env, Q_speaker, Q_listener, n_episodes=100):
    """Evaluate learned policy and compute Nash gap."""
    total_reward = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            s_spk = obs["speaker_0"]
            s_lst = obs["listener_0"]
            
            # Greedy actions
            a_spk = np.argmax(Q_speaker[s_spk])
            a_lst = np.argmax(Q_listener[s_lst])
            
            actions = {"speaker_0": a_spk, "listener_0": a_lst}
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            episode_reward += rewards["speaker_0"]
            done = any(terms.values()) or any(truncs.values())
        
        total_reward += episode_reward
    
    avg_reward = total_reward / n_episodes
    
    # Convert Q-tables to policies
    pi_speaker = np.zeros((S_SPEAKER, A_SPEAKER))
    pi_listener = np.zeros((S_LISTENER, A_LISTENER))
    
    for s in range(S_SPEAKER):
        best_a = np.argmax(Q_speaker[s])
        pi_speaker[s, best_a] = 1.0
    
    for s in range(S_LISTENER):
        best_a = np.argmax(Q_listener[s])
        pi_listener[s, best_a] = 1.0
    
    # Expand speaker policy to joint state space
    S_joint = S_SPEAKER * S_LISTENER
    pi_speaker_joint = np.zeros((S_joint, A_SPEAKER))
    pi_listener_joint = np.zeros((S_joint, A_LISTENER))
    
    for s_spk in range(S_SPEAKER):
        for s_lst in range(S_LISTENER):
            s_joint = s_spk * S_LISTENER + s_lst
            pi_speaker_joint[s_joint] = pi_speaker[s_spk]
            pi_listener_joint[s_joint] = pi_listener[s_lst]
    
    # Compute Nash gap (requires P, R)
    try:
        P = np.load('P_bins6.npy').astype(np.float64)
        R = np.load('R_bins6.npy').astype(np.float64)
        init_dist = np.load('expert_initial_dist_bins6.npy').astype(np.float64)
        
        gap = calc_exploitability_true(pi_speaker_joint, pi_listener_joint, R, P, init_dist, GAMMA)
    except:
        gap = None
    
    return avg_reward, gap

# ============= TRAINING =============
print("\n🚀 Starting Training...")

epsilon = EPSILON_START
episode_rewards = []
eval_episodes = []
eval_gaps = []
eval_rewards = []

start_time = time.time()

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    
    while not done and steps < 50:
        s_spk = obs["speaker_0"]
        s_lst = obs["listener_0"]
        
        # Select actions
        a_spk = select_action(s_spk, Q_speaker, epsilon, A_SPEAKER)
        a_lst = select_action(s_lst, Q_listener, epsilon, A_LISTENER)
        
        actions = {"speaker_0": a_spk, "listener_0": a_lst}
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        
        ns_spk = next_obs["speaker_0"]
        ns_lst = next_obs["listener_0"]
        
        done = any(terms.values()) or any(truncs.values())
        
        # Update Q-tables independently
        update_q_table(Q_speaker, Q_speaker_target, s_spk, a_spk, 
                      rewards["speaker_0"], ns_spk, done, LEARNING_RATE, GAMMA)
        
        update_q_table(Q_listener, Q_listener_target, s_lst, a_lst,
                      rewards["listener_0"], ns_lst, done, LEARNING_RATE, GAMMA)
        
        obs = next_obs
        episode_reward += rewards["speaker_0"]
        steps += 1
    
    episode_rewards.append(episode_reward)
    
    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    # Update target networks
    if episode % TARGET_UPDATE == 0:
        Q_speaker_target = Q_speaker.copy()
        Q_listener_target = Q_listener.copy()
    
    # Evaluation
    if (episode + 1) % EVAL_EVERY == 0:
        avg_reward, gap = evaluate_policy(env, Q_speaker, Q_listener)
        eval_episodes.append(episode + 1)
        eval_rewards.append(avg_reward)
        eval_gaps.append(gap)
        
        elapsed = time.time() - start_time
        print(f"Episode {episode+1:6d} | Reward: {avg_reward:6.3f} | Gap: {gap:.6f if gap else 'N/A'} | ε: {epsilon:.3f} | Time: {elapsed/60:.1f}m")

env.close()

total_time = time.time() - start_time

print(f"\n✅ Training Complete in {total_time/60:.1f} minutes")
print(f"   Final reward: {eval_rewards[-1]:.3f}")
print(f"   Final gap: {eval_gaps[-1]:.6f}")

# ============= COMPARISON WITH MURMAIL =============
print("\n📊 COMPARISON WITH MURMAIL:")

# Load MURMAIL results
try:
    with open('murmail_results_bins6.pkl', 'rb') as f:
        murmail_results = pickle.load(f)
    
    print("\n   MURMAIL:")
    print(f"      Queries: {murmail_results['queries'][-1]}")
    print(f"      Final gap: {murmail_results['exploit'][-1]:.6f}")
    print(f"      Initial gap: {murmail_results['exploit'][0]:.6f}")
    print(f"      Improvement: {murmail_results['exploit'][0] - murmail_results['exploit'][-1]:.6f}")
    
    print("\n   IQL:")
    print(f"      Episodes: {NUM_EPISODES}")
    print(f"      Samples: ~{NUM_EPISODES * 20}")
    print(f"      Final gap: {eval_gaps[-1]:.6f}")
    print(f"      Initial gap: {eval_gaps[0]:.6f}")
    print(f"      Improvement: {eval_gaps[0] - eval_gaps[-1]:.6f}")
    
    # Sample efficiency comparison
    murmail_samples = murmail_results['queries'][-1]
    iql_samples = NUM_EPISODES * 20
    
    print(f"\n   Sample Efficiency:")
    print(f"      MURMAIL: {murmail_samples} samples → gap {murmail_results['exploit'][-1]:.6f}")
    print(f"      IQL: {iql_samples} samples → gap {eval_gaps[-1]:.6f}")
    print(f"      IQL uses {iql_samples/murmail_samples:.1f}x more samples")
    
except Exception as e:
    print(f"   Could not load MURMAIL results: {e}")

# ============= PLOT =============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curve (reward)
ax = axes[0, 0]
window = 100
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax.plot(smoothed)
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward')
ax.set_title('IQL Learning Curve')
ax.grid(True, alpha=0.3)

# Plot 2: Nash gap over time
ax = axes[0, 1]
ax.plot(eval_episodes, eval_gaps, 'o-', markersize=4)
ax.set_xlabel('Episode')
ax.set_ylabel('Nash Gap')
ax.set_title('IQL Convergence (Gap)')
ax.grid(True, alpha=0.3)

# Plot 3: Comparison with MURMAIL (gap)
ax = axes[1, 0]
try:
    # Convert episodes to samples for fair comparison
    iql_samples = np.array(eval_episodes) * 20
    ax.plot(iql_samples, eval_gaps, 'o-', label='IQL', markersize=4)
    ax.plot(murmail_results['queries'], murmail_results['exploit'], 's-', label='MURMAIL', markersize=4)
    ax.set_xlabel('Environment Samples')
    ax.set_ylabel('Nash Gap')
    ax.set_title('IQL vs MURMAIL (Sample Efficiency)')
    ax.legend()
    ax.grid(True, alpha=0.3)
except:
    ax.text(0.5, 0.5, 'MURMAIL data not available', ha='center', va='center')

# Plot 4: Final comparison
ax = axes[1, 1]
try:
    methods = ['MURMAIL', 'IQL']
    final_gaps = [murmail_results['exploit'][-1], eval_gaps[-1]]
    colors = ['green', 'blue']
    ax.bar(methods, final_gaps, color=colors, alpha=0.7)
    ax.set_ylabel('Final Nash Gap')
    ax.set_title('Final Performance Comparison')
    ax.axhline(y=murmail_results['config']['expert_gap'], color='red', linestyle='--', label='Expert')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
except:
    pass

plt.tight_layout()
plt.savefig('iql_vs_murmail_comparison.png', dpi=150)
print("\n📊 Plot saved: iql_vs_murmail_comparison.png")

# ============= SAVE RESULTS =============
results = {
    'eval_episodes': eval_episodes,
    'eval_gaps': eval_gaps,
    'eval_rewards': eval_rewards,
    'episode_rewards': episode_rewards,
    'Q_speaker': Q_speaker,
    'Q_listener': Q_listener,
    'config': {
        'num_episodes': NUM_EPISODES,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'bins': DISCRETIZATION_BINS
    }
}

with open('iql_results_bins6.pkl', 'wb') as f:
    pickle.dump(results, f)

print("💾 Results saved: iql_results_bins6.pkl")
print("\n" + "="*70)
print("✅ BASELINE COMPLETE!")
print("="*70)
"""
test_training_validity.py - Verifica se training ha senso
"""

import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4
from src.discrete_wrapper import DiscretizedSpeakerListenerWrapper

# Crea env ESATTAMENTE come nel training
raw_env = simple_speaker_listener_v4.parallel_env(
    continuous_actions=False,
    render_mode=None,
    max_cycles=25
)

env = DiscretizedSpeakerListenerWrapper(raw_env, bins=6)

print("="*70)
print("VERIFICA TRAINING VALIDITY")
print("="*70)

# Reset e guarda cosa succede SENZA wrapper interference
obs, _ = env.reset(seed=42)

print("\n📊 Initial discrete observations:")
print(f"Speaker: {obs['speaker_0']}")
print(f"Listener: {obs['listener_0']}")

# Accedi all'env RAW dentro il wrapper
raw_parallel = env.env
world = raw_parallel.aec_env.env.world

print("\n📍 Positions in continuous space:")
for agent in world.agents:
    print(f"{agent.name}: {agent.state.p_pos}")

print("\n🎯 Landmarks:")
for i, lm in enumerate(world.landmarks):
    print(f"Landmark {i}: {lm.state.p_pos}")

# Ora STEP con azioni discrete
print("\n🎮 STEP with discrete actions:")
print("Speaker action: 0 (message 0)")
print("Listener action: 4 (UP in discrete)")

# Store positions
pos_before = world.agents[1].state.p_pos.copy()  # Listener

obs_new, rewards, terms, truncs, _ = env.step({
    "speaker_0": 0,
    "listener_0": 4
})

pos_after = world.agents[1].state.p_pos.copy()

print(f"\n📍 Listener position:")
print(f"Before:  {pos_before}")
print(f"After:   {pos_after}")
print(f"Delta:   {pos_after - pos_before}")
print(f"Moved:   {not np.allclose(pos_before, pos_after)}")

print(f"\n💰 Reward: {rewards['listener_0']:.2f}")

print("\n📊 New discrete observations:")
print(f"Speaker: {obs_new['speaker_0']}")
print(f"Listener: {obs_new['listener_0']}")

env.close()
"""
Run MURMAIL algorithm on Speaker-Listener environment.
FIXED: Loads sparse dynamics from pickle and sanitizes inputs.
"""

import numpy as np
from murmail import MaxUncertaintyResponseImitationLearning
from innerloop_rl import UCBVI
from utils import calc_exploitability_true
import matplotlib.pyplot as plt
import pickle
import os
from fast_loader import load_expert_and_dynamics

def run_murmail_experiment(num_iterations=1000, gamma=0.9, learning_rate=10.0):
    """Run MURMAIL algorithm."""
    print("🚀 Running MURMAIL Algorithm...")
    
    expert_speaker, expert_listener, transitions, rewards, initial_dist = load_expert_and_dynamics()
    
    num_states = expert_speaker.shape[0]
    num_actions = expert_speaker.shape[1]
    
    # Initialize inner loop RL algorithm
    # Riduciamo horizon e episodes per velocità, dato che lo spazio è 2048 stati
    inner_algo = UCBVI(
        num_episodes=20,   
        horizon=10         
    )
    
    game_params = {
        'num_states': num_states,
        'num_actions_p1': num_actions,
        'num_actions_p2': num_actions,
        'num_actions': num_actions
    }
    
    # Initialize MURMAIL
    murmail = MaxUncertaintyResponseImitationLearning(
        num_iterations=num_iterations,
        transitions=transitions,
        expert_policies=(expert_speaker, expert_listener),
        innerloop_algo=inner_algo,
        learning_rate=learning_rate,
        gamma=gamma,
        eval_freq=50,  # Evaluate every 50 iterations
        true_rewards=rewards,
        initial_dist=initial_dist,
        game_params=game_params,
        rollout_length=100,
        expert_samples=10
    )
    
    # Run algorithm
    queries, exploitability, policy_speaker, policy_listener = murmail.run()
    
    # Final evaluation
    final_exploit = calc_exploitability_true(
        policy_speaker, policy_listener,
        rewards, transitions, initial_dist, gamma
    )
    
    print(f"\n✅ MURMAIL Results:")
    print(f"  - Final exploitability: {final_exploit:.4f}")
    print(f"  - Total iterations: {num_iterations}")
    
    return {
        'queries': queries,
        'exploitability': exploitability,
        'final_policies': (policy_speaker, policy_listener),
        'final_exploit': final_exploit
    }

def plot_murmail_results(results, save_path='murmail_results.png'):
    """Plot MURMAIL learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['queries'], results['exploitability'],
             label='MURMAIL', color='blue', linewidth=2)
    plt.xlabel('Iterations / Queries', fontsize=12)
    plt.ylabel('Nash Gap (Exploitability)', fontsize=12)
    plt.title('MURMAIL on Speaker-Listener', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Saved plot to {save_path}")

def main():
    """Run MURMAIL experiments with hyperparameter tuning."""
    
    # Grid search over learning rates
    learning_rates = [1.0, 5.0, 10.0] # Rimosso 20.0 per risparmiare tempo
    best_result = None
    best_lr = None
    best_exploit = float('inf')
    
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Testing learning_rate = {lr}")
        print(f"{'='*60}")
        
        try:
            results = run_murmail_experiment(
                num_iterations=500, # Ridotto a 500 per test veloce
                gamma=0.9,
                learning_rate=lr
            )
            
            if results['final_exploit'] < best_exploit:
                best_exploit = results['final_exploit']
                best_lr = lr
                best_result = results
        
        except Exception as e:
            print(f"❌ Failed with lr={lr}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🏆 Best learning rate: {best_lr} (exploitability: {best_exploit:.4f})")
    
    # Save best results
    if best_result:
        plot_murmail_results(best_result)
        
        np.savez('murmail_results.npz',
                 queries=best_result['queries'],
                 exploitability=best_result['exploitability'],
                 final_exploit=best_result['final_exploit'],
                 best_lr=best_lr)
        
        print("\n💾 Saved results to murmail_results.npz")

if __name__ == "__main__":
    main()
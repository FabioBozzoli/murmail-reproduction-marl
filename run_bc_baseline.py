"""
Run Behavioral Cloning baseline on Speaker-Listener environment.
FIXED: Loads sparse dynamics from pickle and reconstructs matrices.
"""

import numpy as np
from behavior_cloning import MultiAgentBehaviorCloning
from utils import calc_exploitability_true
import matplotlib.pyplot as plt
import pickle
import os
from fast_loader import load_expert_and_dynamics

def run_bc_experiment(total_samples=10000, gamma=0.9):
    """Run BC with varying dataset sizes."""
    print("📊 Running Behavioral Cloning Baseline...")
    
    expert_speaker, expert_listener, transitions, rewards, initial_dist = load_expert_and_dynamics()
    
    bc = MultiAgentBehaviorCloning(
        expert_policies=(expert_speaker, expert_listener),
        total_samples=total_samples,
        transition=transitions,
        payoff_matrix=rewards,
        gamma=gamma,
        initial_state_dist=initial_dist
    )
    
    # Train BC
    print("🚀 Training started...")
    policy_speaker, policy_listener, iterations, exploitability = bc.train(
        eval_interval=100 
    )
    
    # Final evaluation
    final_exploit = calc_exploitability_true(
        policy_speaker, policy_listener, 
        rewards, transitions, initial_dist, gamma
    )
    
    print(f"\n✅ BC Results:")
    print(f"  - Final exploitability: {final_exploit:.4f}")
    print(f"  - Dataset size: {total_samples}")
    
    return {
        'iterations': iterations,
        'exploitability': exploitability,
        'final_policies': (policy_speaker, policy_listener),
        'final_exploit': final_exploit
    }

def plot_bc_results(results, save_path='bc_results.png'):
    """Plot BC learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['iterations'], results['exploitability'], 
             label='BC', color='green', linewidth=2)
    plt.xlabel('Dataset Size', fontsize=12)
    plt.ylabel('Nash Gap (Exploitability)', fontsize=12)
    plt.title('Behavioral Cloning on Speaker-Listener', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Saved plot to {save_path}")

def main():
    try:
        results = run_bc_experiment(total_samples=10000, gamma=0.9)
        plot_bc_results(results)
        np.savez('bc_baseline_results.npz',
                 iterations=results['iterations'],
                 exploitability=results['exploitability'],
                 final_exploit=results['final_exploit'])
        print("\n💾 Saved results to bc_baseline_results.npz")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
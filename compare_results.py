"""
Compare BC and MURMAIL results side-by-side.

Generates publication-quality plots comparing both algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results():
    """Load BC and MURMAIL results."""
    bc_data = np.load('bc_baseline_results.npz')
    murmail_data = np.load('murmail_results.npz')
    
    return {
        'bc': {
            'iterations': bc_data['iterations'],
            'exploitability': bc_data['exploitability'],
            'final': bc_data['final_exploit']
        },
        'murmail': {
            'queries': murmail_data['queries'],
            'exploitability': murmail_data['exploitability'],
            'final': murmail_data['final_exploit']
        }
    }


def plot_comparison(results, save_path='comparison_bc_vs_murmail.png'):
    """
    Create side-by-side comparison plot.
    
    Args:
        results: Dict with BC and MURMAIL data
        save_path: Output path for figure
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot BC
    ax.plot(results['bc']['iterations'], 
            results['bc']['exploitability'],
            label='Behavioral Cloning',
            color='#2ecc71',  # Green
            linewidth=2.5,
            marker='o',
            markersize=5,
            markevery=5)
    
    # Plot MURMAIL
    ax.plot(results['murmail']['queries'],
            results['murmail']['exploitability'],
            label='MURMAIL',
            color='#3498db',  # Blue
            linewidth=2.5,
            marker='s',
            markersize=5,
            markevery=5)
    
    # Formatting
    ax.set_xlabel('Queries / Dataset Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nash Gap (Exploitability)', fontsize=14, fontweight='bold')
    ax.set_title('Multi-Agent Imitation Learning: BC vs MURMAIL\n' + 
                 'Speaker-Listener Environment',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    # Add final values as text annotations
    bc_final = results['bc']['final']
    murmail_final = results['murmail']['final']
    
    ax.text(0.02, 0.98, 
            f"BC Final: {bc_final:.4f}\n"
            f"MURMAIL Final: {murmail_final:.4f}\n"
            f"Improvement: {((bc_final - murmail_final)/bc_final * 100):.1f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comparison plot to {save_path}")


def print_summary(results):
    """Print numerical comparison summary."""
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY".center(60))
    print("="*60)
    
    bc_final = results['bc']['final']
    murmail_final = results['murmail']['final']
    
    print(f"\n{'Algorithm':<20} {'Final Exploitability':<25} {'Samples/Queries'}")
    print("-" * 60)
    print(f"{'BC':<20} {bc_final:<25.6f} {results['bc']['iterations'][-1]}")
    print(f"{'MURMAIL':<20} {murmail_final:<25.6f} {results['murmail']['queries'][-1]}")
    print("-" * 60)
    
    improvement = ((bc_final - murmail_final) / bc_final) * 100
    
    if improvement > 0:
        print(f"\n✅ MURMAIL improves over BC by {improvement:.2f}%")
    else:
        print(f"\n⚠️  BC performs better by {abs(improvement):.2f}%")
    
    print("\n" + "="*60)


def plot_convergence_rate(results, save_path='convergence_rate.png'):
    """
    Plot convergence rate comparison (exploitability vs samples).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute moving average for smoother curves
    window = 3
    
    bc_smooth = np.convolve(results['bc']['exploitability'], 
                            np.ones(window)/window, mode='valid')
    bc_x = results['bc']['iterations'][:len(bc_smooth)]
    
    murmail_smooth = np.convolve(results['murmail']['exploitability'],
                                 np.ones(window)/window, mode='valid')
    murmail_x = results['murmail']['queries'][:len(murmail_smooth)]
    
    ax.plot(bc_x, bc_smooth, label='BC (smoothed)', color='#2ecc71', alpha=0.7, linewidth=2)
    ax.plot(murmail_x, murmail_smooth, label='MURMAIL (smoothed)', color='#3498db', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Samples/Queries', fontsize=12)
    ax.set_ylabel('Nash Gap', fontsize=12)
    ax.set_title('Convergence Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Saved convergence plot to {save_path}")


def main():
    """Generate all comparison plots and statistics."""
    try:
        results = load_results()
    except FileNotFoundError as e:
        print(f"❌ Error: Missing results files. Run BC and MURMAIL first.")
        print(f"   {e}")
        return
    
    # Generate plots
    plot_comparison(results)
    plot_convergence_rate(results)
    
    # Print summary
    print_summary(results)
    
    # Save LaTeX table
    with open('results_table.tex', 'w') as f:
        bc_final = results['bc']['final']
        murmail_final = results['murmail']['final']
        
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Algorithm & Final Exploitability & Queries \\\\\n")
        f.write("\\midrule\n")
        f.write(f"BC & {bc_final:.4f} & {results['bc']['iterations'][-1]} \\\\\n")
        f.write(f"MURMAIL & {murmail_final:.4f} & {results['murmail']['queries'][-1]} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    print("\n💾 Saved LaTeX table to results_table.tex")


if __name__ == "__main__":
    main()
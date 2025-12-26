"""
Shared Utilities for Multi-Agent Imitation Learning Algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_exploitability_true( mu_pi: np.ndarray, nu_pi: np.ndarray, reward: np.ndarray, transition: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
        """
        Compute exact exploitability of joint policy under true rewards.
        
        Args:
            mu_pi (np.ndarray): Player μ policy (maximizer), shape (S, A1)
            nu_pi (np.ndarray): Player ν policy (minimizer), shape (S, A2)
            
        Returns:
            float: Exploitability value. Lower values indicate better policies.
                  Zero exploitability corresponds to Nash equilibrium.
        """
        # 1) Compute V^{μ,ν} by solving the full zero‐sum game via simple policy eval
        V_joint = policy_value_zero_sum(mu_pi, nu_pi, transition=transition, reward=reward, initial_dist=initial_dist, gamma=gamma)

        # 2) Build single‐agent MDP for μ as decision‐maker:
        #    R_μ(s,a) = E_{b∼ν_pi(s)}[ r(s,a,b) ]
        R_mu = (reward * nu_pi[:, None, :]).sum(axis=2)  # shape (S, A1)
         #    P_mu[s,a,s'] = ∑_b ν_pi(b|s) P[s,a,b,s']
        P_mu = (transition * nu_pi[:, None, :, None]).sum(axis=2)  # shape (S, A1, S')
        # best‐response value for μ:
        V_br_mu = value_iteration(R_mu, P_mu, initial_dist=initial_dist, gamma=gamma)

        # 3) ν's induced MDP (with negated rewards):
        #    R_nu[s,b] = - E_{a∼μ_pi(s)}[ r(s,a,b) ]
        R_nu = - (reward * mu_pi[:, :, None]).sum(axis=1)  # (S, A2)
        #    P_nu[s,b,s'] = ∑_a μ_pi(a|s) P[s,a,b,s']
        P_nu = (transition * mu_pi[:, :, None, None]).sum(axis=1)  # (S, A2, S')
        V_br_nu = value_iteration(R_nu, P_nu, initial_dist=initial_dist, gamma=gamma)

        return float(max(V_br_mu - V_joint, V_br_nu - V_joint))

def policy_value_zero_sum( mu_pi: np.ndarray, nu_pi: np.ndarray, transition: np.ndarray, reward: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
    """
    Evaluate the *joint* value V^{mu, \nu} = E[ ∑ gamma^t r(s_t,a_t,b_t) ] 
    by solving the linear system (I - gammaP_π)^-1 r_π and averaging over
    initial uniform state.
    """
    # Build 1-step expected reward per state
    r_s = (reward * mu_pi[:, :, None] * nu_pi[:, None, :]).sum(axis=(1,2))
    # Build joint transition matrix P_π[s,s'] = ∑_{a,b} μ(a|s) ν(b|s) P[s,a,b,s']
    P_joint = (transition * mu_pi[:, :, None, None] * nu_pi[:, None, :, None]).sum(axis=(1,2))
    identity_matrix = np.eye(transition.shape[0])  # Avoid ambiguous variable name
    V = np.linalg.solve(identity_matrix - gamma * P_joint, r_s)

    return float(initial_dist @ V)

def value_iteration( R: np.ndarray, P: np.ndarray, initial_dist: np.ndarray,
                        tol: float = 1e-6, max_iter: int = 10_000, gamma:float=0.9) -> float:
    """
    Standard value iteration for single-agent MDP:
        R: shape (S, A)
        P: shape (S, A, S')
    Returns optimal state-value averaged under uniform start.
    """
    S, A = R.shape
    V = np.zeros(S)
    for _ in range(max_iter):
        # Q(s,a) = R[s,a] + gamma ∑ P[s,a,s'] V[s']
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return float(initial_dist @ V)


def plot_results(all_queries_murmail: list[np.ndarray], all_exploits_murmail: list[np.ndarray], all_queries_bc: list[np.ndarray], all_exploits_bc: list[np.ndarray], name: str):
    """
    Plot and save results comparing MURMAIL and Behavior Cloning (BC).
    
    Args:
        all_queries_murmail (list of np.ndarray): List of query counts per run for MURMAIL.
        all_exploits_murmail (list of np.ndarray): List of exploitability values per run for MURMAIL.
        all_queries_bc (list of np.ndarray): List of query counts per run for BC.
        all_exploits_bc (list of np.ndarray): List of exploitability values per run for BC.
        name (str): Filename to save the results CSV.
    """
    # Convert to numpy arrays
    queries_murmail = np.array(all_queries_murmail)
    exploits_murmail = np.array(all_exploits_murmail)

    queries_bc = np.array(all_queries_bc)
    exploits_bc = np.array(all_exploits_bc)

    # Compute means and standard deviations
    mean_queries_murmail = np.mean(queries_murmail, axis=0)
    mean_exploits_murmail = np.mean(exploits_murmail, axis=0)
    std_exploits_murmail = np.std(exploits_murmail, axis=0)

    mean_queries_bc = np.mean(queries_bc, axis=0)
    mean_exploits_bc = np.mean(exploits_bc, axis=0)
    std_exploits_bc = np.std(exploits_bc, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mean_queries_murmail, mean_exploits_murmail, label="MURMAIL", color='blue')
    plt.fill_between(mean_queries_murmail,
                    mean_exploits_murmail - std_exploits_murmail,
                    mean_exploits_murmail + std_exploits_murmail,
                    color='blue', alpha=0.2)

    plt.plot(mean_queries_bc, mean_exploits_bc, label="BC", color='green')
    plt.fill_between(mean_queries_bc,
                    mean_exploits_bc - std_exploits_bc,
                    mean_exploits_bc + std_exploits_bc,
                    color='green', alpha=0.2)

    plt.xlabel("Queries / Iterations")
    plt.ylabel("Exploitability")
    plt.title("BC vs. MURMAIL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mean_exploits_bc = np.insert(mean_exploits_bc, 0,mean_exploits_murmail[0])
    std_exploits_bc = np.insert(std_exploits_bc,0,0)

    # Create a DataFrame for the results
    data_murmail = {
        'Iterations': mean_queries_murmail,
        'MURMAIL_Exploits_Mean': mean_exploits_murmail,
        'MURMAIL_Exploits_Std': std_exploits_murmail,
        'BC_Exploits_Mean': mean_exploits_bc,
        'BC_Exploits_Std': std_exploits_bc
    }


    # Create DataFrame and save to CSV
    results_df_1 = pd.DataFrame(data_murmail)
    results_df_1.to_csv(f'murmail_vs_bc_results_{name}.csv', index=False)

    print(f"Results saved to murmail_vs_bc_results_{name}.csv")





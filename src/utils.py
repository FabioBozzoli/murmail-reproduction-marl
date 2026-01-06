"""
Shared Utilities for Multi-Agent Imitation Learning Algorithms
Fixed for Asymmetric Games (A_mu != A_nu)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_exploitability_true(mu_pi: np.ndarray, nu_pi: np.ndarray, reward: np.ndarray, 
                            transition: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
    """
    Compute exact exploitability of joint policy under true rewards.
    FIXED: Now handles asymmetric action spaces (A_mu != A_nu).
    
    Args:
        mu_pi (np.ndarray): Player μ policy (maximizer), shape (S, A_mu)
        nu_pi (np.ndarray): Player ν policy (minimizer), shape (S, A_nu)
        reward (np.ndarray): Reward tensor, shape (S, A_mu, A_nu)
        transition (np.ndarray): Transition tensor, shape (S, A_mu, A_nu, S)
        initial_dist (np.ndarray): Initial state distribution, shape (S,)
        gamma (float): Discount factor
        
    Returns:
        float: Exploitability value. Lower values indicate better policies.
              Zero exploitability corresponds to Nash equilibrium.
    """
    # 1) Compute V^{μ,ν} by solving the full zero-sum game
    V_joint = policy_value_zero_sum(mu_pi, nu_pi, transition, reward, initial_dist, gamma)

    # 2) Build single-agent MDP for μ as decision-maker:
    #    R_μ(s,a) = E_{b∼ν_pi(s)}[ r(s,a,b) ]
    S, A_mu, A_nu = reward.shape
    R_mu = np.zeros((S, A_mu))
    for s in range(S):
        for a in range(A_mu):
            for b in range(A_nu):
                R_mu[s, a] += reward[s, a, b] * nu_pi[s, b]
    
    #    P_mu[s,a,s'] = ∑_b ν_pi(b|s) P[s,a,b,s']
    P_mu = np.zeros((S, A_mu, S))
    for s in range(S):
        for a in range(A_mu):
            for b in range(A_nu):
                P_mu[s, a, :] += transition[s, a, b, :] * nu_pi[s, b]
    
    # best-response value for μ:
    V_br_mu = value_iteration(R_mu, P_mu, initial_dist, gamma)

    # 3) ν's induced MDP (with negated rewards):
    #    R_nu[s,b] = - E_{a∼μ_pi(s)}[ r(s,a,b) ]
    R_nu = np.zeros((S, A_nu))
    for s in range(S):
        for b in range(A_nu):
            for a in range(A_mu):
                R_nu[s, b] -= reward[s, a, b] * mu_pi[s, a]
    
    #    P_nu[s,b,s'] = ∑_a μ_pi(a|s) P[s,a,b,s']
    P_nu = np.zeros((S, A_nu, S))
    for s in range(S):
        for b in range(A_nu):
            for a in range(A_mu):
                P_nu[s, b, :] += transition[s, a, b, :] * mu_pi[s, a]
    
    V_br_nu = value_iteration(R_nu, P_nu, initial_dist, gamma)

    # Exploitability is the maximum gain from deviating
    return float(max(V_br_mu - V_joint, V_br_nu - V_joint, 0.0))


def policy_value_zero_sum(mu_pi: np.ndarray, nu_pi: np.ndarray, transition: np.ndarray, 
                          reward: np.ndarray, initial_dist: np.ndarray, gamma: float) -> float:
    """
    Evaluate the *joint* value V^{mu, nu} = E[ ∑ gamma^t r(s_t,a_t,b_t) ] 
    by solving the linear system (I - gammaP_π)^-1 r_π.
    
    FIXED: Now handles asymmetric action spaces correctly.
    """
    S, A_mu, A_nu = reward.shape
    
    # Build 1-step expected reward per state
    r_s = np.zeros(S)
    for s in range(S):
        for a in range(A_mu):
            for b in range(A_nu):
                r_s[s] += reward[s, a, b] * mu_pi[s, a] * nu_pi[s, b]
    
    # Build joint transition matrix P_π[s,s'] = ∑_{a,b} μ(a|s) ν(b|s) P[s,a,b,s']
    P_joint = np.zeros((S, S))
    for s in range(S):
        for a in range(A_mu):
            for b in range(A_nu):
                weight = mu_pi[s, a] * nu_pi[s, b]
                P_joint[s, :] += transition[s, a, b, :] * weight
    
    # Solve linear system: V = (I - gamma*P)^-1 @ r
    identity_matrix = np.eye(S)
    try:
        V = np.linalg.solve(identity_matrix - gamma * P_joint, r_s)
    except np.linalg.LinAlgError:
        # Fallback: fixed-point iteration
        V = np.zeros(S)
        for _ in range(1000):
            V_new = r_s + gamma * (P_joint @ V)
            if np.max(np.abs(V_new - V)) < 1e-9:
                break
            V = V_new

    return float(initial_dist @ V)


def value_iteration(R: np.ndarray, P: np.ndarray, initial_dist: np.ndarray,
                   gamma: float = 0.9, tol: float = 1e-6, max_iter: int = 10_000) -> float:
    """
    Standard value iteration for single-agent MDP:
        R: shape (S, A)
        P: shape (S, A, S')
    Returns optimal state-value averaged under initial distribution.
    """
    S, A = R.shape
    V = np.zeros(S)
    
    for iteration in range(max_iter):
        # Q(s,a) = R[s,a] + gamma ∑ P[s,a,s'] V[s']
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    
    return float(initial_dist @ V)


def verify_policy_valid(policy: np.ndarray, name: str = "Policy") -> bool:
    """
    Verify that a policy is valid (sums to 1, non-negative).
    
    Args:
        policy: Shape (S, A)
        name: Name for logging
        
    Returns:
        bool: True if valid, False otherwise
    """
    S, A = policy.shape
    
    # Check non-negativity
    if np.any(policy < -1e-6):
        print(f"⚠️  {name}: valori negativi rilevati! Min: {policy.min()}")
        return False
    
    # Check sum to 1
    row_sums = policy.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-4)[0]
        print(f"⚠️  {name}: somma non unitaria in {len(bad_rows)}/{S} stati!")
        if len(bad_rows) <= 5:
            print(f"    Esempi: {row_sums[bad_rows]}")
        else:
            print(f"    Range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")
        return False
    
    print(f"✓ {name}: Policy valida (shape {policy.shape})")
    return True


def sanitize_policy(policy: np.ndarray) -> np.ndarray:
    """Force a policy to be valid (non-negative, sums to 1)."""
    policy = np.maximum(policy, 0.0)
    row_sums = policy.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return policy / row_sums


def plot_results(all_queries_murmail: list[np.ndarray], all_exploits_murmail: list[np.ndarray], 
                all_queries_bc: list[np.ndarray], all_exploits_bc: list[np.ndarray], name: str):
    """
    Plot and save results comparing MURMAIL and Behavior Cloning (BC).
    
    Args:
        all_queries_murmail: List of query counts per run for MURMAIL.
        all_exploits_murmail: List of exploitability values per run for MURMAIL.
        all_queries_bc: List of query counts per run for BC.
        all_exploits_bc: List of exploitability values per run for BC.
        name: Filename to save the results CSV.
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

    mean_exploits_bc = np.insert(mean_exploits_bc, 0, mean_exploits_murmail[0])
    std_exploits_bc = np.insert(std_exploits_bc, 0, 0)

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
import numpy as np


class UCBVI:
    def __init__(self, num_episodes, horizon):
        """
        Args:
            num_episodes: Number of episodes to run.
            horizon: Length of each episode.
        """
        self.num_episodes = num_episodes
        self.horizon = horizon

    def run_algo(
        self,
        transitions,
        rewards,
        mdp_params,
        initial_dist=None,
        gamma=0.99
    ):
        """
        Simulates UCB-VI for a single-player MDP.
        Args:
            transitions (np.ndarray): Shape (S, A, S).
            rewards (np.ndarray): Shape (S,).
            mdp_params (dict): Contains 'num_states' (S) and 'num_actions' (A).
            initial_dist (np.ndarray, optional): Shape (S,). Probability
                                                 distribution for the starting state.
                                                 Defaults to starting at state 0.
        """
        S = mdp_params['num_states']
        A = mdp_params['num_actions']

        # --- Handle the initial state distribution ---
        if initial_dist is None:
            # Default to a deterministic start at state 0 if not provided
            start_dist = np.zeros(S)
            start_dist[0] = 1.0
        else:
            start_dist = initial_dist

        # --- Initialize empirical model statistics ---
        N_sa = np.zeros((S, A))
        N_sas = np.zeros((S, A, S))
        R_sa = np.zeros((S, A))

        V = np.zeros(S)
        total_steps = 0
        
        # --- Main Learning Loop ---
        for k in range(self.num_episodes):
            epsilon = 1e-10
            P_hat = N_sas / (N_sa[..., np.newaxis] + epsilon)
            R_hat = R_sa / (N_sa + epsilon)
            bonus = np.sqrt(np.log(max(1, total_steps + 1)) / (N_sa + epsilon))

            # --- Optimistic Planning via Value Iteration ---
            for _ in range(self.horizon):
                V_old = V.copy()
                for s in range(S):
                    expected_future_value = np.einsum('ik,k->i', P_hat[s], V_old)
                    Q_s_optimistic = R_hat[s] + bonus[s] + gamma * expected_future_value
                    V[s] = np.max(Q_s_optimistic)

            # --- Policy Execution and Model Update ---
            Q_sa_optimistic = np.zeros((S,A))
            for s in range(S):
                 expected_future_value = np.einsum('ik,k->i', P_hat[s], V)
                 Q_sa_optimistic[s,:] = R_hat[s] + bonus[s] + gamma * expected_future_value
            optimistic_policy = np.argmax(Q_sa_optimistic, axis=1)

            # --- MODIFIED LINE ---
            # Sample the starting state from the provided distribution
            current_state = np.random.choice(S, p=start_dist)
            
            for _ in range(self.horizon):
                action = optimistic_policy[current_state]
                
                reward = rewards[current_state]
                next_state = np.random.choice(S, p=transitions[current_state, action])
                
                N_sa[current_state, action] += 1
                N_sas[current_state, action, next_state] += 1
                R_sa[current_state, action] += reward
                
                current_state = next_state
                total_steps += 1

        # --- Final Policy and Q-Value Calculation ---
        V_final = np.zeros(S)
        for _ in range(self.horizon * 2):
            V_old = V_final.copy()
            for s in range(S):
                expected_future_value = np.einsum('ik,k->i', P_hat[s], V_old)
                Q_s_final = R_hat[s] + gamma * expected_future_value
                V_final[s] = np.max(Q_s_final)

        Q_final = np.zeros((S, A))
        for s in range(S):
            expected_future_value = np.einsum('ik,k->i', P_hat[s], V_final)
            Q_final[s,:] = R_hat[s] + gamma * expected_future_value

        deterministic_policy = np.argmax(Q_final, axis=1)
        policy_final = np.zeros((S, A))
        policy_final[np.arange(S), deterministic_policy] = 1.0
        
        return V_final, policy_final

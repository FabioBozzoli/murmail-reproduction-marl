# 🧠 MuRMAIL in Cooperative Speaker–Listener Games

**Stability vs Performance in Multi-Agent Reinforcement Learning**

## Overview

This project studies **MuRMAIL** (Multi-agent Regularized Model-based Imitation Learning) in a **cooperative Speaker–Listener environment**, with a specific focus on the relationship between:

* **Strategic stability** (Nash exploitability / unilateral improvement gap)
* **Empirical performance** (average episodic reward)

Rather than evaluating agents solely on reward maximization, we analyze how different learning approaches behave with respect to **equilibrium stability** in a cooperative Dec-POMDP.

---

## Environment

* **Task**: Cooperative Speaker–Listener navigation
* **Setting**: Decentralized POMDP (Dec-POMDP)
* **Reward**: Shared team reward (both agents maximize the same signal)
* **Dynamics**: Continuous environment discretized into a finite MDP
* **Episode length**: 25 steps

This is a **team game**, not zero-sum:

$$
R_{\text{speaker}} = R_{\text{listener}}
$$

---

## Compared Methods

| Method      | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| **Expert**  | Model-based joint policy computed on the discretized MDP (upper bound) |
| **MuRMAIL** | Imitation learning from the Expert with regularization                 |
| **MAPPO**   | Model-free multi-agent PPO (independent execution)                     |
| **Random**  | Uniform random baseline                                                |

---

## Evaluation Metrics

### 1. Nash Exploitability (Unilateral Improvement Gap)

We compute a **Nash-style exploitability metric** adapted to cooperative games:

$$
\text{Gap} =
V(\text{BR}_S, \pi_L) +
V(\pi_S, \text{BR}_L) -
2V(\pi_S, \pi_L)
$$

* Measures **how much a single agent can improve the team reward by deviating unilaterally**
* Gap = 0 ⇒ **stable joint policy**
* Lower is better

> In cooperative games, Nash equilibrium corresponds to a **joint policy where no agent can increase the shared reward alone**.

---

### 2. Average Episodic Reward

* **Expert & MuRMAIL**: evaluated model-based using the learned MDP
* **MAPPO & Random**: evaluated empirically in the environment
* Rewards are **raw (non-normalized)** and **negative**, as defined by the task

---

## Final Results

### 📊 Quantitative Comparison

| Method      | Nash Gap ↓ <br>*(Strategic Instability)* | Avg Reward ↑ <br>*(Empirical Performance)* |
| ----------- | :---: | :---: |
| **Expert**  | **0.009**      | -25.90     |
| **MuRMAIL** | 0.058      | -26.71     |
| MAPPO       | 0.320      | **-25.24**     |
| Random      | 0.191      | -39.55     |

### 🔍 Interpretation

The results reveal a critical insight into Multi-Agent learning dynamics:

1.  **Performance Parity:** Both the **Expert** (Game-Theoretic) and **MAPPO** (Deep RL) converge to the same performance ceiling (~ -25). This represents the physical limit of the discretized environment.
2.  **The Hidden Flaw:** Despite achieving the same reward, **MAPPO has a high Nash Gap (0.320)**, while the Expert is near zero (0.009).
3.  **MuRMAIL's Success:** MuRMAIL successfully clones the Expert's behavior, achieving **both** high reward (-26.71) and high stability (0.058), bridging the gap between RL and Game Theory.

> **Crucial Finding:** High empirical reward does **not** guarantee a Nash Equilibrium. MAPPO learns an effective but **strategically fragile** policy, while MuRMAIL inherits the **robustness** of the Expert.

---

## Visualization

The final analysis plots the relationship between theoretical stability and practical performance:

*   **X-axis**: Nash Exploitability Gap (Lower is Better/More Stable)
*   **Y-axis**: Average Reward (Higher is Better)

<p align="center">
  <img src="final_thesis_plot.png" width="600">
</p>

**Analysis of the Plot:**
*   **Expert & MAPPO** are aligned on the Y-axis (Performance).
*   However, they are far apart on the X-axis (Stability).
*   This visualizes the **"Hidden Cost" of Deep RL**: achieving the goal without ensuring equilibrium.

---

## Key Takeaways

*   **Reward is Insufficient:** Evaluating cooperative agents solely on Average Reward hides strategic vulnerabilities.
*   **Robustness via Imitation:** MuRMAIL proves that imitation learning is a viable path to inject game-theoretic stability into neural policies.
*   **Sim-to-Model Alignment:** The convergence of Expert and MAPPO rewards confirms that the discrete model ($P, R$) and the continuous environment are now correctly aligned.

---

## Reproducibility

To reproduce these results, follow the pipeline:

1.  **Generate the Expert (Nash Solver):**
    ```bash
    python generate_expert_FINAL.py
    ```
    *Generates transition matrices P, R and solves for Nash policies.*

2.  **Train Imitation Agent (MuRMAIL):**
    ```bash
    python run_murmail.py
    ```

3.  **Train Deep RL Agent (MAPPO):**
    ```bash
    python ppo.py
    ```

4.  **Run Final Comparative Analysis:**
    ```bash
    python final_analysis_cont.py
    ```

**Outputs:**
*   `final_thesis_plot.png` (Visualization)
*   `final_thesis_table.csv` (Raw Data)

---

## Conclusion

This project demonstrates that while standard Deep MARL algorithms (like MAPPO) can solve complex coordination tasks empirically, they fail to converge to a **Nash Equilibrium**, leaving them potentially vulnerable or non-robust.

**MuRMAIL** offers a solution: by imitating a derived equilibrium, we can obtain policies that are both **effective in practice** and **theoretically sound**.
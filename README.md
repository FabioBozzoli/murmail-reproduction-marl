# MURMAIL & Emergent Communication in Multi-Agent RL 🤖🗣️

This repository contains the implementation of **MURMAIL (Maximum Uncertainty Response Multi-Agent Imitation Learning)** applied to a **Speaker-Listener** environment.

The project consists of two main phases:

1. **Emergent Communication:** Training a PPO Expert to develop a discrete language protocol (Grounding: Word A -> Red, Word B -> Green, Word C -> Blue) while preventing "cheating" strategies (silence/padding abuse) via **Signal Jamming**.
2. **Imitation Learning:** Comparing **MURMAIL** against **Behavioral Cloning (BC)** in learning the Nash Equilibrium policy from the expert demonstrations.

---

## 📊 Key Results

### 1. Emergent Language (The Expert)

We successfully forced the agents to learn a **compositional discrete language**. By using **Entropy Annealing** and a custom **Signal Jamming Wrapper**, we prevented the agents from using "Padding" (silence) to communicate, achieving a perfect diagonal mapping.

*Figure 1: Confusion Matrix showing perfect disentanglement. Target Red → Word B, Target Green → Word C, Target Blue → Word A (Padding is unused).*

### 2. Imitation Learning Performance

Comparison between **MURMAIL** (ours) and **Behavioral Cloning** (baseline) in terms of Exploitability (Nash Gap).

*(Run the pipeline to generate `comparison_bc_vs_murmail.png`)*

---

## 🛠️ Installation

### Prerequisites

* Python 3.8+
* [WandB Account](https://www.google.com/search?q=https://wandb.ai/) (for logging)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MARL-Emergent-Communication.git
cd MARL-Emergent-Communication

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib seaborn gymnasium pettingzoo supersuit stable-baselines3 wandb tqdm numba

```

---

## 🚀 Usage (The Pipeline)

The entire experiment is automated via the `pipeline.sh` script. This script handles training, extraction, estimation, and comparison.

```bash
chmod +x pipeline.sh
./pipeline.sh

```

### Pipeline Steps Breakdown:

1. **Train PPO Expert (`train_expert.py`):**
Trains the Speaker-Listener pair using PPO. Uses **Entropy Annealing** (starts high, ends low) and **Signal Jamming** (replaces padding actions with noise) to force robust communication.
2. **Extract Policies (`extract_expert.py`):**
Converts the deep RL model (neural network) into a tabular policy representation by sampling trajectories.
3. **Estimate Dynamics (`estimate_dynamics.py`):**
Empirically estimates the transition matrix  and reward function  of the environment for the inner-loop solvers.
4. **Run Baselines (`run_bc_baseline.py`):**
Runs **Multi-Agent Behavioral Cloning** to establish a baseline performance.
5. **Run MURMAIL (`run_murmail.py`):**
Executes the MURMAIL algorithm. It uses **UCB-VI** in the inner loop to find uncertainty-maximizing responses.
6. **Compare Results (`compare_results.py`):**
Generates plots and a LaTeX table comparing the final exploitability of BC vs MURMAIL.

---

## 🧠 Algorithms Implemented

### MURMAIL (Maximum Uncertainty Response Multi-Agent Imitation Learning)

Implemented in `murmail.py`.

* **Core Idea:** Learns a Nash Equilibrium from expert data without assuming the expert is perfectly optimal.
* **Mechanism:** Uses an inner loop to solve an induced MDP with exploration bonuses (uncertainty), followed by a Mirror Descent update step.
* **Reward:** Synthetic reward based on expert agreement: .

### UCB-VI (Inner Loop Solver)

Implemented in `innerloop_rl.py`.

* Used within MURMAIL to solve the single-player induced MDPs efficiently.

### Multi-Agent Behavioral Cloning

Implemented in `behavior_cloning.py`.

* Standard Maximum Likelihood Estimation baseline.

---

## 📂 Project Structure

| File | Description |
| --- | --- |
| `pipeline.sh` | Main execution script |
| `murmail.py` | Implementation of the MURMAIL algorithm |
| `behavior_cloning.py` | Implementation of the BC baseline |
| `train_expert.py` | (Expected) Script to train PPO with Signal Jamming |
| `extract_expert.py` | Extracts tabular policies from PPO models |
| `estimate_dynamics.py` | Estimates env dynamics/rewards via sampling |
| `innerloop_rl.py` | UCB-VI implementation for MURMAIL's inner loop |
| `fast_loader.py` | Numba-accelerated loader for heavy matrices |
| `analyze_language.py` | Generates confusion matrices to verify language emergence |
| `compare_results.py` | Plotting and analysis tools |
| `env_wrappers.py` | Visualization script for the trained environment |

---

## 📈 Optimization

* **Numba JIT:** Used in `fast_loader.py` to accelerate the loading and processing of large transition matrices ().
* **Sparse Dynamics:** Dynamics are estimated using sparse dictionaries and converted to dense arrays only when necessary to manage memory.

---

## 👥 Credits & References

* **PettingZoo:** For the Multi-Agent Particle Environment (MPE).
* **Stable Baselines 3:** For the PPO implementation.
* **WandB:** For experiment tracking.

*Project developed for the Distributed AI course.*

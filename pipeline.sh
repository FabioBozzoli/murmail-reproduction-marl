#!/bin/bash
# Complete pipeline for MURMAIL experiments on Speaker-Listener

set -e  # Exit on error

echo "======================================"
echo "MURMAIL Pipeline for Speaker-Listener"
echo "======================================"

# Step 1: Train PPO expert (if not already trained)
if [ ! -f "models/ppo_jamming_normalized/final_model.zip" ]; then
    echo ""
    echo "[1/6] Training PPO expert..."
    python train_expert.py
else
    echo ""
    echo "[1/6] PPO expert already trained, skipping..."
fi

# Step 2: Extract expert policies
echo ""
echo "[2/6] Extracting expert policies..."
python extract_expert.py

# Step 3: Estimate dynamics
echo ""
echo "[3/6] Estimating transition dynamics (this may take a while)..."
python estimate_dynamics.py

# Step 4: Run BC baseline
echo ""
echo "[4/6] Running Behavioral Cloning baseline..."
python run_bc_baseline.py

# Step 5: Run MURMAIL
echo ""
echo "[5/6] Running MURMAIL algorithm..."
python run_murmail.py

# Step 6: Compare results
echo ""
echo "[6/6] Generating comparison plots..."
python compare_results.py

echo ""
echo "======================================"
echo "✅ Pipeline complete!"
echo "======================================"
echo ""
echo "📊 Generated files:"
echo "  - comparison_bc_vs_murmail.png"
echo "  - convergence_rate.png"
echo "  - results_table.tex"
echo ""
echo "📈 Check the plots for detailed comparison!"
#!/bin/bash
cd "D:/projects/Mechanistic-Interpretability-small"
source .venv/Scripts/activate
LOG="experiments/sae_capacity_comparison/logs/seeds_1_2_runs.log"
for seed in 1 2; do
  for w in sae_4096 sae_8192 sae_16384; do
    echo "=== START $w seed=$seed $(date) ===" >> "$LOG"
    python -u experiments/sae_capacity_comparison/scripts/train_sae.py \
      --config experiments/sae_capacity_comparison/configs/$w.json \
      --seed $seed --sparsity-coefficient 0.2 --batch-size 4096 --steps 10000 --amp \
      --log-every 250 --val-every 1000 --save-every 1000 >> "$LOG" 2>&1
    echo "=== END $w seed=$seed exit=$? $(date) ===" >> "$LOG"
  done
done
echo "ALL_SEEDS_DONE" >> "$LOG"

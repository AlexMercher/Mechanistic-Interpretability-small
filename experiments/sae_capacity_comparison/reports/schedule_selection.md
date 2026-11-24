# Training Schedule Selection (post-pilot revision)

The original pilot (`pilot_report.md`) picked λ=0.2 from a 50K-row slice and assumed a
~69-epoch (15,000-step) budget by analogy to the pilot slice's convergence -- but that
analogy ("~61 effective epochs looked converged on 50K rows") is not proof the full
895,502-row training set needs the same epoch count; repeated epochs over the same data
can keep "improving" without that improvement being meaningful. This report replaces
that assumption with a measured plateau analysis and adds a throughput benchmark, per
explicit follow-up guidance before committing to a 9-run (3 width x 3 seed) matrix.

## 1. AMP benchmark (`scripts/benchmark_training.py`, `reports/benchmark_report.json`)

Measured with `torch.cuda.synchronize()`-bounded timing, 20-30 untimed warm-up steps
excluded, batch size 4096, on the real (non-pilot) training tensor:

| Width | FP32 ms/step | AMP (fp16 autocast + GradScaler) ms/step | Speedup | Peak VRAM (AMP) |
|---:|---:|---:|---:|---:|
| 4096 | 110.6 | 53.2 | 2.08x | 0.64 GB |
| 8192 | 230.9 | 96.7 | 2.39x | 1.07 GB |
| 16384 | 363.5 | 177.3 | 2.05x | 2.01 GB |

Loss after the same number of steps was within ~1% between FP32 and AMP at every width
(e.g. 8192: 65.61 FP32 vs 65.85 AMP after 200 steps) -- AMP does not change training
behavior meaningfully here, consistent with PyTorch's documented AMP use case. **Decision:
AMP adopted for all real training runs.**

A secondary fix (unrelated to precision) was also applied to `train_sae.py`: the original
loop called `.cpu()` on a `dict_size`-sized tensor every single step (for dead-feature
tracking) and computed `grad_norm` every step -- both force a host/device synchronization
that serializes the pipeline. Dead-feature tracking now stays GPU-resident (only
materialized to Python at `log_every`), and the NaN/Inf check now runs every 20 steps
(cheap single-scalar sync) instead of every step, with the full diagnostic block still
at `log_every`. This keeps per-step instability detection cheap and frequent without
reintroducing the removed per-step stall.

## 2. Plateau analysis (representative run: SAE-8192, seed 0, AMP, 10,000 steps)

Validation computed every 1,000 steps (full 97,994-row validation split):

| Step | Val recon MSE | Val mean L0 | Dead features (train-time, 2000-step window) |
|---:|---:|---:|---:|
| 1,000 | 4.906 | 205.0 | 0 |
| 2,000 | 3.661 | 91.5 | 0 |
| 3,000 | 3.047 | 69.0 | 1,599 |
| 4,000 | 2.710 | 60.0 | 3,410 |
| 5,000 | 2.463 | 55.0 | 3,525 |
| 6,000 | 2.276 | 51.0 | 3,534 (peak) |
| 7,000 | 2.111 | 47.8 | 3,518 |
| 8,000 | 1.982 | 44.7 | 3,495 |
| 9,000 | 1.867 | 42.0 | 3,476 |
| 9,999 | 1.765 | 40.0 | 3,438 |

Observations:
- Val recon MSE keeps improving through 10,000 steps -- no hard plateau -- but the
  *relative* improvement per 1,000 steps shrinks steadily and stabilizes around ~5.5-6%
  per 1,000 steps by step 7,000-9,999 (vs. 16-25% per 1,000 steps in the first 3,000
  steps). This is the diminishing-returns region, not a true asymptote.
- Mean L0 settles into the task's target band (20-50 active features/token) by step
  ~6,000-7,000 and stays there (51 -> 40 by the end), not still collapsing.
- Dead-feature count is non-monotonic: it climbs fast early (0 -> 3,534 by step 6,000,
  i.e. ~43% of the dictionary), then *declines slightly* for the rest of the run
  (3,534 -> 3,438) -- some features that looked dead under the 2,000-step rolling window
  reactivate later. This is reported as an observed training dynamic, not something
  papered over.

## 3. Decision: 10,000-step shared budget (down from 15,000)

10,000 steps (≈45.7 epochs over the 895,502-row train set, vs. the original 69-epoch
guess) is where: L0 is stably inside the target band, dead-feature growth has leveled
off (even reversed slightly), and reconstruction improvement has visibly slowed to a
small, roughly constant proportional rate rather than the steep early-training drop.
Extending further would still lower validation MSE somewhat (the curve hadn't fully
flattened), but at a shrinking rate for a proportionally larger compute cost -- a
reasonable stopping point given this is a fixed, frozen, identical budget for all three
widths, not a per-width tuned stopping criterion.

This step count, and AMP, are now frozen in `configs/common_config.json` for the SAE-4096
and SAE-16384 training runs.

## 4. Revised cost estimate and seeding plan

At the new 10,000-step/AMP budget, seed-0 wall time per width: 4096 ≈ 8.8 min, 8192 ≈
17.1 min (measured, includes validation overhead), 16384 ≈ 29.5-33 min (estimated from
clean-loop AMP throughput + proportional validation overhead). Total for seed 0 across
all three widths: well under 1 hour -- a 3-seed x 3-width matrix is now only ~2.5-3 hours,
much cheaper than the original (pre-fix, pre-AMP, 15,000-step) ~9-hour estimate.

Given the dramatically lower cost, the full 3-seed x 3-width matrix (9 runs) is run for
the stability analysis, rather than the reduced fallback (3 seeds on the 8192 baseline
only). SAE-8192 seed 0's plateau run above is reused as that width's official seed-0
result (identical config/budget/AMP setting to the other runs -- no separate rerun
needed).

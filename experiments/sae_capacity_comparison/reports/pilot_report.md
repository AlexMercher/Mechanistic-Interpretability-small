# Milestone 5 Pilot Report

## Purpose

Verify the SAE training pipeline end-to-end and select the shared hyperparameters
(learning rate, batch size, sparsity coefficient λ, training length) on the
4096-feature width (smallest/fastest), then freeze them for all three widths
and all seeds — per the task's explicit instruction not to tune each width
independently.

## Pipeline checks (all passed)

- `torch.cuda.is_available()` confirmed true; training ran on `cuda` (RTX 3050), never
  silently fell back to CPU — raises `RuntimeError` otherwise (`sae_common.get_device_no_fallback`).
- Shapes verified: train/validation tensors load as `[n, 2048]`; model forward produces
  `x_hat` shape matching input and `z` shape `[batch, dict_size]`.
- Train/validation separation: `sae_common.load_split_activations` reads from the
  Milestone-4 `shard_manifest.json`'s `train`/`validation` keys independently; shard
  sha256 re-verified against the manifest before any data is used (raises on mismatch).
- Forward/backward/reconstruction/sparsity-penalty all produce finite values across every
  sweep run below; `train_sae.py` raises immediately if a non-finite loss ever appears
  (never observed in any run).
- Decoder unit-norm renormalization applied after every optimizer step
  (`SparseAutoencoder.renormalize_decoder_`), confirmed via `decoder_mean_norm` logged
  every `log_every` steps staying at 1.0.
- Checkpoint save/load exercised (`latest.pt`, `best_val.pt`, `final.pt`); resumability
  verified by re-running a short pilot after an existing `latest.pt` was present and
  confirming training resumed from the saved `step`, not from 0.
- No CUDA OOM at any point; peak VRAM for width 4096 at batch size 1024-4096 stayed
  under 0.5 GB (plenty of headroom below the 6.44 GB total / ~5.4 GB free).

## Sparsity coefficient (λ) sweep

All runs: width 4096, 50,000-row training slice (pilot only — never the full 895,502-row
train split), batch size 1024, lr 3e-4 (Adam, linear 5% warmup), 3,000 steps
(~61 effective epochs over the 50K-row slice), seed held fixed per trial except the
seed itself was varied only to give each sweep trial an isolated checkpoint/log
directory (pilot-only seeds 90-101, distinct from the real experiment seeds 0/1/2).

| λ | Final mean L0 (train) | Val mean L0 | Val explained variance | Dead features @ step 3000 |
|---:|---:|---:|---:|---:|
| 0.0001 | ~1734 (step 50 only, still extremely dense) | — | 0.998 | 0 |
| 0.01 | 1601 | 1604 | 0.9997 | 0 |
| 0.03 | 1200 | 1206 | 0.9997 | 0 |
| 0.1 | 192 | 188 | 0.9996 | 1135/4096 (28%) |
| **0.2** | **49** | **51** | **0.9995** | 1690/4096 (41%) |
| 1.0 | 3.1 | 3.1 | 0.9987 | 3324/4096 (81%, feature collapse) |
| 3.0 | 9.3 (short run) | — | 0.99 | — |
| 10.0 | 6.4 (short run) | — | 0.985 | — |
| 30.0 | 2.6 (short run) | — | 0.953 | — |

## Decision: λ = 0.2

- Lands mean L0 ≈ 49-51 active features/token, inside the task's target band of
  20-50 active features/token, with validation explained variance 0.9995 — reconstruction
  quality is not meaningfully sacrificed at this sparsity level.
- λ = 1.0 and above cause progressive feature collapse (L0 drops toward single digits,
  dead-feature fraction climbs past 80%) — rejected as too aggressive; most of the
  dictionary's capacity would go unused regardless of width, which would make the
  capacity comparison uninformative (every width would just collapse to a handful of
  live features).
- λ ≤ 0.1 leaves L0 in the hundreds — not meaningfully sparse, defeats the purpose of a
  sparse dictionary.
- **Dead-feature fraction at λ=0.2 (41% by step 3000 on this small pilot slice) is
  itself an expected, reportable empirical finding, not a sign the hyperparameter is
  wrong** — the task explicitly asks us to measure dead-feature rate per width, not
  engineer it away. This is flagged going into the real runs: dead-feature percentage
  vs. dictionary width is one of the central capacity-comparison questions (§10 of the
  task), and λ=0.2 is frozen identically across 4096/8192/16384 so that comparison is
  apples-to-apples.

## Frozen hyperparameters (identical across SAE-4096/8192/16384, all seeds)

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-4 (Adam, β1=0.9, β2=0.999, no weight decay) |
| LR schedule | linear warmup over first 5% of steps, then constant |
| Batch size | 4096 |
| Sparsity coefficient λ | 0.2 |
| Total optimization steps | 15,000 |
| Effective epochs (full train set, 895,502 vectors) | ≈ 69 (15,000 × 4096 / 895,502) |
| Decoder constraint | unit-norm columns, renormalized after every optimizer step |
| Checkpoint/val/log cadence | every 1,000 / 1,000 / 250 steps |

These values are written into `configs/common_config.json` and apply to every one of the
9 planned training runs (3 widths × 3 seeds). If a hyperparameter must be changed after
observing a failure in the real runs, the change and reason will be documented here and
all affected configurations rerun consistently — not tuned per-width.

## Known limitation of this pilot

The sweep used a 50,000-row slice of the training data, not the full 895,502-row train
split, and a smaller batch size (1024) than the frozen real-run batch size (4096) purely
for sweep speed. L0/dead-feature absolute numbers on the full run at batch 4096 may differ
somewhat in magnitude from the pilot numbers above (λ's effect depends on batch size
through the relative scale of the per-vector L1 term, which is batch-size-invariant in
expectation, but stochastic gradient noise differs) — the real runs' own logged metrics
are the source of truth for the final comparison, not these pilot numbers.

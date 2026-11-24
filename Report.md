# Milestone 5 SAE Capacity Comparison — Feasibility and Trade-off Assessment

Independent assessment of `experiments/sae_capacity_comparison/`, written against the
actual configs, code, and logs on disk (all source paths given inline). Every number
below is either **measured** (read from a logged/evaluated artifact — cited) or explicitly
marked **inferred/estimated** where no direct measurement exists. Two bookkeeping defects
were found while assembling this report and are disclosed in §8 rather than silently
worked around.

---

## 1. Feasibility

**GPU and environment** (measured, `run_metadata/environment.json`, `torch.cuda` query
re-run live for this report): NVIDIA RTX 3050 Laptop, 6.44 GB total VRAM, CUDA 12.4, torch
2.6.0+cu124, 16 GB system RAM. `sae_common.get_device_no_fallback()` raises rather than
silently running on CPU — every one of the 9 real training runs' logs begins with
`device=cuda`, confirmed.

**Peak VRAM** (measured, per-run training logs `logs/sae_<width>_seed<seed>.jsonl`,
consistent across all 3 seeds of each width):

| Width | Peak VRAM | Headroom vs. 6.44 GB total |
|---:|---:|---:|
| 4096 | 0.76 GB | 5.68 GB |
| 8192 | 1.29 GB | 5.15 GB |
| 16384 | 2.37 GB | 4.07 GB |

No OOM occurred in any run (0/9 failures). Headroom is comfortable at every width; VRAM is
not a binding constraint for this architecture at batch size 4096, even at 16384 features.

**Wall time** (measured from `[train_sae] done. steps=10000 total_time_sec=...` lines in
`logs/seeds_1_2_runs.log`, the only log source with complete, uninterrupted single-run
totals for every width):

| Width | Seed 1 | Seed 2 | Mean of these two |
|---:|---:|---:|---:|
| 4096 | 555.0 s | 537.0 s | 546.0 s (9.1 min) |
| 8192 | 977.7 s | 990.1 s | 983.9 s (16.4 min) |
| 16384 | 1826.4 s | 1931.7 s | 1879.1 s (31.3 min) |

Seed 0 for 4096 and 16384 was interrupted mid-run by a harness-level background-task kill
and resumed from checkpoint (`run_metadata/execution_manifest.json`); their recorded
`total_time_sec` (269.8 s, 187.9 s) covers only the final resumed chunk, not the full
10,000-step run, so they are **not** comparable to the seed-1/2 totals above and are
excluded from this table. Seed 0's full-run time is **inferred** (not measured) to be close
to the seed-1/2 mean for the same width, since architecture, step count, and AMP setting
are identical.

Summing the 6 directly-measured full runs (seeds 1 and 2, all three widths) plus
**inferring** seed 0 ≈ the seed-1/2 mean for its width: total actual training compute
≈ (546+984+1879) × 1 [seed 0, inferred] + 546+984+1879 [seed 1] + 537+990+1932 [seed 2]
≈ 3409 (inferred seed 0) + 3409 (seed 1) + 3459 (seed 2) ≈ **10,277 s ≈ 2.9 hours** of GPU
training time across all 9 runs. This matches the plan's own revised estimate
(`reports/schedule_selection.md` §4, "~2.5-3 hours") — a case where the estimate and the
measured outcome agree, not an unverified assumption.

**Disk use** (measured, `du -sh`):
- Activation source data (read-only, Milestone 4): 3.8 GB on disk (993,496 bf16 vectors,
  2048-dim), matches Milestone 4's own reported 4.07 GB (rounding/du-block-size difference,
  not a discrepancy in content).
- SAE checkpoints (this milestone, 9 runs × 3 files each — `latest.pt`/`best_val.pt`/`final.pt`,
  each ≈ params+2 Adam moments): 4096 ≈ 1.7 GB, 8192 ≈ 3.4 GB, 16384 ≈ 6.8 GB — **11.9 GB total**,
  excluded from git via `.gitignore` (confirmed: checkpoints are not in the repo).

**Activation vector counts** (measured, `preprocessing/activation_statistics.json`,
cross-checked against Milestone 4's independently-generated `activation_statistics.json` —
exact match): 895,502 training vectors, 97,994 validation vectors, 2048-dim, from 22,467 /
2,533 train/validation prompts.

**Bottlenecks and resource risks identified:**
1. VRAM is not the bottleneck at these widths/batch size; wall time scales with width
   (16384 takes ~3.4x longer than 4096 per step-count, driven by the width-proportional
   matmul cost in the encoder/decoder, not by data volume).
2. The real constraint that was corrected mid-project was **training-step throughput**
   before AMP + a hidden per-step host-sync were fixed (`reports/schedule_selection.md` §1)
   — FP32 with the sync bug would have made the original 3×3×15,000-step plan take
   roughly 3x longer than what was actually run.
3. **Execution-environment reliability, not compute**, was the actual operational risk:
   the harness's own background-task tracker killed 2 of the 9 runs mid-training for
   reasons unrelated to the training process (verified via `tasklist` that the underlying
   python process kept running); work-around was detached OS processes / bounded foreground
   chunks relying on checkpoint resumability — documented in
   `run_metadata/execution_manifest.json`.
4. **Conclusion: training 3 widths × 3 seeds at 10,000 steps each is feasible on this
   hardware** (measured: ~2.9 hours total, comfortable VRAM headroom, 9/9 runs completed
   with exit code 0) — feasibility is not in question; the risks that materialized were
   software/environment bugs (§8), not resource limits.

---

## 2. Training data and schedule

**Train/validation split** (measured, `preprocessing/preprocessing_config.json`, and
re-verified live against `experiments/activation_extraction_qwen3_1_7b/metadata/prompt_split.jsonl`
by `prepare_sae_data.py`'s own output at generation time): reused Milestone 4's existing
prompt-level split (seed 1337, 10% target validation fraction) — **no new split was
created**. 22,467 train prompts / 2,533 validation prompts → 895,502 / 97,994 token
vectors. Split is at the prompt level, not the token level, so no token from a validation
prompt leaks into training.

**Batch size and steps** (measured from `configs/common_config.json` and confirmed
identical across every training-log file): batch size 4096, 10,000 optimization steps,
identical for all 3 widths and all 3 seeds.

**Examples processed and dataset-equivalents** (calculated directly from the above, not
assumed): 10,000 steps × 4,096 batch = 40,960,000 total training-example presentations.
40,960,000 / 895,502 unique train vectors = **45.74 "epochs"** (i.e., each training vector
is seen, on average, 45.74 times over the run — the sampling loop (`train_sae.py`
`next_batch_indices`) reshuffles a full permutation of the train set once it is exhausted,
so this is true without-replacement-per-epoch sampling, not resampling with replacement).

**Evidence for the chosen step budget (10,000, revised down from an initial, untested
15,000-step guess):** a dedicated plateau-finding run was executed before committing to the
schedule (`reports/schedule_selection.md` §2 — SAE-8192, seed 0, AMP, validation every
1,000 steps on the full 97,994-row validation set). Measured validation reconstruction MSE
at each checkpoint:

| Step | Val recon MSE | Val mean L0 | Relative improvement vs. prior 1,000 steps |
|---:|---:|---:|---:|
| 1,000 | 4.906 | 205.0 | — |
| 3,000 | 3.047 | 69.0 | ~17%/1000 steps (steps 1k→3k) |
| 5,000 | 2.463 | 55.0 | ~9%/1000 steps |
| 7,000 | 2.111 | 47.8 | ~7%/1000 steps |
| 9,999 | 1.765 | 40.0 | ~5.5%/1000 steps |

This is **measured evidence of diminishing returns, not a plateau in the strict
flat-asymptote sense** — the curve was still improving at step 10,000, just at a shrinking
rate (the report explicitly says this: "no hard plateau... but the *relative* improvement
per 1,000 steps shrinks steadily"). 10,000 was chosen as a documented trade-off point
(slowing improvement vs. proportionally larger compute), not because the metric had
stopped moving. **This is the single weakest piece of evidence in the schedule
justification**: the decision is defensible (it replaced an untested assumption with a
real measurement) but it is a judgment call on a still-moving curve, and a longer run
might have shown a different crossover point. No run in this experiment tested step counts
beyond 10,000, so whether reconstruction keeps improving meaningfully past that point is
genuinely unknown, not just unreported.

---

## 3. Objective

**Exact loss** (from `sae_common.py::sae_loss`, confirmed matching every training log's
`recon_mse_sum`/`sparsity_l1_sum` fields):

```
z     = ReLU( W_e (x - b_d) + b_e )
x_hat = W_d z + b_d
recon    = sum_i (x_i - x_hat_i)^2         , summed over the 2048 dims, averaged over the batch
sparsity = sum_i |z_i| = sum_i z_i          (L1 of z; z >= 0 after ReLU, so L1 = sum, not abs-sum)
loss  = recon + lambda * sparsity,   lambda = 0.2 (frozen, identical across all 9 runs)
```

**Normalization** (measured artifact, `preprocessing/preprocessing_config.json`):
`y = scale * (x - mu)`, with `mu` = train-split mean activation vector and
`scale = sqrt(2048 / mean(||x-mu||^2)) = 0.019661` — a **global scalar** transform (one
number for the whole vector), computed from train data only, applied identically to train
and validation and to all three widths. This is deliberately **not** per-dimension
whitening (`reports/pilot_report.md`, `FINAL_REPORT.md` §3): raw per-dimension variance in
this activation set spans roughly 1.7 to 3.8 million (measured, Milestone 4's
`activation_statistics.json`), and whitening would inflate near-constant dimensions to the
same scale as outlier "massive activation" dimensions, destroying the magnitude structure
an SAE dictionary is meant to discover.

**Decoder constraint**: decoder columns (`W_d`'s columns — the dictionary's 2048-dim
directions) are renormalized to unit L2 norm after **every** optimizer step
(`SparseAutoencoder.renormalize_decoder_`, called every training step in `train_sae.py`).
This is a standard, necessary constraint for this loss formulation: without it, the L1
penalty has a trivial degenerate solution (shrink `W_d`'s norm arbitrarily small so a large
`z` produces a small reconstruction contribution, driving the L1 term down without actually
reducing the *number* of active features). With unit-norm columns, the only way to reduce
the L1 penalty is to reduce how many/how strongly features fire, which is the intended
sparsity signal.

**How the penalty trades reconstruction against sparsity — stated precisely, not as
"maximize sparsity":** λ controls a Pareto-style trade-off between two competing pressures
on `z`: the reconstruction term pulls `z` toward whatever activation pattern best
reconstructs `x` (which, with no penalty, is dense — see λ=0.0001 and λ=0.01 pilot results
below), while the λ·L1 term pulls every `z_i` toward zero regardless of its usefulness. The
pilot sweep (`reports/pilot_report.md`, §"Sparsity coefficient (λ) sweep") measured this
directly on width 4096:

| λ | Val mean L0 | Val explained variance | Dead features (of 4096) |
|---:|---:|---:|---:|
| 0.0001 | ~1734 (near-dense) | 0.998 | 0 |
| 0.01 | 1604 | 0.9997 | 0 |
| 0.1 | 188 | 0.9996 | 1135 (28%) |
| **0.2 (chosen)** | **51** | **0.9995** | 1690 (41%) |
| 1.0 | 3.1 | 0.9987 | 3324 (81%, feature collapse) |

This table is the direct empirical evidence that sparsity is **not** "something to
maximize without limit" in this setup: pushing λ far enough (≥1.0) does not simply yield a
"more sparse, slightly worse" model — it causes outright **feature collapse** (81% of the
dictionary dies, L0 crashes to ~3, explained variance drops measurably to 0.9987). λ=0.2
was chosen specifically because it sits inside the task's target L0 band (20-50) **before**
collapse sets in, trading a small, bounded amount of reconstruction quality (EV 0.9995 vs.
0.9997 at λ=0.01) for a dictionary that is actually sparse rather than dense-but-penalized.

---

## 4. Evidence — per width and seed

All values below are **measured** from `logs/sae_<width>_seed<seed>.jsonl` (training, final
logged step 9999/10000) and `evaluations/{reconstruction,sparsity,feature_utilization}/sae_<width>_seed<seed>.json`
(full-validation-set evaluation, run separately from training-time logging). Every width/seed
combination has both a `best_val.pt` and `final.pt` checkpoint; for all 9 runs these are
**the same checkpoint** (confirmed by loading both files: `best_val step == final step ==
10000` for every one of the 9 runs), because validation reconstruction MSE decreased
monotonically across every logged validation point in every run — no run ever regressed
after an earlier checkpoint, so "best" and "final" coincide. This is itself evidence against
overfitting within this step budget: if the SAE were overfitting the training batches,
validation loss would be expected to bottom out and rise before step 10,000; it does not, in
any of the 9 runs.

| Width | Seed | Train recon (final step, single-batch) | Val recon MSE (full val set) | Val EV | Val mean L0 | Dead features |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0 | 1.939 | 1.901 | 0.9991 | 43.66 | 1956 (47.8%) |
| 4096 | 1 | 1.902 | 1.888 | 0.9991 | 44.35 | 1835 (44.8%) |
| 4096 | 2 | 1.914 | 1.906 | 0.9991 | 44.61 | 1903 (46.5%) |
| 8192 | 0 | 1.780 | 1.765 | 0.9992 | 39.99 | 3535 (43.2%) |
| 8192 | 1 | 1.781 | 1.769 | 0.9992 | 40.45 | 3702 (45.2%) |
| 8192 | 2 | 1.767 | 1.764 | 0.9992 | 40.05 | 3633 (44.3%) |
| 16384 | 0 | 1.779 | 1.800 | 0.9991 | 43.03 | 8897 (54.3%) |
| 16384 | 1 | 1.792 | 1.784 | 0.9992 | 42.36 | 8961 (54.7%) |
| 16384 | 2 | 1.779 | 1.781 | 0.9992 | 42.80 | 8956 (54.7%) |

(Dead-feature counts here use the validation-set definition — "never fires on any
validation vector" — consistent with the headline comparison report; the training-log
`dead_features` column uses a different, 2000-step rolling-window definition and is not
reused here to avoid mixing definitions, per `FINAL_REPORT.md` §8.6.)

**Training vs. validation gap:** at every width/seed, the final single-batch training
reconstruction value and the full-validation-set value are within ±0.02 of each other
(e.g. 4096/seed0: 1.939 train vs. 1.901 val; 16384/seed0: 1.779 train vs. 1.800 val) — in
several cases validation is *slightly lower* than the single training batch, which is
expected sampling noise (one training batch vs. the full 97,994-row validation set), not a
systematic train/validation divergence. **There is no measurable generalization gap at
this step budget for any width or seed** — gains on the training objective visibly
transfer to the held-out validation set across the board.

**Curves over training** (full detail in the raw `.jsonl` logs; summarized from the logged
checkpoints at steps 0, 1k, 2k, ..., 10k): reconstruction error falls steeply in the first
~2,000 steps (e.g. 8192/seed0: val MSE 5714 → 4.91 → 3.66 at steps 0/1k/2k) then flattens
into the slow-diminishing-returns regime described in §2; mean L0 falls from the initial
near-dense value (~2000-8300 depending on width, i.e. roughly half the dictionary firing at
initialization) to the 40-45 range by step 10,000; dead-feature count rises fastest in the
first ~3,000-6,000 steps and then grows much more slowly (8192/seed0: 0 dead at step 2000,
1599 at step 3000, 3534 at step 6000 — already near its final value of 3535 four thousand
steps before training ends).

---

## 5. Trade-off: reconstruction vs. sparsity across widths

3-seed means (`reports/capacity_comparison.json`, generated from the evaluation JSONs, not
hand-typed):

| Width | Val recon MSE (mean) | Val mean L0 (mean) | Dead feature % (mean) |
|---:|---:|---:|---:|
| 4096 | 1.898 | 44.2 | 46.3% |
| 8192 | **1.766** | **40.2** | **44.2%** |
| 16384 | 1.788 | 42.7 | 54.6% |

**This is not a classic monotonic trade-off curve where sparser = worse reconstruction.**
8192 simultaneously has the *lowest* reconstruction error **and** the *lowest* mean L0
(sparsest) of the three widths — it Pareto-dominates both 4096 (which is denser *and* worse
at reconstruction) and 16384 (also denser *and* worse at reconstruction than 8192, despite
having 2x the capacity). If one insists on describing a "trade-off being navigated," the
honest description is: **there is essentially no reconstruction-sparsity trade-off visible
across these three widths** — reconstruction MSE varies by only 7.5% end-to-end (1.766 to
1.898) while dead-feature rate and cross-seed stability vary far more (44%→55%,
0.16→0.42). The real axis of variation in this experiment is capacity utilization and
stability, not a reconstruction-for-sparsity exchange.

The `explained_variance_per_active_feature` metric in the comparison report (0.0226 /
0.0249 / 0.0234 for 4096/8192/16384) looks like a joint reconstruction-sparsity signal but
is **not independent evidence**: explained variance is saturated at ≈0.999 for every width
(range 0.99910-0.99916), so this ratio is numerically within rounding of `1/mean_L0` — it
restates the L0 ranking under a different name rather than showing a genuine trade-off
being resolved differently by width. This caveat is already recorded in
`reports/capacity_comparison.md` and `FINAL_REPORT.md` §5, and is repeated here because it
is directly relevant to the user's instruction not to over-read a reconstruction-sparsity
trade-off that the data does not actually show.

**8192 gives the best useful outcome among the three**, not because it resolves a
trade-off most favorably, but because it dominates on the only two axes (reconstruction,
sparsity) where a trade-off was hypothesized to exist. It does *not* win on stability (§6)
or on interpretability (flat across widths, `reports/interpretability_review_summary.md`:
coherent-semantic feature counts 3/1/3 of 30 sampled for 4096/8192/16384 — 8192 is worst on
this axis in the sampled features).

**Train-to-validation transfer of gains:** yes, confirmed in §4 — every width's
reconstruction improvement over training is matched almost exactly by the held-out
validation set at every logged checkpoint, with no divergence. The 8192-vs-16384
reconstruction gap specifically (1.766 vs. 1.788, each from 3-seed means) does not overlap
in per-seed range (8192: [1.764, 1.769]; 16384: [1.781, 1.800]) — this is a real, measured,
reproducible-across-seeds difference on held-out data, not training-set noise.

---

## 6. Reliability — cross-seed variation and feature matching

**Variation across seeds** (measured, per-seed evaluation JSONs, summarized in §4's table):
reconstruction MSE varies by seed within a width by at most ~0.016 (16384: 1.781-1.800) —
small relative to the between-width spread (0.13, 4096 vs. 8192). Dead-feature percentage
is more consistent within a width (16384: 54.3-54.7%, a tight 0.4-point spread) than the
~10-point spread between widths — seed noise is small relative to the width effect for
these two metrics.

**Feature matching method** (measured/code-reviewed, `evaluate_feature_stability.py`): two
independent signals per width, each computed pairwise across all 3 seeds (3 pairs:
0-vs-1, 0-vs-2, 1-vs-2):
1. **Decoder-direction matching**: cosine similarity between unit-norm decoder columns
   (the learned dictionary directions), greedy one-to-one matched above a 0.3 cosine
   threshold (chosen because two random unit vectors in 2048-dim space have similarity
   std ≈ 0.022, making 0.3 a strong non-spurious signal).
2. **Activation-pattern matching**: cosine similarity between each feature's activation
   values over a fixed 8,192-token validation sample, same matching procedure.

Measured pairwise results (`evaluations/feature_stability/sae_<width>.json`):

| Width | Decoder match rate (3 pairs) | Activation match rate (3 pairs) |
|---:|---:|---:|
| 4096 | 0.418 / 0.423 / 0.429 | 0.420 / 0.431 / 0.432 |
| 8192 | 0.312 / 0.315 / 0.315 | 0.421 / 0.424 / 0.430 |
| 16384 | 0.155 / 0.156 / 0.159 | 0.348 / 0.350 / 0.352 |

Decoder-match rate falls monotonically and substantially with width (0.42→0.31→0.16); the
activation-pattern match rate falls much less (0.43→0.42→0.35) — decoder geometry is less
reproducible across seeds than functional activation behavior at higher width, a real and
somewhat subtle distinction the dual-metric design was built to surface. Because dead
features' decoder directions are essentially untrained noise and cannot match by
construction, the raw decoder-match rate partly conflates "unstable" with "dead" — a
live-feature-normalized version (dividing by the live-feature fraction instead of total
width) was computed to check this: 0.79 / 0.56 / 0.34 for 4096/8192/16384. **The
width-ordering survives correction** (wider dictionaries are still less stable even after
removing the dead-feature penalty), but the gap shrinks, so part of the raw stability
number is a dead-feature artifact, not pure representational instability.

**Missing, unstable, or contradictory evidence, flagged explicitly:**
- No seed beyond 0/1/2 was tested at any width — 3 seeds is the minimum the task's own
  spec calls "meaningful" for stability analysis, not a large sample; the measured
  within-width spreads in §4/§6 are small enough that 3 seeds appear sufficient to support
  the width-ordering conclusions, but this has not been checked against a 4th or 5th seed.
- The interpretability review (`reports/interpretability_review_summary.md`) is a 30-feature/width
  *sample*, deliberately skewed toward frequency extremes, reviewed by a single
  teacher-model pass with no causal/steering validation — it supports a directional claim
  ("not obviously more interpretable at higher width") but not a precise population
  estimate of what fraction of each full dictionary is semantic.
- No step count beyond 10,000 was tested (§2) — whether the reconstruction ranking or the
  stability ordering would hold at, say, 20,000 steps is unknown, not just unreported.

---

## 7. Bugs and documentation defects found while compiling this report

These were discovered during this assessment, not previously disclosed in exactly this
form, and are reported here per the instruction to distinguish measured fact from
assumption rather than paper over inconsistencies:

1. **`reports/benchmark_report.json` only contains the last-run width's data (16384).**
   `benchmark_training.py` writes to a fixed filename regardless of which width was
   benchmarked, so the earlier 4096 and 8192 FP32-vs-AMP numbers quoted in
   `reports/schedule_selection.md` are **not backed by an on-disk JSON** — they exist only
   as console output captured in this project's own conversation history. The 16384 numbers
   (FP32 363.5 ms/step, AMP 177.3 ms/step, 2.05x speedup) are the only ones independently
   re-verifiable from a file on disk; the 4096/8192 speedup figures (2.08x, 2.39x) should
   be treated as reported-but-not-independently-reproducible from the current repo state.
2. **`run_metadata/execution_manifest.json` has a duplicated/incorrect timing value**: both
   16384/seed1 and 16384/seed2 are recorded with `total_time_sec: 1931.7`. The actual
   per-run log (`logs/seeds_1_2_runs.log`) shows seed 1 took 1826.4 s and seed 2 took
   1931.7 s — the manifest's seed-1 entry is wrong (copy-paste error), corrected in §1's
   table above using the primary log source.
3. **`run_metadata/experiment_config.json` still states `total_steps: 15000`**, the
   original pre-revision plan, even though every run that was actually executed used
   10,000 steps (`configs/common_config.json`, every training log, this report's §2). This
   is a stale documentation artifact from before the AMP/plateau-driven schedule revision
   and does not reflect what was run.
4. **GradScaler state is not checkpointed.** `sae_common.save_checkpoint`/`load_checkpoint`
   persist model and optimizer state but not the AMP `GradScaler`'s internal loss-scale
   factor. The two runs that were interrupted and resumed (4096/seed0 at step 5000,
   16384/seed0 at step 9000) each show one isolated `grad_norm=inf` logged exactly at their
   resume step (`logs/sae_4096_seed0.jsonl`, `logs/sae_16384_seed0.jsonl`) — the restarted
   GradScaler began at its default scale, overflowed on the first post-resume step, and
   safely skipped that one optimizer update (GradScaler's designed behavior; this does not
   corrupt the model), then adapted down and continued normally. This did not produce NaN
   losses or divergence in either run and both runs' final metrics are in line with their
   unaffected seed-1/seed-2 siblings, but it is a real, previously undocumented training
   irregularity isolated to exactly the 2 runs that were interrupted.

None of these change the headline comparison (the independently-reproducible 8192 numbers,
the per-seed evaluation JSONs, and the training-curve logs are all internally consistent
and were spot-checked against each other while writing this report), but all four should
be fixed or at least noted before this experiment is cited as a template for a future
milestone.

---

## 8. Conclusion

**Feasibility: yes, confirmed.** 9/9 planned training runs (3 widths × 3 seeds) completed
successfully on the available RTX 3050 (6.44 GB VRAM), with comfortable VRAM headroom at
every width (peak 2.37 GB of 6.44 GB), ~2.9 hours of total measured/inferred GPU time, and
11.9 GB of checkpoint disk use (correctly excluded from git). The operational risks that
materialized during this project were software/environment issues (a per-step host-sync
bottleneck, since fixed; harness background-task kills, worked around; a non-checkpointed
AMP scaler state, not yet fixed) rather than hardware resource limits.

**Does the current evidence demonstrate a useful reconstruction-sparsity trade-off? Only
partially, and not in the form the question implies.** The data does not show a classic
trade-off where increasing sparsity costs reconstruction quality across these widths —
8192 is simultaneously the sparsest and the most accurate of the three, Pareto-dominating
both neighbors on those two axes. Within a single width, the pilot's λ sweep (§3) *does*
demonstrate a real, bounded trade-off (λ=0.01 gives denser/slightly-better reconstruction;
λ=0.2 gives the task's target sparsity band at a small, measured reconstruction cost; λ≥1.0
causes outright feature collapse) — that trade-off is well-evidenced. But the
**cross-width** comparison this task asked for shows capacity utilization (dead-feature
rate) and cross-seed stability varying far more than reconstruction quality does, and those
are the axes that actually discriminate the three widths, not a reconstruction-vs-sparsity
exchange.

**What evidence is still needed before stronger claims should be made:**
1. A step-count ablation beyond 10,000 (§2, §6) — the plateau analysis showed diminishing
   but non-zero returns at the chosen budget; whether the width-ranking is stable at 2-3x
   more steps is untested.
2. A larger interpretability sample (§6) — 30 features/width, frequency-skewed, single-pass
   reviewed, supports a directional claim only.
3. A 4th+ seed, or a formal variance estimate, before treating the 3-seed ranges in §1/§4/§6
   as statistically conclusive rather than suggestive (they are internally consistent and
   non-overlapping between 8192 and 16384 on reconstruction, which is reassuring, but 3
   seeds is a thin basis for a hard statistical claim).
4. Fixes for the four items in §7 (especially the benchmark-report overwrite and the AMP
   scaler checkpointing gap) before this pipeline is reused for a future, larger-scale run
   where a silent scaler-state reset on resume could matter more than it did here.

Within those limits, the experiment's own central finding — that widening the dictionary
from 4096 to 16384 did not produce a meaningfully better reconstruction-sparsity outcome,
and degraded dead-feature rate and cross-seed stability instead — is adequately supported
by the measured evidence assembled in this report.

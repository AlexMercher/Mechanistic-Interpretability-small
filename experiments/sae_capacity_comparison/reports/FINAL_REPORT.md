# Milestone 5 — Controlled SAE Training: Dictionary Capacity Comparison (4096 / 8192 / 16384)

**Status: all Definition-of-Done criteria met**, with explicit caveats below (this report
incorporates a full independent methodology review, `reports/opus_final_review.md`, done
before this report was written).

Reproduce (in order):
```
python experiments/sae_capacity_comparison/scripts/inspect_activations.py
python experiments/sae_capacity_comparison/scripts/prepare_sae_data.py
python experiments/sae_capacity_comparison/scripts/benchmark_training.py --width 8192
# training (see run_metadata/execution_manifest.json for exact invocations):
python experiments/sae_capacity_comparison/scripts/train_sae.py --config configs/sae_<W>.json \
    --seed <S> --sparsity-coefficient 0.2 --batch-size 4096 --steps 10000 --amp
# evaluation, each width x each seed in {0,1,2}:
python experiments/sae_capacity_comparison/scripts/evaluate_reconstruction.py --width <W> --seed <S>
python experiments/sae_capacity_comparison/scripts/evaluate_sparsity.py --width <W> --seed <S>
python experiments/sae_capacity_comparison/scripts/evaluate_feature_utilization.py --width <W> --seed <S>
python experiments/sae_capacity_comparison/scripts/evaluate_feature_stability.py --width <W> --seeds 0 1 2
python experiments/sae_capacity_comparison/scripts/inspect_features.py --width <W> --seed 0
python experiments/sae_capacity_comparison/scripts/compare_sae_capacities.py
```

## 1. Research objective and answer

**Question:** given ~895,502 training activation vectors from Qwen3-1.7B layer-14, which
of 4096/8192/16384-feature SAE dictionaries gives the strongest balance of reconstruction,
sparsity, feature utilization, stability, and interpretability?

**Answer: no width wins on every axis, and the central finding is that widening the
dictionary mostly bought more dead/unstable features and more template/script artifacts,
not meaningfully better reconstruction or more genuine semantic discoveries.**

- **Reconstruction barely differentiates the three widths.** Mean validation MSE (3 seeds):
  4096=1.898, 8192=1.766, 16384=1.788 — only a 7.5% spread end-to-end. 8192 is lowest, and
  its 3-seed range ([1.764, 1.769]) does not overlap 16384's ([1.781, 1.800]), so the
  8192-vs-16384 gap is a real (if small) effect, not seed noise — but it is a small effect.
- **Dead-feature rate gets worse, not better, with width**: 46.3% (4096) → 44.2% (8192) →
  54.6% (16384) of the dictionary never fires on the validation set, mean over 3 seeds.
  16384 wastes the majority of its extra capacity.
- **Cross-seed feature stability gets worse with width**, on both the raw and the
  dead-feature-corrected metric: raw decoder-match rate 0.42 (4096) → 0.31 (8192) → 0.16
  (16384); live-feature-normalized 0.79 → 0.56 → 0.34. Wider dictionaries are less
  reproducible across random seeds here.
- **Interpretability does not improve with width.** A 90-feature teacher-reviewed sample
  (30/width) found coherent, cross-template semantic features at a flat rate — 3/30 (4096),
  1/30 (8192), 3/30 (16384) — with 8192 actually the worst in this sample. The dominant
  categories at every width were template-driven (recurring sentence frames) and
  domain/script markers (mostly Hindi-script detectors), not concepts. See
  `reports/interpretability_review_summary.md`.

**8192 is the most defensible default** (lowest MSE, lowest dead-feature rate, matches the
task's suggested primary-baseline role) but it is not a clean sweep — 4096 is more stable
across seeds, and no width showed a meaningfully higher rate of genuinely interpretable
features. This is reported as the honest empirical answer, not resolved into a false
single winner.

## 2. What was NOT done, per explicit task exclusions

No 32,768 or 131,072-feature SAE was trained. The corpus, embeddings, and Qwen3-1.7B
activations were not regenerated, expanded, or re-extracted. The selected layer (14) and
activation precision (bf16 on disk) were not changed. No feature was forced into the prior
milestones' semantic taxonomy. `git diff --stat HEAD -- raw normalized processed
experiments/activation_extraction_qwen3_1_7b experiments/semantic_topology_25k
experiments/duplicate_language_embedding` is empty (reverified before writing this report).

## 3. Preprocessing

Global scalar normalization: `y = scale * (x - mu)`, `mu` = train-split mean vector,
`scale = sqrt(2048 / mean(||x-mu||^2))` so transformed vectors have mean squared norm
≈ 2048 — computed **from the train split only** (`preprocessing/preprocessing_config.json`),
applied identically to train and validation, identical for all three widths. Rejected
per-dimension whitening: raw per-dimension variance spans 1.71 to 3,831,097 (a ~2.2M×
range, the "massive activation" outlier-dimension phenomenon noted in Milestone 4);
whitening would inflate near-constant dimensions to the same scale as these outliers,
destroying exactly the magnitude structure a dictionary should learn. This is the standard
SAE-literature convention (global scale, not per-dimension), not a per-width tuned choice.
Preprocessing statistics in `preprocessing/activation_statistics.json`.

## 4. Architecture and training protocol (identical across all 9 runs)

`z = ReLU(W_e(x-b_d)+b_e)`, `x_hat = W_d z + b_d`. Decoder columns initialized as random
unit-norm vectors, encoder tied-initialized as `decoder^T` at step 0 then trained
independently; biases zero at init; decoder columns renormalized to unit norm after every
optimizer step. Adam, no weight decay, lr 3e-4 with 5%-step linear warmup. Batch size 4096.
Sparsity coefficient λ=0.2, selected via a pilot sweep (`reports/pilot_report.md`) to land
mean L0 in the task's target band (20-50 active features/token) without the feature
collapse seen at λ≥1.0. 10,000 optimization steps (≈45.7 epochs over the 895,502-row train
set) — this budget was *not* copied from the pilot's epoch count; it was set from a
dedicated plateau analysis (`reports/schedule_selection.md`) showing validation
reconstruction/L0/dead-feature-rate had entered a region of small, roughly-constant
proportional improvement per 1,000 steps by step 7,000-10,000. Mixed precision (AMP,
`torch.autocast('cuda', dtype=torch.float16)` + `GradScaler`) was benchmarked against FP32
(`reports/benchmark_report.json`): ~2.0-2.4x speedup, loss within ~1% of FP32 at matched
steps, adopted for all training runs.

3 seeds (0, 1, 2) trained per width = 9 training runs total, all under the frozen protocol
above. See `run_metadata/execution_manifest.json` for exact per-run timings/VRAM and a note
on two harness-level background-task interruptions during training (checkpoints confirm
training itself was unaffected and resumed correctly from the saved step — see §8 caveats
for one nuance on exactly what "resumed correctly" means).

## 5. Evaluation results (3-seed aggregates on the held-out validation split, 97,994 tokens)

Full table: `reports/capacity_comparison.md` / `.json` (generated from the evaluation JSONs,
not hand-typed). Headline numbers, mean across seeds 0/1/2 (range in brackets):

| Metric | 4096 | 8192 | 16384 |
|---|---:|---:|---:|
| Validation reconstruction MSE | 1.898 [1.888, 1.906] | 1.766 [1.764, 1.769] | 1.788 [1.781, 1.800] |
| Explained variance | 0.9991 | 0.9992 | 0.9992 |
| Mean active features/token (L0) | 44.2 [43.7, 44.6] | 40.2 [40.0, 40.5] | 42.7 [42.4, 43.0] |
| Dead features | 46.3% [44.8, 47.8] | 44.2% [43.2, 45.2] | 54.6% [54.1, 55.2] |
| Active (fired >=1 time) | 53.7% | 55.8% | 45.4% |
| Stability, raw decoder-match | 0.42 | 0.31 | 0.16 |
| Stability, live-normalized | 0.79 | 0.56 | 0.34 |
| Training duration (avg, sec) | 454 | 998 | 1,315 |
| Peak VRAM (avg, GB) | 0.76 | 1.29 | 2.37 |

"MSE" above is sum-of-squared-error per vector (not per-element mean; per-element mean
would be ≈0.0009 — divide by 2048), and explained variance is `1 - sum(||x-x_hat||^2) /
sum(||x||^2)` in the SAE's preprocessed (mu-centered, scaled) input space — a ratio that
also equals the mu-centered raw-activation-space EV, since centering/scaling cancel in a
ratio of sums of squares taken relative to the same centering.

**Important caveat on "explained-variance-per-active-feature":** this ratio is reported in
the comparison JSON/MD for transparency but is **not** an independent signal here — EV is
saturated (~0.999) at every width, so the ratio is numerically ≈1/mean_L0 and just restates
the sparsity ranking. It should not be read as evidence of a genuine reconstruction-vs-
sparsity trade-off being resolved differently by width; in this result, reconstruction
barely moves at all, and the real differentiation across widths lives in sparsity,
dead-feature rate, and stability.

**Important caveat on resumability:** two of the nine training runs (4096/seed0, 16384/seed0)
were interrupted by a harness-level background-task kill and resumed from their last saved
checkpoint. `train_sae.py` checkpoints model/optimizer/step but not the data-shuffle
permutation's position, so a resumed run continues training from the correct step but with
a freshly-reseeded shuffle order from that point, not bit-identical to an uninterrupted
run's batch sequence. This is still valid SGD (every vector is still drawn i.i.d.-ish from
the same training pool) and both affected runs' metrics are in line with their own
unaffected cross-seed siblings, but "resumed correctly" means "continues from the saved
step with valid training," not "reproduces an uninterrupted run bit-for-bit."

## 6. Feature stability (3 seeds per width, pairwise decoder + activation matching)

Two independent matching signals per width (decoder-direction cosine similarity on unit-norm
dictionary columns, and activation-pattern cosine similarity over a fixed 8,192-token
validation sample), greedy one-to-one matched at cosine >= 0.3 threshold (chosen because two
random unit vectors in 2048-dim space have similarity std ≈ 0.022, so 0.3 is a strong,
non-spurious signal). Full detail: `evaluations/feature_stability/sae_*.json`.

Stability decreases monotonically with width on both the raw metric (fraction of the full
dictionary that finds a stable cross-seed match: 0.42/0.31/0.16) and the dead-feature-
corrected live-normalized metric (0.79/0.56/0.34). The correction matters: raw rates
conflate "unstable" with "dead" (a dead feature's near-random, untrained decoder direction
cannot match across seeds by construction), but the ordering survives correction — wider
dictionaries are genuinely less reproducible here, not just more dead.

## 7. Interpretability and semantic selectivity

Full detail: `evaluations/interpretability/sae_*_feature_stats.json` (automated stats: top
activating tokens/prompts, domain distribution — no auto-generated labels),
`sae_*_reviewed.json` (teacher-model/Opus-reviewed labels per feature), and
`reports/interpretability_review_summary.md` (cross-width comparison).

30 features per width were sampled (10 highest-frequency + 10 median-frequency + 10
rarest-but-alive) and independently labeled by a teacher model into
template_driven / domain_or_script_marker / coherent_semantic / uninterpretable_or_mixed,
with justification tied to the actual top examples (not keyword-matching). Result:
coherent-semantic counts were 3/30, 1/30, 3/30 for 4096/8192/16384 — flat-to-noisy, with
8192 the worst in this sample. The dominant failure modes were (a) recurring sentence
templates firing identically across unrelated domains (e.g. "Summarize the established
historical information about X that illuminates..." across multiple GANDHI subtopics,
same activation value regardless of X) and (b) Hindi-script detectors that key on
Devanagari Unicode ranges, not meaning — exactly the template-phrasing and script-driven
confounds flagged in Milestones 3/4. **No claim is made that any feature represents a
concept beyond what its own top examples visibly support**, and the sample is explicitly
frequency-skewed (not a uniform random draw of the dictionary), so absolute percentages
should be read as directional, not precise population estimates.

## 8. Known limitations (full list)

1. **Reconstruction/sparsity/dead-feature evaluation is now a 3-seed aggregate (fixed
   during review)** — an earlier draft of the comparison script wrote all three widths'
   per-seed evaluation JSONs to the same filename (`sae_<width>.json` regardless of seed),
   so seed 0's results were silently overwritten by seed 1 then seed 2 before this was
   caught. Fixed: output filenames now include the seed (`sae_<width>_seed<seed>.json`),
   all three seeds were re-evaluated, and `compare_sae_capacities.py` now aggregates
   mean+range across seeds. The stale pre-fix files are preserved under
   `evaluations/*/​_legacy_overwritten_no_seed_suffix/` for audit, not deleted.
2. **Resumability caveat** — see §5; two runs resumed from checkpoint with a freshly-seeded
   shuffle order rather than bit-identical continuation.
3. **EV-per-active-feature is not an independent axis** — see §5; it is ≈1/L0 here because
   EV is saturated across all widths.
4. **Interpretability review is a 30-feature/width sample**, frequency-skewed, single
   teacher-model pass, no causal/steering validation — directional evidence, not a
   dictionary-wide census.
5. **Two explained-variance formulas exist in the codebase**: `sae_common.explained_variance`
   (centered-variance form, used only in training-time logs) and the uncentered
   sum-of-squares form in `evaluate_reconstruction.py` (used in every number in this report
   and in `capacity_comparison.md`). Both are internally consistent within their own use,
   but should not be cross-compared.
6. **Dead-feature definitions differ by context, intentionally**: training-time logs use a
   2000-step rolling-inactivity window; evaluation scripts use "never fires on any
   validation-set token." Every number in this report uses the evaluation (validation-set)
   definition consistently across all three widths.
7. **Hardware**: RTX 3050 laptop GPU, 6.44GB VRAM, peak usage 2.37GB (16384) — comfortable
   headroom throughout; no OOM, no CPU fallback at any point (`sae_common.get_device_no_fallback`
   raises rather than falls back).
8. **Harness background-task reliability**: the run_in_background task tracker killed two
   long-running training jobs for reasons unrelated to the training process itself (verified
   via `tasklist` that detached nohup processes continued running correctly once no longer
   tracked by that mechanism). This is an execution-environment quirk, not a methodology
   issue, documented in `run_metadata/execution_manifest.json`.

## 9. Independent review

A full independent methodology/correctness review (`reports/opus_final_review.md`) was
performed before this report was written: recomputed headline ratios from raw JSONs
(confirmed no transcription errors), checked train/validation separation end-to-end
(confirmed respected), checked the preprocessing rationale quantitatively (confirmed
sound), and surfaced the single-seed aggregation gap (fixed, see §8.1) and the two
secondary metric caveats (EV-per-feature, stability normalization) incorporated into this
report.

## 10. Definition of Done checklist

- [x] Activation dataset verified unchanged (sha256-checked every shard before use; `git diff` empty).
- [x] Documented preprocessing method selected and justified (§3).
- [x] Pilot confirmed stable training (`reports/pilot_report.md`) before any full run.
- [x] SAE-4096 trained and evaluated (3 seeds).
- [x] SAE-8192 trained and evaluated (3 seeds).
- [x] SAE-16384 trained and evaluated (3 seeds).
- [x] All three evaluated on the same held-out validation vectors (97,994 tokens, identical set).
- [x] Reconstruction and sparsity metrics available, treated as co-equal axes (§5).
- [x] Feature utilization / dead-feature statistics available (§5, §6).
- [x] Feature stability evaluated across 3 seeds per width, with two independent matching signals (§6).
- [x] Representative feature interpretations inspected and teacher-reviewed (§7).
- [x] Capacity comparison report generated from evaluation JSONs, not hand-typed (§5, `reports/capacity_comparison.{md,json}`).
- [x] Training configs, checkpoints, and reproduction commands saved (`configs/`, `checkpoints/`, top of this report).
- [x] No source corpus, activation shard, or previous-milestone output modified (§2).

**No criterion failed.** The milestone's central finding — that increasing dictionary width
from 4096 to 16384 did not produce a clear improvement in reconstruction, sparsity,
stability, or interpretability under this ~895K-vector training budget, and in several
respects (dead-feature rate, cross-seed stability) made things worse — is the honest,
evidence-based answer to the research question, not a conclusion reached by default.

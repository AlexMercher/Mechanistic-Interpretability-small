# Milestone 5 — Final Methodology/Results Review (Opus, read-only)

Reviewer pass over `experiments/sae_capacity_comparison/` prior to writing `FINAL_REPORT.md`.
No files were modified except this review document.

## Summary verdict

The evidence base is **mostly sound and the headline verdict is faithfully supported by the
underlying JSONs** — every central number in `capacity_comparison.md` transcribes and
recomputes correctly, train/validation separation is respected everywhere, and the "no single
width wins on every axis" conclusion (8192 on reconstruction / sparsity-efficiency / dead-rate,
4096 on cross-seed stability) is exactly what the data shows. However, the comparison's
reconstruction/sparsity/dead-feature winner claims rest on a **single seed (seed 0) per width**
even though three seeds were trained, and the "explained-variance-per-active-feature" lens is
numerically a pure L0 ranking because explained variance is saturated (~0.999) across all three
widths. These two things, plus a stability metric that is confounded by dead-feature count,
should be caveated before the final report crowns any axis winner. **Writable as-is only if these
caveats are added; a modest amount of extra evaluation (all 3 seeds) would materially strengthen it.**

## Confirmed findings (verified, correct)

- **Numeric transcription is clean.** Recomputed from raw JSONs:
  - `explained_variance_per_active_feature`: 8192 = 0.9991632/39.9929 = **0.024983** (reported 0.02498);
    4096 = 0.9990988/43.6621 = **0.022883** (reported 0.02288); 16384 = 0.9991467/43.0294 = **0.023220**
    (reported 0.02322). All match.
  - `dead_feature_pct`: 8192 3535/8192 = **43.15%**, 4096 1956/4096 = **47.75%**, 16384 8897/16384 = **54.30%**. All match.
  - 16384 `mean_decoder_match_rate`: (0.15533+0.15546+0.15857)/3 = **0.15645**. Matches.
  - Reconstruction MSE ordering 8192 (1.7652) < 16384 (1.8000) < 4096 (1.9011) — matches table and verdict.
- **The verdict genuinely matches the data.** `compare_sae_capacities.py` picks axis winners by `min`/`max`
  over the loaded JSONs (not hand-typed), and the "co-equal axes, do not default to widest" framing is enforced
  in code (lines 117-143): it only prints a single-winner line if the axis-winner set collapses to one, which it does not.
- **Train/validation separation is respected end-to-end.** Shards are read from the Milestone-4
  `shard_manifest.json` `train`/`validation` keys independently (`sae_common.load_split_activations`),
  sha256-verified before use (`verify_and_load_shard`), the preprocessing `mu`/`scale` were computed from
  **train only** (`preprocessing_config.json: computed_from = train split`), and all four eval scripts load only the
  `validation` split; `evaluate_reconstruction.py:48` asserts every validation row carries a domain label.
- **Preprocessing rationale is quantitatively sound.** `activation_statistics.json` shows per-dimension variance
  spanning 1.71 → 3,811,791 (a ~2.2M× ratio). Per-dimension whitening would hoist the lowest-variance dims to
  parity with the "massive activation" outlier dims, which is exactly the magnitude structure an SAE dictionary
  should preserve. Global-scalar centering+scaling to `mean_sq_norm ≈ 2048` is a defensible, standard choice. The
  stated rationale holds.
- **Dead-feature definition mismatch is handled honestly, not a bug.** Training-time logging uses a 2000-step
  rolling window (`train_sae.py:191`); the evaluation uses "never fires on any validation vector"
  (`evaluate_feature_utilization.py:53`). Both scripts explicitly document the difference and the final table uses
  only the eval-based definition consistently across all three widths. Correct.
- **Architecture/stability mechanics are correct.** Decoder columns renormalized to unit norm after every optimizer
  step; decoder cosine-similarity matrix is a true cosine (unit-norm columns); dead features are naturally excluded
  from activation-pattern matching (all-zero activation column → ~0 vector → sub-threshold cosine); the `UNKNOWN`
  domain bucket avoids the numpy negative-index aliasing trap (`build_domain_ids`). AMP keeps loss/reduction in fp32
  (`train_sae.py:169`), and the schedule_selection benchmark showed FP32-vs-AMP loss within ~1%, so the FP32 pilot's
  λ=0.2 choice is not materially invalidated by the switch to AMP.
- **EV magnitude is internally consistent:** 1 − 1.80/2048 ≈ 0.99912, matching the reported 16384 EV of 0.99915 —
  confirms the ~0.999 EV is real given the scaling, not a bug.
- **Interpretability counts verified** directly from the `*_reviewed.json` files: coherent_semantic = **3 / 1 / 3**
  for 4096 / 8192 / 16384, matching `interpretability_review_summary.md`. The "widening mostly multiplied template/
  script artifacts, not concepts" read is supported and appropriately caveated (30-feature frequency-skewed sample).

## Issues found (ranked by severity)

### 1. (HIGH) Reconstruction / sparsity / dead-feature winners rest on a single seed per width
`evaluate_reconstruction.py`, `evaluate_sparsity.py`, `evaluate_feature_utilization.py` all default to
`--seed 0`, and every written JSON has `"seed": 0`. `compare_sae_capacities.py:72-74` reads only
`sae_{w}.json` (the seed-0 output). So three of the four verdict axes — "lowest MSE: 8192", "best
efficiency: 8192", "lowest dead%: 8192" — are **n=1 per width**, despite 9 trained checkpoints existing.
The 8192-vs-16384 reconstruction gap is only 1.7652 vs 1.8000 (~2%), and cross-seed decoder stability is
0.16–0.42, i.e. seeds demonstrably diverge. With no cross-seed variance on these axes, the headline winner
could plausibly be within seed noise. **Fix:** run the three eval scripts for seeds 1 and 2 as well and
report mean ± range per width; at minimum, state loudly in FINAL_REPORT that recon/sparsity/dead figures are
single-seed point estimates while only stability is multi-seed.

### 2. (MEDIUM) "Explained-variance-per-active-feature" is a disguised L0 ranking, not a joint lens
EV is saturated: 0.99910 / 0.99916 / 0.99915 across widths — varying only in the 4th decimal. Therefore
`efficiency = EV / mean_l0` (`compare_sae_capacities.py:95`) is numerically ≈ `1 / mean_l0`, so "best
efficiency: 8192" is just a restatement of "8192 has the lowest mean L0 (39.99)". It is not independent
corroboration of the reconstruction result and does not represent a genuine reconstruction↔sparsity
trade-off being navigated. More broadly, because reconstruction barely moves across widths, the *real*
discriminating signal in this experiment lives almost entirely in **sparsity, dead-feature rate, and
stability**, not reconstruction. **Fix:** FINAL_REPORT should state EV is saturated and that the
"efficiency" winner is L0-driven, rather than presenting it as a separate axis that happens to agree.

### 3. (MEDIUM) Feature-stability match rate is confounded by dead-feature count
`decoder_match_rate = n_matched / width` (`evaluate_feature_stability.py:96`) divides by the **full**
dictionary. Dead features (54% at 16384) keep near-random, untrained decoder directions that cannot match
across seeds, so wider dictionaries are penalized twice — once in the dead-feature metric and again here.
Normalizing by *live* features instead: 4096 = 0.4233/0.5225 = **0.81**, 8192 = 0.3138/0.5685 = **0.55**,
16384 = 0.1565/0.4570 = **0.34**. The "4096 most stable" ordering survives, but the gap shrinks substantially
and part of the headline number is a dead-feature artifact, not genuine representational instability.
**Fix:** report the live-normalized rate alongside, or caveat that the raw rate compounds the dead-feature penalty.

### 4. (MEDIUM) Resume was verified as "continues from the right step", not "reproduces an uninterrupted run"
`execution_manifest.json` notes 4096/seed0 resumed from step 5000 and 16384/seed0 from step 9000 after
background-task kills, and these two seed-0 runs are exactly the ones feeding the headline recon/sparsity/dead
numbers (Issue 1). But `train_sae.py:137-138` always rebuilds the shuffle permutation from a fresh
`Generator(seed)` with `cursor=0` **after** the resume block — the data-iteration state (perm position) is not
checkpointed, only model/optimizer/step. So a resumed run replays the permutation stream from the top at the
resume step, seeing a **different batch order** than an uninterrupted run would. This is still valid SGD and
unlikely to change results much, but `pilot_report.md:28`'s "resumability verified" and the manifest's
"checkpoint resumability verified" overstate it: what was verified is that training continues from the saved
step, *not* that the resumed run is identical to an uninterrupted one. **Fix:** state this honestly, or (better)
spot-check the two resumed seed-0 checkpoints against a clean re-run.

### 5. (LOW) Two explained-variance formulas coexist; the eval docstring's invariance claim is imprecise
`sae_common.explained_variance` uses centered variance `1 − var(x−x̂)/var(x)` (used in training logs,
`train_sae.py:193`), while the final table uses the uncentered `1 − Σ‖x−x̂‖²/Σ‖x‖²`
(`evaluate_reconstruction.py:54`). They are consistent across the three widths in the final comparison (all
from the eval script), so the comparison is valid — but the report should not conflate the two. Also the
`evaluate_reconstruction.py:13-18` docstring claims the preprocessed-space ratio "equals the
raw-activation-space explained variance too"; it actually equals the **mu-centered** raw-space EV (the
denominator becomes Σ‖x_raw − mu‖², not Σ‖x_raw‖²). Numerically fine (and the more standard definition), but the
invariance statement as written is loose.

### 6. (LOW) "MSE" is actually sum-of-squared-error per vector
`validation_reconstruction_mse` = Σ over 2048 dims of squared error, averaged over vectors (≈1.8), not a
per-element mean (which would be ≈0.0009). Consistent across widths so the comparison is unaffected, but
FINAL_REPORT should define it (and note `EV ≈ 1 − SSE/2048`) to avoid a reader misreading a "1.8 MSE" as a
poor reconstruction.

## Recommendations for FINAL_REPORT.md

1. **Add seed coverage or an explicit single-seed caveat (Issue 1).** Ideally evaluate all 3 seeds for
   reconstruction/sparsity/utilization and report mean ± range; otherwise state plainly that these three axes
   are seed-0 point estimates and that the ~2% 8192-vs-16384 MSE gap is within plausible seed noise.
2. **Reframe the efficiency metric (Issue 2).** State that EV is saturated (~0.999) at every width, that the
   discriminating signal is sparsity/dead-rate/stability, and that "EV-per-active-feature" is L0-driven — not
   an independent third axis.
3. **Report live-normalized stability (Issue 3)** or caveat that the raw decoder-match rate compounds the
   dead-feature penalty against wider dictionaries.
4. **State resume honestly (Issue 4):** training resumed from the correct step but not with identical data
   ordering; results are valid SGD, not a bit-reproduction of an uninterrupted run.
5. **Define EV and MSE precisely (Issues 5–6):** which formula, which space (mu-centered, scaled SAE input
   space), and that "MSE" is per-vector SSE.
6. **Keep the core verdict — it is well supported.** "No single width wins on every axis" is correct: 8192 leads
   reconstruction/sparsity-efficiency/dead-rate, 4096 leads cross-seed stability, and interpretability (coherent
   semantic 3/1/3) is flat across widths. The honest conclusion — wider dictionaries given this ~895K-vector
   budget mostly bought more dead/unstable features and template/script artifacts rather than better
   reconstruction or more genuine concepts — is defensible and should be stated confidently, with the caveats above.

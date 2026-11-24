# Milestone 6 — Phase 1 Report (Exact split, inventory, candidate selection, Discovery evidence)

## Completed

1. **Discovery/Confirmation split** (`build_discovery_confirmation_split.py`): exact
   rank-based per-domain partition of the 2,533 Milestone-4 validation prompts.
   Result: 1,265 Discovery / 1,268 Confirmation prompts (49,615 / 48,379 tokens).
   Per-domain counts exactly `floor(n/2)`/`ceil(n/2)` (verified, not approximate).
   Exhaustive token-inheritance check passed on all 97,994 validation tokens.
2. **Full SAE-8192 seed-0 feature inventory, Discovery subset** (`build_feature_inventory.py`):
   all 8,192 features, token/prompt-level frequency, activation quantiles, domain/
   subdomain/language_form firing distributions. 3,535 dead features (never fire in
   Discovery) — matches Milestone 5's full-validation-set dead count for this exact
   checkpoint almost exactly, a useful cross-check. 1,983,718 total firing events.
3. **Tiered template clusters** (`build_template_clusters.py`): Tier 1 (confirmed
   near-duplicate families, reused from `corpus_validation_25k`) = 1,905 families /
   6,333 prompts. Tier 2 (skeleton fallback) = 151 families / 1,533 prompts, spot-checked
   on 20 examples (`reports/template_tier2_spotcheck.json`) — correctly recovers known
   template families (e.g. GANDHI's "A historian examines X through political context...").
   Tier 3 (unknown/unclustered) = 17,134 prompts, correctly never counted as independent
   templates downstream.
4. **Candidate selection** (`select_candidates.py`): 121 candidates across 10 strata, all
   formulas/thresholds frozen in `configs/candidate_selection_criteria.json` before
   selection. `always_on_near_ubiquitous` correctly found only 1 candidate (feature 972 —
   the same near-ubiquitous unit Milestone 5 independently identified). The
   Discovery-only/Confirmation-only boundary for the two cross-seed strata is documented
   in `reports/candidate_selection_report.md`.
5. **Discovery-subset evidence** (`extract_candidate_evidence.py --mode discovery`):
   0/121 candidates with zero examples; top/mid/low activation-quantile examples plus
   one-per-distinct-template evidence, each tagged with its Tier 1/2/3 template label,
   subdomain diversity, and lexical type-token-ratio diversity. All examples traceable to
   `prompt_id`/`token_position`/`activation_row_index` and resolve against
   `normalized/*.jsonl` (spot-checked, including correct Hindi/Devanagari text recovery).

## Discovered (worth flagging now, not burying until the final report)

- A first version of `select_candidates.py` produced only 94 candidates with two strata
  at zero unique members, because later strata's "top N" pools were fully claimed by
  earlier strata's already-selected features (e.g. script/language-dependence candidates
  heavily overlapped with domain-concentration candidates). Fixed by excluding
  already-selected features from each subsequent stratum's pool before ranking — this is
  now the implemented behavior, not a pending issue.
- The `cross_seed_prescreened_reproducible` stratum found only ~10 features passing the
  combined decoder+activation-pattern Discovery-only screen out of 4,657 alive features —
  cross-seed reproducibility is rare even before Confirmation-subset validation, consistent
  with Milestone 5's own low raw stability numbers for width 8192 (0.31).

## Failed

Nothing failed outright; one design bug (stratum overlap exhaustion, above) was caught via
the resulting candidate count falling short of the target range and corrected before
proceeding, rather than silently accepted.

## Remaining (Phase 2 onward)

Confound diagnostics on both subsets, Confirmation-subset evidence extraction,
counterexample generation, the Confirmation-only feature-coverage statistic, then Phase 3
review/cross-seed/cross-capacity work.

## Safe to proceed to Phase 2

Yes. All Phase 1 checkpoints in the approved plan passed: zero candidates with zero
Discovery evidence, traceability resolves, split-exactness verified, and the cross-seed
stratum selection is confirmed and documented to have used Discovery data only.

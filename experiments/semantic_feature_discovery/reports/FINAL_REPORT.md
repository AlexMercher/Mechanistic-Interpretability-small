# Milestone 6 — Final Execution Report

This is the execution/methodology log for Milestone 6 ("Semantic Feature Discovery,
Validation & Relationship Analysis"). The scientific narrative — what the results mean
about Qwen3-1.7B layer-14 representations — lives in `analysis/findings.md`; this
document covers configuration, checkpoints, validation, bugs/corrections, reproducibility,
and artifact locations only.

## 1. Methodology summary

| Stage | Method | Output |
|---|---|---|
| Discovery/Confirmation split | Exact, deterministic, rank-based per domain: `sha256("m6_discovery_confirmation:" + prompt_id)`, sorted ascending, `floor(n/2)`->Discovery / `ceil(n/2)`->Confirmation per domain | `configs/discovery_confirmation_split_config.json`, `configs/discovery_confirmation_assignment.jsonl` (1,265/1,268 prompts, 49,615/48,379 tokens) |
| Template detection | Tier 1 (confirmed near-dup connected components) / Tier 2 (skeleton fallback) / Tier 3 (unclustered, never counted as independent) | `configs/template_clustering_config.json`, `analysis/data/aggregate_statistics/template_clusters.jsonl` |
| Candidate selection | 10 pre-frozen quota strata, formulas fixed before selection | `configs/candidate_selection_criteria.json`, `analysis/data/feature_analysis/candidates.jsonl` (121 candidates) |
| Primary + second review | Batched structured review (9 Sonnet batches, 121 candidates) + blind 40-candidate second opinion (Opus, condensed-digest input only) | `analysis/data/validation_results/primary_reviews.jsonl`, `second_reviewer_blind.jsonl`, `reviewer_agreement.json` |
| Cross-seed/capacity matching | Full-target-dictionary decoder search + full-source-dictionary reverse check, Confirmation-subset-only functional validation, 12-combination sensitivity sweep | `analysis/data/validation_results/cross_seed_reproducibility.json`, `cross_capacity_check.json`, `cross_seed_capacity_reconciliation.json`, `cross_seed_sensitivity_sweep.json` |
| Relationship analysis | Co-activation (confound-controlled cascade, prompt-level primary), decoder/activation/lexical comparison, M3-seeded cross-domain hypotheses, typed graph | `analysis/data/relationship_analysis/*.json` |
| Visualization | 21 Seaborn/Matplotlib figures across 7 categories (PNG 300dpi + SVG) | `analysis/plots/*/`, `analysis/plots/figure_manifest.json` |
| Integrity verification | 31 independent re-derivation checks against raw outputs | `scripts/validate_integrity.py`, `analysis/data/aggregate_statistics/integrity_report.json` |

## 2. Experiment configuration

- **Model / layer**: Qwen3-1.7B, layer-14 residual-stream activations (frozen, from
  `experiments/activation_extraction_qwen3_1_7b/`, Milestone 4 — not re-extracted).
- **Primary dictionary**: SAE-8192, seed 0, checkpoint `experiments/sae_capacity_comparison/checkpoints/sae_8192/seed_0/best_val.pt` (frozen, Milestone 5 — not retrained).
- **Cross-seed checkpoints**: SAE-8192 seed 1, seed 2 (same directory structure).
- **Cross-capacity checkpoints**: SAE-4096 seed 0, SAE-16384 seed 0.
- **Validation substrate**: 97,994 activation vectors / 2,533 prompts
  (`experiments/activation_extraction_qwen3_1_7b/activations/validation/`), partitioned
  within this milestone only into Discovery/Confirmation (Section 1); the original
  validation split file was never altered.
- **New dependencies installed**: `seaborn`, `pandas` (approved in the Phase 0 plan).
- **All new work isolated under** `experiments/semantic_feature_discovery/`; nothing under
  `raw/`, `normalized/`, `processed/`, or any other Milestone 1-5 experiment directory was
  written to (verified in Section 5).

## 3. Phase-by-phase checkpoints

- **Phase 1** (`reports/phase1_report.md`): exact split built and token-inheritance-verified
  exhaustively (not sampled); full SAE-8192 seed-0 Discovery-subset inventory (8,192
  features, 3,535 dead); Tier 1/2/3 template tagging (6,333 / 1,533 / 17,134 prompts);
  121 candidates selected across all 10 quota strata (one stratum, `always_on_near_ubiquitous`,
  legitimately capped at 1 member, matching Milestone 5's known single near-ubiquitous feature).
- **Phase 2** (`reports/phase2_report.md`): confound diagnostics computed independently on
  both subsets; Confirmation-subset evidence extracted; counterexamples generated (after
  two bug fixes, Section 4); Confirmation-only coverage pass run (0/3,535 Discovery-dead
  features fire in Confirmation).
- **Phase 3** (`reports/phase3_report.md`, later appended with the cross-seed/capacity
  reconciliation section): primary + blind second review completed and agreement computed
  (original kappa 0.390, normalized 0.524); cross-seed matching run for seeds 1/2 against
  the complete 8,192-feature target dictionaries; cross-capacity matching run for widths
  4096/16384 against the 61 >=medium-confidence candidates; 12-combination sensitivity
  sweep run; explicit reconciliation of the 121-candidate and 61-candidate denominators.
- **Phase 4** (`reports/phase4_report.md`): co-activation analysis (1,830 pooled pairs, 126
  selected for the confound-controlled cascade, 104/126 survived conditioning); decoder-
  space analysis (37/1,830 decoder-vs-functional disagreements); cross-domain analysis (5
  single-feature generalizers, 1,107 raw / top-20 reported two-feature hypotheses); typed
  relationship graph built (61 nodes, 139 edges).
- **Phase 5** (this report + `analysis/findings.md`): 21 figures generated across 7
  categories; `validate_integrity.py` run with 31/31 checks passing; `findings.md`
  completed; this file completed.

## 4. Known bugs and corrections (all caught before being trusted, all fixed and re-run)

1. **`select_candidates.py` stratum exhaustion** (Phase 1): later strata's "top N" pool
   wasn't excluding already-selected features, yielding only 94 candidates with 2 empty
   strata. Fixed by filtering each stratum's ranking pool to remaining (unselected)
   features before ranking. Re-run: 121 candidates, all strata populated.
2. **`generate_counterexamples.py` embedding-index bug** (Phase 2): the nearest-neighbor
   index was built only from Confirmation-subset prompts, so Discovery-prompt seed queries
   (2 of 3 counterexample types) always returned empty. Fixed by indexing the full
   embedding set for seed lookup while still restricting *returned* neighbors to the
   Confirmation subset. Re-run: 315 unique prompts, all 121 candidates covered.
3. **`generate_counterexamples.py` Tier-3 aliasing bug** (caught during Phase 3 pre-flight
   verification, per explicit request): Tier-3 prompts' `(3, None)` comparison keys
   compared equal to each other, so a Tier-3-seeded candidate could never find a
   "different template" neighbor. Fixed: `key = (tier, cluster) if tier in (1, 2) else
   (3, prompt_id)`. Pre-fix output preserved at
   `analysis/data/feature_analysis/_superseded/counterexamples_v2_tier3_aliasing_bug.jsonl`
   for audit. Verified post-fix: zero Discovery-prompt leakage into Confirmation-only
   evidence, `same_concept_different_template` coverage 120/121 -> 121/121.
4. **`cross_seed_reproducibility.py` minimum-evidence-check bug** (Phase 3): the
   `insufficient_evidence` check (<5 firing tokens OR <3 unique firing prompts) was only
   evaluated inside the "if a target match exists" branch, so unmatched-decoder candidates
   with thin Confirmation evidence were mislabeled `not_reproduced`. Fixed by moving the
   check to run first, unconditionally. Re-run: exactly 25 `insufficient_evidence`
   candidates per seed, matching Phase 2's independently-known count of 25 low-evidence
   candidates.
5. **First blind second-reviewer attempt failed**: a single Opus agent spawned directly on
   the raw 40-candidate evidence bundle hit a session limit mid-digestion before writing
   any output. No data was lost (9 primary batches were already safe; no partial
   second-reviewer file existed). Added a standing rule (now in the project's `CLAUDE.md`):
   never spawn an Opus agent directly on raw/heavy data; have Sonnet/Haiku gather and
   condense evidence first. Built `scripts/condense_second_reviewer_bundles.py`
   (mechanical, non-agent condensation to ~706KB for 40 candidates); re-ran the second
   review as two 20-candidate Opus batches reading only the condensed digest — both
   succeeded.
6. **`cross_domain_analysis.py` wrong-source-file bug** (Phase 4): read
   `cross_domain_entropy_norm` from the raw per-feature inventory (which never has that
   field) instead of `candidates.jsonl` (where `select_candidates.py` actually wrote it),
   silently emptying the entire single-feature-generalizer section. Fixed by reading from
   `candidates.jsonl` directly; re-verified against the known 5-candidate
   `coherent_semantic` list (features 95, 1974, 4485, 7399, 7640).
7. **`cross_domain_analysis.py` noisy-threshold issue** (Phase 4): a lenient 5%
   per-feature firing-rate floor for the M3-seeded two-feature check produced 1,107 raw
   intersections, recognized as mostly noise. Added an `evidence_strength_score` ranking
   and capped the reported list at the top 20, keeping the raw count visible as explicit
   context rather than hidden or presented as 1,107 discoveries.
8. **User-flagged cross-seed/cross-capacity denominator discrepancy** (Phase 3 -> 4 gate):
   the 121-candidate cross-seed denominator and the 61-candidate (>=medium-confidence-only)
   cross-capacity denominator were initially presented without making the difference
   explicit, risking confusion between seed_2's `reproduced_strong`=67 (out of 121) and the
   cross-capacity eligible count of 61. Fixed by writing
   `cross_seed_capacity_reconciliation.json` with a full, explicit accounting of all 121
   candidates across all six classifications for both seeds, plus a programmatic identity
   check that the same 25 `insufficient_evidence` feature indices appear under both seeds.

No bug was patched over silently; every one above is also documented in its originating
phase report, and superseded/buggy outputs were preserved rather than deleted.

## 5. Reproducibility and frozen-artifact verification

- `git status --porcelain` on every frozen Milestone 1-5 directory (`raw/`, `normalized/`,
  `processed/`, `experiments/activation_extraction_qwen3_1_7b/`,
  `experiments/sae_capacity_comparison/`, `experiments/semantic_topology_25k/`,
  `experiments/duplicate_language_embedding/`, `experiments/corpus_validation_25k/`)
  returns clean — verified both manually throughout the project and programmatically by
  `validate_integrity.py`'s `frozen_milestone_dirs_unmodified` check.
- All candidate-selection formulas were frozen in `configs/candidate_selection_criteria.json`
  before any candidate was selected and never tuned afterward.
- The Discovery/Confirmation split is a content-free, deterministic function of each
  `prompt_id`; re-running `build_discovery_confirmation_split.py` reproduces the identical
  1,265/1,268 partition.
- Every plot in `analysis/plots/` is generated purely from already-written JSON/JSONL
  files by `scripts/make_plots_*.py`; no plot script performs new activation computation.
- `run_metadata/` was not populated as a separate `environment.json`/`execution_manifest.json`
  pair in the final execution (the Discovery/Confirmation methodological boundary it was
  meant to document is instead recorded directly in `configs/discovery_confirmation_split_config.json`
  and `reports/candidate_selection_report.md`, which fully state which script/subset
  produced which number). This is a minor documentation-location gap relative to the
  original Phase 0 directory plan, noted here for completeness; it did not affect any
  phase's actual execution or any reported number, since the boundary is documented
  elsewhere.

## 6. Artifact locations

All paths below are relative to `experiments/semantic_feature_discovery/`.

- **Configs**: `configs/candidate_selection_criteria.json`, `template_clustering_config.json`,
  `discovery_confirmation_split_config.json` + `discovery_confirmation_assignment.jsonl`,
  `second_reviewer_sample.json`.
- **Feature analysis**: `analysis/data/feature_analysis/` (inventory, candidates, evidence,
  confound diagnostics, counterexamples, superseded buggy outputs).
- **Validation results**: `analysis/data/validation_results/` (reviews, agreement,
  cross-seed/capacity matching + reconciliation + sensitivity sweep).
- **Relationship analysis**: `analysis/data/relationship_analysis/` (activation matrices,
  co-activation, decoder relationships, cross-domain hypotheses, typed graph).
- **Aggregate statistics**: `analysis/data/aggregate_statistics/` (domain/token baselines,
  template clusters, Confirmation-only coverage, integrity report).
- **Figures**: `analysis/plots/{feature_distributions,semantic_selectivity,confound_analysis,
  cross_seed_stability,cross_domain_analysis,feature_relationships,model_representation}/`
  + `figure_manifest.json`.
- **Reports**: `reports/phase1_report.md` through `phase4_report.md`,
  `candidate_selection_report.md`, `template_tier2_spotcheck.json`, this file.
- **Scientific narrative**: `analysis/findings.md`.

## 7. Final scientific-discipline confirmation

No causal claim appears anywhere in this milestone's outputs; co-activation and decoder
similarity are used as diagnostic/correlational signals only, never as proof of semantic
or functional identity; no feature was forced into the Milestone-3 taxonomy; the
121-candidate review (1.5% of the dictionary) is never extrapolated to the full 8,192
features; every negative finding (three zero-count edge types, 25 insufficient-evidence
candidates, 22 confound-explained co-activation pairs, moderate 0.39-0.52 inter-reviewer
kappa) is reported explicitly in `analysis/findings.md` rather than omitted.

# Milestone 6 — Phase 2 Report (Confound diagnostics, Confirmation evidence, counterexamples, coverage)

The 121-candidate cohort and selection criteria from Phase 1 were not altered. All Phase 2
work reads Phase 1 outputs as frozen inputs.

## Completed

1. **Confound diagnostics on both subsets separately** (`confound_diagnostics.py`):
   domain-normalized concentration, cross-domain entropy, and script/language score
   computed independently on Discovery and on Confirmation (each using that subset's own
   domain/language baseline, never the other subset's or a prior-milestone baseline), plus
   the Tier-1/2/3 template-diversity and subdomain/lexical-diversity numbers already
   produced in Phase 1. Wrote `confound_diagnostics.jsonl` (121 rows, both subsets'
   numbers side by side, used as a robustness check per the plan).
2. **Confirmation-subset evidence** (`extract_candidate_evidence.py --mode confirmation`):
   independent top/mid/low/one-per-template examples per candidate, pulled only from the
   1,268-prompt Confirmation subset, tagged `subset: confirmation`, never merged with
   Discovery evidence.
3. **Systematic counterexample testing** (`generate_counterexamples.py`), Confirmation-
   subset only, three types per candidate: same-concept-different-template,
   different-domain-same-surface-form, same-script-unrelated-semantics. 121/121 candidates
   now have at least one measured counterexample (after a bug fix -- see below).
4. **Confirmation-only coverage statistic** (`build_confirmation_only_coverage.py`,
   full 8192-feature dictionary pass): of 3,535 Discovery-dead features, **0 fire in
   Confirmation**. Documented in `candidate_selection_report.md` as a measured result, not
   assumed.

## Important findings

- **19/121 candidates have zero firing tokens in Confirmation; 6 more fall below the
  minimum-evidence rule (≥5 tokens, ≥3 prompts)** -- 25/121 (~21%) will be recorded as
  `insufficient_evidence` for any Confirmation-dependent check in Phase 3, not forced into
  a pass/fail. This is expected and heaviest among the `rare_but_active` and
  `cross_seed_prescreened_nonreproducible_control` strata by construction.
- **25/121 candidates (not the same 25 as above, overlapping but not identical) show
  >50% relative divergence between Discovery and Confirmation on at least one confound
  metric** (domain concentration, cross-domain entropy, or script/language score).
  Preserved as-is, not smoothed over -- several of these are exactly the low-evidence
  candidates where a ratio computed over a handful of Confirmation tokens is expected to
  be noisy; this will be weighed explicitly in Phase 3's confidence assignment rather than
  treated as disqualifying.
- **Counterexample fire rates are informative and plausible**: same-concept-different-
  template fired 291/560 (52%) -- meaningful partial generalization signal;
  different-domain-same-surface-form fired only 49/233 (21%) -- consistent with Milestone
  3's finding that cross-domain embedding neighbors are often template-surface matches
  rather than shared meaning, and most candidate features do not treat them as the "same"
  content; same-script-unrelated-semantics fired 62/195 (32%) -- most Hindi-adjacent
  candidates do NOT fire uniformly on unrelated same-script content, i.e. most are not pure
  script detectors, though a meaningful minority (32%) do fire, which Phase 3 will need to
  weigh per-candidate rather than treat as a blanket result either way.
- **The Discovery-dead-feature-is-also-Confirmation-dead result (0 Confirmation-only
  features) is a genuine, reportable finding**, not just a technicality: it suggests dead
  features in this SAE checkpoint are robustly dead (consistent with M5's own finding that
  dead features largely stabilize early in training), not an artifact of which half of the
  validation set happened to be sampled.

## Failed / bugs caught and fixed

- The first version of `generate_counterexamples.py` built its embedding nearest-neighbor
  index from Confirmation-subset prompts only, but the seed query for each candidate (its
  top Discovery-firing example) is itself a Discovery prompt and was therefore never found
  in that restricted index -- every `same_concept_different_template` and
  `different_domain_same_surface_form` lookup silently returned zero results (only
  `same_script_unrelated_semantics`, which doesn't depend on embedding lookup, worked).
  Fixed by indexing the full embedding set for seed-vector lookup while still restricting
  *returned neighbors* to the Confirmation subset. Re-ran and verified: 121/121 candidates
  now have counterexamples of multiple types with plausible, non-degenerate fire rates
  (reported above) -- caught before being mistaken for a real "no generalization evidence
  anywhere" finding.
- **Second bug, caught during Phase 3 pre-flight verification**: `same_concept_different_template`
  compared raw `(tier, template_cluster_id)` tuples to decide "different template," but
  Tier-3 ("template relationship unknown") prompts all carry `template_cluster_id = None`,
  so two unrelated Tier-3 prompts compared equal -- a Tier-3-seeded candidate could never
  find a qualifying "different template" neighbor among other Tier-3 prompts, even though
  Tier 3 means *unknown*, not *same*. This affected exactly 1/121 candidates (feature 4657,
  whose seed example `DNA_002318` is Tier 3) at the aggregate-count level, but was a latent
  risk for any Tier-3-seeded candidate. Fixed using the same convention already applied in
  `extract_candidate_evidence.py`'s one-per-template logic: Tier-3 prompts use
  `(3, prompt_id)` as their comparison key, so two different Tier-3 prompts are correctly
  treated as different templates. Re-ran: `same_concept_different_template` coverage is
  now 121/121 candidates (up from 120/121), fire rate 334/605 = 55% (consistent with the
  pre-fix 52%, confirming this was a coverage completeness fix, not a result-changing one).
  The pre-fix output is preserved at
  `analysis/data/feature_analysis/_superseded/counterexamples_v2_tier3_aliasing_bug.jsonl`
  for audit; the corrected file at the canonical path is what Phase 3 uses.

## Remaining (Phase 3)

Primary structured review of all 121 candidates (Discovery hypothesis + Confirmation
test), the blind 40-candidate second-reviewer pass, full-dictionary cross-seed matching
(with the full-source-dictionary reverse check for split/merge cases) validated on
Confirmation-subset activations only, cross-capacity checks against SAE-4096/16384, and
the threshold sensitivity sweep.

## Safe to proceed to Phase 3

Yes, pending your review of this report. All Phase 2 outputs are written under
`experiments/semantic_feature_discovery/` only; no Phase 1 output, prior-milestone
artifact, or the 121-candidate cohort was altered. The Discovery/Confirmation boundary was
followed strictly throughout (confound diagnostics kept the two subsets separate rather
than pooling them; evidence and counterexamples were drawn from the subset the plan
specifies for each purpose).

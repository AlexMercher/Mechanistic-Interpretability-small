# Milestone 6 — Phase 3 Report (Independent interpretation, cross-seed/capacity matching, sensitivity sweep)

The 121-candidate cohort and Phase 1/2 outputs were not altered. All Phase 3 work reads
Phase 1/2 outputs as frozen inputs. Confirmed: `git diff --stat` against `raw/`,
`normalized/`, `processed/`, and all four prior-milestone `experiments/` directories is
empty; `candidates.jsonl`'s checksum is unchanged from Phase 1.

## Incident: first second-reviewer attempt failed, nothing lost

The first attempt to run the blind second review spawned a single Opus agent directly on
the raw 40-candidate evidence bundle. It hit a session/API limit mid-task (while still
digesting the raw file, before writing any output) and terminated with no deliverable.
**Verified nothing was lost**: the 9 primary-review batches (already merged into
`primary_reviews.jsonl`, 121/121 candidates, 0 duplicates, 0 missing) were produced by
separate, already-completed agent calls and were unaffected. The failed agent's own
partial state was its internal context only; no file was written or corrupted.

**Root-caused and fixed**: added a new `CLAUDE.md` rule ("Opus agent usage") — Opus agents
must never be spawned directly on raw/heavy data-reading work; a cheaper
digestion/condensation step must happen first, with Opus receiving only the
pre-gathered, condensed evidence. Implemented via `condense_second_reviewer_bundles.py`
(pure mechanical extraction — top-5 examples per quantile instead of full example sets,
same confound/counterexample data, still blind-compatible, no primary-reviewer labels),
reducing the per-candidate payload substantially, and split the Opus review itself into
two 20-candidate batches instead of one 40-candidate call. Both batches completed
successfully on retry.

## Completed

1. **Primary structured review, all 121 candidates** (9 batched Sonnet agents, merged
   into `primary_reviews.jsonl`): 10-field structured judgment + 9-class classification
   per candidate, using Discovery evidence as the hypothesis and Confirmation evidence +
   counterexamples as the independent test, per the type-conditional domain-concentration
   check and tiered-template-evidence rules frozen in the plan.

   | final_classification | count |
   |---|---:|
   | coherent_domain_specific | 35 |
   | template_driven | 20 |
   | insufficient_evidence | 18 |
   | polysemantic_or_mixed | 16 |
   | rare_candidate | 11 |
   | uninterpretable | 8 |
   | coherent_semantic | 5 |
   | script_or_language_marker | 5 |
   | domain_marker | 3 |

   | confidence | count |
   |---|---:|
   | medium | 35 |
   | low | 31 |
   | unvalidated | 29 |
   | high | 26 |

   Only 40/121 (33%) landed in a positive semantic category (`coherent_semantic` +
   `coherent_domain_specific`); the remainder are confounds, mixed/polysemantic, rare, or
   insufficiently evidenced — consistent with the project's explicit instruction not to
   inflate validation counts.

2. **Blind second review, 40-candidate deterministic sample** (two 20-candidate Opus
   batches, reviewing only the condensed digest, with no access to primary labels):
   merged into `second_reviewer_blind.jsonl`.

3. **Inter-reviewer agreement** (`compute_reviewer_agreement.py`), both original and
   convention-normalized:

   | Metric | Original | Normalized (rare_candidate + insufficient_evidence merged) |
   |---|---:|---:|
   | Exact-match rate | 50.0% (20/40) | 62.5% (25/40) |
   | Cohen's kappa | 0.390 (fair) | 0.524 (moderate) |

   The normalized metric exists because `rare_candidate` and `insufficient_evidence` both
   mean "too little evidence to judge" — they differ only in which exact label a reviewer
   reached for, not in substance. **Both numbers are reported; neither replaces the
   other** — the original metric is the literal agreement on the 9-class schema as
   specified, the normalized metric isolates how much of the disagreement was a labeling
   convention versus a real difference in judgment.

   Of the 20 original disagreements:
   - **5 labeling-convention disagreements** (features 1324, 1717, 1869, 2154, 2155): all
     `rare_candidate` (primary) vs. `insufficient_evidence` (second) — same underlying
     finding (near-zero Discovery/Confirmation evidence), different label choice.
   - **15 substantive disagreements** (listed verbatim in `reviewer_agreement.json`,
     `substantive_disagreements` field) — genuine differences in interpretation, not
     label-convention artifacts.

   **Systematic tendency, quantified**: among the 15 substantive disagreements, the second
   (Opus) reviewer upgraded 7 candidates to a coherent label (`coherent_domain_specific` or
   `coherent_semantic`) where the primary reviewer had assigned a confound/mixed/
   uninterpretable label (features 1974, 2201, 2281, 2920, 2922, 3160, 3734, 3908 — note
   3160 is primary `polysemantic_or_mixed` → second `coherent_semantic`, not
   `coherent_domain_specific`, but counted under "any coherent label"), versus only 3 cases
   of the second reviewer downgrading a primary coherent call (features 95, 720, 838 — all
   primary `coherent_domain_specific`/`coherent_semantic` → second
   `polysemantic_or_mixed`/`script_or_language_marker`). **Net 7-vs-3 asymmetry**: the
   second reviewer was more willing to credit domain-specific/semantic coherence than the
   primary reviewer in this comparison. This is reported as an observed tendency in this
   specific pairing, not a claim that either reviewer is "more correct" — Milestone 6's
   final report states plainly that roughly a third of this sample's classifications are
   reviewer-dependent, not a single settled fact.

### Cross-seed / cross-capacity reconciliation (corrected — see `cross_seed_capacity_reconciliation.json`)

The original Phase 3 report presented the cross-seed (121-candidate denominator) and
cross-capacity (61-candidate denominator) tables without making the denominator
difference explicit, and the numeric closeness of cross-seed seed_2's `reproduced_strong`
count (67) and cross-capacity's eligible-candidate count (61) created a real risk of
conflating two differently-scoped analyses. Corrected here with an explicit, fully
reconciled accounting.

**Cross-seed matching — full 121-candidate cohort, six mutually exclusive classifications:**

| Classification | seed 1 | seed 2 |
|---|---:|---:|
| reproduced_strong | 60 | 67 |
| numerically_matched_semantically_inconsistent | 26 | 20 |
| possible_split | 0 | 0 |
| possible_merge | 2 | 0 |
| not_reproduced | 8 | 9 |
| insufficient_evidence | 25 | 25 |
| **Sum** | **121** | **121** |

- **The "67" is a cell value, not a denominator**: it is seed 2's `reproduced_strong`
  count, out of the full 121-candidate cohort (67/121 = 55.4%) — not a count out of any
  smaller eligible subset.
- **The 25 `insufficient_evidence` candidates are the SAME 25 feature_idx values under
  both seed 1 and seed 2** (verified programmatically: the two 25-element sets are
  identical). This is expected: insufficiency is a property of the candidate's own
  Confirmation-subset firing volume (<5 tokens or <3 unique prompts) — a fact about the
  candidate itself, independent of which target seed's dictionary it's being matched
  against — so it cannot differ by seed.
- **No double-counting**: each of the 121 candidates receives exactly one of the six
  classifications per seed; every row sums to exactly 121.

**Cross-capacity matching — restricted 61-candidate subset, a DIFFERENT population:**

| Classification | width 4096 | width 16384 |
|---|---:|---:|
| reproduced_strong | 35 | 49 |
| numerically_matched_semantically_inconsistent | 24 | 11 |
| possible_split | 0 | 0 |
| possible_merge | 1 | 0 |
| not_reproduced | 1 | 1 |
| insufficient_evidence | 0 | 0 |
| **Sum** | **61** | **61** |

- **The 61-candidate denominator is NOT the full cohort.** It is exactly the subset of the
  121 candidates whose *primary review* reached >= medium confidence (26 `high` + 35
  `medium` = 61), per the plan's explicit restriction of cross-capacity checking to
  confidence-worthy candidates. The other 60 of 121 (29 `unvalidated` + 31 `low`) are
  **not** cross-capacity-checked, but they **are** fully present and classified in the
  121-row cross-seed tables above — every one of the 121 candidates is accounted for
  somewhere in this phase's work, just not in the same table.
- **Why zero `insufficient_evidence` in the cross-capacity table**: the >=medium-confidence
  filter that defines this 61-candidate population already excludes every candidate that
  would hit the <5-token/<3-prompt floor (thin Confirmation evidence was already reflected
  in a low/unvalidated confidence at review time) — so by construction, none of the 61
  fall below that floor again here. This is a consequence of the confidence filter, not an
  independent finding that cross-capacity matching never encounters sparse evidence.
- **The two denominators (121 vs. 61) must never be combined, averaged, or compared
  cell-for-cell** — they describe different populations answering different questions
  (cross-seed: "does this candidate reproduce in another SAE-8192 seed, at all 121
  candidates?"; cross-capacity: "does this candidate reproduce at a different dictionary
  width, among only the candidates confident enough to be worth checking?").

4. **Cross-seed reproducibility** (`cross_seed_reproducibility.py`, full SAE-8192 seed-1/
   seed-2 target dictionaries, full-source-dictionary reverse check, Confirmation-subset-
   only activation-pattern/top-prompt-overlap validation): a real bug was caught and fixed
   before this result was trusted (see below). Final numbers:

   | Classification | seed 1 | seed 2 |
   |---|---:|---:|
   | reproduced_strong | 60 | 67 |
   | insufficient_evidence | 25 | 25 |
   | numerically_matched_semantically_inconsistent | 26 | 20 |
   | not_reproduced | 8 | 9 |
   | possible_merge | 2 | 0 |

   **Bug caught**: the first run classified every unmatched-decoder candidate as
   `not_reproduced` without checking the minimum-evidence rule first, so all 25
   insufficient-Confirmation-evidence candidates were silently mislabeled `not_reproduced`
   — exactly the "never downgrade insufficient evidence" violation the plan explicitly
   forbids. Fixed by checking minimum evidence before any other branch, regardless of
   decoder-match status. Re-run confirms 25 `insufficient_evidence` per seed, matching
   Phase 2's known count of low-Confirmation-evidence candidates exactly.

5. **Cross-capacity check** (`cross_capacity_check.py`, same methodology, SAE-4096/16384
   seed 0, restricted to the 61/121 candidates at >= medium confidence):

   | Classification | width 4096 | width 16384 |
   |---|---:|---:|
   | reproduced_strong | 35 | 49 |
   | numerically_matched_semantically_inconsistent | 24 | 11 |
   | possible_merge | 1 | 0 |
   | not_reproduced | 1 | 1 |

6. **Threshold sensitivity sweep** (`sensitivity_sweep.py`, cosine ∈ {0.25,0.3,0.4,0.5} x
   margin ∈ {0.05,0.1,0.2}, 12 combinations x 2 target seeds): `insufficient_evidence`
   count is exactly 25 at every single threshold combination (as it must be — it's gated
   on Confirmation-subset firing volume, not on the matching threshold at all). The
   `reproduced_strong` count is stable at cosine<=0.3 (60/67 for seeds 1/2) and degrades
   gracefully as the threshold tightens (58/65 at cosine=0.4, 58/63 at cosine=0.5) — no
   cliff-edge sensitivity to the exact 0.3/0.1 choice, supporting (but not proving) that
   the headline reproducibility numbers are not an artifact of one arbitrary threshold.

## Important findings surfaced by reviewers (illustrative, not exhaustive — full detail in
`primary_reviews.jsonl` / `second_reviewer_blind.jsonl`)

- Several features initially flagged by automated selection statistics as "broadly
  cross-domain" or "domain-concentrated" were reclassified by reviewers after inspecting
  actual text: e.g. feature 6128 looked cross-domain by aggregate stats but counterexamples
  showed it keys narrowly on the literal entity "Indian Opinion" (Gandhi's newspaper), not
  a general concept.
- Feature 7210 (highest raw token frequency in the entire pool, firing on 83% of prompts)
  is a constant-activation lock to one recurring GANDHI carrier sentence — its "200+
  template families" count was an artifact of per-topic clustering, not genuine
  generalization, exactly the template-vs-semantics confound this milestone was built to
  catch.
- Feature 6403 and feature 720 are clean script-form detectors (Hinglish/Romanized Hindi
  vs. Devanagari), directly confirmed by ~1000x activation drops on same-topic,
  different-script counterexamples.
- Several small clusters of candidates independently converged on overlapping territory
  (e.g. three Hindi grammar/syntax features 3498/3997/4094; two DNA single-strand features
  4455/4769; two Hindi grammar features 2728/2974) — flagged by reviewers as likely
  correlated/redundant rather than independent discoveries, feeding directly into Phase
  4's relationship analysis.
- Feature 95 is the most defensible cross-domain discovery in the pool: a lexical-semantic
  feature for "physical/physically" in its material/bodily sense, firing across DNA,
  EMOTIONS, and SCIENCE, and correctly silent on structurally identical prompts that don't
  contain the word — though the second reviewer rated this one `polysemantic_or_mixed`
  rather than `coherent_semantic` (one of the 20 disagreements).

## Failed / bugs caught and fixed (this phase)

1. Direct Opus-agent spawn on raw data hit a session limit — process fixed per the new
   CLAUDE.md rule (digest first, then Opus), no data lost.
2. `cross_seed_reproducibility.py`'s minimum-evidence check was only evaluated inside the
   "decoder match found" branch, silently mislabeling all unmatched low-evidence
   candidates as `not_reproduced` instead of `insufficient_evidence` — fixed, verified
   against Phase 2's independently-known count of 25 low-evidence candidates.

## Remaining (Phase 4)

Co-activation analysis (token+prompt level, frozen fallback hierarchy), decoder-space
relationship analysis, cross-domain relationship analysis (DNA<->SCIENCE,
EMOTIONS<->HINDI), and the feature relationship graph — restricted to candidates with a
real classification, confidence-gated.

## Safe to proceed to Phase 4

Yes, pending your review. All Phase 3 deliverables are complete and internally
cross-checked (cross-seed insufficient-evidence counts match Phase 2's independently
known figure; sensitivity sweep shows no threshold cliff-edge; reviewer disagreements are
logged verbatim rather than resolved by fiat). The two bugs caught this phase were fixed
and the affected outputs re-run before being trusted, not patched over silently.

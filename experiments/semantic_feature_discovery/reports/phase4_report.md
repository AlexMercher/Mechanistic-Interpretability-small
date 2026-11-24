# Milestone 6 — Phase 4 Report (Feature relationship analysis)

The 121-candidate cohort remains frozen. Phase 4 analysis is restricted to the 61
candidates at >= medium primary-review confidence (`analysis/data/relationship_analysis/eligible_candidates.json`)
— the same population already defined and used for cross-capacity checking in Phase 3,
reused here per the plan's instruction that relationship confidence is gated by each
candidate's own individual validation confidence. No code in this phase re-selects,
re-scores, or alters the 121-candidate pool or Phase 1-3 outputs.

## 1. Four distinct evidence types (kept separate throughout, never collapsed)

1. **Co-activation / statistical correlation** (`coactivation_analysis.py`) — pooled and
   confound-controlled Pearson/Spearman correlation of activation values. A statistical
   fact about when two features fire together; no claim about mechanism.
2. **Decoder-space similarity** (`decoder_relationship_analysis.py`, `decoder_cosine`) —
   geometric similarity of the two features' learned dictionary directions. A property of
   the trained weights, computed with zero activation data — used as a diagnostic, never
   as a semantic verdict on its own.
3. **Activation-pattern similarity / top-prompt overlap** (same script,
   `activation_pattern_cosine`, `top_prompt_jaccard`) — functional similarity measured
   from behavior on held-out data, independent of decoder geometry.
4. **Reviewer-judged semantic similarity** (same script, `interpretation_lexical_jaccard`,
   plus the qualitative primary-review text) — the only evidence type that reflects an
   actual read of what each feature means, and a crude, deterministic proxy for it (word
   overlap between two independently-written interpretations), not an NLP-level judgment.

Every relationship claim in this report, and every edge in the graph, states which of
these four (or which combination) supports it.

## 2. Co-activation analysis

**Methodology (frozen before any pair was inspected, per B.7):** pooled correlation
computed first, both subsets, both levels (1,830 candidate pairs); pairs selected for the
expensive confound-controlled analysis solely by `|Confirmation-subset, prompt-level,
pooled Pearson r| >= 0.3` — a compute-bounding filter decided before any stratified
number existed, not a conditioning-strategy choice. For each selected pair, the
**frozen three-level cascade** (domain x template_family x language_form -> domain x
language_form -> domain, excluding below 20 prompts at every level, never choosing a
level because it produces a stronger correlation) was applied per domain.

- **1,830/1,830 pairs** got pooled correlations on both subsets, both levels.
- **126/1,830 pairs (6.9%)** passed the Confirmation-prompt-level pooled filter and went
  through the controlled cascade.
- **Prompt-level correlation is the primary inferential result at every level** reported
  above; **token-level correlation is secondary/descriptive only**, and the one
  token-level number reported per domain-stratum (the domain-level/level-3 token-pooled
  figure) carries a **prompt-cluster bootstrap 95% CI** (1,000 resamples of *prompts*, not
  tokens) rather than any token-count-based interval — token counts and contributing
  prompt counts are both reported side by side for every correlation at every level.
- **Confound effect, measured directly**: of the 126 selected pairs, **104 (82.5%)
  survived conditioning** (at least one domain-stratum correlation kept the same sign and
  >=50% of the pooled magnitude); **22 (17.5%) disappeared or weakened substantially**
  once conditioned on domain x template x language — a real, reportable confound effect,
  not a null result to be explained away.
- **Discovery-vs-Confirmation robustness**: of the pairs with |pooled Confirmation r| >=
  0.3, **114 also showed |pooled Discovery r| >= 0.3 with the same sign** (consistent
  across both independent halves of the validation set); **12 were strong in Confirmation
  only** (not replicated in the independent Discovery-subset correlation) — reported as a
  weaker, single-subset finding, not elevated to the same status as the 114 consistent
  pairs.
- Some fine-grained strata produced `NaN` correlations (one feature constant/all-zero
  within that bucket) — expected for sparse features in small buckets, left as `NaN`
  (undefined), never silently treated as zero or excluded without the explicit
  `ConstantInputWarning`-driven reason.

## 3. Decoder-space relationship analysis

All 1,830 pairs scored on `decoder_cosine`, `activation_pattern_cosine`,
`top_prompt_jaccard` (Confirmation subset), and `interpretation_lexical_jaccard`
(primary-reviewer text).

- **37 pairs (2.0%) show an explicit disagreement** between decoder similarity and the
  other three signals: either high decoder cosine with low functional/lexical agreement
  (`decoder_high_semantic_low`), or the reverse (`semantic_high_decoder_low`) — logged
  individually in `decoder_relationship_analysis.json`'s `disagreements` field. This
  directly operationalizes "decoder similarity is a diagnostic, not a semantic verdict":
  in these 37 cases, trusting decoder cosine alone would have given the wrong answer
  about functional/interpretive relatedness.

## 4. Cross-domain relationship analysis

Two distinct kinds of evidence, not conflated:

- **(a) Single-feature cross-domain generalization**: 5 candidates in the eligible pool
  were independently classified `coherent_semantic` by primary review (features 95, 1974,
  4485, 7399, 7640) — a single feature firing meaningfully across more than one domain.
  This is a property of one feature, not a relationship between two.
- **(b) Two-feature parallel-concept hypotheses**, seeded from Milestone 3's
  teacher-reviewed DNA<->SCIENCE / EMOTIONS<->HINDI concept pairs (26 non-`no_relationship`
  pairs available): checked whether a domain-A-dominant eligible candidate fires
  above-floor on concept A's Confirmation-subset prompts AND a domain-B-dominant eligible
  candidate fires above-floor on concept B's prompts. **A lenient 5% firing-rate floor
  produced 1,107 raw (feature, feature, concept-pair) intersections** — this count is
  explicitly flagged as mostly noise from the permissive floor combined with ~26 concept
  pairs x many domain-matched candidates, **not 1,107 discoveries**. Only the **top 20,
  ranked by combined evidence strength** (`rate_a x rate_b x log(min concept-prompt
  count)`), are reported as candidate hypotheses, every one explicitly labeled
  `low_hypothesis_only` confidence with the caveat that firing on "parallel" content in
  two different domains is not evidence the two features represent the same concept.

A real bug was caught and fixed here: the single-feature generalizer list initially came
back empty because the script read `cross_domain_entropy_norm` from the wrong source file
(the raw per-feature inventory, which never has that field) instead of `candidates.jsonl`
(where `select_candidates.py` actually wrote it) — fixed and re-verified against the known
5-candidate `coherent_semantic` list from Phase 3.

## 5. Typed relationship graph

**Nodes: 61** (the eligible candidates only — not all 8192 features). **Edges: 139**,
across the edge types that actually found qualifying evidence:

| Edge type | Count | Rule |
|---|---:|---|
| `co_activation` | 64 | Pooled Confirmation-prompt-level \|r\|>=0.3, not already typed by a stronger rule |
| `semantic_similarity` | 62 | Both endpoints independently reviewed `coherent_*`, AND confound-controlled correlation holds (\|r\|>=0.3) in >=1 domain stratum |
| `cross_domain_connection` | 13 | Top-ranked M3-seeded parallel-concept hypothesis, both endpoints in the eligible pool |
| `possible_merge` | 0 | No pair reached decoder_cosine>=0.5 AND activation_pattern_cosine>=0.5 AND top_prompt_jaccard>=0.3 simultaneously |
| `shared_surface_form` | 0 | No pair of template_driven/domain_marker candidates reached the decoder+activation agreement threshold |
| `mutual_exclusion` | 0 | No pair showed negative controlled correlation in a majority of its qualifying domain strata |

**The three zero-count edge types are a reported negative finding, not an omission**: the
rules were applied to the full 1,830-pair space exactly as defined; no pair in this
specific 61-candidate pool met the (deliberately strict) possible-merge, shared-surface-
form, or mutual-exclusion thresholds. This is worth stating plainly rather than silently
dropping the edge types from the schema.

Every edge carries `evidence` (the raw metric values), `confidence`
(`medium`/`low_to_medium`/`low`/`low_hypothesis_only`), and `provenance` (which script
produced it) — exported in full in `relationship_graph.json` (node-link format,
reproducible from the upstream JSON files, never hand-edited).

**No causal claim is made anywhere in this phase.** Every edge description explicitly
states what it does and does not establish (e.g. `semantic_similarity`'s note: "Correlational
evidence only -- not a causal or 'same concept' claim"; `co_activation`'s note: "no claim
about why -- shared template/domain/script confound is plausible and not ruled out").

## 6. Failed / bugs caught and fixed this phase

1. `cross_domain_analysis.py` read `cross_domain_entropy_norm` from the wrong source file
   (inventory instead of `candidates.jsonl`), silently zeroing out the entire
   single-feature-generalizer section — caught by cross-checking against the known
   5-candidate list from Phase 3, fixed, re-verified.
2. The M3-seeded two-feature cross-domain check's initial 5% firing-rate floor produced
   1,107 raw intersections — recognized as noise rather than presented as findings;
   added an explicit evidence-strength ranking and capped the reported list at 20, with
   the raw count kept visible as context, not hidden.

## 7. Safe to proceed to Phase 5

Yes, pending your review. All Phase 4 deliverables are complete, the 121-candidate
cohort and all Phase 1-3 outputs remain unmodified, and both negative findings (zero
possible_merge/shared_surface_form/mutual_exclusion edges; 22 pairs whose correlation
disappeared under conditioning) are reported explicitly rather than omitted.

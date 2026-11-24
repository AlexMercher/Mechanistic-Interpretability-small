# Milestone 6 — Findings: Semantic Feature Discovery, Validation & Relationship Analysis

**Scope.** This document reports what was measured, by whom (mechanical script vs. LLM
reviewer), and with what confidence, about a stratified sample of 121 SAE-8192 (seed 0)
features extracted from **Qwen3-1.7B's layer-14 residual stream**. All findings are
specific to this one layer and this one dictionary unless explicitly stated otherwise.
**No claim here generalizes to any other layer of Qwen3-1.7B's 28 transformer layers,**
nor to the full 8,192-feature dictionary beyond the 121 candidates actually reviewed
(1.5% of the dictionary). Every figure referenced below lives under `analysis/plots/`
and is listed with its source data in `analysis/plots/figure_manifest.json`; every
number below is reproducible from a named file under `analysis/data/`.

---

## 1. What this milestone did and did not do

Milestone 6 built a full feature inventory of SAE-8192 seed 0 on an exact,
deterministic Discovery/Confirmation split of the frozen Milestone-4 validation set
(1,265 / 1,268 prompts), selected 121 candidates under ten pre-frozen quota rules,
independently reviewed every candidate's semantics (plus a blind 40-candidate second
opinion), checked cross-seed and cross-capacity reproducibility against the complete
target dictionaries (never index-aligned), and analyzed relationships among the
61 candidates that reached >=medium review confidence. It did not retrain Qwen3, retrain
any SAE, regenerate any corpus/embedding/activation artifact, or modify any Milestone
1-5 output — verified programmatically in Section 11 below.

---

## 2. Domain-specific semantic features

35/121 candidates (29%) were classified `coherent_domain_specific`: a single, well-defined
concept, consistently domain-concentrated, and that domain-concentration is *expected and
not penalized* per the type-conditional confidence criteria (B.4). Representative examples:

- **Feature 720** — fires on HINDI prompts asking the model to distinguish the
  nuance/register between two near-synonymous Hindi words (e.g. "Anurodh" vs. "Prarthana",
  "Maafi" vs. "kshama"). A genuine lexical-semantic task-type feature, not a script marker.
- **Feature 838** — fires on HINDI prompts involving context-dependent/pragmatic meaning:
  deixis ("यहाँ"/"वहाँ"), context-shifted word meaning, politeness/reassurance formulas.
- **Feature 6815** — fires strongly (activation 1.5-2.0) specifically on the literal
  SCIENCE subdomain label "Astrophysics" appearing in a prompt — reviewed as
  `domain_marker` rather than `coherent_domain_specific` because it tracks a label string,
  not the underlying concept.

See `analysis/plots/semantic_selectivity/b1_classification_counts.png` for the full
9-class breakdown and `b2_concentration_vs_entropy_by_class.png` for where
domain-specific features sit on the concentration/entropy plane relative to generalists
and artifacts.

**Domain imbalance context (confound, not a finding about meaning):** the Discovery
subset's token-level domain share is far from uniform (HINDI 38.6%, SCIENCE 22.0%,
EMOTIONS 16.3%, GANDHI 11.7%, DNA 11.3% — `analysis/plots/model_representation/g3_domain_token_baseline.png`),
because HINDI's native-script tokenization produces more tokens per prompt despite equal
prompt counts per domain. The B.1 domain-concentration formula normalizes by this
baseline specifically so a feature is not called "domain-concentrated" merely because its
domain has more tokens.

## 3. Cross-domain semantic representations

Two structurally different kinds of cross-domain evidence were measured and never
conflated (`analysis/plots/cross_domain_analysis/`):

**(a) Single-feature generalization — measured, moderate-confidence finding.**
5/61 eligible candidates (8%) were independently classified `coherent_semantic`: one
feature, one concept, firing across multiple domains. Two examples:
- **Feature 95** — fires on the literal lexeme "physical"/"physically" in its
  material/bodily sense ("physical presence," "physically bridge," "physical danger"),
  appearing incidentally across SCIENCE, EMOTIONS, and GANDHI prompts. A lexical-semantic
  feature for one word sense, not a topic feature.
- **Feature 1974** — fires on tokens where prompts juxtapose or comparatively evaluate two
  named entities/concepts (e.g. "Nehru vs. documentary evidence," "Criticism of Gandhi
  alongside Smuts negotiations") — a structural/relational pattern that recurs across
  GANDHI and other domains' comparative-reasoning prompts.

**(b) Two-feature parallel-concept hypotheses — exploratory, low-confidence only.**
Seeded from Milestone 3's teacher-reviewed DNA<->SCIENCE / EMOTIONS<->HINDI concept pairs,
a lenient 5% per-feature firing-rate floor produced **1,107 raw (feature, feature,
concept-pair) intersections** — explicitly recognized as mostly noise from that
permissive floor combined with ~26 concept pairs x many domain-matched candidates, **not
1,107 discoveries**. Only the top 20 by evidence-strength score are reported
(`e2_top_cross_domain_hypotheses.png`), every one labeled `low_hypothesis_only`: firing on
"parallel" content in two domains is not evidence the two features represent the same
concept, and no causal or shared-representation claim is made for any of them.

## 4. Template and script artifacts (confounds, explicitly quantified)

20/121 candidates (17%) were classified `template_driven` and 5/121 `script_or_language_marker`,
plus 3 `domain_marker` — together confirming Milestone 5's qualitative flag that these
confounds are real and non-negligible in this dictionary, now with concrete evidence:

- **Feature 1093** — fires almost exclusively on one fixed EMOTIONS template skeleton
  ("Scenario: [filler]. Task: Reason about plausible perspectives..."), confirmed via the
  Tier-1 near-duplicate-family template_concentration metric.
- **Feature 1135** — position/template-locked on a fixed DNA skeleton, with activation
  value nearly *identical* (1.0143) across dozens of otherwise-unrelated completions — the
  strongest kind of template-artifact evidence available (numerical identity, not just
  topical similarity).
- **Feature 553** — a broad Hindi-language/script marker: fires across a huge,
  heterogeneous range of Hindi content with no shared topic, reviewed `script_or_language_marker`
  rather than any Hindi-semantic label.

Template evidence is explicitly **tiered**, never collapsed into one count
(`analysis/plots/confound_analysis/c1_template_tier_population.png`): of 25,000 corpus
prompts, 6,333 (25.3%) sit in a Tier-1 confirmed near-duplicate family, 1,533 (6.1%) are
Tier-2 skeleton-fallback matches, and 17,134 (68.5%) are Tier-3 unclustered —
**their template relationship to any other prompt is unknown and is never counted as "an
independent template."** `c2_template_concentration_by_class.png` shows `template_driven`
candidates cluster at markedly higher Tier-1 template-concentration than every other
class, which is the designed, falsifiable check that the label is doing real work rather
than being assigned on vibes.

**Discovery-vs-Confirmation confound stability** (`c4_confound_divergence_flags.png`):
confound diagnostics were computed independently on both subsets for every candidate;
the vast majority of domain-concentration / entropy / script-score values are flagged
`consistent` between subsets, with divergences reported explicitly rather than hidden —
this is a direct check that the confound story is not an artifact of the Discovery subset
alone.

## 5. Feature relationships

Four evidence types were kept structurally distinct throughout and are never merged into
one "relatedness" score: co-activation/statistical correlation, decoder-direction
similarity, activation-pattern/top-prompt similarity, and reviewer-judged semantic
similarity (lexical-overlap proxy). The resulting typed graph has **61 nodes (the
>=medium-confidence candidates) and 139 edges**
(`analysis/plots/feature_relationships/f1_relationship_graph.png`):

| Edge type | Count | What it means |
|---|---:|---|
| `co_activation` | 64 | Statistical co-firing above a pre-registered threshold; explicitly not a mechanistic claim |
| `semantic_similarity` | 62 | Both features independently reviewed coherent AND confound-controlled correlation survives in >=1 domain stratum |
| `cross_domain_connection` | 13 | Top-ranked M3-seeded hypothesis pairs (Section 3b), hypothesis-only |
| `possible_merge` | **0** | No pair reached joint decoder+activation+top-prompt agreement — negative finding |
| `shared_surface_form` | **0** | No template/domain-artifact pair reached decoder+activation agreement — negative finding |
| `mutual_exclusion` | **0** | No pair showed consistent negative controlled correlation — negative finding |

**Confound-controlled co-activation is a real, measured effect, not a null result to
explain away**: of the 126 pairs whose pooled Confirmation-subset correlation passed the
screening filter, **104 (82.5%) survived conditioning** on domain x template x language,
while **22 (17.5%) disappeared or weakened substantially** — meaning roughly 1 in 6 raw
co-activation signals in this candidate pool was at least partly a shared-confound
artifact rather than a feature-level relationship. Prompt-level correlation was the
primary inferential statistic throughout; token-level correlation is reported only
descriptively, with any interval coming from a prompt-cluster bootstrap (never a
token-count-based formula), because tokens within a prompt are not independent draws.

**Decoder geometry and functional/semantic similarity disagree on 37/1,830 pairs (2.0%)**
(`f3_decoder_vs_activation_disagreements.png`) — concrete, counted evidence that decoder
cosine alone would have given the wrong answer about relatedness in those cases, which is
why it was used only as a diagnostic signal throughout, never a semantic verdict.

**No causal claim is made about any edge.** `co_activation`'s provenance note states
plainly that a shared template/domain/script confound remains plausible and unruled-out
for any individual pair even after the controlled analysis; `semantic_similarity` is
stated as correlational evidence only.

## 6. Reproducibility across seeds and dictionary widths

**Cross-seed** (full 121-candidate cohort, matched against the *complete* 8,192-feature
target dictionary, Confirmation-subset-validated — `analysis/plots/cross_seed_stability/d1_cross_seed_classification.png`):

| Classification | Seed 1 | Seed 2 |
|---|---:|---:|
| reproduced_strong | 60 | 67 |
| numerically_matched_semantically_inconsistent | 26 | 20 |
| possible_split | 0 | 0 |
| possible_merge | 2 | 0 |
| not_reproduced | 8 | 9 |
| insufficient_evidence | 25 | 25 |

The same 25 feature indices are `insufficient_evidence` under **both** seeds (verified:
the two 25-element sets are identical) — insufficiency is a property of the candidate's
own Confirmation-subset firing volume, independent of which seed it is matched against.
**26-35 candidates per seed are a decoder match that does not survive functional/semantic
validation** (`numerically_matched_semantically_inconsistent`) — a sizable minority for
which decoder-cosine matching alone would have overstated reproducibility.

**Cross-capacity** (restricted to the 61 >=medium-confidence candidates — a *smaller,
differently-defined* population, never conflated with the 121-candidate cross-seed
denominator — `d2_cross_capacity_classification.png`): width 4096 reproduces 35/61
strongly and 24/61 numerically-but-not-semantically; width 16384 reproduces 49/61 strongly
and 11/61 numerically-but-not-semantically. `insufficient_evidence`=0 at both widths by
construction (the >=medium filter already excludes thin-evidence candidates), not an
independent finding.

**Threshold sensitivity** (`d3_sensitivity_sweep_heatmap.png`): the headline 0.3
cosine / 0.1 margin thresholds are explicitly heuristics. Across all 12 cosine x margin
combinations tested, `reproduced_strong` counts shift moderately (reported in the heatmap)
while `insufficient_evidence` stays fixed at exactly 25/121 at every combination —
confirming the sweep only moves the match/no-match boundary, never the underlying
evidence-floor outcome.

## 7. Independent review and inter-reviewer agreement

121 candidates were reviewed by a primary (batched Sonnet) reviewer; a blind, independent
40-candidate second opinion (Opus, reading only raw evidence, never the primary label) was
collected per the pre-registered B.5 sample. Agreement:

- **Original**: exact-match rate 0.500, Cohen's kappa 0.390.
- **Normalized** (merging `rare_candidate`/`insufficient_evidence`, since both mean "too
  little evidence" and differ only by labeling convention): exact-match rate 0.625,
  Cohen's kappa 0.524.
- **5 of 20 disagreements** were this exact labeling-convention split (`rare_candidate` vs.
  `insufficient_evidence` on the same evidence); **15 were substantive** label disagreements.
- **Systematic tendency, this comparison only**: among the 15 substantive disagreements,
  the second reviewer upgraded 7 candidates *to* a coherent label and downgraded 3 *from*
  one — net more willing to credit domain-specific/semantic coherence than the primary
  reviewer. This is reported as an observed tendency in this specific comparison, not a
  claim about which reviewer is more "correct."

A kappa in the 0.39-0.52 range indicates moderate, not strong, agreement — interpretive
labeling of SAE features at this granularity is not a fully reliable, mechanically
reproducible process even with two independent LLM reviewers, and this uncertainty is
reported rather than resolved by picking one reviewer's labels as ground truth.

## 8. Taxonomy correspondence

Milestone 3's teacher-reviewed semantic hierarchy and cross-domain concept-relationship
pairs were used **only as a search seed** for Section 3(b)'s cross-domain hypotheses —
never as ground truth for any SAE feature's identity, and never used to force a feature
into the taxonomy. No SAE feature was relabeled, merged, or validated against the
taxonomy as if the taxonomy were a gold standard; the taxonomy itself was not modified.
Where a feature's reviewer interpretation happens to resemble a taxonomy concept (e.g.
feature 1974's comparative-juxtaposition pattern touching GANDHI-domain argumentative
prompts), that resemblance is reported as a qualitative observation, not a validated
correspondence.

## 9. Negative findings and limitations (stated, not omitted)

1. **Three of six relationship-edge types found zero qualifying pairs** among the 61
   eligible candidates: `possible_merge`, `shared_surface_form`, `mutual_exclusion`. The
   pre-registered thresholds were applied to the full 1,830-pair space; nothing in this
   specific pool met them. This milestone did not find redundant/mergeable feature pairs,
   confirmed surface-form-sharing pairs, or mutually-exclusive pairs in this candidate set
   — a real result about this pool, not a gap in the method.
2. **25/121 candidates (21%) are `insufficient_evidence`** for cross-seed/capacity
   validation — too few Confirmation-subset firing tokens/prompts to test at all. This is
   a first-class outcome, never silently downgraded to "not reproduced."
3. **0 Confirmation-only features** exist among the 3,535 Discovery-dead SAE-8192 features
   (`analysis/plots/model_representation/g2_confirmation_only_coverage.png`) — but rare-feature
   *selection* in this milestone was structurally conditional on firing in Discovery; this
   milestone makes no claim of exhaustive rare-feature coverage of the dictionary.
4. **Only 121/8,192 features (1.5%) were ever reviewed.** The funnel in
   `g1_representation_funnel.png` must not be read as "X% of the dictionary is coherent" —
   it describes the reviewed stratified sample only.
5. **Inter-reviewer kappa (0.39-0.52) indicates only moderate agreement** — labeling
   uncertainty is real and reported, not resolved.
6. **1,107 raw cross-domain two-feature intersections are explicitly noise**, not
   findings; only the top-20 ranked, hypothesis-only results are reported.
7. **16/121 candidates (13%) are `polysemantic_or_mixed`** — e.g. feature 309 bundles a
   template-locked GANDHI trigger with a weaker inference-cue signal; feature 1855 shows no
   single coherent concept across unrelated DNA/EMOTIONS/SCIENCE examples. These are
   reported as genuinely mixed, not forced into a single clean label.
8. **8/121 candidates (7%) are `uninterpretable`**, including one always-on,
   near-ubiquitous generic feature (972) and one generic early-sentence positional feature
   (1876) — both largely independent of topic or domain, i.e. not semantic at all.

## 10. Scope statement

Every result above concerns **Qwen3-1.7B's layer-14 residual-stream activations**, as
captured by **one SAE-8192 dictionary (seed 0)**, on **one specific 1,265/1,268-prompt
Discovery/Confirmation split** of the Milestone-4 validation set. Cross-seed and
cross-capacity checks test reproducibility of *this* layer's representation under
different SAE training conditions, not generalization to other layers. **No statement in
this document should be read as a claim about layers other than 14, about the full
8,192-feature dictionary beyond the 121 reviewed candidates, or about Qwen3 model
behavior at the token-generation level.**

## 11. Independent integrity verification

A dedicated script (`scripts/validate_integrity.py`) re-derived every headline count
directly from the raw JSON/JSONL outputs — never trusting a prior phase report's prose —
and ran **31 independent checks**, all passing: 121-candidate accounting across every
downstream artifact, both cross-seed/cross-capacity reconciliation tables summing
correctly with no double-counting, all 139 graph edges having valid endpoints/types/evidence/
confidence/provenance, every one of the 21 saved figures tracing to an existing source
file, Discovery/Confirmation separation holding with zero prompt-ID leakage across every
evidence and counterexample file, and `git status --porcelain` confirming zero
modifications to any frozen Milestone 1-5 directory. Full results:
`analysis/data/aggregate_statistics/integrity_report.json`.

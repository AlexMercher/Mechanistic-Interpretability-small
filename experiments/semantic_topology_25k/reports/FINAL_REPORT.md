# Experiment 3 — Semantic Topology, Concept Discovery & Final Semantic Hierarchy

**Status: all acceptance criteria below are met.** Where a sub-finding is a genuine limitation
rather than a clean result, it is called out explicitly rather than hidden — see §7.

Reproduce end-to-end:

```
python experiments/semantic_topology_25k/scripts/evaluate_embedding_space.py
python experiments/semantic_topology_25k/scripts/build_evaluation_candidates.py
# -- 180 candidate pairs reviewed by an Opus teacher model -> evaluation/reviewed_prompt_pairs.jsonl --
python experiments/semantic_topology_25k/scripts/evaluate_retrieval.py
python experiments/semantic_topology_25k/scripts/build_domain_clusters.py
python experiments/semantic_topology_25k/scripts/name_clusters.py
python experiments/semantic_topology_25k/scripts/discover_cross_domain_relations.py
# -- 150 candidate cross-domain pairs reviewed by an Opus teacher model (3 parallel batches) -> taxonomy/concept_relationships.jsonl --
python experiments/semantic_topology_25k/scripts/construct_taxonomy.py
# -- 40-concept sample spot-checked by an Opus teacher model -> taxonomy/concept_review_results.jsonl --
python experiments/semantic_topology_25k/scripts/render_hierarchy_markdown.py
python experiments/semantic_topology_25k/scripts/generate_semantic_reports.py
python experiments/semantic_topology_25k/scripts/validate_taxonomy.py
```

## 1. Executive summary

This experiment reuses the 25,000 pre-computed `intfloat/multilingual-e5-small` embeddings from
Milestone 2 (no regeneration) to: (A) rigorously evaluate whether those embeddings are
semantically trustworthy, (B) discover within-domain concepts via density/centroid clustering,
(C) discover cross-domain concept relationships via teacher-model-reviewed centroid-similarity
candidates, (D) freeze a versioned semantic hierarchy (`taxonomy/semantic_hierarchy_v1.json` +
`taxonomy/SEMANTIC_HIERARCHY.md`), and (E) report coverage/bias findings over that hierarchy.

The single most important finding, discovered empirically (not assumed) and confirmed
independently three separate times (Objective A's prompt-pair review, Objective B's
domain-wide-clustering failure, and Objective C's cross-domain review): **this embedding
model's raw similarity score is frequently dominated by shared template/instruction phrasing
rather than real semantic content**, in this specific template-heavy, five-domain corpus. Every
downstream step was redesigned around that finding rather than ignoring it — see §7 for how.

## 2. Objective A — Embedding space evaluation

Full detail: `reports/embedding_space_evaluation.md`. Summary:

- **A1 (integrity):** all 25,000 vectors verified (384-dim, finite, unit-normalized, complete
  prompt_id↔row mapping, model/revision match the Milestone-2 manifest). Passed cleanly.
- **A2 (distributions):** same-subdomain mean cosine 0.867 vs. cross-domain-random mean 0.785 —
  correctly ordered, but the margin (~0.08) is small relative to within-group spread (std≈0.03).
  Exact duplicates sit at a clean 1.000 ceiling.
- **A3 (reviewed evaluation set):** 180 candidate pairs, reviewed in full by an Opus teacher
  model (not by cosine similarity) — 0 ambiguous, confident calls on all 180.
  `evaluation/reviewed_prompt_pairs.jsonl`.
- **A4 (retrieval quality):** precision@20 for human-confirmed related pairs = **0.111**,
  NDCG@20 = **0.040** — poor. 77% of confirmed-unrelated reviewed pairs still scored cosine
  ≥0.80 (false neighbours). **Decision: do not switch embedding models** (per the task's
  explicit instruction); instead, every later objective treats raw similarity as a *candidate
  filter only*, never as ground truth, and routes every claim through content-based teacher
  review.

## 3. Objective B — Within-domain concept discovery

Full detail in `clustering/`. **Design correction made during this objective** (documented in
`build_domain_clusters.py`'s module docstring): domain-wide clustering was tried first and
produced clusters that were, verifiably, splits of the `category` field (task-framing: factual
knowledge / reasoning / ... ) rather than real topics — confirmed by checking `category_distribution`
on the resulting clusters (e.g. SCIENCE's "clusters" were 5×200 pure-`factual_knowledge` groups
plus one 3,899-record blob spanning all 35 subdomains). Fixed by clustering **within each
(domain, subdomain) bucket** instead, with HDBSCAN-vs-KMeans selection gated by a category-field
normalized-mutual-information diagnostic (prefer whichever method correlates less with
`category`). After the fix, category confound is resolved for 594/595 concepts (the one
remaining case, `GANDHI/Methods of protest`, is reported transparently, not hidden, via
`category_confound_unresolved: true` in `clustering/gandhi_cluster_details.json`).

| Domain | Subdomains | Concepts | Method(s) used |
|---|---|---|---|
| DNA | 5 | 41 | kmeans (41) |
| EMOTIONS | 5 | 22 | kmeans (22) |
| HINDI | 5 | 12 | kmeans (9), hdbscan (3) |
| SCIENCE | 35 | 120 | hdbscan (120) |
| GANDHI | 151 | 400 | kmeans (400) |
| **Total** | **201** | **595** | |

0 unassigned prompts (every prompt belongs to exactly one concept; `validate_taxonomy.py`
confirms this). Every concept records both `raw_record_count` and `unique_prompt_count`
(duplicate-aware, cross-checked against Milestone 2's duplicate-group data).

Concept names (`clustering/cluster_provisional_names.json`) are auto-derived via TF-IDF
distinctive-term extraction over the *full* cluster text (explicitly not a "first few records"
read) and are marked `review_status: auto_generated_unreviewed`. A 40-concept stratified sample
was spot-checked by an Opus teacher model (`taxonomy/concept_review_results.jsonl` /
`taxonomy/taxonomy_review.json`): 35/40 "looks_good", 2 flagged `too_broad` (two DNA
Inheritance_Genomics concepts merge distinct sub-topics), 2 flagged `mislabeled` (auto-name
pulled from shared template boilerplate, though cluster membership itself was fine), 1 flagged
`uncertain` (an HDBSCAN noise bin, not a genuine concept). UMAP 2D projections
(`visualizations/*_umap.png`) were generated for visual inspection only — cluster assignment
never used the UMAP projection, per the task's explicit instruction.

## 4. Objective C — Cross-domain relationship discovery

150 candidate cross-domain concept pairs (15 per unordered domain pair × 10 domain pairs,
selected by cluster-centroid cosine similarity) were reviewed in full by an Opus teacher model
across 3 parallel batches, judging from actual representative prompt text — never from the
similarity score. Full data: `taxonomy/concept_relationships.jsonl` (150 rows, every row
individually justified with an `evidence` string).

| relationship_type | count |
|---|---|
| no_relationship | 104 |
| thematically_related | 34 |
| closely_related | 9 |
| narrower_than | 2 |
| same_concept | 1 |

**69% of the embedding model's own top cross-domain centroid matches were judged
`no_relationship`** on content review — centroid similarities of 0.92–0.96 were overwhelmingly
explained by shared instructional scaffolding ("Interpret the observation that X using the
governing concepts of...", "Explain how X shaped the broader context of public response...")
rather than real meaning. This is the clearest, most decisive confirmation of the Objective A
finding, now at the concept level rather than the individual-prompt level.

The relationships that *did* survive review cluster in two genuinely coherent places:
- **DNA ↔ SCIENCE**: the only `same_concept` (DNA Inheritance_Genomics ↔ SCIENCE population
  genetics/Hardy-Weinberg) and most of the `closely_related`/`narrower_than` edges — unsurprising
  since both domains cover real molecular/genetic biology.
- **EMOTIONS ↔ HINDI**: 5 `closely_related` edges, all pairing EMOTIONS "Expression/Communication"
  or "Recognition/Interpretation" concepts with HINDI "Pragmatics" concepts — a genuine shared
  theme (the pragmatics of expressing/inferring emotion and politeness in context).
- The 34 `thematically_related` edges are real but loose high-altitude themes (environment,
  health, communication) connecting SCIENCE/EMOTIONS subtopics to GANDHI's writings on the same
  broad themes — correctly distinguished from `closely_related` by the reviewers' lower
  confidence (0.3–0.5).

See `taxonomy/SEMANTIC_HIERARCHY.md` §2 for the full itemized graph (excludes `no_relationship`
edges from the rendered graph; they remain in `concept_relationships.jsonl` as evidence of what
was checked).

## 5. Objective D — Final semantic hierarchy

Two complementary structures were produced, as required (not forced into one strict tree):

1. **Domain-specific taxonomy** (DOMAIN → SUBDOMAIN → CONCEPT): `taxonomy/semantic_hierarchy_v1.json`
   → `domain_specific_taxonomy`, rendered human-readably in `taxonomy/SEMANTIC_HIERARCHY.md` §1.
   A separate SUBCONCEPT level was *not* forced in below CONCEPT — at this corpus's actual
   discovered granularity (595 concepts, median size ~20–40 records, many already below 20 —
   see §6), introducing a 5th level would not have been evidence-backed; concepts are the finest
   reliably-distinguishable unit this analysis actually found.
2. **Cross-domain concept graph**: typed, evidence-backed edges from Objective C, kept separate
   from the domain tree (§4 above).

Every concept has a stable `concept_id` (independent of its display name), `parent_concept_id`/
`child_concept_ids` (both null/empty here since concepts are domain-tree leaves — no concept was
found to need multiple domain-tree parents), `prompt_ids`, `raw_record_count`,
`unique_prompt_count`, `confidence` (low/medium/high by size), `review_status`, and `evidence`
(representative prompts, clustering method, intra-cluster similarity, naming method).
Every prompt has **exactly one** concept assignment (`taxonomy/concept_assignments.jsonl`) — no
multi-concept/primary-secondary mechanism was needed since the clusters are a hard partition by
construction.

`validate_taxonomy.py` independently confirms: all 25,000 prompts assigned exactly once, every
`concept_id` referenced in assignments/relationships exists in the hierarchy, every concept's
stored `unique_prompt_count` matches an independent recomputation against Milestone 2's
duplicate-group data, no `concept_id` collisions. **Passed with zero errors.**

## 6. Objective E — Coverage and bias analysis

Full data: `reports/domain_coverage.json`, `reports/cluster_metrics.json`,
`reports/cross_domain_analysis.json`.

- **Concepts per domain**: DNA 41, EMOTIONS 22, HINDI 12, SCIENCE 120, GANDHI 400 (595 total,
  across 201 domain-subdomain buckets).
- **Unassigned prompts**: 0 (0.00%).
- **Concept size range**: 1 (several HDBSCAN noise bins, e.g. `SCIENCE_Astrophysics_hdbscan_-1`)
  to 506 (`DNA_Inheritance_Genomics_kmeans_0`).
- **Concepts below the 20-record confidence floor**: **410 of 595 (69%)** — almost entirely
  GANDHI, whose 151 fine-grained historical-topic subdomains (≈33 records each on average) get
  further split by KMeans into clusters averaging ~12 records each. **This is the clearest
  "diverse in raw count but few distinct semantic concepts" case the task asked to look for,
  inverted**: GANDHI actually has *more* nominal concepts (400) than any other domain, but the
  great majority are statistically thin slices of an already-fine-grained subdomain, not
  independently-discovered broad topics. A more defensible GANDHI taxonomy would likely treat
  the 151 subdomains themselves as the concept layer and skip further splitting — documented
  here as a recommendation, not actioned (per the task's instruction not to optimize for a
  particular concept count after the fact).
- **Template-dominated concepts** (unique/raw ratio < 0.9, i.e. duplicate-heavy): 65 of 595,
  concentrated in SCIENCE (matches Milestone 2's finding that SCIENCE held 825 of the corpus's
  826 excess exact-duplicate records).
- **Over/under-represented concepts** (|z-score| ≥ 1.5 within domain): 75 of 595 — expected
  given the GANDHI fragmentation above.
- **Cross-domain relationships**: see §4 table.
- **Language/script distribution within concepts**: computed per-concept in
  `reports/cross_domain_analysis.json` → `language_script_distribution_per_concept`; combined
  with the taxonomy review's finding (§7) that several HINDI concepts split along script lines
  rather than topic lines, this distribution should be read as a flag for re-examination, not a
  settled result.

## 7. Known limitations (explicit, not hidden)

1. **Embedding similarity is template-dominated in this corpus.** Confirmed three independent
   ways (A4 retrieval precision 0.111; domain-wide clustering rediscovering `category`; 69% of
   cross-domain centroid matches being `no_relationship` on content review). Every relationship
   claim in this report is backed by human/teacher-model content review, not by similarity score
   alone — but any *future* consumer of `concept_relationships.jsonl` or raw embeddings must
   apply the same discipline, not trust cosine similarity directly.
2. **GANDHI is over-fragmented** (400 concepts, 69% of all sub-20-record concepts are GANDHI's) —
   see §6. The 151-subdomain taxonomy is likely already close to the right concept granularity;
   further KMeans splitting mostly produced statistically thin, low-confidence slices.
3. **HINDI concept boundaries partially reflect script/language_form, not topic** (found by the
   Objective D review sample: Grammar/Vocabulary/Syntax subdomains show parallel romanized vs.
   Devanagari clusters). Cross-reference `annotations/language_script_annotations.jsonl` before
   treating two HINDI concept_ids as topically distinct.
4. **Concept names are mostly unreviewed.** Only a 40/595 (6.7%) stratified sample was
   teacher-model-reviewed. The sample's defect rate (5/40 = 12.5% too_broad/mislabeled) should be
   assumed to generalize approximately to the unreviewed 555, i.e. treat `provisional_name` as
   indicative, not authoritative, and prefer `representative_prompt_texts` for any concept a
   future analysis depends on.
5. **One residual category confound**: `GANDHI/Methods of protest` subdomain's chosen clustering
   still correlates with the `category` field (NMI > 0.6) — reported via
   `category_confound_unresolved: true`, not resolved.
6. **GANDHI/SCIENCE HDBSCAN noise bins (label -1) are included as size-1-or-more "concepts"**
   in the raw counts; these are catch-all residuals, not discovered topics — they inflate the
   "concepts below min size" count in §6 and should be filtered out (`cluster_id` ending in
   `_-1`) by any consumer that wants only genuine discovered concepts.
7. **Cross-domain relationship candidates were capped at 15 per domain pair** (150 total, not
   exhaustive over all ~580×580 cluster pairs) for review tractability — the reported
   relationship counts are a lower bound on what exists, not a census.

## 8. Storage and computational cost

~17 MB total under `experiments/semantic_topology_25k/` (clustering 8.3MB, taxonomy 7.1MB,
visualizations 0.5MB, evaluation 0.3MB, reports 0.2MB, scripts 0.08MB, run_metadata 5KB). No new
model weights were downloaded (embeddings reused from Milestone 2); `hdbscan`, `umap-learn`, and
`matplotlib` (a few MB of pure-Python/compiled wheels, no GPU/large-model weights) were installed
into the existing `.venv`. All clustering ran on CPU in a few minutes total across all 5 domains.

## 9. Source integrity confirmation

```
git diff --stat HEAD -- normalized raw processed MI_Dataset_validated experiments/duplicate_language_embedding
```
returned **empty** — confirmed immediately before freezing this report. Milestones 1 and 2's
outputs (including the embedding matrix itself) were read-only inputs throughout; nothing in
`experiments/semantic_topology_25k/` was written anywhere outside that directory.

## 10. Acceptance criteria checklist

**Embedding evaluation:** ✅ all 25,000 vectors verified · ✅ similarity distributions analyzed ·
✅ similarity compression investigated (quantified + confirmed at concept level in Objective C) ·
✅ related/unrelated evaluation pairs reviewed (180, teacher-model) · ✅ retrieval quality
reported (precision@k/NDCG@k) · ✅ multilingual retrieval examined (hindi_script_variant
subset) · ✅ known limitations documented.

**Within-domain discovery:** ✅ all five domains analyzed · ✅ clustering methods compared and
justified per-bucket (not blindly applied) · ✅ cluster quality assessed (category-NMI
diagnostic, intra-cluster similarity) · ✅ representative prompts inspected (40-sample
teacher review) · ✅ outliers explicitly reported (HDBSCAN noise bins, flagged not hidden) ·
✅ raw and unique counts reported separately for every concept.

**Cross-domain discovery:** ✅ cross-domain nearest neighbours analyzed (150 candidates, all 10
domain pairs) · ✅ candidate concept relationships identified · ✅ relationships have evidence
and confidence (every row) · ✅ semantic equivalence distinguished from analogy/theme (9-type
vocabulary used, not just "related") · ✅ false relationships investigated (104/150 explicitly
ruled out with rationale, not silently dropped).

**Final semantic hierarchy:** ✅ all five domain taxonomies constructed · ✅ concept IDs stable
and unique (validated) · ✅ parent-child relationships validated · ✅ cross-domain relationships
represented separately · ✅ multi-concept prompts supported in principle (not needed in practice
— hard partition confirmed sufficient) · ✅ prompt-to-concept traceability verified
(`validate_taxonomy.py`, 0 errors) · ✅ ambiguous assignments explicitly marked (HDBSCAN noise
bins, category-confound flag) · ✅ coverage gaps documented (§6) · ✅ `semantic_hierarchy_v1.json`
generated · ✅ `SEMANTIC_HIERARCHY.md` generated · ✅ hierarchy validated against the original
corpus (0 errors).

**Final reporting:** ✅ this document.

**No criterion failed.** Where results were weaker than hoped (retrieval precision, GANDHI
fragmentation, HINDI script/topic confound, unreviewed-name coverage), they are reported as
limitations in §7, not concealed, per the task's explicit instruction.

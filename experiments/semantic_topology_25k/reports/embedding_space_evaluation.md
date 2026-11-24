# Embedding Space Evaluation (Objective A)

Source embeddings: `experiments/duplicate_language_embedding/embeddings/` (25,000 vectors,
`intfloat/multilingual-e5-small`@`614241f...`, 384-dim, L2-normalized). Not regenerated.
Reproduce with:

```
python experiments/semantic_topology_25k/scripts/evaluate_embedding_space.py
python experiments/semantic_topology_25k/scripts/build_evaluation_candidates.py
# -- candidate pairs reviewed by an Opus teacher-model pass (see evaluation/reviewed_prompt_pairs.jsonl) --
python experiments/semantic_topology_25k/scripts/evaluate_retrieval.py
```

## A1 — Integrity

All checks passed (`reports/similarity_distributions.json` → `integrity`): 25,000 rows, 384
dimensions, prompt_id mapping complete and bijective, all vectors finite and unit-normalized,
model/revision match the Milestone-2 manifest exactly. **No regeneration was needed or performed.**

## A2 — Similarity distributions (seed=42, see `reports/similarity_distributions.json` for full stats)

| Comparison group | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| Random pairs, whole corpus | 20,000 | 0.794 | 0.794 | 0.748 | 0.837 |
| Random pairs, same domain | 20,000 | 0.828 | 0.828 | 0.787 | 0.866 |
| Random pairs, cross-domain | 16,022 | 0.785 | 0.787 | 0.744 | 0.824 |
| Random pairs, same subdomain | 20,100 | 0.867 | 0.865 | 0.831 | 0.902 |
| Exact duplicate pairs | 2,401 | 1.000 | 1.000 | 1.000 | 1.000 |
| Template-variant pairs (proxy) | 5,035 | 0.847 | 0.848 | 0.815 | 0.878 |

**This confirms, and quantifies, the similarity-compression finding flagged in Milestone 2.**
The entire corpus — unrelated cross-domain pairs included — sits in a narrow band
(roughly 0.75–0.90 mean cosine similarity). The *ordering* of the groups is directionally
correct (same-subdomain > same-domain > cross-domain, and exact duplicates are a clean 1.000
ceiling), so the embedding space does carry a real, usable signal — but the absolute margin
between "same subdomain" (mean 0.867) and "cross-domain random" (mean 0.785) is only ~0.08,
and both distributions have enough spread (std ≈ 0.03) to overlap substantially. **Any
downstream clustering or relationship-discovery step must treat cosine similarity as a
relative ranking signal only, never as an absolute threshold** — this is the same conclusion
the Milestone-2 pilot reached from a 20-item sample; A2 now confirms it at full-corpus scale.

## A3 — Reviewed evaluation set

Deterministic candidate generation (`build_evaluation_candidates.py`, seed=42) produced 180
candidate pairs, 30 each across 6 categories (`same_concept_proxy`, `same_domain_diff_concept`,
`cross_domain_related_cand`, `cross_domain_unrelated_cand`, `hindi_script_variant`,
`template_variant`). **Cosine similarity was used only to select `cross_domain_related_cand`
candidates (each query's top cross-domain neighbour) — never to assign labels.**

All 180 pairs were independently read and labeled by an Opus teacher-model reviewer
(`evaluation/reviewed_prompt_pairs.jsonl`, `reviewer: "opus_teacher_model_v1"`), each with a
content-specific rationale. Label distribution:

| Label | Count |
|---|---|
| strongly_related | 16 |
| related | 29 |
| weakly_related | 38 |
| unrelated | 97 |
| ambiguous | 0 |

By candidate category (see `evaluation/embedding_retrieval_metrics.json` →
`candidate_category_x_label_counts` for the full cross-tab):

- `same_concept_proxy` (same domain+subdomain+category): 12 strongly_related, 18 related, **0 unrelated**
  — confirms the normalized schema's fine-grained category field is a decent proxy for shared concept.
- `cross_domain_related_cand` (embedding's own top cross-domain pick per query): only 1
  strongly_related + 3 related + 8 weakly_related out of 30 — **18/30 (60%) were unrelated on
  manual review.** The reviewer's rationales consistently point to shared *template phrasing*
  ("what characterizes X", "best-supported conclusion about Y") as the cause, not real content
  overlap. This is the single most important finding of Objective A: **the embedding model's
  own nearest-cross-domain-neighbour choices are majority false positives**, driven by surface
  template structure rather than meaning.
- `hindi_script_variant`: only 7/30 were related/strongly_related; most (19/30) were
  weakly_related (same pragmatic/grammar *template*, different specific vocabulary/phenomenon)
  and 4/30 fully unrelated. True same-concept, different-script pairs exist in the corpus but
  are the minority even among pairs selected specifically to probe for them.
- `template_variant` (same fine-grained category, high text overlap): 22/30 unrelated — confirms
  the project's standing instruction that template slot-fills frequently encode real semantic
  differences, not cosmetic variation (particularly pronounced in DNA/Structure, which dominated
  this sample).

## A4 — Retrieval quality (`evaluation/embedding_retrieval_metrics.json`)

For the 45 reviewed pairs labeled `related` or `strongly_related`, each pair's first prompt was
used as a query and its full-corpus rank (by cosine similarity, among the other 24,999 prompts)
for the second prompt was computed:

| k | Precision@k | NDCG@k |
|---|---|---|
| 1 | 0.000 | 0.000 |
| 5 | 0.089 | 0.034 |
| 10 | 0.089 | 0.034 |
| 20 | 0.111 | 0.040 |

**This is poor retrieval performance.** A human-confirmed "related" partner is almost never
in the top 20 nearest neighbours by raw cosine similarity — e.g. one HINDI script-variant pair
confirmed `strongly_related` by the reviewer ranked at position 4,280 out of 24,999. The
dominant cause, visible directly in the corpus structure, is that each fine-grained
`(domain, subdomain, category)` bucket contains hundreds of templated prompts that are
*lexically/structurally* closer to the query than the one semantically-related prompt from a
different bucket — the embedding space is dominated by surface template similarity within a
bucket, which outranks genuine cross-bucket conceptual relatedness.

**Multilingual retrieval** (`hindi_script_variant_relevant_pairs`): the 7 pairs confirmed
related/strongly_related show cosine similarities of 0.81–0.88, statistically indistinguishable
from the unrelated/weakly_related hindi_script_variant pairs (0.79–0.92) — i.e. **cosine
similarity alone cannot separate true cross-script equivalents from same-template-different-
content pairs in this corpus.** This is a genuine, documented limitation of using this
embedding model's raw similarity for multilingual retrieval here, not merely a theoretical
caveat.

**False-neighbour check:** of the 97 pairs confirmed `unrelated`, 75 (77%) still scored cosine
similarity ≥ 0.80 — above the corpus-wide same-subdomain p25 (0.848) in many cases, i.e. squarely
inside the range that would normally be read as "quite similar." Representative false
neighbours are listed in `evaluation/embedding_retrieval_metrics.json` →
`false_neighbours_unrelated_but_high_similarity` (e.g. `SCIENCE_002279`/`SCIENCE_001487`,
astrophysics vs evolutionary biology template prompts, similarity 0.851).

## Conclusion — is the embedding model semantically reliable?

**Partially, and only for coarse-grained, relative comparisons — not for fine-grained retrieval
or absolute-threshold decisions.** Evidence for a real signal: the four similarity-distribution
groups are correctly ordered, exact duplicates sit at a clean ceiling, and `same_concept_proxy`
pairs scored 0% unrelated. Evidence against naive reliance on it: cross-domain "nearest
neighbour" picks are majority false positives, precision@20 for genuinely related pairs is
~11%, and the false-neighbour rate among confirmed-unrelated pairs is 77%.

**Decision: do not switch embedding models** (per the task's explicit instruction not to switch
automatically). Instead, Objective B/C/D rely on (a) **density-based clustering within each
domain** (which only needs locally-dense neighbourhoods, not globally-separated absolute
similarity) and (b) **teacher-model-reviewed evidence for every cross-domain relationship
claim**, rather than thresholding raw cosine similarity. This evaluation is the documented
justification for that design choice.

# flow.md — How this project evolved, and what it represents

This document traces the project from its starting dataset to its current end-state (the
completed Milestone 6 feature-relationship analysis), and maps which features/artifacts
each stage produced and which later stage consumed them. It is a narrative companion to
`File_context.md` (which lists files) — this file answers *why each stage exists* and
*what it unlocked next*.

## What the project represents, in one paragraph

This is a disciplined, evidence-gated mechanistic-interpretability pipeline: a
25,000-prompt, 5-domain, bilingual (English/Hindi) corpus is built and progressively
quality-controlled; a real model (Qwen3-1.7B) is run over it and its internal
representations are extracted; a dictionary-learning model (sparse autoencoders) is
trained on those representations at multiple capacities; and the resulting SAE features
are then rigorously discovered, validated, and related to each other — with every claim
gated behind pre-registered criteria, independent review, reproducibility checks across
seeds/capacities, and explicit reporting of negative findings. Each milestone exists
because the previous one surfaced a question it could not answer itself, and each
milestone's artifacts are frozen once complete so later stages can build on them without
ever being allowed to retroactively alter the evidence beneath them.

## The flow, stage by stage

### Stage 0 — Raw dataset assembly (`raw/`, `scripts/build_raw_snapshot.py`)
**Why:** before anything else, there needs to be a byte-for-byte-provenanced source of
truth. Five domains (DNA, EMOTIONS, HINDI, SCIENCE, GANDHI; 5,000 prompts each) were
copied verbatim from their original Markdown sources (two different legacy conventions —
`MI_Dataset_validated/` for DNA/EMOTIONS/HINDI, extracted Monisha zips for SCIENCE/GANDHI)
with a sha256 manifest per file.
**Produces:** `raw/<DOMAIN>/`, `raw/MANIFEST.json`.
**Feeds into:** Stage 1 (normalization never rewrites prompt text — only this raw layer
is the ground truth for it).

### Stage 1 — Normalization (`normalized/`, `scripts/build_normalized.py`, `validate_normalized.py`)
**Why:** the two legacy source conventions needed one canonical schema so every later
script could assume a single record shape, without ever touching prompt wording.
**Produces:** `normalized/{domain}.jsonl` — one record per prompt with a stable
`prompt_id`, `source_prompt_id`, `prompt_sha256`, deterministic train/val/test split
(hash-based, reproducible).
**Feeds into:** every single downstream stage. This is the one dataset layer every later
experiment reads from.

### Stage 2 — Processed derivatives (`processed/`, `scripts/build_processed.py`)
**Why:** some non-destructive annotation (clustering by subdomain, batching for model
runs, exact-duplicate flags) was needed without ever mutating the normalized layer.
**Produces:** `processed/{clusters,chunks,annotations,reports}/`.
**Feeds into:** mostly a convenience layer; informs later duplicate/quality work.

### Stage 3 — Corpus quality validation (`experiments/corpus_validation_25k/`)
**Why:** a 25K-prompt corpus assembled from two source conventions needed an independent
quality gate before any model ever saw it — are there invalid records, exact or near
duplicates, suspicious length outliers, or script/language edge cases?
**Produces:** 826 exact duplicates, 35,938 near-duplicate *candidate* pairs (flagged, not
deleted), a teacher-model-adjudicated sample of those pairs, 1,401 Romanized-Hindi review
flags, 0 length outliers.
**Feeds into:** the near-duplicate candidates (`near_duplicate_candidates.csv`) become
Milestone 6's **Tier-1 template-family** signal — the single most important artifact this
stage contributes downstream, five milestones later.

### Stage 4 — Duplicate/language audit + embeddings (`experiments/duplicate_language_embedding/`)
**Why:** three distinct open questions remained after Stage 3: (a) a second, independent
exact-duplicate audit (recomputing hashes from scratch, not trusting the stored field),
(b) a deterministic (non-ML) Hindi/English script-and-language classifier per prompt, and
(c) a semantic embedding of every prompt, needed before any clustering/topology work could
begin.
**Produces:** `reports/exact_duplicate_groups.csv`, `annotations/language_script_annotations.jsonl`
(language/script/`language_form` per prompt), `embeddings/prompt_embeddings.npy` (25,000×384,
multilingual-e5-small).
**Feeds into:** Milestone 3's clustering (needs embeddings), and directly into Milestone 6
as the **script/language confound signal** (`script_language_score`) and the
**counterexample-retrieval index** (nearest-neighbor search over these embeddings).

### Stage 5 — Semantic topology & taxonomy (`experiments/semantic_topology_25k/`)
**Why:** having an embedding space raised the natural next question — does it actually
capture *meaning*, or mostly template phrasing? And if concepts exist, can a taxonomy and
cross-domain relationship graph be built and teacher-reviewed?
**Produces:** a critical **negative finding** (precision@20≈0.11 — this embedding space's
raw similarity is frequently template-driven, not meaning-driven), per-domain clusters
named via TF-IDF, a frozen `taxonomy/semantic_hierarchy_v1.json`, and
`taxonomy/concept_relationships.jsonl` (150 teacher-reviewed cross-domain concept pairs,
104 ruled `no_relationship`).
**Feeds into:** Milestone 6 reuses the non-`no_relationship` concept pairs *only as a
search seed* for cross-domain SAE-feature hypotheses (Stage B3) — explicitly never as
ground truth for what an SAE feature means, precisely because Stage 5 already proved this
embedding space cannot be trusted at face value.

### Stage 6 — Activation extraction (`experiments/activation_extraction_qwen3_1_7b/`)
**Why:** mechanistic interpretability needs an actual model's internal activations, not
just embeddings of the text. Qwen3-1.7B was chosen, layer 14 (the middle of 28 layers) was
selected as the residual-stream hook point, and a pilot run investigated a real bf16
numerical-drift anomaly before committing to a full run.
**Produces:** 993,496 token-level residual-stream activation vectors (2048-dim) across all
25,000 prompts, safetensors-sharded, with a deterministic train/validation split.
**Feeds into:** Milestone 5 (SAE training substrate) and, transitively, Milestone 6 (which
never re-extracts activations — it only ever reuses this frozen shard set via Milestone
5's loading code).

### Stage 7 — SAE dictionary-capacity comparison (`experiments/sae_capacity_comparison/`)
**Why:** once activations exist, the natural interpretability tool is a sparse
autoencoder (dictionary learning) — but at what width? Three capacities (4096/8192/16384)
were trained, 3 seeds each (9 runs total), and compared on reconstruction, sparsity,
cross-seed feature stability, and a 90-feature teacher-reviewed interpretability sample.
**Produces:** a second critical **negative finding** — no width wins on every axis, and
widening the dictionary did not yield more genuinely semantic features; it mostly
multiplied template-driven and Hindi-script-marker features. Four confounds were flagged
qualitatively but never rigorously quantified: template-driven activations, script/language
markers, domain markers, and duplicated prompt structures.
**Feeds into:** Milestone 6 directly — this milestone's unresolved, *qualitative* confound
flag is exactly the open question Milestone 6 was commissioned to answer *quantitatively*.
The SAE-8192 seed-0/1/2 and SAE-4096/16384 seed-0 checkpoints, and the `sae_common.py`
loading code, are reused read-only.

### Stage 8 — Semantic Feature Discovery, Validation & Relationship Analysis (`experiments/semantic_feature_discovery/`) — Milestone 6, this project's current end-state
**Why:** Milestone 5 left four confounds unquantified and zero validated claims about
what any individual SAE feature actually represents, or how features relate to each
other. Milestone 6 exists to close exactly that gap, under a much stricter evidentiary
bar than any prior milestone: pre-registered selection criteria frozen before use, a
genuine Discovery/Confirmation split to prevent selection-bias self-confirmation,
independent blind second review, full-dictionary (never index-aligned) cross-seed/
cross-capacity reproducibility checks, and a typed, evidence-and-provenance-tagged
relationship graph — never allowing correlation, decoder similarity, or co-activation to
be read as a causal or "same concept" claim.

**How it unfolded internally (its own five-phase flow, gated by explicit approval at
every boundary):**
1. **Phase 0** — inspection of every prior milestone's reusable artifact (table in
   Section A1 of the approved plan) + three rounds of plan correction before any code was
   written, closing precision gaps in the split method, matching methodology, candidate-
   selection formulas, and co-activation statistics.
2. **Phase 1** — built the exact Discovery/Confirmation split (1,265/1,268 prompts),
   the full SAE-8192 seed-0 feature inventory, tiered template clusters (reusing Stage 3's
   near-duplicate graph as Tier 1), and selected 121 candidates under 10 frozen quota rules.
3. **Phase 2** — confound diagnostics on both subsets independently, Confirmation-subset
   evidence extraction, counterexample generation (reusing Stage 4's embeddings for
   nearest-neighbor retrieval; two real bugs found and fixed here), and a Confirmation-
   only coverage statistic documenting a real rare-feature-selection limitation.
4. **Phase 3** — primary + blind second-opinion review of all 121 candidates (moderate
   inter-reviewer kappa 0.39–0.52, reported rather than resolved), full-target-dictionary
   cross-seed matching (seeds 1/2) and cross-capacity matching (widths 4096/16384), plus a
   12-combination threshold-sensitivity sweep.
5. **Phase 4** — relationship analysis among the 61 >=medium-confidence candidates:
   confound-controlled co-activation (104/126 pairs survived conditioning), decoder-vs-
   functional-similarity disagreement (37/1,830 pairs), cross-domain two-feature
   hypotheses (seeded from Stage 5's concept pairs, with 1,107 raw noisy intersections
   explicitly distinguished from the top-20 reported hypotheses), and a typed 61-node/
   139-edge relationship graph.
6. **Phase 5** — 21 Seaborn/Matplotlib figures across 7 categories, a 31-check independent
   integrity verification (all passing), the scientific narrative (`analysis/findings.md`),
   and this milestone's own execution log (`reports/FINAL_REPORT.md`).

**What it found, headline:** 35/121 candidates are genuinely domain-specific coherent
features; only 5/121 are single-feature cross-domain generalizers; 20/121 are
template-driven artifacts and 5+3/121 are script/domain markers — confirming Milestone
5's qualitative flag with hard numbers. Three of six possible relationship-edge types
(`possible_merge`, `shared_surface_form`, `mutual_exclusion`) found **zero** qualifying
pairs — a reported negative finding, not a gap in the method.

## Feature map — what each stage built, and who consumes it

| Stage | Key artifact(s) produced | Consumed by |
|---|---|---|
| 0. Raw snapshot | `raw/<DOMAIN>/`, sha256 manifest | Stage 1 (sole source of truth for prompt text) |
| 1. Normalization | `normalized/{domain}.jsonl`, stable `prompt_id` | Every later stage (universal join key) |
| 2. Processed derivatives | clusters/chunks/annotations | Convenience layer; informs Stage 3/4 |
| 3. Corpus validation | `near_duplicate_candidates.csv` (35,938 pairs) | **Milestone 6 Tier-1 template families** |
| 4. Duplicate/language/embeddings | `language_script_annotations.jsonl`, `prompt_embeddings.npy` | **Milestone 6 script/language confound score + counterexample retrieval**; Milestone 3 clustering |
| 5. Semantic topology | `semantic_hierarchy_v1.json`, `concept_relationships.jsonl` | **Milestone 6 cross-domain hypothesis search seed (never ground truth)** |
| 6. Activation extraction | 993,496 layer-14 residual-stream vectors (sharded) | Milestone 5 (training substrate); Milestone 6 (via M5's loaders, never re-extracted) |
| 7. SAE capacity comparison | 9 trained SAE checkpoints (3 widths × 3 seeds), qualitative confound flag | **Milestone 6's entire primary dictionary + cross-seed/capacity targets** |
| 8. Semantic feature discovery (M6) | 121-candidate cohort, review labels, reproducibility classifications, typed relationship graph, 21 figures, `findings.md` | End-state: the project's current deliverable |

## What this represents, taken as a whole

The project is not a single analysis but a chain of falsifiable checkpoints: every
milestone's "final" report explicitly states what it did *not* establish, and the next
milestone exists specifically to either establish it rigorously or document that it
remains unresolved. Two deliberate negative findings anchor the whole chain — the
embedding space's template-vs-meaning confound (Stage 5) and the SAE dictionary's failure
to produce more genuine semantics at higher capacity (Stage 7) — and Milestone 6 is the
project's answer to both: a quantified, reproducibility-checked, independently-reviewed
account of which SAE features are real semantics, which are confounds, and how the real
ones relate to each other, with every uncertain or negative result preserved rather than
smoothed over.

# Experiment: Structural & Semantic Validation of the 25K Prompt Corpus

**Date:** 2026-11-24
**Status:** complete (Step 2 of the dataset pipeline; precedes any semantic/activation processing)
**Corpus under test:** `normalized/*.jsonl`, 25,000 records, 5 domains (DNA, EMOTIONS, HINDI, SCIENCE, GANDHI), built by `scripts/build_raw_snapshot.py` -> `scripts/build_normalized.py` (see `CLAUDE.md`, `File_context.md`).

## Motivation

Before this corpus is used for any mechanistic-interpretability work (activation probing, feature attribution, model comparison), it needs a dataset-quality gate: are the records structurally sound, free of silent corruption, and is duplication in the data understood well enough that it won't quietly bias downstream metrics (e.g. a model "performing well" because 800 identical prompts dominate one bucket)? This experiment is that gate. It is intentionally conservative: it flags, it does not delete. Per instruction, near-duplicate prompts are never auto-removed, because two textually similar prompts can probe different reasoning patterns (e.g. swap one entity, keep the template) and collapsing them would destroy a legitimate variation axis in the corpus.

## Method

Script: `scripts/validate_corpus.py` (pure stdlib, re-runnable, read-only against `normalized/`).

1. **Record count** — total records and per-domain counts read directly from `normalized/*.jsonl`.
2. **ID integrity** — every `prompt_id` matched against the expected `{DOMAIN}_{000001..00N}` sequence; checked for gaps and duplicates.
3. **Required fields** — `domain`, `subdomain`, `prompt`, `source_model`, `source_file` must be present and non-empty on every record.
4. **Empty records** — `prompt.strip()` must be non-empty.
5. **Encoding** — scanned every prompt for the U+FFFD replacement character (a reliable signal of real mojibake/corruption); separately checked HINDI prompts for at least one Devanagari codepoint (U+0900-U+097F).
6. **Exact duplicates** — grouped by `prompt_sha256` within each domain (already computed by `scripts/build_processed.py`; aggregated here).
7. **Near duplicates** — blocked by `(category, subdomain)` (confirmed by inspection that template reuse is localized to this grouping — see Results), then pairwise `difflib.SequenceMatcher` ratio with `real_quick_ratio()`/`quick_ratio()` used as cheap upper-bound filters before the full ratio computation. Threshold: **0.72**. Flagged, not deleted.
8. **Length** — word-count distribution (mean/median/p90/p99/max) per domain; records above an absolute 120-word ceiling flagged as outliers for review.

All outputs are written under `experiments/corpus_validation_25k/reports/` and `metrics.json` in this folder; nothing in `raw/`, `normalized/`, or `processed/` is modified.

## Results

```
total_records: 25000
valid_records: 25000
invalid_records: 0
review_flags: 1401   (content characteristic, not a defect -- see below)
exact_duplicates: 826
near_duplicate_candidates: 35938
length_outliers: 0   (max observed: 57 words, well under any reasonable "unusually long" bar)
```

Full breakdown in `metrics.json`; per-report detail in `reports/`.

| Domain | Records | Exact dups | Near-dup candidates | Mean words | Max words |
|---|---|---|---|---|---|
| DNA | 5000 | 1 | 11,933 | 15.44 | 40 |
| EMOTIONS | 5000 | 0 | 18,848 | 26.31 | 57 |
| HINDI | 5000 | 0 | 2,266 | 20.97 | 55 |
| SCIENCE | 5000 | 825 | 1,666 | 34.64 | 49 |
| GANDHI | 5000 | 0 | 1,225 | 18.13 | 25 |

**ID integrity:** zero missing IDs, zero duplicate IDs, across all 5 domains — the `prompt_id` sequence is fully intact.

**Required fields / empty records:** zero failures corpuswide.

**Encoding:** zero U+FFFD (mojibake) occurrences found. One false alarm during manual review: Devanagari em-dashes and curly quotes render as "�" in this Windows terminal's codepage but are correct Unicode codepoints on disk (verified via `ord()` inspection, e.g. U+2014, U+2018) — not a defect, a display artifact of the review environment, noted here so it isn't re-flagged in a future pass.

**Hindi script check (1,401 "review_flags"):** ~28% of HINDI prompts contain no Devanagari codepoints — these are Romanized/Hinglish prompts (e.g. *"'Usne baat ko taal diya' mein 'taal diya' ka sandarbhik arth kya hai?"*). Checked whether this traces to one faulty source generator: it does not — the flagged records split 572/501/328 across Gemini/Claude/ChatGPT respectively, roughly proportional to each generator's overall share of the HINDI domain. This is a genuine mixed-script characteristic of the HINDI domain as delivered, not corruption. **Not counted against `valid_records`.** Worth knowing before tokenization/embedding experiments, since mixed-script input can behave differently than pure-Devanagari input in some models.

**Exact duplicates (826):** concentrated almost entirely in SCIENCE (825 of 826). Root cause (confirmed by inspection): SCIENCE's own `GENERATION_REPORT.md` already notes template reuse across its 35 subdomains, and the duplicate rows confirm it — e.g. `SCIENCE_002001`/`002006`/`002355`/`002360`/`002709` are five verbatim copies of the same `factual_knowledge` × `Mechanics` template question. DNA has exactly one incidental exact duplicate; EMOTIONS, HINDI, GANDHI have none. Per-domain lists: `reports/{domain}_exact_duplicates.csv` (via `processed/reports/`, aggregated here).

**Near duplicates (35,938 candidates, threshold 0.72):** EMOTIONS and DNA account for the bulk (18,848 and 11,933). Manual spot checks (3 pairs across EMOTIONS/DNA/HINDI) during development, confirmed that flagged pairs are template-slot variations — same sentence frame, different scenario/entity/subdomain swapped in (e.g. *"...characterize DNA curvature in DNA?"* vs *"...characterize nucleotide solubility in DNA?"*; *"colleague is unusually quiet after..."* vs *"person feels awkward receiving praise..."* under an identical task sentence). These are legitimately distinct test items and were correctly **not** auto-removed.

### Teacher-model (Opus) adjudication

Reviewing all ~36K candidate pairs individually isn't tractable or necessary; `scripts/sample_near_duplicates_for_review.py` drew a representative sample (top-15-by-similarity + 5 random per domain = 100 pairs, `reports/near_duplicate_review_sample.json`) and an Opus agent adjudicated each pair as `true_duplicate` / `distinct_template_variant` / `ambiguous` (verdicts: `reports/near_duplicate_opus_verdicts.json`).

```
true_duplicate:          0 / 100
distinct_template_variant: 99 / 100
ambiguous:                1 / 100
```

Opus's calibration judgment: even the highest-similarity pairs (ratio > 0.98) swap a load-bearing slot — a chemical entity, a scenario actor, an analytic lens, a grammatical category — so a model could answer one correctly and the other incorrectly. **The 0.72-threshold heuristic is, if anything, conservative in the right direction: it over-flags for review but the dataset owner's instruction (never auto-delete) means that's the safe failure mode.** The single ambiguous pair (`SCIENCE_000069`/`SCIENCE_000769`) differs only in a trailing meta-instruction with an identical scenario — the one pattern worth a targeted human look if any pruning is ever considered.

## Conclusion

The corpus passes all structural gates: 25,000/25,000 valid, zero missing/duplicate IDs, zero required-field failures, zero encoding corruption, zero unusually long prompts. Two things are documented rather than fixed, by design: (1) SCIENCE's 825 exact-duplicate template questions and (2) HINDI's ~28% Romanized-script subset. Near-duplicate candidates (35,938) are flagged in `reports/near_duplicate_candidates.csv` for future reference but intentionally left in the corpus — teacher-model adjudication on a representative sample confirms they are overwhelmingly distinct reasoning probes, not redundant copies.

## Reproduce

```
python scripts/build_raw_snapshot.py
python scripts/build_normalized.py
python scripts/validate_normalized.py
python scripts/build_processed.py
python scripts/validate_corpus.py
python scripts/sample_near_duplicates_for_review.py
# then dispatch an opus-model agent against reports/near_duplicate_review_sample.json
```

## Files

- `metrics.json` — machine-readable summary (the numbers quoted above).
- `reports/invalid_records.csv` — structural failures (empty in this run).
- `reports/review_flags.csv` — content-characteristic flags (Romanized Hindi).
- `reports/near_duplicate_candidates.csv` — all 35,938 flagged pairs.
- `reports/near_duplicate_review_sample.json` — 100-pair stratified sample sent for adjudication.
- `reports/near_duplicate_opus_verdicts.json` — Opus's per-pair verdicts.
- `reports/length_stats.csv`, `reports/length_outliers.csv` — word-count distribution and outliers (empty in this run).

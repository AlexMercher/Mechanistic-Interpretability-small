# Final Report: Exact-Duplicate Audit, Language/Script Classification & Semantic Embedding Generation

**Date:** 2026-11-24
**Scope:** data-processing and research-preparation only. No activation extraction, SAE training, semantic clustering, or prompt rewriting was performed.
**Status: all three objectives complete, all acceptance criteria met.** See "Acceptance criteria checklist" at the end for the itemized confirmation.

## 1. What was implemented

Four new scripts under `scripts/`, all read-only against `raw/`, `normalized/`, and `processed/`, writing only into `experiments/duplicate_language_embedding/`:

| Script | Purpose |
|---|---|
| `scripts/audit_exact_duplicates.py` | Objective 1: independent hash verification + exact-duplicate multiplicity audit. |
| `scripts/lang_script_utils.py` + `scripts/classify_language_script.py` | Objective 2: deterministic language/script/language-form classification for all 25,000 records. |
| `scripts/embed_utils.py` + `scripts/generate_embeddings.py` | Objective 3: model loading (HF cache redirected into `weights/`), pilot and full embedding generation, resumable. |
| `scripts/validate_embedding_outputs.py` | Independent post-hoc validation of the embedding outputs (doesn't re-run the model). |
| `scripts/sample_near_duplicates_for_review.py` | (carried over from the prior validation pass; reused, not reimplemented, for Objective 1 context) |

Every script has `--help`, explicit CLI flags, sensible defaults, and refuses to silently overwrite existing outputs (`--overwrite` required to replace prior results).

## 2. Exact-duplicate audit (Objective 1)

Method: recomputed `sha256(prompt_text)` independently for every one of the 25,000 records and used that recomputed hash -- not the corpus's stored `prompt_sha256` field -- as the grouping key. The stored field was only cross-checked against it, never trusted blindly.

```
total_records:              25000
unique_prompt_texts:        24174
duplicate_group_count:      176     (distinct prompt texts that repeat)
duplicate_record_count:     1002    (records belonging to a repeated-text group)
duplicate_excess_count:     826     (records beyond the first in each group)
max_multiplicity:           6
stored_hash_mismatch_count: 0
cross_domain_duplicate_group_count: 0
reconciliation: 25000 == 24174 + 826  -> TRUE
```

- Every stored `prompt_sha256` matched its independently recomputed hash -- zero inconsistencies between stored hash and actual text.
- Zero duplicate groups span more than one domain.
- Duplicates are almost entirely in SCIENCE (175 of 176 groups, 1000 of 1002 records, 825 of 826 excess) -- confirmed to be template reuse within specific `(category, subdomain)` pairs (e.g. five verbatim copies of the same `factual_knowledge` × `Mechanics` question). DNA has exactly one incidental duplicate pair; EMOTIONS, HINDI, GANDHI have zero.
- `826` matches the `exact_duplicates` figure from the prior structural-validation pass (`experiments/corpus_validation_25k/`) -- a cross-check that the earlier count and this one's *excess* count agree.
- **No records were deleted, merged, or rewritten.** Full group-by-group detail is in `reports/exact_duplicate_groups.csv`.

## 3. Language / script classification (Objective 2)

Method: `deterministic_unicode_script_plus_lexical_markers_v1` -- no language-identification model, no LLM call per record (see `scripts/lang_script_utils.py` module docstring for the full rationale). Every one of the 25,000 records got exactly one classification.

```
language_totals:       English=19997  Hindi=4141  mixed=833  uncertain=29
script_totals:         Latin=21564  Devanagari=3000  mixed=436
language_form_totals:  english=19997  native_hindi=3132  romanized_hindi=1009
                        hinglish=529  mixed_multilingual=304  uncertain=29
requires_manual_review: 508 (2.03%)
```

By domain: DNA, EMOTIONS, GANDHI, and SCIENCE are ~100% English (DNA has 3 `uncertain` outliers, see limitations). HINDI splits into 4,141 Hindi (native + romanized + hinglish + mixed), 833 further flagged `mixed` at the script level, and 26 `uncertain`.

**Two real classification bugs were found and fixed during development** (not just accepted on the first pass):
1. The word `"the"` was initially included as a Hindi transliteration marker (Hindi थे/"were") but collides catastrophically with the single most common English word, along with `"to"` and `"men"` -- this initially misclassified 58% of the entire corpus as `uncertain`. All three were removed from the marker list.
2. `"hi"` (Hindi casual marker) collided with the English interjection "hi" and, via hyphen-splitting, with "Hi-C" (a real DNA-domain method name) -- removed.
3. A narrower, accepted residual: `"ne"` (a genuinely common and useful Hindi marker) collides with the population-genetics abbreviation `Ne`/`N_e`, affecting exactly 3 DNA records -- left in place (removing it would cost far more true-Hindi recall than it's worth) and those 3 records correctly land in `requires_manual_review` rather than being silently mislabeled.
4. Distinguishing `romanized_hindi` from `hinglish` using closed-class markers alone is fundamentally approximate (Hindi's open-class vocabulary is much larger than any compact marker list can cover) -- the heuristic requires corroborating English *content* loanwords before calling `hinglish`, and routes borderline cases to manual review rather than guessing. 26 HINDI records use a distinct, non-standard transliteration spelling convention (e.g. "snbndha" instead of "sambandh", "aura" instead of "aur") not covered by either marker list -- correctly flagged `uncertain` rather than mismatched.

Full per-domain breakdown: `reports/language_script_domain_breakdown.csv`. Records needing human judgment: `reports/language_script_review.csv` (508 rows). Annotations (joinable by `prompt_id`, original text never touched): `annotations/language_script_annotations.jsonl`.

## 4. Embedding model selection and rationale (Objective 3)

**Chosen:** `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, MIT license, 384-dim, ~118M params (~470 MB). Full rationale, alternatives considered, and hardware-fit analysis: `run_metadata/model_selection.md`. In short: it's the only candidate with Hindi *explicitly* documented in its language coverage (not just "multilingual" generically), MIT-licensed, and small enough to run comfortably on CPU (chosen over GPU to avoid an unnecessary 2+ GB CUDA PyTorch download for a 118M-parameter inference-only model). Only one model was downloaded.

**Weights location:** `weights/hf_cache/` (471 MB) -- `HF_HOME`/`HF_HUB_CACHE`/`TRANSFORMERS_CACHE`/`SENTENCE_TRANSFORMERS_HOME` are all redirected there in `scripts/embed_utils.py` *before* any `transformers`/`sentence_transformers`/`torch` import. Verified empirically: the default `C:\Users\...\.cache\huggingface` directory has no new content from this task (its `hub/` subfolder's last-modified date predates this session).

**Documented limitation (stated up front, in `model_selection.md`, before generating anything):** multilingual embedding models are trained overwhelmingly on native-script text; Romanized Hindi/Hinglish embedding quality is not something these models are typically evaluated on, and multilingual support does not guarantee equal quality across scripts. The pilot step checks this empirically rather than assuming it.

## 5. Pilot results

20 prompts, stratified by `(domain, language_form)` so that every domain (DNA, EMOTIONS, GANDHI, HINDI, SCIENCE) and every language form (native_hindi, romanized_hindi, hinglish, english, mixed_multilingual) is represented, including short and long prompts (token range 11-104). First pilot attempt only covered 3 of 5 domains (a selection-logic bug -- it exhausted the first domain's quota before reaching later ones alphabetically); fixed and rerun before accepting the result.

```
dimension_check_passed:      true
finite_check_passed:         true
normalization_check_passed:  true
overall_pass:                true
```

- Nearest-neighbour spot checks are topically coherent: DNA Hi-C/genome-assembly questions cluster together; HINDI grammar questions about related constructs (aspect, oblique case, politeness) cluster together.
- Cross-script comparison (closest available same-domain, different-script pair in the pilot): `native_hindi` vs `romanized_hindi` similarity = 0.81 -- not a true parallel pair (no exact translation pairs exist in the corpus), so this is indicative only, not conclusive.
- **Real finding, not hidden:** the "unrelated" GANDHI-vs-SCIENCE pair scored 0.78 cosine similarity -- close to the pilot's own median pairwise similarity (0.798), i.e. `similarity_compression_observed: true`. This is the anisotropy commonly seen in sentence-embedding models, where even unrelated texts score high raw cosine similarity. **Recorded as explicit guidance for the next milestone: treat cosine similarity as a relative ranking signal for nearest-neighbour ordering, not an absolute relatedness threshold**, unless/until further calibration is done.

Full detail: `reports/embedding_pilot_report.json`.

## 6. Full embedding generation results

```
total_records:                      25000
valid_embeddings:                   25000
invalid_embeddings:                 0
records_exceeding_max_sequence_length: 0   (longest prompt: 57 words, well under the 512-token limit)
embedding_dimension:                384
normalized (L2):                    true
device:                             cpu
```

- Deterministic ordering: output rows are sorted by `prompt_id` ascending, independent of batch processing order.
- **Resume behavior tested explicitly**, not just claimed: a second invocation of `--mode full` against the already-complete output detected `already_done=25000, to_embed=0` and skipped all model inference, re-validating and re-writing the same (byte-identical, sha256-confirmed) output in seconds instead of ~10 minutes.
- Independent post-hoc validation (`scripts/validate_embedding_outputs.py`, which does not re-run the model) confirms: row count matches the normalized corpus exactly, `prompt_id` → row mapping is complete and gap/duplicate-free, every vector is finite, every vector is unit-normalized, and the `.npy` file's sha256 matches the manifest's recorded hash.
- Preprocessing (recorded in `embeddings/embedding_manifest.json`): complete original `prompt` field, verbatim, with the required `"query: "` e5 prefix prepended; no domain/subdomain/category/annotation text appended; no truncation occurred.

## 7. Storage consumed

```
weights/hf_cache/                                    471 MB   (model weights + tokenizer, NOT in C: cache)
experiments/duplicate_language_embedding/annotations/  16 MB   (language_script_annotations.jsonl)
experiments/duplicate_language_embedding/embeddings/   40 MB   (prompt_embeddings.npy + index + manifest)
experiments/duplicate_language_embedding/reports/     206 KB   (all CSV/JSON reports)
experiments/duplicate_language_embedding/run_metadata/  8 KB
--------------------------------------------------------------
Total new disk usage: ~527 MB, well within the ~100 GB budget.
```

## 8. Scripts and commands used (reproducible end to end)

```
python scripts/audit_exact_duplicates.py
python scripts/classify_language_script.py --overwrite
python ./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m pip install sentence-transformers
./.venv/Scripts/python.exe scripts/generate_embeddings.py --mode pilot
./.venv/Scripts/python.exe scripts/generate_embeddings.py --mode full
./.venv/Scripts/python.exe scripts/validate_embedding_outputs.py
```

## 9. Known limitations and unresolved cases

1. **`romanized_hindi` vs `hinglish` boundary is approximate.** Closed-class lexical markers can't fully distinguish "Hindi sentence using uncommon vocabulary" from genuine code-switched Hinglish. Ambiguous cases are flagged (`requires_manual_review=true`), not guessed.
2. **3 DNA records** misclassify-resistant to a residual `Ne`/`ne` collision (population-genetics abbreviation vs. Hindi postposition) -- correctly routed to manual review, not silently mislabeled.
3. **26 HINDI records** use a non-standard transliteration spelling convention not covered by the marker lists -- correctly flagged `uncertain`.
4. **No true parallel corpus** exists across Devanagari/Romanized/English for this dataset, so the pilot's cross-script comparison is indicative, not a rigorous equivalence test.
5. **Similarity-score compression**: raw cosine similarities across this embedding model cluster in a narrow high range (~0.73-0.96 observed in the pilot); the next milestone should use relative nearest-neighbour ranking, not absolute cosine thresholds, until recalibrated (e.g. via centering/whitening).
6. **Romanized Hindi/Hinglish embedding quality is unverified** beyond the pilot's basic technical checks -- per the task's explicit caution, multilingual model support does not guarantee equal quality across scripts, and no rigorous evaluation of this was in scope here.
7. SCIENCE's 825 exact-duplicate records (template reuse) remain in the embedding set unmodified -- downstream clustering work should be aware that a handful of SCIENCE clusters will contain literal duplicate points.

## 10. Source data integrity -- explicit confirmation

- `git diff --stat HEAD -- normalized raw processed MI_Dataset_validated` returns **empty** -- zero bytes changed in any source-data directory since the last commit, for the entirety of this task.
- Every script in this task opens `raw/` and `normalized/` read-only; all new content was written exclusively under `experiments/duplicate_language_embedding/`.
- No prompt text was ever rewritten, paraphrased, truncated, or deleted at any stage of the duplicate audit, language classification, or embedding generation.

## Acceptance criteria checklist

**Duplicate audit:** all 25,000 accounted for ✅ · hashes independently verified ✅ · multiplicity metrics reconcile mathematically (25000 = 24174 + 826) ✅ · cross-domain duplicates checked (0 found) ✅ · metadata conflicts checked (0 found) ✅ · no source records changed ✅

**Language/script classification:** every record has exactly one annotation ✅ · every annotation maps to a valid `prompt_id` ✅ · all controlled values are schema-valid (enforced in-script, run exits non-zero otherwise) ✅ · Hindi/Romanized Hindi/Hinglish/English distinguished where evidence permits, with documented approximation at the romanized/hinglish boundary ✅ · ambiguous cases flagged (508, 2.03%), not forced ✅ · metrics available by domain ✅ · original text unchanged ✅

**Embeddings:** model + revision documented (`intfloat/multilingual-e5-small`, `614241f622f53c4eeff9890bdc4f31cfecc418b3`) ✅ · pilot passed (after fixing a domain-coverage selection bug) ✅ · exactly 25,000 valid embeddings, one per prompt ✅ · consistent dimension (384) ✅ · all finite ✅ · normalization verified (L2, confirmed independently) ✅ · all map correctly to `prompt_id` (verified independently) ✅ · resume tested and confirmed working ✅ · no prompt text or source dataset modified ✅

No acceptance criterion failed. No blockers remain open.

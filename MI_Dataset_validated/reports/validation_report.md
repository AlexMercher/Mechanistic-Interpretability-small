# MI Dataset — normalization and validation report

## Scope and outcome

Processed the supplied archive containing 45 Markdown source files and 15,000 prompt records across DNA, EMOTIONS and HINDI. The original source files are preserved byte-for-byte under `source_originals/`.

**Important limitation:** structural integrity and metadata normalization were performed programmatically. This is not an independent expert certification of all scientific claims or Hindi linguistic correctness. All prompts remain unchanged and are marked `structural_pass_content_review_required` so downstream users do not mistake them for expert-validated content.

## Domain counts

| Domain | Prompts |
|---|---:|
| DNA | 5000 |
| EMOTIONS | 5000 |
| HINDI | 5000 |
| **Total** | **15,000** |

SCIENCE and GANDHI were not present in the uploaded ZIP and were not invented or added.

## Structural checks

- All 45 Markdown files parsed using the archive's actual `PROMPT:` format.
- All per-file declared TARGET_COUNT values matched parsed prompt counts.
- No empty prompt bodies were found.
- Canonical IDs were assigned uniquely per domain.
- Original IDs, generator names, source paths and original prompt text were retained.
- The prompt text was not paraphrased or scientifically edited.
- JSONL records include SHA-256 hashes of the extracted prompt text.
- Deterministic train/validation/test splits were assigned within domain/subdomain/category/generator groups.

## Metadata normalization

- Categories normalized to lowercase snake_case; original category retained.
- Hindi script labels mapped conservatively to controlled labels; original labels retained in `source_script_label`.
- Missing `DOMAIN_CLASS` values remain null rather than being guessed.
- Source IDs are retained separately because they collide across files and generators.

## Category distribution

| Domain | Category | Count |
| DNA | CONTEXTUAL_UNDERSTANDING | 2359 |
| DNA | CONTRASTIVE_UNDERSTANDING | 629 |
| DNA | FACTUAL_KNOWLEDGE | 1579 |
| DNA | REAL_WORLD_APPLICATIONS | 363 |
| DNA | REASONING | 70 |
| EMOTIONS | CONTEXTUAL_UNDERSTANDING | 1000 |
| EMOTIONS | CONTRASTIVE_UNDERSTANDING | 1000 |
| EMOTIONS | FACTUAL_KNOWLEDGE | 1000 |
| EMOTIONS | REAL_WORLD_APPLICATIONS | 1000 |
| EMOTIONS | REASONING | 1000 |
| HINDI | CONTEXTUAL_UNDERSTANDING | 1001 |
| HINDI | CONTRASTIVE_UNDERSTANDING | 1000 |
| HINDI | FACTUAL_KNOWLEDGE | 999 |
| HINDI | REAL_WORLD_APPLICATIONS | 1000 |
| HINDI | REASONING | 1000 |

DNA has 70 reasoning prompts (1.4% of its 5,000 records), so task-category comparisons for DNA will be imbalanced. This is documented, not silently rebalanced.

## Duplicate and template review

- Exact duplicate normalized prompt bodies: 1 pair (2 records). See `exact_duplicates.csv`.
- Near-duplicate candidate pairs at character n-gram cosine similarity >= 0.82: 874 pairs found; top 500 are exported in `near_duplicate_candidates_top500.csv`.
- Near-duplicate candidates are not confirmed duplicates. Review before excluding anything.
- Repeated prompt templates can create lexical shortcuts in activation analysis. Use the candidate report and source category/model fields to design robustness checks.

## Scientific and linguistic content review — limits

- The archive was checked for structural and metadata integrity, but all 15,000 prompts were not independently fact-checked by a subject-matter expert.
- DNA scientific correctness and Hindi grammar/semantics/pragmatics cannot honestly be certified from automated structural checks alone.
- No prompts were rewritten or removed to avoid silently changing the dataset.
- Use `validation_status` as a clear warning; perform targeted human/expert review before treating any prompt as a gold-standard evaluation item.
- Emotion prompts also need construct-validity review; their labels are intended dataset topics, not ground truth about actual human mental states.

## Split counts

| Domain | Train | Validation | Test |
| DNA | 4018 | 491 | 491 |
| EMOTIONS | 3950 | 525 | 525 |
| HINDI | 3972 | 514 | 514 |

## Files

- `datasets/jsonl/dna.jsonl`, `emotions.jsonl`, `hindi.jsonl`: canonical records.
- `schemas/prompt.schema.json`: JSON Schema.
- `reports/exact_duplicates.csv`: exact duplicate records.
- `reports/near_duplicate_candidates_top500.csv`: high-similarity candidates for review.
- `reports/metadata_review_flags.csv`: Hindi script metadata review queue.
- `source_originals/`: byte-preserved Markdown inputs.

## Recommended use

Use the JSONL as a normalized staging dataset, not a scientifically certified benchmark. For SAE training, consider balancing sampling across domains and subdomains while preserving the natural dataset and split labels. Do not use the held-out validation/test prompts to fit the SAE if you want unbiased feature evaluation.
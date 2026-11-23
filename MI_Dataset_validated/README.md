# Mechanistic Interpretability Dataset

## Current status

This repository contains version 1.0 of the normalized prompt dataset for three domains: **DNA**, **Emotions**, and **Hindi**. The current release contains **15,000 prompts** in total, with 5,000 prompts per domain.

The dataset is part of a broader five-domain mechanistic interpretability project. The **Science** and **Gandhi** domains have not yet been added and are planned for future releases.

## Data sources

The raw prompt files were generated using three large language models (LLMs): **ChatGPT**, **Claude**, and **Gemini**. The original Markdown source files are retained under `source_originals/` for provenance and reference.

## Processed data

The normalized, machine-readable datasets are provided as JSONL files:

- `datasets/jsonl/dna.jsonl` — 5,000 prompts
- `datasets/jsonl/emotions.jsonl` — 5,000 prompts
- `datasets/jsonl/hindi.jsonl` — 5,000 prompts

Together, these files contain **15,000 prompts**.

## Preservation and normalization

- **Original prompt wording has been preserved.** Prompt text was not rewritten during normalization.
- **Metadata has been normalized** into a consistent record structure, with source/provenance fields retained where available.
- **IDs have been normalized** to provide unique canonical identifiers in the processed dataset. Original source IDs are retained separately where available.
- The schema is available at `schemas/prompt.schema.json`, and dataset-level metadata is recorded in `manifest.json`.

## Validation status and limitations

The current validation covers structural parsing and consistency checks, including record counts, required fields, metadata, and duplicate/near-duplicate screening. It does **not** constitute scientific or linguistic expert review.

**Scientific and linguistic expert validation is pending.** In particular, the scientific correctness of DNA content and the linguistic quality and appropriateness of Hindi content have not been certified by domain experts. Treat the dataset as a structured research draft rather than an expert-validated benchmark.

Review outputs and validation notes are provided in `reports/`. Similarity-based near-duplicate entries are candidates for human review, not confirmed duplicates.

## Directory overview

```text
MI_Dataset_validated/
├── README.md
├── manifest.json
├── datasets/
│   └── jsonl/
│       ├── dna.jsonl
│       ├── emotions.jsonl
│       └── hindi.jsonl
├── schemas/
│   └── prompt.schema.json
├── source_originals/
│   └── ... original Markdown source files ...
└── reports/
    └── ... validation and distribution reports ...
```

## Planned additions

The current release includes only DNA, Emotions, and Hindi. The **Science** and **Gandhi** domains remain to be collected, processed, and added in a future dataset release.

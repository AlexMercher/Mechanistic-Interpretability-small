# File Context

One-line description of each file/folder in this repo. Check here first before exploring the codebase.

## Root

- `CLAUDE.md` — guidance for Claude Code: pipeline overview + repo conventions (no destructive commands, temp/, plans/, weights/, venv policy).
- `File_context.md` — this file.
- `.gitignore` — excludes `temp/`, `plans/`, `.venv/`, model weight binaries (weights/* except .gitkeep), Python caches.
- `raw/`, `normalized/`, `processed/` — the three frozen layers of the 25K-prompt dataset (see "Dataset layers" below).
- `MI_Dataset_validated/` — legacy, untouched snapshot of the original DNA/EMOTIONS/HINDI dataset release (its own source_originals/datasets/jsonl/reports convention). Superseded by `raw/`/`normalized/`/`processed/` for day-to-day use; kept for provenance.
- `MONISHA_SCIENCE_AND_gANDHI_10000_PROMPTS (1)/` — original delivered zips + their extracted SCIENCE/GANDHI Markdown prompt files (no JSONL). Source for `raw/SCIENCE/` and `raw/GANDHI/`.
- `scripts/` — the build pipeline (see below).
- `plans/` — Claude's planning docs (gitignored). Subdirs: `active/`, `completed/`, `archive/`.
- `temp/` — scratch/ephemeral work (gitignored). Subdirs: `scratch/`, `experiments/`, `downloads/`.
- `weights/` — model weight files live here only (gitignored except `.gitkeep`).
- `.venv/` — project virtual environment (gitignored). See "Environment setup" below.

## scripts/ (dataset build pipeline, stdlib-only, run in order)

- `build_raw_snapshot.py` — `main()` copies original unmodified Markdown for all 5 domains into `raw/<DOMAIN>/`; `sha256_of()` hashes each file; writes `raw/MANIFEST.json`.
- `build_normalized.py` — `parse_md_blocks()` parses `KEY: value` Markdown record blocks; `build_md_domain()` turns SCIENCE/GANDHI blocks into canonical records with stable `prompt_id`; `deterministic_split()` assigns train/validation/test by hashing the prompt_id (reproducible, not random); `snake_case()` normalizes category labels; `main()` copies legacy DNA/EMOTIONS/HINDI JSONL verbatim and writes `science.jsonl`/`gandhi.jsonl` into `normalized/`.
- `validate_normalized.py` — `main()` checks every record in `normalized/*.jsonl` for required fields, valid domain/split enums, and globally unique `prompt_id`s; exits non-zero on any error.
- `build_processed.py` — `main()` derives, per domain: `clusters/` (grouped by subdomain via `snake_case()`), `chunks/` (fixed-size `CHUNK_SIZE=50` batches per category for batched model runs), exact-duplicate detection by `prompt_sha256` collisions, `annotations/` (normalized record + cluster_id/chunk_id/is_exact_duplicate), and cross-domain `reports/category_distribution.csv` + `subdomain_distribution.csv`.

## Dataset layers

- `raw/<DOMAIN>/...` — byte-identical copies of original source Markdown (DNA/EMOTIONS/HINDI copied from `MI_Dataset_validated/source_originals/`, SCIENCE/GANDHI copied from the Monisha extracted folders). `raw/MANIFEST.json` — sha256 + byte size per copied file.
- `normalized/{dna,emotions,hindi,science,gandhi}.jsonl` — one canonical JSON record per prompt; prompt text never rewritten. `normalized/prompt.schema.json` — JSON Schema for a record (5-domain enum). `normalized/manifest.json` — dataset-level counts/sources index.
- `processed/clusters/<domain>_clusters.jsonl` — `{prompt_id, cluster_id, subdomain, category}` grouping by subdomain.
- `processed/chunks/<domain>_chunks.jsonl` — `{chunk_id, domain, category, size, prompt_ids}` fixed-size batches per category.
- `processed/annotations/<domain>_annotated.jsonl` — full normalized record plus `cluster_id`, `chunk_id`, `is_exact_duplicate`.
- `processed/reports/category_distribution.csv`, `subdomain_distribution.csv` — counts across all 5 domains.
- `processed/reports/<domain>_exact_duplicates.csv` — `{prompt_id, duplicate_of, prompt_sha256}` for exact-text duplicates within a domain (SCIENCE has 825 — formulaic template reuse noted in its own GENERATION_REPORT.md; others have 0-1).

## Environment setup

`.venv/` already exists (created with `python -m venv .venv`, Python 3.12.7) but has no packages installed yet — nothing in this repo currently needs anything beyond the standard library. When a task needs packages, install into this same `.venv` rather than recreating it:

```
# activate (Windows, git-bash)
source .venv/Scripts/activate

# everyday light deps
pip install numpy pandas

# torch (CPU) -- large download, only do this once per machine
pip install torch --index-url https://download.pytorch.org/whl/cpu

# torch (CUDA 12.1 build) -- only if a GPU is confirmed available
pip install torch --index-url https://download.pytorch.org/whl/cu121

# transformer-lens / HF stack, if/when needed for interpretability work
pip install transformers transformer_lens
```

If a large install (torch, CUDA builds) fails, stop and tell the user the exact failing command rather than retrying it repeatedly.

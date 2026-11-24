# File Context

One-line description of each file/folder in this repo. Check here first before exploring the codebase.

## Root

- `CLAUDE.md` — guidance for Claude Code: pipeline overview + repo conventions (no destructive commands, temp/, plans/, weights/, venv policy).
- `File_context.md` — this file.
- `.gitignore` — excludes `temp/`, `plans/`, `.venv/`, model weight binaries (weights/* except .gitkeep), Python caches.
- `raw/`, `normalized/`, `processed/` — the three frozen layers of the 25K-prompt dataset (see "Dataset layers" below).
- `experiments/` — dated/named experiment records (method + results + repro steps), e.g. `experiments/corpus_validation_25k/` (see below). Read-only against raw/normalized/processed.
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
- `validate_corpus.py` — Step-2 quality gate over `normalized/*.jsonl`: `check_record()` splits issues into hard (missing fields, blank prompt, actual U+FFFD mojibake) vs soft review flags (e.g. Romanized Hindi with no Devanagari); `check_id_sequence()` checks for gaps/dupes in the per-domain ID sequence; `near_duplicate_pairs()` blocks by `(category, subdomain)` and uses `difflib.SequenceMatcher` with `quick_ratio`/`real_quick_ratio` pre-filters (threshold 0.72) to flag near-duplicates *without deleting anything*; `length_stats_and_outliers()` computes word-count distributions. Writes everything under `experiments/corpus_validation_25k/`.
- `sample_near_duplicates_for_review.py` — draws a tractable stratified sample (top-15-by-similarity + 5 random, per domain) from the near-duplicate candidates for a teacher-model (Opus) adjudication pass; writes `experiments/corpus_validation_25k/reports/near_duplicate_review_sample.json`.
- `audit_exact_duplicates.py` — recomputes `sha256(prompt)` independently of the stored `prompt_sha256` field and groups by that recomputed hash; reports multiplicity distribution, domain/subdomain/category breakdowns, cross-domain groups, metadata conflicts, and a mathematical reconciliation check. Writes `experiments/duplicate_language_embedding/reports/`.
- `lang_script_utils.py` — deterministic (no ML model) Unicode-script-range + closed-class lexical-marker classifier (`classify(text)`); distinguishes Hindi/English/mixed language, Devanagari/Latin/mixed script, and native_hindi/romanized_hindi/hinglish/english/mixed_multilingual language_form. See its module docstring for two real marker-collision bugs found and fixed ("the"/"to"/"men"/"hi" colliding with common English words).
- `classify_language_script.py` — CLI wrapper running `lang_script_utils.classify()` over all 25,000 records; writes `experiments/duplicate_language_embedding/annotations/language_script_annotations.jsonl` + domain-breakdown/review-queue reports.
- `embed_utils.py` — **must be imported before `torch`/`transformers`/`sentence_transformers`** anywhere in a process: redirects all HF/sentence-transformers caches into `weights/hf_cache/` (never the default `C:\Users\...\.cache`). Defines the chosen embedding model (`intfloat/multilingual-e5-small`) and the `embed_texts()` helper (applies the required `"query: "` e5 prefix, L2-normalizes).
- `generate_embeddings.py` — `--mode pilot` (stratified sample across domain × language_form × length, writes a sanity-check report incl. a similarity-compression diagnostic) and `--mode full` (all 25,000 records, deterministic `prompt_id`-sorted output, resumable -- skips records already embedded under an identical model/config, errors on a silent config change unless `--overwrite`). Writes `experiments/duplicate_language_embedding/embeddings/`.
- `validate_embedding_outputs.py` — independent post-hoc check (doesn't re-run the model): row count, prompt_id↔row mapping completeness, finiteness, normalization, and `.npy` sha256 vs. manifest.

## Dataset layers

- `raw/<DOMAIN>/...` — byte-identical copies of original source Markdown (DNA/EMOTIONS/HINDI copied from `MI_Dataset_validated/source_originals/`, SCIENCE/GANDHI copied from the Monisha extracted folders). `raw/MANIFEST.json` — sha256 + byte size per copied file.
- `normalized/{dna,emotions,hindi,science,gandhi}.jsonl` — one canonical JSON record per prompt; prompt text never rewritten. `normalized/prompt.schema.json` — JSON Schema for a record (5-domain enum). `normalized/manifest.json` — dataset-level counts/sources index.
- `processed/clusters/<domain>_clusters.jsonl` — `{prompt_id, cluster_id, subdomain, category}` grouping by subdomain.
- `processed/chunks/<domain>_chunks.jsonl` — `{chunk_id, domain, category, size, prompt_ids}` fixed-size batches per category.
- `processed/annotations/<domain>_annotated.jsonl` — full normalized record plus `cluster_id`, `chunk_id`, `is_exact_duplicate`.
- `processed/reports/category_distribution.csv`, `subdomain_distribution.csv` — counts across all 5 domains.
- `processed/reports/<domain>_exact_duplicates.csv` — `{prompt_id, duplicate_of, prompt_sha256}` for exact-text duplicates within a domain (SCIENCE has 825 — formulaic template reuse noted in its own GENERATION_REPORT.md; others have 0-1).

## experiments/corpus_validation_25k/ (Step 2 dataset-quality experiment)

- `EXPERIMENT.md` — full write-up: motivation, method, results, Opus teacher-model adjudication of near-duplicates, conclusion, repro steps. Read this first for the validation results.
- `metrics.json` — machine-readable summary (25,000/25,000 valid, 826 exact dups, 35,938 near-dup candidates, 1,401 Romanized-Hindi review flags, 0 length outliers).
- `reports/invalid_records.csv`, `review_flags.csv`, `near_duplicate_candidates.csv`, `near_duplicate_review_sample.json`, `near_duplicate_opus_verdicts.json`, `length_stats.csv`, `length_outliers.csv` — see `EXPERIMENT.md` "Files" section for what each contains.

## experiments/duplicate_language_embedding/ (exact-duplicate audit + language/script classification + embeddings)

- `FINAL_REPORT.md` — full write-up: all 3 objectives, results, model selection rationale, pilot findings, full-run results, storage, limitations, explicit source-data-unchanged confirmation. Read this first.
- `reports/duplicate_metrics.json`, `exact_duplicate_groups.csv`, `duplicate_multiplicity_distribution.csv`, `duplicate_domain_breakdown.csv`, `duplicate_metadata_conflicts.csv` — Objective 1 outputs.
- `annotations/language_script_annotations.jsonl`, `reports/language_script_metrics.json`, `language_script_domain_breakdown.csv`, `language_script_review.csv` — Objective 2 outputs.
- `run_metadata/model_selection.md` — embedding model choice (`intfloat/multilingual-e5-small`, MIT, 384-dim, Hindi explicitly supported) with hardware-fit and license analysis, done *before* downloading weights.
- `embeddings/prompt_embeddings.npy` (25000×384 float32, L2-normalized), `embedding_index.csv` (prompt_id↔row mapping + status), `embedding_manifest.json` (model/revision/config/sha256) — Objective 3 full-run output.
- `reports/embedding_pilot_report.json` — pilot sanity-check results (dimension/finite/normalization checks, nearest-neighbour spot checks, a documented similarity-compression caveat).

## experiments/semantic_topology_25k/ (Milestone 3: embedding evaluation, concept discovery, cross-domain relationships, final semantic hierarchy)

- `reports/FINAL_REPORT.md` — full write-up: all 5 objectives (A-E), results, limitations, storage, repro commands, source-integrity confirmation. Read this first.
- `scripts/semantic_common.py` — shared loaders (`load_normalized_records`, `load_annotations`, `load_embeddings` with sha256 re-verification, `load_duplicate_group_map`, `unique_text_count`) for every script in this experiment; read-only against Milestones 1-2.
- `scripts/evaluate_embedding_space.py` + `scripts/build_evaluation_candidates.py` + `scripts/evaluate_retrieval.py` — Objective A: integrity re-check, sampled cosine-similarity distributions across 8 comparison groups, a 180-pair teacher-model-reviewed evaluation set (`evaluation/reviewed_prompt_pairs.jsonl`), and precision@k/NDCG@k retrieval metrics. Key finding: this embedding space's raw similarity is frequently template-phrasing-driven, not meaning-driven (precision@20≈0.11); see `reports/embedding_space_evaluation.md`.
- `scripts/build_domain_clusters.py` — Objective B: clusters each (domain, subdomain) bucket independently (HDBSCAN vs. silhouette-swept KMeans, chosen via a category-field normalized-mutual-information confound diagnostic) after domain-wide clustering was tried and rejected for rediscovering the `category` field instead of real topics. Writes `clustering/<domain>_cluster_details.json`, `cluster_assignments.jsonl`, `cluster_metadata.json`, `visualizations/*_umap.png` (viz-only, never used for assignment).
- `scripts/name_clusters.py` — TF-IDF distinctive-term concept naming over full cluster text (not a first-record read); writes `clustering/cluster_provisional_names.json`, marked `auto_generated_unreviewed`.
- `scripts/discover_cross_domain_relations.py` — Objective C: shortlists 150 cross-domain cluster-centroid candidate pairs (15 per domain pair) for teacher-model review; writes `taxonomy/cross_domain_candidates.jsonl`. Reviewed output (`taxonomy/concept_relationships.jsonl`) labels each with a 9-type relationship vocabulary + evidence; 104/150 were ruled `no_relationship` (template artifacts).
- `scripts/construct_taxonomy.py` + `scripts/render_hierarchy_markdown.py` — Objective D: freezes `taxonomy/semantic_hierarchy_v1.json` (domain-specific taxonomy + cross-domain graph, stable concept_ids, full prompt traceability) and renders `taxonomy/SEMANTIC_HIERARCHY.md`. `taxonomy/taxonomy_review.json` records a 40-concept teacher-model spot-check (35/40 looks_good; found GANDHI over-fragmentation and HINDI script-vs-topic confound — see FINAL_REPORT.md limitations).
- `scripts/generate_semantic_reports.py` — Objective E: `reports/domain_coverage.json`, `cluster_metrics.json`, `cross_domain_analysis.json` (concepts per domain/subdomain, size extremes, template-dominated/under-sized/over-under-represented concepts, language/script distribution per concept).
- `scripts/validate_taxonomy.py` — independent post-hoc check (prompt-assignment completeness/uniqueness, concept_id referential integrity, duplicate-aware count consistency); passes with 0 errors.

## experiments/activation_extraction_qwen3_1_7b/ (Milestone 4: Qwen3-1.7B middle-layer residual-stream activation extraction)

- `reports/FINAL_REPORT.md` — full write-up: model/layer selection, pilot findings (incl. an investigated bf16 batched-vs-unbatched numerical-drift finding), full-run results, validation, statistics, limitations, repro commands. Read this first.
- `scripts/model_utils.py` — **must be imported before `transformers`/`torch`** anywhere in a process: redirects all HF caches into `weights/hf_cache/`. `get_device_and_check_vram()` raises (never silently falls back to CPU) if CUDA/VRAM is insufficient. `load_model_and_tokenizer()` loads `Qwen/Qwen3-1.7B` directly onto CUDA in bf16, inference mode, gradients disabled, resolves and pins the exact commit sha.
- `scripts/inspect_model.py` — downloads (once, cached) and inspects the model: confirms hidden_size=2048, 28 layers, selects layer 14 (`n_layers // 2`) as the middle layer, documents the exact hook location (`model.model.layers[14]` output = residual stream after that block), special tokens (no bos_token exists for this model). Writes `metadata/model_manifest.json`.
- `scripts/extraction_common.py` — shared extraction core: `ShardWriter` (safetensors shards, bf16-native, atomic tmp-then-rename writes, a prompt's tokens never split across shards), `ResidualStreamHook` (forward hook on one decoder block, avoids retaining all 28 layers via `output_hidden_states=True`), `compute_prompt_split` (deterministic sha256-based prompt-level train/val split, seed 1337), `build_batches` (length-bucketed), `run_extraction()` (the full batched/resumable/incremental-write extraction loop used by both the pilot and the full run).
- `scripts/run_activation_pilot.py` — mandatory pre-full-run pilot (60 prompts, stratified across all 5 domains and a length range), writing to an isolated `reports/pilot_data/` directory. Found and resolved a real anomaly: a naive disk-vs-fresh-forward-pass round-trip check failed under strict `torch.allclose`; root-caused to expected bf16 batched-vs-unbatched numerical drift (<0.15% of elements affected by a small absolute amount), not a metadata bug — documented in `reports/pilot_report.md`.
- `scripts/extract_activations.py` — full 25,000-prompt run. Resumable: re-verifies every existing shard's sha256 against disk (not just the manifest's say-so) before skipping its prompts; empirically re-tested by running it twice (second run correctly reported "0 remaining").
- `scripts/validate_activations.py` — independent post-hoc check covering all 12 required items (prompt/token completeness, no padding leakage via re-tokenization, no duplicate rows, dim/dtype/finite checks, shard-manifest-vs-disk consistency, train/val split integrity, all 5 domains present, `git diff` source-integrity check). Passes with 0 errors.
- `scripts/generate_activation_report.py` — activation statistics (tokens per domain/subdomain, length percentiles, storage size, activation-norm and per-dimension-variance summaries — found a few large-magnitude "outlier" dimensions, a known LLM residual-stream phenomenon, left un-normalized per the task's instruction to preserve raw activations).
- Results: 25,000/25,000 prompts, 993,496 tokens (corpus average 39.7 tokens/prompt, far below the task's 256-token planning estimate), 4.07 GB storage (10 safetensors shards), peak VRAM 4.11GB/6.44GB, ~9.4 min wall-clock.

## experiments/sae_capacity_comparison/ (Milestone 5: SAE dictionary-capacity comparison, 4096/8192/16384, on Qwen3-1.7B layer-14 activations)

- `reports/FINAL_REPORT.md` — full write-up: preprocessing rationale, frozen architecture/training protocol, 3-seed-aggregated results, feature stability, interpretability review, all caveats. Read this first.
- `reports/pilot_report.md` — λ (sparsity coefficient) sweep on a 50K-row slice; chose λ=0.2 for mean L0≈20-50 without feature collapse.
- `reports/schedule_selection.md` — AMP benchmark (2.0-2.4x speedup vs FP32, adopted) + plateau analysis that set the shared step budget to 10,000 (not the pilot's naive ~69-epoch guess), with a real representative validation curve.
- `reports/benchmark_report.json` — raw FP32-vs-AMP timing numbers (CUDA-event-bounded, warm-up excluded).
- `reports/capacity_comparison.{md,json}` — the final comparison table, generated by `scripts/compare_sae_capacities.py` from the per-seed evaluation JSONs (3-seed mean+range per width, never hand-typed). Headline: no width wins on every axis; 8192 has lowest MSE/dead-rate, 4096 is most stable across seeds, interpretability is flat-to-noisy across widths.
- `reports/interpretability_review_summary.md` — teacher-model (Opus) review of 90 sampled features (30/width): coherent-semantic rate flat at 3/1/3 of 30 — wider dictionaries mostly multiplied template-driven and Hindi-script-marker features, not concepts.
- `reports/opus_final_review.md` — independent methodology/correctness review performed before FINAL_REPORT.md was written; recomputed headline numbers from raw JSONs, caught a single-seed-aggregation gap (since fixed) and two secondary-metric caveats (EV-per-feature ≈1/L0, stability-metric dead-feature confound).
- `preprocessing/` — `activation_statistics.json` (train-only per-dim variance/norm stats), `preprocessing_config.json` + `mu_vector.pt` (frozen global-scalar centering+scaling transform, NOT per-dimension whitening — see FINAL_REPORT §3).
- `configs/common_config.json` + `sae_{4096,8192,16384}.json` — frozen architecture/hyperparameters (lr 3e-4, batch 4096, λ=0.2, 10,000 steps, AMP) shared across all 9 training runs; only dictionary width varies.
- `scripts/sae_common.py` — shared code: `SparseAutoencoder` (pre-encoder-bias ReLU SAE, tied decoder-init, unit-norm decoder columns renormalized every step), `Preprocessor`, `load_split_activations` (sha256-verified shard loading from the read-only Milestone-4 shards), `run_validation_pass` (single shared forward pass producing every raw stat the eval scripts need), `load_model_for_eval`.
- `scripts/train_sae.py` — generic trainer (config+seed-driven; every real run goes through this one script). Supports `--amp`; dead-feature tracking is GPU-resident (no per-step host sync); NaN/Inf check every `--nan-check-every` steps (cheap, frequent) with full diagnostics at `--log-every`; checkpoints `latest/best_val/final`, resumable.
- `scripts/benchmark_training.py` — CUDA-event-bounded FP32-vs-AMP step-time benchmark with explicit warm-up exclusion.
- `scripts/evaluate_reconstruction.py`, `evaluate_sparsity.py`, `evaluate_feature_utilization.py` — per-(width,seed) validation-set evaluation; write `evaluations/{reconstruction,sparsity,feature_utilization}/sae_<width>_seed<seed>.json`. (A pre-fix version wrote all seeds to the same filename, silently overwriting seed 0 — stale files preserved under each dir's `_legacy_overwritten_no_seed_suffix/` for audit.)
- `scripts/evaluate_feature_stability.py` — cross-seed decoder-direction + activation-pattern greedy cosine matching (threshold 0.3); writes `evaluations/feature_stability/sae_<width>.json`.
- `scripts/inspect_features.py` — automated per-feature stats (top-activating tokens/prompts/domains) for 30 representative features/width; writes `evaluations/interpretability/sae_<width>_feature_stats.json` (no auto-generated semantic labels — see `sae_<width>_reviewed.json` for the separate teacher-reviewed pass).
- `scripts/compare_sae_capacities.py` — builds the final comparison table/verdict from the evaluation JSONs; treats reconstruction and sparsity as co-equal axes and never defaults to crowning the widest dictionary.
- `run_metadata/execution_manifest.json` — per-run timings/VRAM/exit codes for all 9 training runs, including a note on two harness background-task interruptions (training itself unaffected, resumed from checkpoint).
- Results: 9/9 training runs completed (3 widths x 3 seeds, 10,000 AMP steps each); full 3-seed-aggregated evaluation; no source corpus, activation shard, or prior-milestone output modified (`git diff` empty).

## Environment setup

`.venv/` has torch **2.6.0+cu124 (CUDA-enabled, replacing the earlier CPU-only build)**, `sentence-transformers`, `scikit-learn`, `scipy`, `hdbscan`, `umap-learn`, `matplotlib`, `transformers` 5.24, and `safetensors` installed. HF/sentence-transformers/transformers caches are redirected into `weights/hf_cache/` via `scripts/embed_utils.py` (Milestone 2) / `scripts/model_utils.py` (Milestone 4) -- never the default `C:\Users\...\.cache`. **Note: the C: drive has very little free space (~4GB) -- any future large pip installs should redirect `PIP_CACHE_DIR`/`TMPDIR`/`TMP`/`TEMP` to a folder under `temp/` on the D: project drive first (see Milestone 4 session notes), or the install can fail by filling C:.** Install commands used so far, reusable for future packages:

```
# activate (Windows, git-bash)
source .venv/Scripts/activate

# everyday light deps
pip install numpy pandas

# torch with CUDA (replaces the CPU-only build) -- already installed; don't reinstall unless missing
# IMPORTANT: redirect pip's cache/temp off the (nearly-full) C: drive first, e.g.:
#   export PIP_CACHE_DIR="D:/projects/Mechanistic-Interpretability-small/temp/pip_cache"
#   export TMPDIR="D:/projects/Mechanistic-Interpretability-small/temp/pip_tmp"
#   export TMP="$TMPDIR"; export TEMP="$TMPDIR"
pip install torch --index-url https://download.pytorch.org/whl/cu124

# transformers + safetensors -- already installed; don't reinstall unless missing
pip install transformers safetensors

# sentence-transformers -- already installed; don't reinstall unless missing
pip install sentence-transformers

# scikit-learn, scipy, hdbscan, umap-learn, matplotlib -- already installed (Milestone 3 clustering/viz); don't reinstall unless missing
pip install scikit-learn scipy hdbscan umap-learn matplotlib

# torch (CPU-only build) -- only if a future environment has no GPU; NOT needed here anymore

# transformer-lens, if/when needed for actual activation-level interpretability work
pip install transformer_lens
```

If a large install (torch, CUDA builds) fails, stop and tell the user the exact failing command rather than retrying it repeatedly.

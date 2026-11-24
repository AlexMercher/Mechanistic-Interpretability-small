# Milestone 4 — Qwen3-1.7B Middle-Layer Residual Stream Activation Extraction

**Status: all acceptance criteria in the Definition of Done are met.** This milestone covers
extraction, validation, and storage only — no SAE training, feature discovery, or neuron
interpretation was performed, per the task's explicit instruction.

Reproduce / resume:

```
python experiments/activation_extraction_qwen3_1_7b/scripts/inspect_model.py
python experiments/activation_extraction_qwen3_1_7b/scripts/run_activation_pilot.py --fresh
python experiments/activation_extraction_qwen3_1_7b/scripts/extract_activations.py
python experiments/activation_extraction_qwen3_1_7b/scripts/validate_activations.py
python experiments/activation_extraction_qwen3_1_7b/scripts/generate_activation_report.py
```
`extract_activations.py` is safely re-runnable: it re-verifies every existing shard's sha256
before trusting it and only processes prompts not already completed.

## 1. Model: exact ID, revision, precision

- **Model:** `Qwen/Qwen3-1.7B`
- **Resolved revision (commit sha):** `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- **Loading precision:** bfloat16 (loaded directly onto CUDA; `torch_dtype`/`dtype=bfloat16`)
- **Activation storage precision:** bfloat16 (same as loading precision — chosen because it is
  the model's native training/inference dtype, so no additional rounding is introduced beyond
  what the model itself already uses; stored losslessly via safetensors, which supports bf16
  natively, unlike numpy `.npy`)
- **Weights cache:** `weights/hf_cache/` (verified empirically — nothing new appeared in the
  default `C:\Users\...\.cache`; checkpoint downloaded once, reused on every subsequent run)

## 2. Architecture and hidden dimension

| Property | Value |
|---|---|
| num_hidden_layers | 28 |
| hidden_size | **2048** (matches the task's expectation exactly) |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (grouped-query attention) |
| vocab_size | 151,936 |
| max_position_embeddings | 40,960 |
| attn_implementation | sdpa (memory-efficient fused attention) |
| bos_token | **none** — this model has no BOS token at all |
| eos_token | `<|im_end|>` (id 151645) |
| pad_token | `<|endoftext|>` (id 151643) — tokenizer already defines one, no workaround needed |

Full detail: `metadata/model_manifest.json`.

## 3. Selected layer and hook location

- **Total layers:** 28 (zero-based indexing).
- **Selected layer:** index **14** = `n_layers // 2`, the architectural midpoint.
- **Module:** `model.model.layers[14]`.
- **Hook location:** a `register_forward_hook` on that decoder block, capturing **element 0 of
  its output tuple** — the residual stream **after** block 14 has been applied. This is
  explicitly **not** attention output, not MLP output alone, not embeddings, not logits.
  Verified independently against `output_hidden_states` indexing in `inspect_model.py`
  (`output_hidden_states[15]` == the hook's captured tensor, since index 0 of that tuple is the
  embedding-layer output and index `k` is the output after decoder block `k-1`).
- A forward hook (not `output_hidden_states=True` for the whole model) was used for the actual
  extraction run specifically to avoid retaining all 28 layers' activations in VRAM
  simultaneously — this GPU has only 6.44 GB total.

## 4. Tokenization and extraction protocol

- Every prompt tokenized with the Qwen3 tokenizer, **no chat template, no added prefix** — the
  raw `prompt` field, verbatim.
- Verified empirically: plain `tokenizer(text)` calls add **no BOS/EOS automatically** for this
  model (no bos_token exists; eos is not auto-appended either) — every retained token id is a
  real content token, not a special token.
- **No truncation occurred anywhere in the real run.** Corpus-wide tokenized length: mean 39.7,
  median 31, P90 75, P95 114, **max 266** — far below `max_position_embeddings` (40,960). The
  pipeline supports an optional `--max-length` truncation cap with explicit before/after logging
  (see `extraction_common.run_extraction`'s `truncated_prompts` tracking), but it was left unset
  since truncation was never needed.
- Every non-padding token retained (one row per token); padding excluded via the attention mask,
  cross-checked against an **independent re-tokenization** of every prompt (not merely trusting
  the extraction run's own token count).
- Right-padding used (verified correct for a single non-generation forward pass on a causal
  model: real tokens occupy the sequence prefix, so causal attention makes trailing padding
  irrelevant to every real token's activation; left-padding would have corrupted RoPE position
  alignment).

## 5. Batching, memory management, and GPU usage

- Length-bucketed batching: prompts sorted by tokenized length before batching, so each batch
  pads only to its own longest member, not the corpus-wide max.
- `batch_size=8`, `use_cache=False` (no KV cache needed — single forward pass, not generation),
  `output_hidden_states=False` (only the one hooked layer is retained), `torch.inference_mode()`
  throughout, no gradients anywhere (`requires_grad_(False)` on every parameter at load time).
- Completed batches' activations moved to CPU immediately; GPU tensors deleted and
  `torch.cuda.empty_cache()` called every batch.
- **Peak VRAM usage over the full run: 4.11 GB** of 6.44 GB total (comfortable headroom; grew
  slowly from 3.5 GB to 4.1 GB over the run as length-bucketing reached the longest prompts last).
- The model was never moved to or run on CPU at any point. `model_utils.get_device_and_check_vram()`
  raises (not falls back) if CUDA is unavailable or free VRAM is below a safety floor.

## 6. Pilot findings

Full detail: `reports/pilot_report.md`, `reports/pilot_report_data.json`.

- 60-prompt stratified pilot (12 per domain, isolated in `reports/pilot_data/`, never mixed with
  the real run) passed every check: shapes, dims, dtype, finiteness, padding exclusion, shard
  sha256, independent token-count cross-check.
- **Real finding, investigated and resolved rather than hidden:** a strict disk-vs-fresh-forward-
  pass round-trip check initially failed. Root-caused to expected bf16 batched-vs-unbatched
  numerical drift (different SDPA reduction path, compounding slightly across 14 layers and
  token position) — confirmed by direct inspection: only 0.005%–0.14% of elements exceed small
  absolute-difference thresholds, concentrated partly on a few very-large-magnitude "outlier"
  residual-stream dimensions where even normal bf16 rounding looks large in absolute terms. This
  is not a metadata/alignment bug (row/token/shape/dtype alignment all independently verified
  exact). See §9 for the implication.
- Pilot throughput (1,319 tokens/sec) correctly predicted the full run's actual duration within
  ~1.5x (pilot estimated 12.6 min; actual full run took 9.1 min — faster, likely due to GPU/driver
  warm-up effects in the smaller pilot run).

## 7. Full-run results

| Metric | Value |
|---|---|
| Prompts processed | **25,000 / 25,000** |
| Domains represented | DNA, EMOTIONS, GANDHI, HINDI, SCIENCE (all 5) |
| Total retained (non-padding) tokens | **993,496** |
| Failed/skipped prompts | **0** |
| Truncated prompts | **0** |
| Peak VRAM | 4.11 GB / 6.44 GB |
| Wall-clock extraction time | **566.3 sec (~9.4 minutes)** |
| Activation storage on disk | **4.0694 GB** (10 safetensors shards: 9 train, 1 validation) |
| Train / validation prompts | 22,467 / 2,533 (9.0% / 10% target — see §8) |
| Train / validation tokens | 895,502 / 97,994 |

**Note on the task's planning estimate:** the task's worst-case storage/time planning assumed a
256-token average prompt length (→ ~6.4M tokens, ~26.2 GB). This corpus's actual average is 39.7
tokens (→ 993,496 tokens, 4.07 GB) — measured directly by tokenizing all 25,000 prompts before
the pilot, not assumed. The task explicitly frames its numbers as "only a planning estimate";
this is the measured reality for this specific corpus.

## 8. Train/validation split

- **Level:** prompt (never token) — every token from a given prompt stays in the same split,
  verified by `validate_activations.py` check #10 (0 violations).
- **Method:** `sha256(f"{seed}:{prompt_id}")` → deterministic pseudo-random fraction in [0,1);
  `< val_fraction` → validation. Seed **1337**, `val_fraction` **0.1**.
- **Actual split:** 22,467 train (89.9%) / 2,533 validation (10.1%) — close to the 90/10 target;
  the small deviation from exactly 10.0% is expected from a per-prompt hash-based split rather
  than an exact stratified sample.
- Recorded in `metadata/prompt_split.jsonl` (one row per prompt_id, written once and reused on
  every subsequent resumed run so a restart can never silently use a different split).
- Both splits' activations are fully retained on disk (`activations/train/`,
  `activations/validation/`) — nothing was discarded.

## 9. Validation results

`validate_activations.py` (independent of the extraction code, does not re-run the model) —
**all 12 required checks passed, 0 errors:**

```json
{
  "n_normalized_prompts": 25000, "n_prompts_in_activations": 25000,
  "n_metadata_rows": 993496, "n_rows_on_disk": 993496, "n_shards": 10,
  "domains_present": ["DNA","EMOTIONS","GANDHI","HINDI","SCIENCE"],
  "source_corpus_unchanged": true, "errors": [], "warnings": [], "passed": true
}
```

Specifically verified: every normalized prompt accounted for (0 missing/extra); every retained
token's count matches an independent re-tokenization (0 mismatches); no gaps/duplicates in
token_position sequences; no duplicate `(prompt_id, token_position)` metadata rows; every shard's
sha256 matches its manifest entry; every shard's dimension (2048) and dtype (bfloat16) correct;
0 non-finite values anywhere; every metadata row resolves to a real shard; 0 train/validation
split violations; all 5 domains present; `git diff --stat HEAD -- normalized raw processed
MI_Dataset_validated` returns empty (source corpus untouched).

## 10. Activation statistics

Full detail: `reports/activation_statistics.json`.

- Tokenized prompt length: mean 39.7, median 31, P90 75, P95 114, max 266.
- Activation norm (sampled 200,000 rows): mean 550.5, std 2279.9, min 125.9, **max 16,509.4**.
  The large std/max relative to the mean is a known, expected property of LLM residual
  streams — a small number of "outlier"/"massive activation" dimensions carry disproportionately
  large magnitude (consistent with the per-dimension variance summary: max per-dim variance
  3,831,097 vs. mean per-dim variance 2,599.8, i.e. a handful of dimensions dominate). This is
  recorded as a raw characteristic of the extracted data, **not corrected, normalized, or
  whitened** — per the task's explicit instruction to preserve raw captured activations and treat
  any such transform as a separate, future, explicitly-justified step.
- Tokens per domain: DNA 162,170 / EMOTIONS 149,636 / GANDHI 180,923 / HINDI 294,837 / SCIENCE
  205,930 (full subdomain breakdown in `activation_statistics.json`). HINDI's share is largest
  despite having only 5,000 prompts like every other domain, because HINDI prompts are
  substantially longer on average than the other four domains' templated prompts.

## 11. Storage layout

```
experiments/activation_extraction_qwen3_1_7b/
  activations/train/shard_00000.safetensors ... shard_00008.safetensors   (9 shards, ~3.5 GB)
  activations/validation/shard_00000.safetensors                          (1 shard, ~0.38 GB)
  metadata/model_manifest.json, extraction_config.json, shard_manifest.json,
           prompt_token_metadata.jsonl (993,496 rows), prompt_split.jsonl (25,000 rows)
  reports/pilot_report.md, pilot_report_data.json, activation_validation.json,
          activation_statistics.json, FINAL_REPORT.md (this file)
  logs/extraction_run_<timestamp>.json
```

Format choice (safetensors over `.npy`): numpy has no native bfloat16 dtype (would force a lossy
fp16 downcast or a nonstandard uint16 bit-cast); safetensors supports bf16 natively, supports
fast memory-mapped/zero-copy reads via `safe_open` without loading a whole shard into RAM, and is
already the format this project uses for model weights. A prompt's tokens are never split across
two shards (a shard is flushed before it would overflow), so every shard is either fully valid or
does not exist yet — this is what makes the sha256-based resumability check simple and reliable.

## 12. Known limitations and unresolved issues

1. **Batched-vs-unbatched bf16 numerical drift** (§6): disk-stored activations are reproducible
   and internally consistent with each other (verified via repeated resumability runs reporting
   "0 remaining"), but are not bit-identical to what an unbatched forward pass of the same prompt
   would produce. Measured at <0.15% of elements differing by a small absolute amount. This is
   expected bf16 behavior, not a pipeline defect, but should be kept in mind if exact
   single-prompt reproduction outside the batched pipeline is ever needed.
2. **GQA note:** Qwen3-1.7B uses grouped-query attention (8 KV heads for 16 query heads) — this
   does not affect the residual-stream extraction itself (the hook captures the decoder block's
   output after attention+MLP have already been applied, independent of how attention computed
   internally), but is recorded here for completeness since it affects compute/memory behavior.
3. **Outlier/massive-activation dimensions** (§10) were observed but not investigated further —
   out of scope for this extraction-only milestone; flagged for whoever designs SAE
   preprocessing next, since un-normalized raw activations with a few huge-magnitude dimensions
   are a well-known challenge for dictionary learning.
4. **Train/validation ratio is 89.9%/10.1%, not exactly 90%/10%** — expected from a per-prompt
   hash-based split rather than an exact stratified draw; documented, not corrected.
5. **GPU was a laptop RTX 3050 (6.44 GB total VRAM)** shared with the OS/display at the time of
   the run (~240 MB already in use by the browser before this process started) — peak usage
   (4.11 GB) leaves meaningful headroom, but a machine with less free VRAM at run time might need
   a smaller `--batch-size` than the default of 8.

## 13. Source integrity confirmation

```
git diff --stat HEAD -- normalized raw processed MI_Dataset_validated
```
returned **empty**, confirmed by `validate_activations.py` (also re-confirmed manually
immediately before writing this report). All three prior milestones' outputs
(`experiments/corpus_validation_25k/`, `experiments/duplicate_language_embedding/`,
`experiments/semantic_topology_25k/`) were read-only inputs and were not modified.

## 14. Definition of Done checklist

- [x] Qwen3-1.7B verified and loaded onto the GPU (never CPU, never silently offloaded).
- [x] Selected middle-layer residual-stream hook documented (`model.model.layers[14]` output).
- [x] Representative pilot passed all checks (including an investigated, resolved anomaly).
- [x] Full 25,000-prompt corpus processed (0 incomplete, 0 failed).
- [x] Every non-padding token has a stored activation vector (993,496/993,496, cross-checked).
- [x] Activations saved in validated, resumable shards (safetensors, sha256-verified,
      resumability empirically re-tested — a second run correctly found "0 remaining").
- [x] Every vector traceable to its source prompt and token position
      (`prompt_token_metadata.jsonl`).
- [x] Train/validation separation recorded at the prompt level (`prompt_split.jsonl`).
- [x] Activation statistics and resource measurements reported (§10, §5, §7).
- [x] A separate validation script confirms storage integrity (`validate_activations.py`, 0 errors).
- [x] Source data and previous milestone outputs remain untouched (§13).
- [x] Reproduction/resume commands documented (top of this report).

**No criterion failed. No SAE training, feature generation, or neuron interpretation was
performed in this milestone.**

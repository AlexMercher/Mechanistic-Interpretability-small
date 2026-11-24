# Pilot Report — Qwen3-1.7B Residual-Stream Extraction

Pilot sample: 60 prompts (12 per domain, stratified by prompt length), isolated in
`reports/pilot_data/` (never mixed with the real `activations/`/`metadata/` the full run uses).
Full data: `reports/pilot_report_data.json`.

## Checks performed and results

| Check | Result |
|---|---|
| Model loads onto CUDA | ✅ RTX 3050 6GB Laptop GPU |
| Correct residual-stream hook captured | ✅ `model.model.layers[14]` output (verified against `output_hidden_states` indexing in `inspect_model.py`) |
| Activation shape correct | ✅ `(n_tokens, 2048)` per shard |
| Activation dimension matches hidden_size | ✅ 2048 == config.hidden_size |
| Every non-padding token has exactly one activation | ✅ 2330 meta rows == 2330 tokens from independent re-tokenization |
| Padding excluded | ✅ via attention_mask; row counts match retained-token counts exactly |
| No NaN/Inf | ✅ |
| Activation precision correct | ✅ bfloat16 throughout (loading + storage) |
| Batch size safe for VRAM | ✅ batch_size=8, peak 3.64 GB / 6.44 GB total |
| Peak VRAM measured | ✅ 3.64 GB |
| Throughput measured | ✅ ~1,319 tokens/sec |
| Disk-write throughput | ✅ safetensors shard writes negligible vs. GPU compute time |
| Full-run time/storage estimated | ✅ ~12.6 min, ~4.07 GB (see below) |
| Shard sha256 verified | ✅ |
| Independent reload-and-compare | ✅ (see finding below) |

## Finding: batched-vs-unbatched bf16 numerical drift (investigated, not a bug)

A strict `torch.allclose` round-trip check (disk-stored activation vs. a **fresh, independent,
unbatched** forward pass of the same prompt) initially failed. Investigated directly rather than
loosening the check blindly:

- Root cause: batched + right-padded SDPA attention takes a different internal numerical
  reduction path than an unbatched, unpadded forward pass. bfloat16 (~3 significant digits) lets
  this compound slightly across the model's 14 decoder layers down to the extraction point, and
  more at later token positions (which causally depend on more prior tokens).
- Measured on a 10-prompt sample (561,152 total elements): **0.14% of elements differ by >0.3
  absolute, 0.016% by >0.5, 0.005% by >1.0.** The single largest deviations (~64 absolute) occur
  on a small number of well-known very-large-magnitude "outlier" residual-stream dimensions
  (~12,000 in magnitude), where even the expected ~0.5% bf16 relative rounding is numerically
  large in absolute terms.
- **This is not a metadata, alignment, or hook-location bug** — row/token/shape/dtype alignment
  is independently and exactly verified elsewhere in this report (prompt counts, token counts via
  re-tokenization, shard sha256, dimension checks all pass with zero mismatches).
- Final round-trip check uses a measured, documented fractional tolerance (pass if <0.5% of a
  prompt's elements exceed absolute difference 1.0 vs. a fresh unbatched pass) rather than exact
  equality. All 5 round-trip sample prompts pass under this criterion (worst observed:
  0.0092% of elements, `DNA_001623`).
- **Implication for downstream SAE work (documented, not actioned here):** activations extracted
  via batched inference are reproducible and internally consistent with each other, but are not
  bit-identical to what an unbatched forward pass of the same prompt would produce. This is
  expected and typically immaterial for SAE training, but should be kept in mind if exact
  reproduction of a single prompt's activation is ever needed outside the batched pipeline.

## Throughput / full-run estimate

- Measured: 1,318.7 tokens/sec, peak VRAM 3.64 GB (of 6.44 GB total — comfortable headroom).
- Full corpus: 993,496 tokens (measured directly by tokenizing all 25,000 normalized prompts,
  **not** the task's 6.4M-token planning estimate — this corpus's prompts average ~40 tokens,
  not 256; see `reports/FINAL_REPORT.md` for the full discrepancy discussion).
- **Estimated full-run duration: ~12.6 minutes. Estimated storage: ~4.07 GB** (bfloat16,
  2048-dim, safetensors). Both far smaller than the task's worst-case planning numbers, which
  assumed a 256-token average.

## Decision

**Pilot passed all checks. Proceeding to full-corpus extraction** with the same configuration
(layer 14, bfloat16, batch_size=8, right-padding, length-bucketed batching, safetensors shards).

#!/usr/bin/env python
"""
Milestone 4, step 5 -- activation statistics report (does not re-run the model).

Computes: tokens per domain/subdomain, tokenized-length percentiles,
storage size, activation-norm mean/std/min/max, per-dimension variance
summary, and (from the extraction log) GPU memory/throughput numbers.

Usage:
    python generate_activation_report.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_common as ec  # noqa: E402


def main():
    exp_dir = ec.EXP_DIR
    metadata_dir = exp_dir / "metadata"
    records = ec.load_normalized_records()
    meta_rows = [json.loads(l) for l in (metadata_dir / "prompt_token_metadata.jsonl").open(encoding="utf-8")]
    manifest = json.loads((metadata_dir / "shard_manifest.json").read_text(encoding="utf-8"))

    import numpy as np
    import torch
    from safetensors.torch import load_file

    tokens_per_domain = Counter(m["domain"] for m in meta_rows)
    tokens_per_subdomain = Counter((m["domain"], m["subdomain"]) for m in meta_rows)

    tokens_per_prompt = defaultdict(int)
    for m in meta_rows:
        tokens_per_prompt[m["prompt_id"]] += 1
    lengths = np.array(list(tokens_per_prompt.values()))

    log_files = sorted((exp_dir / "logs").glob("extraction_run_*.json"))
    extraction_logs = [json.loads(p.read_text(encoding="utf-8")) for p in log_files]

    storage_bytes = 0
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            path = exp_dir / "activations" / split / Path(s["file"]).name
            storage_bytes += path.stat().st_size

    # activation norm statistics -- sample up to 200,000 rows across shards for tractable computation
    rng = np.random.default_rng(42)
    norms = []
    per_dim_values = []
    for split in ("train", "validation"):
        for s in manifest.get(split, {}).get("shards", []):
            path = exp_dir / "activations" / split / Path(s["file"]).name
            t = load_file(str(path))["activations"].float()
            n = t.shape[0]
            sample_n = min(n, 20000)
            idx = rng.choice(n, size=sample_n, replace=False)
            sample = t[idx]
            norms.append(torch.linalg.vector_norm(sample, dim=1).numpy())
            per_dim_values.append(sample.numpy())
    norms = np.concatenate(norms)
    per_dim_stack = np.concatenate(per_dim_values, axis=0)
    per_dim_variance = per_dim_stack.var(axis=0)

    split_counts = Counter(m["split"] for m in meta_rows)
    split_prompt_counts = Counter()
    seen_pids_by_split = defaultdict(set)
    for m in meta_rows:
        seen_pids_by_split[m["split"]].add(m["prompt_id"])

    stats = {
        "total_prompts_processed": len(set(m["prompt_id"] for m in meta_rows)),
        "total_retained_tokens": len(meta_rows),
        "tokens_per_domain": dict(tokens_per_domain),
        "tokens_per_subdomain": {f"{d}::{s}": n for (d, s), n in tokens_per_subdomain.items()},
        "tokenized_prompt_length": {
            "mean": float(lengths.mean()), "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)), "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()), "min": int(lengths.min()),
        },
        "activation_storage_bytes": storage_bytes,
        "activation_storage_gb": round(storage_bytes / 1e9, 4),
        "activation_norm_stats": {
            "mean": float(norms.mean()), "std": float(norms.std()),
            "min": float(norms.min()), "max": float(norms.max()),
            "n_sampled": int(norms.shape[0]),
        },
        "per_dimension_variance_summary": {
            "mean_variance_across_dims": float(per_dim_variance.mean()),
            "min_variance_across_dims": float(per_dim_variance.min()),
            "max_variance_across_dims": float(per_dim_variance.max()),
            "n_dims": int(per_dim_variance.shape[0]),
        },
        "split_token_counts": dict(split_counts),
        "split_prompt_counts": {k: len(v) for k, v in seen_pids_by_split.items()},
        "extraction_runs": [
            {"wall_clock_sec": log.get("wall_clock_sec"), "n_prompts": log.get("n_prompts"),
             "n_tokens_retained": log.get("n_tokens_retained"), "peak_vram_gb": log.get("peak_vram_gb"),
             "failed_prompts": len(log.get("failed_prompts", []))}
            for log in extraction_logs
        ],
        "failed_or_skipped_prompts": sum(len(log.get("failed_prompts", [])) for log in extraction_logs),
    }

    out_path = exp_dir / "reports" / "activation_statistics.json"
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\n[generate_activation_report] wrote {out_path}")


if __name__ == "__main__":
    main()

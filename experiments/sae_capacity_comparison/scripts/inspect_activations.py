"""
Milestone 5, step 1: inspect the frozen Milestone-4 activation shards (train
split only) and decide the SAE preprocessing transform.

Re-derives activation-norm and per-dimension variance statistics directly
from the train shards (not just trusting Milestone 4's activation_statistics.json,
though we cross-check against it), then computes the global scalar
normalization constant chosen in the project plan:

    mu    = mean(x) over all training token vectors              (shape [2048])
    scale = sqrt(input_dim / mean(||x - mu||^2))                  (scalar)

This is the standard SAE-literature convention (Anthropic/OpenAI-style:
scale activations so mean squared norm ~= hidden_dim) and is deliberately
NOT per-dimension whitening -- see reports/pilot_report.md for the full
rationale (per-dim standardization would inflate the near-constant
dimensions to the same scale as the "massive activation" outlier
dimensions, destroying the magnitude structure a dictionary should learn).

Streams shard-by-shard (never materializes the full 895,502x2048 array at
once) to stay within the 16GB system RAM budget. Train-only: validation
statistics are never used to compute mu/scale, per the task's explicit
instruction.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
ROOT = EXP_DIR.parent.parent
EXTRACTION_DIR = ROOT / "experiments" / "activation_extraction_qwen3_1_7b"
ACTIVATIONS_DIR = EXTRACTION_DIR / "activations"
SHARD_MANIFEST_PATH = EXTRACTION_DIR / "metadata" / "shard_manifest.json"
MILESTONE4_STATS_PATH = EXTRACTION_DIR / "reports" / "activation_statistics.json"
OUT_PATH = EXP_DIR / "preprocessing" / "activation_statistics.json"

INPUT_DIM = 2048


def load_train_shard_paths():
    manifest = json.loads(SHARD_MANIFEST_PATH.read_text(encoding="utf-8"))
    shards = manifest["train"]["shards"]
    return [(s["shard_id"], ACTIVATIONS_DIR / s["file"], s["n_rows"], s["sha256"]) for s in shards]


def main():
    shard_paths = load_train_shard_paths()
    print(f"[inspect_activations] {len(shard_paths)} train shards found")

    n_total = 0
    sum_vec = torch.zeros(INPUT_DIM, dtype=torch.float64)
    sumsq_vec = torch.zeros(INPUT_DIM, dtype=torch.float64)
    norm_sum = 0.0
    norm_sumsq = 0.0
    norm_min = float("inf")
    norm_max = float("-inf")
    sampled_norms = []

    for shard_id, path, n_rows_expected, sha256_expected in shard_paths:
        import hashlib
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != sha256_expected:
            raise RuntimeError(f"shard {path} sha256 mismatch vs manifest -- refusing to use unverified data")

        with safe_open(str(path), framework="pt", device="cpu") as f:
            x = f.get_tensor("activations").to(torch.float32)
        assert x.shape[1] == INPUT_DIM, f"unexpected hidden dim in {path}: {x.shape}"
        assert x.shape[0] == n_rows_expected, f"row count mismatch in {path}"

        n_total += x.shape[0]
        sum_vec += x.sum(dim=0).to(torch.float64)
        sumsq_vec += (x.double() ** 2).sum(dim=0)

        norms = x.norm(dim=1)
        norm_sum += norms.sum().item()
        norm_sumsq += (norms.double() ** 2).sum().item()
        norm_min = min(norm_min, norms.min().item())
        norm_max = max(norm_max, norms.max().item())
        if len(sampled_norms) < 200_000:
            take = min(20_000, norms.shape[0])
            idx = torch.randperm(norms.shape[0])[:take]
            sampled_norms.extend(norms[idx].tolist())

        print(f"[inspect_activations] shard {shard_id}: {x.shape[0]} rows processed, running n={n_total}")
        del x

    mu = (sum_vec / n_total).to(torch.float32)
    per_dim_var = (sumsq_vec / n_total - (sum_vec / n_total) ** 2).to(torch.float32)
    mean_sq_norm_centered = float(per_dim_var.sum())  # E[||x-mu||^2] = sum of per-dim variances
    scale = (INPUT_DIM / mean_sq_norm_centered) ** 0.5

    norm_mean = norm_sum / n_total
    norm_var = norm_sumsq / n_total - norm_mean ** 2
    norm_std = norm_var ** 0.5

    stats = {
        "n_train_tokens": n_total,
        "input_dim": INPUT_DIM,
        "raw_activation_norm_stats": {
            "mean": norm_mean, "std": norm_std, "min": norm_min, "max": norm_max,
        },
        "per_dimension_variance_summary": {
            "mean_variance_across_dims": float(per_dim_var.mean()),
            "min_variance_across_dims": float(per_dim_var.min()),
            "max_variance_across_dims": float(per_dim_var.max()),
        },
        "preprocessing_transform": {
            "method": "global_scalar_normalization_with_centering",
            "mu_norm": float(mu.norm()),
            "scale": scale,
            "mean_sq_norm_after_transform_check": mean_sq_norm_centered * (scale ** 2),
            "rationale": (
                "Per-dimension whitening was rejected: it would inflate the "
                "lowest-variance dims (min per-dim variance observed here) to the "
                "same scale as 'massive activation' outlier dims (max per-dim "
                "variance observed here), destroying the magnitude structure a "
                "dictionary is supposed to discover. A single global scalar scale "
                "(after centering) preserves relative direction/magnitude structure "
                "while putting ||x|| on a scale where a chosen sparsity coefficient "
                "and reconstruction loss are numerically comparable across widths."
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    mu_path = EXP_DIR / "preprocessing" / "mu_vector.pt"
    torch.save(mu, mu_path)

    print(f"[inspect_activations] n_train_tokens={n_total}")
    print(f"[inspect_activations] scale={scale:.6f}, mu_norm={float(mu.norm()):.4f}")
    print(f"[inspect_activations] wrote {OUT_PATH} and {mu_path}")

    if MILESTONE4_STATS_PATH.exists():
        m4 = json.loads(MILESTONE4_STATS_PATH.read_text(encoding="utf-8"))
        m4_train_tokens = m4.get("split_token_counts", {}).get("train")
        if m4_train_tokens != n_total:
            print(f"[inspect_activations] WARNING: train token count mismatch vs Milestone 4 "
                  f"report ({n_total} vs {m4_train_tokens})", file=sys.stderr)
        else:
            print("[inspect_activations] cross-check vs Milestone 4 report: train token count matches.")


if __name__ == "__main__":
    main()

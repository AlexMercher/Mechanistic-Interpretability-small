"""
Milestone 5, evaluation B: sparsity metrics on the held-out validation split.

"Active feature" convention (stated once, used consistently across every
evaluation script and width): a feature is active for a given token if its
post-ReLU activation z_i > 0. L0 for a token is the count of active features;
L1 is sum(|z_i|) = sum(z_i) since z >= 0 after ReLU.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="best_val")
    args = ap.parse_args()

    device = sc.get_device_no_fallback()
    model, config, ckpt = sc.load_model_for_eval(args.width, args.seed, args.checkpoint, device)
    preprocessor = sc.Preprocessor.load(EXP_DIR / "preprocessing")

    val_x = sc.load_split_activations("validation")
    n = val_x.shape[0]
    domain_labels = sc.load_domain_labels("validation", n)
    domain_ids, domain_list = sc.build_domain_ids(domain_labels)

    raw = sc.run_validation_pass(model, val_x, domain_ids, len(domain_list), preprocessor, device)

    l0 = raw["per_row_l0"].astype(np.float64)
    l1 = raw["per_row_l1"]
    activation_frequency = raw["feature_active_count"] / n  # fraction of val tokens each feature fires on

    out = {
        "experiment_id": config["experiment_id"],
        "dictionary_width": args.width,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "n_validation_rows": n,
        "active_feature_convention": "z_i > 0 (post-ReLU)",
        "mean_active_features_per_token": float(l0.mean()),
        "median_active_features_per_token": float(np.median(l0)),
        "p90_active_features_per_token": float(np.percentile(l0, 90)),
        "mean_l0": float(l0.mean()),
        "mean_l1_activation_magnitude": float(l1.mean()),
        "l0_percentiles": {str(p): float(np.percentile(l0, p)) for p in [10, 25, 50, 75, 90, 95, 99]},
        "feature_activation_frequency_summary": {
            "mean": float(activation_frequency.mean()),
            "median": float(np.median(activation_frequency)),
            "p10": float(np.percentile(activation_frequency, 10)),
            "p90": float(np.percentile(activation_frequency, 90)),
            "max": float(activation_frequency.max()),
            "min": float(activation_frequency.min()),
        },
        "feature_activation_frequency_histogram": {
            "bin_edges": list(np.histogram(activation_frequency, bins=20, range=(0, max(activation_frequency.max(), 1e-9)))[1].tolist()),
            "counts": list(np.histogram(activation_frequency, bins=20, range=(0, max(activation_frequency.max(), 1e-9)))[0].tolist()),
        },
    }

    out_path = EXP_DIR / "evaluations" / "sparsity" / f"sae_{args.width}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[evaluate_sparsity] wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if "histogram" not in k}, indent=2))


if __name__ == "__main__":
    main()

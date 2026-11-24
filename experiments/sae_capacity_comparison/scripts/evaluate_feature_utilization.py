"""
Milestone 5, evaluation C: feature utilization / dead-feature statistics on
the held-out validation split.

Convention: a feature is "dead" if it never fires (z_i > 0) on any of the
97,994 validation vectors. "Rare" = fires on <0.01% of validation tokens but
at least once; "frequent" = fires on >10% of validation tokens. These
thresholds are fixed and applied identically to all three widths.

Note: this is a validation-set-based dead-feature count. The training logs
(logs/sae_<width>_seed<k>.jsonl) separately track a train-time rolling dead
count (no activation in the last 2000 training steps); the two can differ
and both are reported in the final comparison, not conflated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent

RARE_THRESHOLD_FRAC = 0.0001
FREQUENT_THRESHOLD_FRAC = 0.10


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

    width = args.width
    active_count = raw["feature_active_count"]
    freq = active_count / n

    dead_mask = active_count == 0
    rare_mask = (~dead_mask) & (freq < RARE_THRESHOLD_FRAC)
    frequent_mask = freq > FREQUENT_THRESHOLD_FRAC
    mid_mask = (~dead_mask) & (~rare_mask) & (~frequent_mask)

    n_dead = int(dead_mask.sum())
    n_rare = int(rare_mask.sum())
    n_frequent = int(frequent_mask.sum())
    n_active_at_least_once = width - n_dead

    out = {
        "experiment_id": config["experiment_id"],
        "dictionary_width": width,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "n_validation_rows": n,
        "thresholds": {"rare_below_frac": RARE_THRESHOLD_FRAC, "frequent_above_frac": FREQUENT_THRESHOLD_FRAC},
        "total_dictionary_features": width,
        "features_active_at_least_once": n_active_at_least_once,
        "features_active_at_least_once_pct": 100.0 * n_active_at_least_once / width,
        "dead_features": n_dead,
        "dead_features_pct": 100.0 * n_dead / width,
        "rare_features": n_rare,
        "rare_features_pct": 100.0 * n_rare / width,
        "frequent_features": n_frequent,
        "frequent_features_pct": 100.0 * n_frequent / width,
        "mid_frequency_features": int(mid_mask.sum()),
        "mid_frequency_features_pct": 100.0 * int(mid_mask.sum()) / width,
        "activation_frequency_percentiles": {
            str(p): float(np.percentile(freq, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        "activation_count_distribution_summary": {
            "mean": float(active_count.mean()), "max": int(active_count.max()), "min": int(active_count.min()),
        },
    }

    out_path = EXP_DIR / "evaluations" / "feature_utilization" / f"sae_{width}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[evaluate_feature_utilization] wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

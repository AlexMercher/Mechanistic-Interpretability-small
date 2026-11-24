"""
Milestone 5, evaluation A: reconstruction quality on the held-out validation
split, for the seed-0 checkpoint of each width. Reads only validation-split
activation rows (asserted below), never train.

Explained variance definition (stated explicitly per the task's requirement
to be unambiguous about the denominator): for a set of vectors x and their
reconstructions x_hat,

    EV = 1 - sum(||x - x_hat||^2) / sum(||x||^2)

computed in the SAE's own input space (the preprocessed/transformed space:
centered by train mu, scaled by the frozen scalar). This ratio is invariant
to the affine (center+scale) transform as long as it's applied identically
to x and x_hat (which it is here), so it equals the raw-activation-space
explained variance too -- but we report it as computed in the SAE's native
input space, per sae_common.run_validation_pass, and say so explicitly here
rather than silently mixing spaces.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
    assert all(lab is not None for lab in domain_labels), "every validation row must have a domain label"
    domain_ids, domain_list = sc.build_domain_ids(domain_labels)

    raw = sc.run_validation_pass(model, val_x, domain_ids, len(domain_list), preprocessor, device)

    overall_mse_per_vector = float(raw["per_row_mse"].mean())
    overall_ev = 1.0 - raw["sum_sq_error"] / raw["sum_sq_x"]
    per_row_mse = raw["per_row_mse"]
    percentiles = {p: float(np.percentile(per_row_mse, p)) for p in [10, 25, 50, 75, 90, 95, 99]}

    per_domain = {}
    for i, dom in enumerate(domain_list):
        if raw["domain_count"][i] == 0:
            continue
        per_domain[dom] = {
            "n_rows": int(raw["domain_count"][i]),
            "mse_per_vector": float(raw["domain_sq_error"][i] / raw["domain_count"][i]),
            "explained_variance": float(1.0 - raw["domain_sq_error"][i] / raw["domain_sq_x"][i]),
        }

    out = {
        "experiment_id": config["experiment_id"],
        "dictionary_width": args.width,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "n_validation_rows": n,
        "evaluation_space": "preprocessed (centered+scaled) SAE input space -- numerically equal to raw-space ratio, see module docstring",
        "mean_squared_reconstruction_error_per_vector": overall_mse_per_vector,
        "explained_variance_overall": overall_ev,
        "explained_variance_denominator": "sum(||x||^2) over all validation vectors, in SAE input space",
        "reconstruction_error_percentiles_per_vector": percentiles,
        "explained_variance_per_domain": per_domain,
    }

    out_path = EXP_DIR / "evaluations" / "reconstruction" / f"sae_{args.width}_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[evaluate_reconstruction] wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "explained_variance_per_domain"}, indent=2))


if __name__ == "__main__":
    main()

"""
Milestone 6, Phase 2 step 4 (plan B.9): full-dictionary Confirmation-only coverage
statistic.

Candidate selection (select_candidates.py) runs entirely on the Discovery-subset
inventory, so a feature that never fires in Discovery but does fire in Confirmation
cannot enter the rare-but-active stratum, or any stratum. This is a real coverage gap,
documented with a measured count rather than silently worked around (working around it
would reintroduce the exact Discovery/Confirmation leakage the split was built to
prevent).

One additional full-dictionary forward pass over the Confirmation subset (all 8192
features) -- same cost profile as the Discovery inventory pass.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
WIDTH = 8192
SEED = 0
BATCH_SIZE = 4096


def main():
    assignment = {}
    with (SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assignment[row["prompt_id"]] = row["subset"]

    rows = []
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "validation" and assignment.get(row["prompt_id"]) == "confirmation":
                rows.append(row["activation_row_index"])
    row_idx = np.array(sorted(rows))
    print(f"[coverage] {len(row_idx)} Confirmation tokens")

    device = sf.get_device_no_fallback()
    model, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    full_val = sf.load_split_activations("validation")
    x_all = full_val[row_idx.copy()]
    del full_val

    discovery_inventory = pd.read_json(
        SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl", lines=True)
    discovery_dead = set(discovery_inventory[discovery_inventory["is_dead_in_discovery"]]["feature_idx"])
    print(f"[coverage] {len(discovery_dead)} features dead in Discovery")

    fires_in_confirmation = np.zeros(WIDTH, dtype=bool)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(row_idx), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(row_idx))
            raw = x_all[start:end].to(device, torch.float32)
            x = preprocessor.transform(raw)
            z = model.encode(x).cpu().numpy()
            fires_in_confirmation |= (z > 0).any(axis=0)

    confirmation_only = [f for f in discovery_dead if fires_in_confirmation[f]]
    dead_in_both = [f for f in discovery_dead if not fires_in_confirmation[f]]

    result = {
        "n_discovery_dead_features": len(discovery_dead),
        "n_confirmation_only_features": len(confirmation_only),
        "n_dead_in_both_subsets": len(dead_in_both),
        "confirmation_only_feature_indices": sorted(confirmation_only),
        "note": ("Rare-feature discovery in this milestone is conditional on firing in the "
                 "Discovery subset. These confirmation_only_feature_indices fired zero times "
                 "in Discovery and therefore could not enter the rare_but_active (or any) "
                 "candidate stratum -- the 121-candidate cohort is a stratified sample of "
                 "Discovery-visible features, not an exhaustive census of all rare features "
                 "in the SAE-8192 seed-0 dictionary."),
    }
    out_path = SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "confirmation_only_feature_coverage.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "confirmation_only_feature_indices"}, indent=2))
    print(f"[coverage] wrote {out_path}")


if __name__ == "__main__":
    main()

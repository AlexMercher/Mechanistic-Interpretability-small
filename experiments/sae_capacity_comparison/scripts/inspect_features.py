"""
Milestone 5, evaluation E (automated half): per-feature statistics for a
representative sample of features per width -- top-activating validation
tokens/prompts, domain distribution of firing tokens. Automated statistics
ONLY; no semantic label is generated here. A separate human/teacher-reviewed
pass (see reports/pilot_report.md-style write-up in FINAL_REPORT.md section
on interpretability) adds candidate interpretations into a reviewed file,
explicitly kept apart from these raw stats.

Feature selection: NOT just the top-activating features (that would bias
toward frequent/generic features) -- 10 highest-frequency + 10
median-frequency + 10 lowest-nonzero-frequency ("rare but alive") features,
so the sample spans the dictionary's activity spectrum.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_common as sc  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent.parent
ROOT = EXP_DIR.parent.parent
NORMALIZED_DIR = ROOT / "normalized"

N_TOP_TOKENS = 20
N_FEATURES_PER_TIER = 10


def load_normalized_text():
    texts = {}
    for p in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts[rec["prompt_id"]] = rec["prompt"]
    return texts


def select_representative_features(freq: np.ndarray):
    alive = np.where(freq > 0)[0]
    if alive.size == 0:
        return []
    order = alive[np.argsort(-freq[alive])]
    n = len(order)
    top = order[:N_FEATURES_PER_TIER]
    mid_start = max(0, n // 2 - N_FEATURES_PER_TIER // 2)
    mid = order[mid_start: mid_start + N_FEATURES_PER_TIER]
    rare = order[-N_FEATURES_PER_TIER:]
    chosen = sorted(set(top.tolist()) | set(mid.tolist()) | set(rare.tolist()))
    return chosen


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
    freq = raw["feature_active_count"] / n

    chosen_features = select_representative_features(freq)
    print(f"[inspect_features] selected {len(chosen_features)} features for width {args.width}")

    texts = load_normalized_text()
    feat_idx_t = torch.tensor(chosen_features, dtype=torch.long, device=device)
    top_tokens = {f: [] for f in chosen_features}
    domain_fire_counts = {f: defaultdict(int) for f in chosen_features}

    batch_size = 4096
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            raw_batch = val_x[start:end].to(device, torch.float32)
            x = preprocessor.transform(raw_batch)
            z = model.encode(x)
            z_sel = z[:, feat_idx_t].cpu().numpy()
            for ci, f in enumerate(chosen_features):
                col = z_sel[:, ci]
                nz = np.nonzero(col)[0]
                for i in nz:
                    row_idx = start + int(i)
                    lab = domain_labels[row_idx]
                    if lab is not None:
                        domain_fire_counts[f][lab[0]] += 1
                    top_tokens[f].append((float(col[i]), row_idx))

    feature_reports = {}
    for f in chosen_features:
        lst = sorted(top_tokens[f], key=lambda t: -t[0])[:N_TOP_TOKENS]
        examples = []
        for val, row_idx in lst:
            lab = domain_labels[row_idx]
            if lab is None:
                continue
            domain, subdomain, prompt_id, token_pos = lab
            examples.append({
                "activation": val, "prompt_id": prompt_id, "domain": domain,
                "subdomain": subdomain, "token_position": token_pos,
                "prompt_text": texts.get(prompt_id, "")[:400],
            })
        feature_reports[str(f)] = {
            "activation_frequency": float(freq[f]),
            "n_firing_tokens_in_validation": int(raw["feature_active_count"][f]),
            "domain_distribution_of_firing_tokens": dict(domain_fire_counts[f]),
            "top_activating_examples": examples,
        }

    out = {
        "dictionary_width": args.width, "seed": args.seed, "checkpoint": args.checkpoint,
        "selection_method": "10 highest-frequency + 10 median-frequency + 10 lowest-nonzero-frequency alive features",
        "n_features_inspected": len(chosen_features),
        "features": feature_reports,
        "note": "Automated statistics only. No semantic label is auto-generated here -- "
                "see sae_<width>_reviewed.json for the separate human/teacher-reviewed "
                "interpretation pass.",
    }

    out_path = EXP_DIR / "evaluations" / "interpretability" / f"sae_{args.width}_feature_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[inspect_features] wrote {out_path}")


if __name__ == "__main__":
    main()

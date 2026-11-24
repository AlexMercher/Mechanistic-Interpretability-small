"""
Milestone 6, Phase 3 step 3 (plan B.2/D.3.3): full-dictionary cross-CAPACITY matching
against SAE-4096 and SAE-16384 (seed 0), for candidates whose primary review reached
>= medium confidence. Same methodology as cross_seed_reproducibility.py (full-target
decoder search, full-SOURCE reverse check for split/merge, Confirmation-subset-only
activation-pattern/top-prompt-overlap validation) -- just a different target
dictionary width instead of a different seed.

Restricted to >=medium confidence candidates per the plan (cheap either way
computationally; the restriction keeps the deliverable focused on candidates worth a
cross-capacity claim).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
SOURCE_WIDTH = 8192
SOURCE_SEED = 0
TARGET_WIDTHS = [4096, 16384]
TARGET_SEED = 0
COSINE_THRESHOLD = 0.3
MARGIN_THRESHOLD = 0.1
MIN_TOKENS = 5
MIN_PROMPTS = 3
ACTIVATION_PATTERN_THRESHOLD = 0.3
TOP_PROMPT_K = 20


def load_confirmation_tensor():
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
                rows.append((row["activation_row_index"], row["prompt_id"]))
    meta = pd.DataFrame(rows, columns=["activation_row_index", "prompt_id"]).sort_values("activation_row_index").reset_index(drop=True)
    full_val = sf.load_split_activations("validation")
    x = full_val[meta["activation_row_index"].to_numpy().copy()]
    del full_val
    return meta, x


def classify_matches(decoder_sim, cosine_thr, margin_thr):
    results = []
    for row in decoder_sim:
        above = np.where(row > cosine_thr)[0]
        if len(above) == 0:
            results.append({"status": "unmatched", "best_target": None, "above_threshold_targets": []})
            continue
        order = above[np.argsort(-row[above])]
        top1_val = row[order[0]]
        if len(order) == 1:
            results.append({"status": "single_clear_match", "best_target": int(order[0]),
                             "above_threshold_targets": [int(i) for i in order]})
            continue
        top2_val = row[order[1]]
        status = "single_clear_match" if (top1_val - top2_val) >= margin_thr else "ambiguous_match"
        results.append({"status": status, "best_target": int(order[0]),
                         "above_threshold_targets": [int(i) for i in order]})
    return results


@torch.no_grad()
def reverse_check(target_feature_idxs, target_Wd, source_Wd_full, cosine_thr, margin_thr):
    if not target_feature_idxs:
        return {}
    target_cols = target_Wd[:, target_feature_idxs]
    sim = (source_Wd_full.t() @ target_cols).cpu().numpy()
    out = {}
    for i, t in enumerate(target_feature_idxs):
        col = sim[:, i]
        above = np.where(col > cosine_thr)[0]
        if len(above) == 0:
            out[t] = {"best_source": None, "status": "unmatched"}
            continue
        order = above[np.argsort(-col[above])]
        if len(order) == 1 or (col[order[0]] - col[order[1]]) >= margin_thr:
            out[t] = {"best_source": int(order[0]), "status": "single_clear_match"}
        else:
            out[t] = {"best_source": int(order[0]), "status": "ambiguous_match"}
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-confidence", choices=["high", "medium", "low", "unvalidated"], default="medium")
    args = ap.parse_args()

    confidence_rank = {"unvalidated": 0, "low": 1, "medium": 2, "high": 3}
    reviews = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]
    eligible = {r["feature_idx"] for r in reviews if confidence_rank[r["confidence"]] >= confidence_rank[args.min_confidence]}
    cand_idxs = sorted(eligible)
    print(f"[cross_capacity] {len(cand_idxs)}/{len(reviews)} candidates at >= {args.min_confidence} confidence")

    device = sf.get_device_no_fallback()
    source_model, _, _ = sf.load_model_for_eval(SOURCE_WIDTH, SOURCE_SEED, "best_val", device)
    source_Wd_full = source_model.W_d.detach()
    source_cand_Wd = source_Wd_full[:, cand_idxs]

    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    conf_meta, conf_x_raw = load_confirmation_tensor()
    conf_x = preprocessor.transform(conf_x_raw.to(device, torch.float32))
    z_source_conf = source_model.encode(conf_x)[:, cand_idxs].cpu().numpy()

    results = {feat: {} for feat in cand_idxs}
    for width in TARGET_WIDTHS:
        target_model, _, _ = sf.load_model_for_eval(width, TARGET_SEED, "best_val", device)
        target_Wd = target_model.W_d.detach()

        decoder_sim = (source_cand_Wd.t() @ target_Wd).cpu().numpy()
        matches = classify_matches(decoder_sim, COSINE_THRESHOLD, MARGIN_THRESHOLD)

        implicated = sorted({m["best_target"] for m in matches if m["best_target"] is not None} |
                             {t for m in matches for t in m["above_threshold_targets"]})
        reverse = reverse_check(implicated, target_Wd, source_Wd_full, COSINE_THRESHOLD, MARGIN_THRESHOLD)

        z_target_conf = target_model.encode(conf_x).cpu().numpy()

        for i, feat in enumerate(cand_idxs):
            m = matches[i]
            entry = {"decoder_match_status": m["status"], "best_target": m["best_target"],
                     "above_threshold_targets": m["above_threshold_targets"]}

            src_col = z_source_conf[:, i]
            n_src_firing = int((src_col > 0).sum())
            n_src_firing_prompts = len(set(conf_meta["prompt_id"].to_numpy()[src_col > 0]))
            insufficient = n_src_firing < MIN_TOKENS or n_src_firing_prompts < MIN_PROMPTS
            entry["confirmation_source_firing_tokens"] = n_src_firing
            entry["confirmation_source_firing_prompts"] = n_src_firing_prompts

            if insufficient:
                entry["classification"] = "insufficient_evidence"
            elif m["best_target"] is not None:
                rev = reverse.get(m["best_target"], {})
                reverse_confirms = rev.get("best_source") == feat
                entry["reverse_check"] = {"best_source_from_target": rev.get("best_source"),
                                           "reverse_confirms_this_candidate": reverse_confirms}

                tgt_col = z_target_conf[:, m["best_target"]]
                src_norm = src_col / (np.linalg.norm(src_col) + 1e-8)
                tgt_norm = tgt_col / (np.linalg.norm(tgt_col) + 1e-8)
                act_sim = float(np.dot(src_norm, tgt_norm))
                entry["confirmation_activation_pattern_cosine"] = act_sim

                src_top = set(conf_meta["prompt_id"].to_numpy()[np.argsort(-src_col)[:TOP_PROMPT_K]])
                tgt_top = set(conf_meta["prompt_id"].to_numpy()[np.argsort(-tgt_col)[:TOP_PROMPT_K]])
                entry["confirmation_top_prompt_jaccard"] = len(src_top & tgt_top) / max(len(src_top | tgt_top), 1)

                if not reverse_confirms:
                    entry["classification"] = "possible_split" if rev.get("best_source") is None else "numerically_matched_semantically_inconsistent"
                elif act_sim >= ACTIVATION_PATTERN_THRESHOLD:
                    entry["classification"] = "reproduced_strong"
                else:
                    entry["classification"] = "numerically_matched_semantically_inconsistent"
            else:
                entry["classification"] = "not_reproduced"

            if m["best_target"] is not None:
                merge_partners = [cand_idxs[j] for j, mm in enumerate(matches)
                                   if mm["best_target"] == m["best_target"] and cand_idxs[j] != feat]
                if merge_partners:
                    entry["possible_merge_partners"] = merge_partners
                    if entry["classification"] == "reproduced_strong":
                        entry["classification"] = "possible_merge"

            results[feat][f"width_{width}"] = entry

    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / "cross_capacity_check.json"
    out_path.write_text(json.dumps({"min_confidence": args.min_confidence, "eligible_candidates": cand_idxs,
                                     "results": results}, indent=2), encoding="utf-8")
    print(f"[cross_capacity] wrote {out_path}")
    from collections import Counter
    for width in TARGET_WIDTHS:
        c = Counter(results[feat][f"width_{width}"]["classification"] for feat in cand_idxs)
        print(f"width_{width}: {dict(c)}")


if __name__ == "__main__":
    main()

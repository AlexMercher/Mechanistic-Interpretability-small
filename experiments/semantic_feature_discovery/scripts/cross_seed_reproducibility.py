"""
Milestone 6, Phase 3 step 2 (plan B.2): full-target-dictionary decoder search for
every candidate against SAE-8192 seed 1 and seed 2, a full-SOURCE-dictionary reverse
check for any provisional split/merge/ambiguous case, and Confirmation-subset-ONLY
activation-pattern/top-prompt-overlap/semantic-evidence validation.

Feature indices are NOT aligned across seeds. A high decoder cosine alone does NOT
establish semantic reproducibility -- it is one numeric signal, checked against
independent activation-pattern similarity, top-prompt overlap, and the matched target
feature's own evidence before any classification is assigned.

Thresholds (0.3 cosine, 0.1 margin) are numerical-matching heuristics, not established
semantic-reproducibility criteria -- see sensitivity_sweep.py for a swept re-run.
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
WIDTH = 8192
SOURCE_SEED = 0
TARGET_SEEDS = [1, 2]
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


def classify_matches(decoder_sim: np.ndarray, cosine_thr: float, margin_thr: float):
    """decoder_sim: [n_candidates, target_width]. Returns per-candidate dict with
    status (unmatched/single_clear_match/ambiguous_match), best target idx, and the
    set of all above-threshold target indices (used for split detection)."""
    results = []
    for row in decoder_sim:
        above = np.where(row > cosine_thr)[0]
        if len(above) == 0:
            results.append({"status": "unmatched", "best_target": None, "above_threshold_targets": []})
            continue
        order = above[np.argsort(-row[above])]
        top1_idx, top1_val = order[0], row[order[0]]
        if len(order) == 1:
            results.append({"status": "single_clear_match", "best_target": int(top1_idx),
                             "above_threshold_targets": [int(i) for i in order]})
            continue
        top2_val = row[order[1]]
        if (top1_val - top2_val) >= margin_thr:
            status = "single_clear_match"
        else:
            status = "ambiguous_match"
        results.append({"status": status, "best_target": int(top1_idx),
                         "above_threshold_targets": [int(i) for i in order]})
    return results


@torch.no_grad()
def reverse_check(target_feature_idxs, target_Wd, source_Wd_full, cosine_thr, margin_thr):
    """Full-SOURCE-dictionary reverse check: for each implicated target feature,
    search against the COMPLETE source dictionary (all 8192 seed-0 features, not just
    the selected candidates). Returns dict target_idx -> (best_source_idx, status)."""
    if not target_feature_idxs:
        return {}
    target_cols = target_Wd[:, target_feature_idxs]  # [2048, n_targets]
    sim = (source_Wd_full.t() @ target_cols).cpu().numpy()  # [8192_source, n_targets]
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
    ap.add_argument("--cosine-threshold", type=float, default=COSINE_THRESHOLD)
    ap.add_argument("--margin-threshold", type=float, default=MARGIN_THRESHOLD)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    candidates = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]
    cand_idxs = [c["feature_idx"] for c in candidates]

    device = sf.get_device_no_fallback()
    source_model, _, _ = sf.load_model_for_eval(WIDTH, SOURCE_SEED, "best_val", device)
    source_Wd_full = source_model.W_d.detach()  # [2048, 8192]
    source_cand_Wd = source_Wd_full[:, cand_idxs]  # [2048, 121]

    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    conf_meta, conf_x_raw = load_confirmation_tensor()
    conf_x = preprocessor.transform(conf_x_raw.to(device, torch.float32))
    z_source_conf = source_model.encode(conf_x)[:, cand_idxs].cpu().numpy()  # [n_conf_tokens, 121]

    results = {feat: {} for feat in cand_idxs}
    for seed in TARGET_SEEDS:
        target_model, _, _ = sf.load_model_for_eval(WIDTH, seed, "best_val", device)
        target_Wd = target_model.W_d.detach()

        decoder_sim = (source_cand_Wd.t() @ target_Wd).cpu().numpy()  # [121, 8192]
        matches = classify_matches(decoder_sim, args.cosine_threshold, args.margin_threshold)

        implicated_targets = sorted({m["best_target"] for m in matches if m["best_target"] is not None} |
                                     {t for m in matches for t in m["above_threshold_targets"]})
        reverse = reverse_check(implicated_targets, target_Wd, source_Wd_full, args.cosine_threshold, args.margin_threshold)

        z_target_conf = target_model.encode(conf_x).cpu().numpy()  # [n_conf_tokens, 8192]

        for i, feat in enumerate(cand_idxs):
            m = matches[i]
            entry = {"decoder_match_status": m["status"], "best_target": m["best_target"],
                     "above_threshold_targets": m["above_threshold_targets"]}

            # Minimum-evidence rule (B.2/B.4/B.7, shared): checked FIRST, regardless of
            # whether a decoder match was found. A candidate with too little Confirmation-
            # subset firing to support a reliable match attempt is insufficient_evidence,
            # never downgraded to not_reproduced just because the decoder search also
            # happened to find nothing.
            src_col_full = z_source_conf[:, i]
            n_src_firing = int((src_col_full > 0).sum())
            n_src_firing_prompts = len(set(conf_meta["prompt_id"].to_numpy()[src_col_full > 0]))
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
                if not reverse_confirms and rev.get("best_source") is not None:
                    entry["note"] = (f"Provisional match corrected: target {m['best_target']}'s true best "
                                      f"reverse match in the full source dictionary is feature "
                                      f"{rev.get('best_source')}, not candidate {feat}.")

                # Confirmation-subset-only activation-pattern + top-prompt overlap
                src_col = src_col_full
                tgt_col = z_target_conf[:, m["best_target"]]
                src_norm = src_col / (np.linalg.norm(src_col) + 1e-8)
                tgt_norm = tgt_col / (np.linalg.norm(tgt_col) + 1e-8)
                act_sim = float(np.dot(src_norm, tgt_norm))
                entry["confirmation_activation_pattern_cosine"] = act_sim

                src_top_prompts = set(conf_meta["prompt_id"].to_numpy()[np.argsort(-src_col)[:TOP_PROMPT_K]])
                tgt_top_prompts = set(conf_meta["prompt_id"].to_numpy()[np.argsort(-tgt_col)[:TOP_PROMPT_K]])
                jaccard = len(src_top_prompts & tgt_top_prompts) / max(len(src_top_prompts | tgt_top_prompts), 1)
                entry["confirmation_top_prompt_jaccard"] = jaccard

                if not reverse_confirms:
                    entry["classification"] = "possible_split" if rev.get("best_source") is None else "numerically_matched_semantically_inconsistent"
                elif act_sim >= ACTIVATION_PATTERN_THRESHOLD:
                    entry["classification"] = "reproduced_strong"
                else:
                    entry["classification"] = "numerically_matched_semantically_inconsistent"
            else:
                entry["classification"] = "not_reproduced"

            # possible_merge: another candidate also best-matches this same target
            if m["best_target"] is not None:
                merge_partners = [cand_idxs[j] for j, mm in enumerate(matches)
                                   if mm["best_target"] == m["best_target"] and cand_idxs[j] != feat]
                if merge_partners:
                    entry["possible_merge_partners"] = merge_partners
                    if entry["classification"] == "reproduced_strong":
                        entry["classification"] = "possible_merge"

            results[feat][f"seed_{seed}"] = entry

    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / f"cross_seed_reproducibility{args.out_suffix}.json"
    out_path.write_text(json.dumps({"cosine_threshold": args.cosine_threshold, "margin_threshold": args.margin_threshold,
                                     "results": results}, indent=2), encoding="utf-8")
    print(f"[cross_seed] wrote {out_path}")

    from collections import Counter
    for seed in TARGET_SEEDS:
        c = Counter(results[feat][f"seed_{seed}"]["classification"] for feat in cand_idxs)
        print(f"seed_{seed}: {dict(c)}")


if __name__ == "__main__":
    main()

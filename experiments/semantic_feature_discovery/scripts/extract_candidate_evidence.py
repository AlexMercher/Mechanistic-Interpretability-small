"""
Milestone 6, Phase 1 step 5 / Phase 2 step 2 (plan D.1.5 / D.2.2): stratified,
traceable evidence extraction per candidate.

--mode discovery: initial/exploratory evidence used to FORM a hypothesis about what a
  candidate might represent. Pulled only from the Discovery subset.
--mode confirmation: independent evidence used to TEST that hypothesis. Pulled only
  from the Confirmation subset. Every record is tagged subset:"confirmation" and is
  never merged with Discovery-subset evidence in the same pool.

For every candidate: top/mid/low activation-quantile examples (not just top-activating
-- a feature firing on 20 near-identical top examples is not strong evidence of
generalization), one-example-per-distinct-template-family up to a cap, and three
template-diversity numbers (Tier-1 confirmed, Tier-2 skeleton, Tier-3 unknown/unclustered
-- never conflated into one count), plus subdomain diversity and a simple lexical
type-token-ratio diversity signal. Every example carries full traceability (prompt_id,
token_position, activation_row_index, domain/subdomain, source prompt text).
"""
import argparse
import json
import re
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
N_TOP = 8
N_MID = 6
N_LOW = 6
N_PER_TEMPLATE_CAP = 15
WORD_RE = re.compile(r"[a-zA-Zऀ-ॿ]+")


def load_candidates():
    rows = []
    with (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_template_tiers():
    tiers = {}
    with (SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "template_clusters.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tiers[row["prompt_id"]] = (row["tier"], row["template_cluster_id"])
    return tiers


def compute_firing_events(mode: str, candidate_feature_idxs: list):
    """Runs a forward pass over the requested subset's tokens and returns a DataFrame
    of firing events restricted to the candidate feature indices (events for ALL
    other features are discarded -- this script does not need the full inventory)."""
    assignment = {}
    with (SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assignment[row["prompt_id"]] = row["subset"]

    rows = []
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "validation":
                continue
            if assignment.get(row["prompt_id"]) != mode:
                continue
            rows.append((row["activation_row_index"], row["prompt_id"], row["domain"],
                         row["subdomain"], row["token_position"]))
    meta = pd.DataFrame(rows, columns=["activation_row_index", "prompt_id", "domain", "subdomain", "token_position"])
    meta = meta.sort_values("activation_row_index").reset_index(drop=True)
    n_tokens = len(meta)
    print(f"[evidence:{mode}] {n_tokens} tokens")

    device = sf.get_device_no_fallback()
    model, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    full_val = sf.load_split_activations("validation")
    x_all = full_val[meta["activation_row_index"].to_numpy()]
    del full_val

    feat_idx_t = torch.tensor(candidate_feature_idxs, dtype=torch.long, device=device)
    events = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_tokens, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_tokens)
            raw = x_all[start:end].to(device, torch.float32)
            x = preprocessor.transform(raw)
            z_full = model.encode(x)
            z_sel = z_full[:, feat_idx_t].cpu().numpy()
            rows_nz, cols_nz = np.nonzero(z_sel)
            if len(rows_nz) == 0:
                continue
            vals = z_sel[rows_nz, cols_nz]
            meta_slice = meta.iloc[start:end].reset_index(drop=True)
            events.append(pd.DataFrame({
                "feature_idx": [candidate_feature_idxs[c] for c in cols_nz],
                "value": vals,
                "domain": meta_slice["domain"].to_numpy()[rows_nz],
                "subdomain": meta_slice["subdomain"].to_numpy()[rows_nz],
                "prompt_id": meta_slice["prompt_id"].to_numpy()[rows_nz],
                "token_position": meta_slice["token_position"].to_numpy()[rows_nz],
                "activation_row_index": meta_slice["activation_row_index"].to_numpy()[rows_nz],
            }))
    if not events:
        return pd.DataFrame(columns=["feature_idx", "value", "domain", "subdomain", "prompt_id", "token_position", "activation_row_index"])
    return pd.concat(events, ignore_index=True)


def lexical_diversity(texts: list) -> dict:
    words = []
    for t in texts:
        words.extend(w.lower() for w in WORD_RE.findall(t))
    if not words:
        return {"n_tokens": 0, "n_distinct": 0, "type_token_ratio": None}
    distinct = len(set(words))
    return {"n_tokens": len(words), "n_distinct": distinct, "type_token_ratio": distinct / len(words)}


def build_examples(feat_events: pd.DataFrame, records: dict, tiers: dict, subset_label: str) -> list:
    feat_events = feat_events.sort_values("value", ascending=False).reset_index(drop=True)
    n = len(feat_events)

    def make_example(row):
        rec = records.get(row["prompt_id"], {})
        tier, cluster_id = tiers.get(row["prompt_id"], (3, None))
        return {
            "subset": subset_label,
            "prompt_id": row["prompt_id"], "activation_row_index": int(row["activation_row_index"]),
            "token_position": int(row["token_position"]), "domain": row["domain"], "subdomain": row["subdomain"],
            "activation_value": float(row["value"]),
            "template_tier": tier, "template_cluster_id": cluster_id,
            "prompt_text": rec.get("prompt", "")[:400],
        }

    top = [make_example(r) for _, r in feat_events.head(N_TOP).iterrows()]
    mid_start, mid_end = int(n * 0.40), int(n * 0.60)
    mid_pool = feat_events.iloc[mid_start:mid_end] if mid_end > mid_start else feat_events
    mid = [make_example(r) for _, r in mid_pool.head(N_MID).iterrows()]
    low = [make_example(r) for _, r in feat_events.tail(N_LOW).iterrows()]

    # one-per-distinct-template (prefer Tier-1, then Tier-2, then Tier-3), highest activation within each
    seen_templates = set()
    per_template = []
    for _, r in feat_events.iterrows():
        tier, cluster_id = tiers.get(r["prompt_id"], (3, None))
        key = (tier, cluster_id) if tier in (1, 2) else (3, r["prompt_id"])  # tier-3: each prompt is its own "template"
        if key in seen_templates:
            continue
        seen_templates.add(key)
        per_template.append(make_example(r))
        if len(per_template) >= N_PER_TEMPLATE_CAP:
            break

    return {"top": top, "mid": mid, "low": low, "one_per_template": per_template}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["discovery", "confirmation"], required=True)
    args = ap.parse_args()

    candidates = load_candidates()
    feature_idxs = [c["feature_idx"] for c in candidates]
    records = sf.load_normalized_records()
    tiers = load_template_tiers()

    events = compute_firing_events(args.mode, feature_idxs)

    # Persisted so confound_diagnostics.py / generate_counterexamples.py can reuse these
    # candidate-restricted firing events without re-running the forward pass.
    events_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / f"candidate_firing_events_{args.mode}.pkl"
    events.to_pickle(events_path)
    print(f"[evidence:{args.mode}] wrote {events_path} ({len(events)} rows, for downstream reuse)")

    out_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / f"candidate_evidence_{args.mode}.jsonl"
    n_zero_evidence = 0
    with out_path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            feat = cand["feature_idx"]
            feat_events = events[events["feature_idx"] == feat]
            if len(feat_events) == 0:
                n_zero_evidence += 1
                record = {
                    "feature_idx": feat, "subset": args.mode, "n_firing_tokens": 0,
                    "n_unique_firing_prompts": 0, "examples": {"top": [], "mid": [], "low": [], "one_per_template": []},
                    "template_diversity": {"tier1_families": 0, "tier2_families": 0, "tier3_unclustered_prompts": 0},
                    "subdomain_diversity": 0, "lexical_diversity": lexical_diversity([]),
                }
                f.write(json.dumps(record) + "\n")
                continue

            unique_prompts = feat_events["prompt_id"].unique().tolist()
            prompt_tiers = [tiers.get(pid, (3, None)) for pid in unique_prompts]
            tier1_families = len({c for t, c in prompt_tiers if t == 1})
            tier2_families = len({c for t, c in prompt_tiers if t == 2})
            tier3_count = sum(1 for t, _ in prompt_tiers if t == 3)
            subdomain_diversity = feat_events["subdomain"].nunique()
            texts = [records.get(pid, {}).get("prompt", "") for pid in unique_prompts]

            examples = build_examples(feat_events, records, tiers, args.mode)

            record = {
                "feature_idx": feat, "subset": args.mode,
                "n_firing_tokens": int(len(feat_events)), "n_unique_firing_prompts": len(unique_prompts),
                "examples": examples,
                "template_diversity": {
                    "tier1_confirmed_families": tier1_families, "tier2_skeleton_families": tier2_families,
                    "tier3_unclustered_unknown_prompts": tier3_count,
                    "caveat": "tier3 prompts have UNKNOWN template relationships -- never counted as independent templates",
                },
                "subdomain_diversity": int(subdomain_diversity),
                "lexical_diversity": lexical_diversity(texts),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[evidence:{args.mode}] wrote {out_path}; {n_zero_evidence}/{len(candidates)} candidates had zero firings in this subset")


if __name__ == "__main__":
    main()

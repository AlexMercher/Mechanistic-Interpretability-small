"""
Milestone 6, Phase 4 pre-step: builds the shared activation matrices used by every
Phase 4 relationship script. Restricted to the 61 candidates at >=medium primary-review
confidence (same population as cross_capacity_check.py) -- relationship confidence is
explicitly gated by each candidate's own individual validation confidence, per the plan.

Writes, for BOTH Discovery and Confirmation subsets separately (never pooled across the
two -- Discovery is exploratory, Confirmation is confirmatory):
  - token-level activation matrix [n_tokens, 61] + per-token domain/template_family/
    language_form/prompt_id metadata
  - prompt-level activation matrix [n_prompts, 61] (max activation per prompt), covering
    EVERY prompt in that subset (not just firing ones -- needed for valid correlation,
    since restricting to firing-only prompts would introduce selection bias)
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
MIN_CONFIDENCE = "medium"
CONFIDENCE_RANK = {"unvalidated": 0, "low": 1, "medium": 2, "high": 3}


def eligible_candidates():
    reviews = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]
    return sorted(r["feature_idx"] for r in reviews if CONFIDENCE_RANK[r["confidence"]] >= CONFIDENCE_RANK[MIN_CONFIDENCE])


def load_subset_meta(subset_label, template_tier):
    assignment = {}
    with (SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assignment[row["prompt_id"]] = row["subset"]
    lang_annot = sf.load_language_script_annotations()

    rows = []
    with sf.PROMPT_TOKEN_METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "validation" or assignment.get(row["prompt_id"]) != subset_label:
                continue
            tier, cluster_id = template_tier.get(row["prompt_id"], (3, None))
            template_family = cluster_id if tier == 1 else "none"
            rows.append((row["activation_row_index"], row["prompt_id"], row["domain"],
                         template_family, lang_annot.get(row["prompt_id"], {}).get("language_form", "unknown")))
    meta = pd.DataFrame(rows, columns=["activation_row_index", "prompt_id", "domain", "template_family", "language_form"])
    return meta.sort_values("activation_row_index").reset_index(drop=True)


@torch.no_grad()
def main():
    cand_idxs = eligible_candidates()
    print(f"[rel_matrix] {len(cand_idxs)} eligible candidates (>= {MIN_CONFIDENCE} confidence)")

    template_tier = {}
    with (SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "template_clusters.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            template_tier[row["prompt_id"]] = (row["tier"], row["template_cluster_id"])

    device = sf.get_device_no_fallback()
    model, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    full_val = sf.load_split_activations("validation")
    cand_idxs_t = torch.tensor(cand_idxs, dtype=torch.long, device=device)

    out_dir = SFD_DIR / "analysis" / "data" / "relationship_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eligible_candidates.json").write_text(json.dumps({"candidates": cand_idxs, "min_confidence": MIN_CONFIDENCE}, indent=2))

    for subset in ("discovery", "confirmation"):
        meta = load_subset_meta(subset, template_tier)
        x = full_val[meta["activation_row_index"].to_numpy().copy()]
        z_all = []
        batch = 4096
        for start in range(0, len(meta), batch):
            end = min(start + batch, len(meta))
            raw = x[start:end].to(device, torch.float32)
            xp = preprocessor.transform(raw)
            z_all.append(model.encode(xp)[:, cand_idxs_t].cpu().numpy())
        z = np.concatenate(z_all, axis=0)  # [n_tokens, 61]

        token_df = meta.copy()
        for i, feat in enumerate(cand_idxs):
            token_df[f"f{feat}"] = z[:, i]
        token_path = out_dir / f"token_activation_matrix_{subset}.pkl"
        token_df.to_pickle(token_path)

        prompt_group = token_df.groupby("prompt_id")
        prompt_max = prompt_group[[f"f{feat}" for feat in cand_idxs]].max()
        prompt_domain = prompt_group["domain"].first()
        prompt_template = prompt_group["template_family"].first()
        prompt_lang = prompt_group["language_form"].first()
        prompt_df = prompt_max.copy()
        prompt_df["domain"] = prompt_domain
        prompt_df["template_family"] = prompt_template
        prompt_df["language_form"] = prompt_lang
        prompt_path = out_dir / f"prompt_activation_matrix_{subset}.pkl"
        prompt_df.to_pickle(prompt_path)

        print(f"[rel_matrix] {subset}: {len(token_df)} tokens, {len(prompt_df)} prompts -> wrote {token_path.name}, {prompt_path.name}")


if __name__ == "__main__":
    main()

"""
Milestone 6, Phase 1 step 2 (plan D.1.2): full SAE-8192 seed-0 feature inventory,
computed on the Discovery subset only (per the B.3 split -- selection must never see
Confirmation-subset statistics).

One forward pass over every Discovery-subset validation token, producing per-feature:
  - token-level and prompt-level activation frequency
  - mean activation when active, max activation, activation quantiles
  - domain and (domain,subdomain) firing-count distributions (raw counts; normalization
    against the Discovery subset's own domain token baseline happens in select_candidates.py,
    using the baseline this script also writes out)
  - language_form firing-count distribution (joined from M2's language/script annotations)
  - unique firing prompt count

Dead features (zero firings in Discovery) are included with all-zero/NaN-safe stats --
never silently dropped from the 8192-row output.
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
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def load_discovery_rows():
    """Returns a DataFrame with one row per Discovery-subset validation token:
    activation_row_index, prompt_id, domain, subdomain, token_position."""
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
            if assignment.get(row["prompt_id"]) != "discovery":
                continue
            rows.append((row["activation_row_index"], row["prompt_id"], row["domain"],
                         row["subdomain"], row["token_position"]))
    df = pd.DataFrame(rows, columns=["activation_row_index", "prompt_id", "domain", "subdomain", "token_position"])
    return df.sort_values("activation_row_index").reset_index(drop=True)


def main():
    device = sf.get_device_no_fallback()
    model, config, ckpt = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")

    discovery_meta = load_discovery_rows()
    n_discovery_tokens = len(discovery_meta)
    n_discovery_prompts = discovery_meta["prompt_id"].nunique()
    print(f"[inventory] Discovery subset: {n_discovery_tokens} tokens, {n_discovery_prompts} prompts")

    lang_annot = sf.load_language_script_annotations()
    discovery_meta["language_form"] = discovery_meta["prompt_id"].map(
        lambda pid: lang_annot.get(pid, {}).get("language_form", "unknown"))
    discovery_meta["domain_subdomain"] = discovery_meta["domain"] + "::" + discovery_meta["subdomain"]

    full_val = sf.load_split_activations("validation")
    row_idx = discovery_meta["activation_row_index"].to_numpy()
    discovery_x = full_val[row_idx]
    del full_val

    # Domain/language-form token baselines, computed from the Discovery subset itself
    # (not M4/M2's full-dataset stats) -- needed by select_candidates.py's
    # concentration formulas, each over ALL Discovery tokens, not just firing ones.
    domain_baseline = (discovery_meta["domain"].value_counts() / n_discovery_tokens).to_dict()
    language_baseline = (discovery_meta["language_form"].value_counts() / n_discovery_tokens).to_dict()
    (SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "discovery_domain_token_baseline.json").write_text(
        json.dumps({"n_discovery_tokens": int(n_discovery_tokens),
                    "n_discovery_prompts": int(n_discovery_prompts),
                    "domain_token_share": {k: float(v) for k, v in domain_baseline.items()},
                    "language_form_token_share": {k: float(v) for k, v in language_baseline.items()}}, indent=2),
        encoding="utf-8")

    events = []  # (feature_idx, value, domain, domain_subdomain, language_form, prompt_id, token_position, activation_row_index)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_discovery_tokens, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_discovery_tokens)
            raw = discovery_x[start:end].to(device, torch.float32)
            x = preprocessor.transform(raw)
            z = model.encode(x).cpu().numpy()
            rows_nz, cols_nz = np.nonzero(z)
            if len(rows_nz) == 0:
                continue
            vals = z[rows_nz, cols_nz]
            meta_slice = discovery_meta.iloc[start:end].reset_index(drop=True)
            events.append(pd.DataFrame({
                "feature_idx": cols_nz,
                "value": vals,
                "domain": meta_slice["domain"].to_numpy()[rows_nz],
                "domain_subdomain": meta_slice["domain_subdomain"].to_numpy()[rows_nz],
                "language_form": meta_slice["language_form"].to_numpy()[rows_nz],
                "prompt_id": meta_slice["prompt_id"].to_numpy()[rows_nz],
                "token_position": meta_slice["token_position"].to_numpy()[rows_nz],
                "activation_row_index": meta_slice["activation_row_index"].to_numpy()[rows_nz],
            }))
            if (start // BATCH_SIZE) % 5 == 0:
                print(f"[inventory] processed {end}/{n_discovery_tokens} tokens")

    events_df = pd.concat(events, ignore_index=True)
    print(f"[inventory] total firing events (nonzero activations): {len(events_df)}")

    # Persisted so downstream scripts (select_candidates.py, confound_diagnostics.py)
    # can join template-cluster tiers etc. without re-running this forward pass.
    events_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_discovery_firing_events.pkl"
    events_df.to_pickle(events_path)
    print(f"[inventory] wrote {events_path} ({len(events_df)} rows, for downstream reuse)")

    grouped = events_df.groupby("feature_idx")
    stats = grouped["value"].agg(["count", "mean", "max"]).rename(
        columns={"count": "n_firing_tokens", "mean": "mean_activation_when_active", "max": "max_activation"})
    quantiles = grouped["value"].quantile(QUANTILES).unstack()
    quantiles.columns = [f"q{int(q*100)}" for q in QUANTILES]
    unique_prompts = grouped["prompt_id"].nunique().rename("n_unique_firing_prompts")

    domain_counts = events_df.groupby(["feature_idx", "domain"]).size().unstack(fill_value=0)
    subdomain_counts = events_df.groupby(["feature_idx", "domain_subdomain"]).size().unstack(fill_value=0)
    language_counts = events_df.groupby(["feature_idx", "language_form"]).size().unstack(fill_value=0)

    full_index = pd.RangeIndex(WIDTH, name="feature_idx")
    stats = stats.reindex(full_index, fill_value=0)
    quantiles = quantiles.reindex(full_index, fill_value=0.0)
    unique_prompts = unique_prompts.reindex(full_index, fill_value=0)
    domain_counts = domain_counts.reindex(full_index, fill_value=0)
    subdomain_counts = subdomain_counts.reindex(full_index, fill_value=0)
    language_counts = language_counts.reindex(full_index, fill_value=0)

    out_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for feat in range(WIDTH):
            n_firing = int(stats.loc[feat, "n_firing_tokens"])
            record = {
                "feature_idx": feat,
                "dictionary_width": WIDTH,
                "seed": SEED,
                "subset": "discovery",
                "n_firing_tokens": n_firing,
                "token_level_frequency": n_firing / n_discovery_tokens,
                "n_unique_firing_prompts": int(unique_prompts.loc[feat]),
                "prompt_level_frequency": int(unique_prompts.loc[feat]) / n_discovery_prompts,
                "mean_activation_when_active": float(stats.loc[feat, "mean_activation_when_active"]) if n_firing else None,
                "max_activation": float(stats.loc[feat, "max_activation"]) if n_firing else None,
                "activation_quantiles": {c: float(quantiles.loc[feat, c]) for c in quantiles.columns} if n_firing else None,
                "domain_firing_counts": {d: int(domain_counts.loc[feat, d]) for d in domain_counts.columns if domain_counts.loc[feat, d] > 0},
                "domain_subdomain_firing_counts": {d: int(subdomain_counts.loc[feat, d]) for d in subdomain_counts.columns if subdomain_counts.loc[feat, d] > 0},
                "language_form_firing_counts": {d: int(language_counts.loc[feat, d]) for d in language_counts.columns if language_counts.loc[feat, d] > 0},
                "is_dead_in_discovery": n_firing == 0,
            }
            f.write(json.dumps(record) + "\n")

    n_dead = int((stats["n_firing_tokens"] == 0).sum())
    print(f"[inventory] wrote {out_path} -- {WIDTH} features, {n_dead} dead in Discovery subset")


if __name__ == "__main__":
    main()

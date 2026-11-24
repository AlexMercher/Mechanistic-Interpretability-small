"""
Milestone 6, Phase 1 step 4 (plan B.1 + D.1.4): applies the frozen quota table in
configs/candidate_selection_criteria.json to the Discovery-subset inventory.

Selects ~120-150 candidates across 10 strata. A candidate matching multiple strata's
rules keeps ALL secondary-stratum memberships but is assigned exactly one
primary_stratum (first-matching row in the quota table's fixed order) -- this is what
Phase 3's second-reviewer sampling stratifies on.

The two cross-seed strata are selected using a Discovery-subset-ONLY preliminary
screen (decoder cosine + activation-pattern cosine against the complete SAE-8192
seed-1/seed-2 dictionaries). This preliminary result is never reused as, or confused
with, the final Confirmation-subset-only validation done in Phase 3
(cross_seed_reproducibility.py) -- the boundary is logged explicitly below and in
candidate_selection_report.md.
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
RARE_FLOOR = 0.001


def load_inventory():
    rows = []
    with (SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows).set_index("feature_idx")


def compute_domain_concentration(inv: pd.DataFrame, domain_baseline: dict) -> pd.Series:
    scores = {}
    for feat, row in inv.iterrows():
        counts = row["domain_firing_counts"]
        total = row["n_firing_tokens"]
        if not counts or total == 0:
            scores[feat] = 0.0
            continue
        best = 0.0
        for d, c in counts.items():
            p_d = c / total
            b_d = domain_baseline.get(d, 1e-9)
            best = max(best, p_d / b_d)
        scores[feat] = best
    return pd.Series(scores, name="domain_concentration")


def compute_cross_domain_entropy(inv: pd.DataFrame) -> pd.Series:
    scores = {}
    for feat, row in inv.iterrows():
        counts = row["domain_firing_counts"]
        total = row["n_firing_tokens"]
        if not counts or total == 0:
            scores[feat] = 0.0
            continue
        h = 0.0
        for d, c in counts.items():
            p = c / total
            if p > 0:
                h -= p * np.log2(p)
        scores[feat] = h / np.log2(5)
    return pd.Series(scores, name="cross_domain_entropy_norm")


def compute_template_concentration(inv: pd.DataFrame, events: pd.DataFrame, template_tier: dict) -> pd.Series:
    events = events.copy()
    events["tier1_cluster"] = events["prompt_id"].map(
        lambda pid: template_tier.get(pid, (3, None))[1] if template_tier.get(pid, (3, None))[0] == 1 else None)
    has_cluster = events.dropna(subset=["tier1_cluster"])
    cluster_counts = has_cluster.groupby(["feature_idx", "tier1_cluster"]).size()
    max_cluster_count = cluster_counts.groupby("feature_idx").max() if len(cluster_counts) else pd.Series(dtype=float)
    scores = {}
    for feat, row in inv.iterrows():
        total = row["n_firing_tokens"]
        if total < 3:
            scores[feat] = 0.0
            continue
        best = max_cluster_count.get(feat, 0)
        scores[feat] = best / total
    return pd.Series(scores, name="template_concentration")


def compute_script_language_score(inv: pd.DataFrame, language_baseline: dict) -> pd.Series:
    scores = {}
    for feat, row in inv.iterrows():
        counts = row["language_form_firing_counts"]
        total = row["n_firing_tokens"]
        if not counts or total == 0:
            scores[feat] = 0.0
            continue
        best = 0.0
        for form in ("native_hindi", "romanized_hindi"):
            c = counts.get(form, 0)
            p = c / total
            b = language_baseline.get(form, 1e-9)
            best = max(best, p / b)
        scores[feat] = best
    return pd.Series(scores, name="script_language_score")


@torch.no_grad()
def preliminary_cross_seed_screen(candidate_pool_idx, device):
    """Discovery-subset-only decoder+activation-pattern screen against the complete
    seed-1 and seed-2 SAE-8192 dictionaries. Returns dict feat_idx -> bool (passes
    both checks against both seeds)."""
    model0, _, _ = sf.load_model_for_eval(WIDTH, SEED, "best_val", device)
    preprocessor = sf.Preprocessor.load(
        Path(__file__).resolve().parent.parent.parent / "sae_capacity_comparison" / "preprocessing")
    val_x = sf.load_split_activations("validation")
    sample_idx = torch.arange(min(8192, val_x.shape[0]))
    raw = val_x[sample_idx].to(device, torch.float32)
    x = preprocessor.transform(raw)
    z0 = model0.encode(x)
    Wd0 = model0.W_d.detach()  # [2048, 8192] unit-norm columns

    passes = {}
    for seed in (1, 2):
        model_s, _, _ = sf.load_model_for_eval(WIDTH, seed, "best_val", device)
        z_s = model_s.encode(x)
        Wd_s = model_s.W_d.detach()

        decoder_sim = (Wd0.t() @ Wd_s).cpu().numpy()  # [8192, 8192], candidate(source) x target
        top1 = decoder_sim.max(axis=1)
        top2 = np.partition(decoder_sim, -2, axis=1)[:, -2]
        decoder_ok = (top1 > 0.3) & ((top1 - top2) >= 0.1)
        best_target = decoder_sim.argmax(axis=1)

        z0_np = z0.cpu().numpy()
        zs_np = z_s.cpu().numpy()
        z0_norm = z0_np / (np.linalg.norm(z0_np, axis=0, keepdims=True) + 1e-8)
        zs_norm = zs_np / (np.linalg.norm(zs_np, axis=0, keepdims=True) + 1e-8)
        act_sim_full = z0_norm.T @ zs_norm  # [8192, 8192]
        act_sim_at_best = act_sim_full[np.arange(WIDTH), best_target]
        act_ok = act_sim_at_best > 0.3

        for feat in range(WIDTH):
            passes.setdefault(feat, {})[seed] = bool(decoder_ok[feat] and act_ok[feat])

    return {feat: all(passes[feat].values()) for feat in passes}


def main():
    inv = load_inventory()
    baseline = json.loads((SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "discovery_domain_token_baseline.json").read_text())
    domain_baseline = baseline["domain_token_share"]
    language_baseline = baseline["language_form_token_share"]
    events = pd.read_pickle(SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_discovery_firing_events.pkl")

    template_rows = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "template_clusters.jsonl").open(encoding="utf-8")]
    template_tier = {r["prompt_id"]: (r["tier"], r["template_cluster_id"]) for r in template_rows}

    inv["domain_concentration"] = compute_domain_concentration(inv, domain_baseline)
    inv["cross_domain_entropy_norm"] = compute_cross_domain_entropy(inv)
    inv["template_concentration"] = compute_template_concentration(inv, events, template_tier)
    inv["script_language_score"] = compute_script_language_score(inv, language_baseline)

    device = sf.get_device_no_fallback()
    print("[select] running Discovery-only preliminary cross-seed screen (full target dictionaries)...")
    cross_seed_pass = preliminary_cross_seed_screen(inv.index.tolist(), device)
    inv["prelim_cross_seed_pass"] = inv.index.map(lambda f: cross_seed_pass.get(f, False))

    alive = inv[inv["n_firing_tokens"] > 0].copy()
    nonzero_freq = alive["token_level_frequency"]

    # Strata are applied in fixed quota-table order; each stratum's candidate pool
    # EXCLUDES features already selected by an earlier-order stratum before taking its
    # top N. This is what guarantees every stratum gets to contribute its own unique
    # primary members (a feature can still accumulate secondary-stratum memberships
    # for strata it would also have qualified for) -- without this exclusion, a later
    # stratum whose top-ranked features are all correlated with an earlier stratum's
    # criterion (e.g. domain-concentration and script-dependence both peaking on the
    # same Hindi-heavy features) would contribute zero new primaries, which happened
    # in an earlier run of this script and is corrected here.
    selected = {}  # feat_idx -> list of strata (in quota order)
    already_selected = set()

    def add(feat_idx, stratum):
        selected.setdefault(feat_idx, []).append(stratum)
        already_selected.add(feat_idx)

    def remaining(df):
        return df[~df.index.isin(already_selected)]

    # 1. high_frequency (excl >0.90)
    pool = remaining(alive[alive["token_level_frequency"] <= 0.90]).sort_values("token_level_frequency", ascending=False)
    for feat in pool.index[:15]:
        add(feat, "high_frequency")

    # 2. always_on_near_ubiquitous
    ubiq = remaining(alive[alive["token_level_frequency"] > 0.90]).sort_values("token_level_frequency", ascending=False)
    for feat in ubiq.index[:5]:
        add(feat, "always_on_near_ubiquitous")

    # 3. medium_frequency (40th-60th percentile of nonzero freq)
    p40, p60 = nonzero_freq.quantile(0.40), nonzero_freq.quantile(0.60)
    med_pool = remaining(alive[(alive["token_level_frequency"] >= p40) & (alive["token_level_frequency"] <= p60)])
    med_pool = med_pool.sort_values("token_level_frequency", ascending=False)
    for feat in med_pool.index[:15]:
        add(feat, "medium_frequency")

    # 4. rare_but_active (bottom decile)
    p10 = nonzero_freq.quantile(0.10)
    rare_pool = remaining(alive[alive["token_level_frequency"] <= p10]).sort_values("token_level_frequency", ascending=True)
    for feat in rare_pool.index[:20]:
        add(feat, "rare_but_active")

    # 5. strong_domain_concentration
    dom_pool = remaining(alive).sort_values("domain_concentration", ascending=False)
    for feat in dom_pool.index[:15]:
        add(feat, "strong_domain_concentration")

    # 6. cross_domain_broadly_distributed (above rare floor)
    cross_pool = remaining(alive[alive["token_level_frequency"] >= RARE_FLOOR]).sort_values("cross_domain_entropy_norm", ascending=False)
    for feat in cross_pool.index[:15]:
        add(feat, "cross_domain_broadly_distributed")

    # 7. suspicious_template_concentration (>=3 firing tokens)
    templ_pool = remaining(alive[alive["n_firing_tokens"] >= 3]).sort_values("template_concentration", ascending=False)
    for feat in templ_pool.index[:10]:
        add(feat, "suspicious_template_concentration")

    # 8. script_language_dependence
    lang_pool = remaining(alive).sort_values("script_language_score", ascending=False)
    for feat in lang_pool.index[:10]:
        add(feat, "script_language_dependence")

    # 9/10. cross-seed prescreened reproducible / non-reproducible control.
    # Frequency-matched via decile bucket (not a narrow min-max range), since the
    # reproducible pool can be small and a min-max range would be too narrow to find
    # matches once already-selected features are also excluded.
    alive_decile = pd.qcut(nonzero_freq, 10, labels=False, duplicates="drop")
    repro_pool_all = alive[alive["prelim_cross_seed_pass"]]
    repro_pool = remaining(repro_pool_all).sort_values("token_level_frequency", ascending=False)
    for feat in repro_pool.index[:10]:
        add(feat, "cross_seed_prescreened_reproducible")

    repro_deciles = set(alive_decile.loc[repro_pool_all.index].dropna().unique())
    nonrepro_pool_all = alive[~alive["prelim_cross_seed_pass"]]
    nonrepro_in_deciles = nonrepro_pool_all[alive_decile.loc[nonrepro_pool_all.index].isin(repro_deciles)]
    nonrepro_matched = remaining(nonrepro_in_deciles).sort_values("token_level_frequency", ascending=False)
    for feat in nonrepro_matched.index[:10]:
        add(feat, "cross_seed_prescreened_nonreproducible_control")

    quota_order = ["high_frequency", "always_on_near_ubiquitous", "medium_frequency", "rare_but_active",
                   "strong_domain_concentration", "cross_domain_broadly_distributed",
                   "suspicious_template_concentration", "script_language_dependence",
                   "cross_seed_prescreened_reproducible", "cross_seed_prescreened_nonreproducible_control"]

    out_path = SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for feat, strata in sorted(selected.items()):
            primary = next(s for s in quota_order if s in strata)
            f.write(json.dumps({
                "feature_idx": int(feat), "dictionary_width": WIDTH, "seed": SEED,
                "primary_stratum": primary, "secondary_strata": [s for s in strata if s != primary],
                "token_level_frequency": float(inv.loc[feat, "token_level_frequency"]),
                "prompt_level_frequency": float(inv.loc[feat, "prompt_level_frequency"]),
                "domain_concentration": float(inv.loc[feat, "domain_concentration"]),
                "cross_domain_entropy_norm": float(inv.loc[feat, "cross_domain_entropy_norm"]),
                "template_concentration": float(inv.loc[feat, "template_concentration"]),
                "script_language_score": float(inv.loc[feat, "script_language_score"]),
                "prelim_cross_seed_pass_discovery_only": bool(inv.loc[feat, "prelim_cross_seed_pass"]),
            }) + "\n")

    n_by_primary = pd.Series([next(s for s in quota_order if s in strata) for strata in selected.values()]).value_counts()
    report_lines = [
        "# Milestone 6 Candidate Selection Report\n",
        f"Total candidates selected: {len(selected)}\n",
        "## Primary-stratum counts\n",
        n_by_primary.to_string(), "\n",
        "## Discovery/Confirmation boundary (mandatory disclosure)\n",
        "The two cross-seed strata (`cross_seed_prescreened_reproducible`, "
        "`cross_seed_prescreened_nonreproducible_control`) were selected using a "
        "**Discovery-subset-only** preliminary decoder+activation-pattern screen "
        "(0.3 cosine / 0.1 margin thresholds) against the complete SAE-8192 seed-1 and "
        "seed-2 dictionaries. This preliminary result decided ONLY which features enter "
        "these two strata -- it is never reused as, or averaged into, the final "
        "cross-seed validation number. The independent, final cross-seed "
        "activation-pattern match, top-prompt overlap, and matched-target semantic "
        "evidence pull (for every selected candidate, not just these two strata) are "
        "computed fresh in Phase 3 using **Confirmation-subset-only** activations.\n",
        "## Rare-feature coverage limitation (mandatory disclosure)\n",
        "Candidate selection runs entirely on the Discovery-subset inventory. A feature "
        "that never fires in Discovery but does fire in Confirmation cannot enter the "
        "`rare_but_active` stratum, or any stratum. This cohort is a stratified sample "
        "of Discovery-visible features, not an exhaustive census of all rare features "
        "in the dictionary. See Phase 2's Confirmation-only coverage statistic.\n",
    ]
    (SFD_DIR / "reports" / "candidate_selection_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[select] selected {len(selected)} candidates")
    print(n_by_primary)
    print(f"[select] wrote {out_path}")
    print(f"[select] wrote {SFD_DIR / 'reports' / 'candidate_selection_report.md'}")


if __name__ == "__main__":
    main()

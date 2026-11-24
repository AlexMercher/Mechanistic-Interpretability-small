"""
Milestone 6, Phase 4 step 1 (plan B.1/B.7): co-activation analysis among the 61
eligible candidates.

Methodology, frozen and not chosen per-result:
  1. Pooled (raw, unconditioned) correlation, both subsets, both levels -- the
     exploratory/baseline number.
  2. A pair is selected for the EXPENSIVE controlled (stratified) analysis only if its
     Confirmation-subset, PROMPT-LEVEL pooled |Pearson r| >= 0.3 (the primary
     inferential signal, decided before any stratified numbers exist -- this is a
     compute-bounding filter, not a conditioning-strategy choice).
  3. For each selected pair, PER DOMAIN, the B.7 fallback hierarchy is applied as a fixed
     cascade that never discards data: bucket prompts by (template_family, language_form);
     any bucket with >=20 prompts is reported at "level 1"; prompts left over (buckets
     <20) are re-pooled by (language_form) alone -- buckets >=20 reported at "level 2";
     whatever remains is pooled at the (domain) level -- reported at "level 3" if >=20,
     else excluded with an explicit reason. The level actually used, and used ONLY
     because of its position in this fixed cascade (never because it produced a
     stronger correlation), is recorded for every stratum.
  4. PROMPT-LEVEL correlation is the primary number at every level. TOKEN-LEVEL
     correlation is reported only as a secondary, descriptive statistic; any token-level
     interval uses a prompt-cluster bootstrap (resampling prompts, never individual
     tokens) rather than a token-count-based formula.
  5. Both the number of prompts AND the number of tokens are reported for every
     correlation, at every level.

template_family here = the Tier-1 confirmed-near-duplicate cluster id, or "none" for
everything else (Tier 2/3 are per-prompt guesses/unknowns, not reusable grouping keys).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
REL_DIR = SFD_DIR / "analysis" / "data" / "relationship_analysis"
MIN_STRATUM_PROMPTS = 20
POOLED_FILTER_THRESHOLD = 0.3
N_BOOTSTRAP = 1000
RNG_SEED = 1337


def load_matrices(subset):
    token_df = pd.read_pickle(REL_DIR / f"token_activation_matrix_{subset}.pkl")
    prompt_df = pd.read_pickle(REL_DIR / f"prompt_activation_matrix_{subset}.pkl")
    return token_df, prompt_df


def pooled_correlations(df, feat_cols, level_label, n_prompts_lookup=None):
    """df: rows = observations (tokens or prompts), columns include feat_cols.
    Returns dict (feat_a, feat_b) -> {pearson, spearman, n}."""
    mat = df[feat_cols].to_numpy()
    pearson = np.corrcoef(mat.T)
    spearman, _ = stats.spearmanr(mat)
    if spearman.ndim == 0:
        spearman = np.array([[1.0, spearman], [spearman, 1.0]])
    n = mat.shape[0]
    out = {}
    for i, a in enumerate(feat_cols):
        for j, b in enumerate(feat_cols):
            if j <= i:
                continue
            out[(a, b)] = {"pearson": float(pearson[i, j]), "spearman": float(spearman[i, j]), "n": int(n)}
    return out


def prompt_cluster_bootstrap_ci(token_df, feat_a, feat_b, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Resamples PROMPTS with replacement (never individual tokens), re-pools that
    resample's tokens, recomputes the token-level Pearson r each time. Returns
    (point_estimate, (p2.5, p97.5))."""
    rng = np.random.default_rng(seed)
    prompt_ids = token_df["prompt_id"].unique()
    by_prompt = {pid: g[[feat_a, feat_b]].to_numpy() for pid, g in token_df.groupby("prompt_id")}
    point_vals = np.concatenate(list(by_prompt.values()))
    point_r = float(np.corrcoef(point_vals[:, 0], point_vals[:, 1])[0, 1]) if len(point_vals) > 1 else float("nan")

    boot_rs = []
    for _ in range(n_boot):
        sample_pids = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        vals = np.concatenate([by_prompt[p] for p in sample_pids])
        if len(vals) < 2 or np.std(vals[:, 0]) == 0 or np.std(vals[:, 1]) == 0:
            continue
        boot_rs.append(np.corrcoef(vals[:, 0], vals[:, 1])[0, 1])
    if not boot_rs:
        return point_r, (float("nan"), float("nan"))
    lo, hi = np.percentile(boot_rs, [2.5, 97.5])
    return point_r, (float(lo), float(hi))


def cascaded_stratified_correlation(prompt_df, token_df, feat_a, feat_b):
    """Implements the frozen B.7 cascade per domain. Returns list of stratum results."""
    results = []
    for domain in sorted(prompt_df["domain"].unique()):
        dom_prompts = prompt_df[prompt_df["domain"] == domain]
        remaining = dom_prompts.copy()

        # Level 1: (template_family, language_form) buckets with >=20 prompts
        for (tf, lf), group in remaining.groupby(["template_family", "language_form"]):
            if len(group) < MIN_STRATUM_PROMPTS:
                continue
            r, p = stats.pearsonr(group[feat_a], group[feat_b])
            sr, sp = stats.spearmanr(group[feat_a], group[feat_b])
            results.append({"domain": domain, "level": 1, "level_def": "domain x template_family x language_form",
                             "stratum_key": {"template_family": str(tf), "language_form": lf},
                             "n_prompts": len(group), "pearson": float(r), "spearman": float(sr)})
            remaining = remaining.drop(group.index)

        # Level 2: remaining prompts pooled by language_form, >=20
        for lf, group in remaining.groupby("language_form"):
            if len(group) < MIN_STRATUM_PROMPTS:
                continue
            r, _ = stats.pearsonr(group[feat_a], group[feat_b])
            sr, _ = stats.spearmanr(group[feat_a], group[feat_b])
            results.append({"domain": domain, "level": 2, "level_def": "domain x language_form",
                             "stratum_key": {"language_form": lf},
                             "n_prompts": len(group), "pearson": float(r), "spearman": float(sr)})
            remaining = remaining.drop(group.index)

        # Level 3: whatever is left, pooled at domain level
        if len(remaining) >= MIN_STRATUM_PROMPTS:
            r, _ = stats.pearsonr(remaining[feat_a], remaining[feat_b])
            sr, _ = stats.spearmanr(remaining[feat_a], remaining[feat_b])
            results.append({"domain": domain, "level": 3, "level_def": "domain only",
                             "stratum_key": {}, "n_prompts": len(remaining), "pearson": float(r), "spearman": float(sr)})
        elif len(remaining) > 0:
            results.append({"domain": domain, "level": "excluded", "level_def": None,
                             "stratum_key": {}, "n_prompts": len(remaining), "pearson": None, "spearman": None,
                             "exclusion_reason": f"only {len(remaining)} prompts remain after levels 1-2, below the {MIN_STRATUM_PROMPTS}-prompt minimum even at domain level"})

        # Token-level secondary + prompt-cluster bootstrap, for this domain's full population (not sub-strata, to bound compute)
        dom_tokens = token_df[token_df["domain"] == domain]
        if len(dom_tokens) > 0:
            point_r, ci = prompt_cluster_bootstrap_ci(dom_tokens, feat_a, feat_b)
            n_contributing_prompts = dom_tokens["prompt_id"].nunique()
            for res in results:
                if res["domain"] == domain and res["level"] == 3:
                    res["token_level_secondary"] = {
                        "pearson_pooled": point_r, "bootstrap_95ci": ci, "n_tokens": len(dom_tokens),
                        "n_contributing_prompts": n_contributing_prompts,
                        "method": "prompt-cluster bootstrap (resamples prompts, not tokens), 1000 resamples",
                    }
    return results


def main():
    eligible = json.loads((REL_DIR / "eligible_candidates.json").read_text())["candidates"]
    feat_cols = [f"f{c}" for c in eligible]

    pooled = {}
    for subset in ("discovery", "confirmation"):
        token_df, prompt_df = load_matrices(subset)
        pooled[f"{subset}_token"] = pooled_correlations(token_df, feat_cols, "token")
        pooled[f"{subset}_prompt"] = pooled_correlations(prompt_df, feat_cols, "prompt")
        print(f"[coactivation] {subset}: pooled correlations computed for {len(pooled[f'{subset}_prompt'])} pairs")

    # Filter: Confirmation-subset, prompt-level pooled |r| >= 0.3 decides eligibility
    # for the expensive controlled analysis -- decided BEFORE any stratified result exists.
    selected_pairs = [pair for pair, v in pooled["confirmation_prompt"].items() if abs(v["pearson"]) >= POOLED_FILTER_THRESHOLD]
    print(f"[coactivation] {len(selected_pairs)}/{len(pooled['confirmation_prompt'])} pairs pass the "
          f"pooled-Confirmation-prompt-level |r|>={POOLED_FILTER_THRESHOLD} filter for controlled analysis")

    confirmation_token_df, confirmation_prompt_df = load_matrices("confirmation")
    discovery_token_df, discovery_prompt_df = load_matrices("discovery")

    controlled_results = {}
    for feat_a, feat_b in selected_pairs:
        controlled_results[f"{feat_a}__{feat_b}"] = {
            "confirmation_controlled": cascaded_stratified_correlation(confirmation_prompt_df, confirmation_token_df, feat_a, feat_b),
            "discovery_controlled_exploratory": cascaded_stratified_correlation(discovery_prompt_df, discovery_token_df, feat_a, feat_b),
        }

    out = {
        "eligible_candidates": eligible,
        "pooled_filter_threshold": POOLED_FILTER_THRESHOLD,
        "min_stratum_prompts": MIN_STRATUM_PROMPTS,
        "pooled_correlations": {
            level: {f"{a}__{b}": v for (a, b), v in d.items()} for level, d in pooled.items()
        },
        "selected_pairs_for_controlled_analysis": [f"{a}__{b}" for a, b in selected_pairs],
        "controlled_results": controlled_results,
    }
    out_path = REL_DIR / "coactivation_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[coactivation] wrote {out_path}")


if __name__ == "__main__":
    main()

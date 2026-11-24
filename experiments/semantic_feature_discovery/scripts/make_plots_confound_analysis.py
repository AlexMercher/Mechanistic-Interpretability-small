"""
Phase 5, category C: confound_analysis. Template-tier population structure
(B.6), per-candidate template concentration vs. review classification, script/
language-dependence scores, and Discovery-vs-Confirmation confound divergence
flags (B.1/B.4 confound diagnostics). Reads only Phase 1-2 outputs.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "confound_analysis"
TEMPLATE_CFG_PATH = pc.SFD_DIR / "configs" / "template_clustering_config.json"
CAND_PATH = pc.DATA_DIR / "feature_analysis" / "candidates.jsonl"
REVIEW_PATH = pc.DATA_DIR / "validation_results" / "primary_reviews.jsonl"
CONFOUND_PATH = pc.DATA_DIR / "feature_analysis" / "confound_diagnostics.jsonl"


def main():
    tcfg = pc.load_json(TEMPLATE_CFG_PATH)
    cand = pd.DataFrame(pc.load_jsonl(CAND_PATH))
    rev = pd.DataFrame(pc.load_jsonl(REVIEW_PATH))
    df = cand.merge(rev, on="feature_idx", how="inner")
    confound = pd.DataFrame(pc.load_jsonl(CONFOUND_PATH))

    # C1: corpus-wide template tier population (25,000 prompts)
    fig, ax = plt.subplots(figsize=(9, 6))
    tiers = ["Tier 1\n(confirmed near-dup family)", "Tier 2\n(skeleton fallback)", "Tier 3\n(unclustered/unknown)"]
    counts = [tcfg["tier1_prompt_count"], tcfg["tier2_prompt_count"], tcfg["tier3_prompt_count"]]
    ax.bar(tiers, counts, color=[pc.PALETTE[0], pc.PALETTE[1], pc.PALETTE[7]])
    for i, v in enumerate(counts):
        ax.text(i, v + 150, str(v), ha="center", fontsize=12)
    ax.set_ylabel("Number of prompts (out of 25,000)")
    ax.set_title("Corpus-wide template-tier evidence strength (B.6)")
    fig.text(0.5, -0.06,
              "Caption: Tier 3 (unclustered) prompts have an UNKNOWN template relationship to "
              "any other prompt and are never counted as an independent template in any "
              "downstream template-diversity statistic.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "c1_template_tier_population",
                "Corpus-wide prompt counts by template-detection tier (Tier 1 confirmed / Tier 2 skeleton-fallback / Tier 3 unknown).",
                [TEMPLATE_CFG_PATH], n=tcfg["total_prompts"])

    # C2: template_concentration by final classification (strip/box)
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 6))
    order = (df.groupby("final_classification")["template_concentration"].median()
             .sort_values(ascending=False).index.tolist())
    sns.boxplot(data=df, x="final_classification", y="template_concentration", order=order, ax=ax,
                color=pc.PALETTE[4], showfliers=False)
    sns.stripplot(data=df, x="final_classification", y="template_concentration", order=order, ax=ax,
                  color="black", size=4, alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_xlabel("Primary-review final classification")
    ax.set_ylabel("Template concentration (Tier-1-family activation-mass share)")
    ax.set_title(f"Template concentration by review classification (n={len(df)})")
    fig.text(0.5, -0.12,
              "Caption: template_concentration is computed only from Tier-1 (confirmed "
              "near-duplicate) families -- the designed check for whether template_driven "
              "candidates are, in fact, concentrated on confirmed shared templates.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "c2_template_concentration_by_class",
                "Template concentration score distribution by final review classification.",
                [CAND_PATH, REVIEW_PATH], n=len(df))

    # C3: script_language_score by classification, highlighting script_or_language_marker
    fig, ax = plt.subplots(figsize=(11, 6))
    is_marker = df["final_classification"] == "script_or_language_marker"
    ax.scatter(df.loc[~is_marker, "feature_idx"], df.loc[~is_marker, "script_language_score"],
               label=f"other (n={(~is_marker).sum()})", color=pc.PALETTE[8], alpha=0.5, s=40)
    ax.scatter(df.loc[is_marker, "feature_idx"], df.loc[is_marker, "script_language_score"],
               label=f"script_or_language_marker (n={is_marker.sum()})", color=pc.PALETTE[3], s=90, marker="D")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Script/language-dependence score (Discovery subset)")
    ax.set_title("Script/language-dependence score by candidate, reviewed script_or_language_marker highlighted")
    ax.legend(fontsize=10)
    fig.text(0.5, -0.06,
              f"Caption: n={len(df)}. A high score alone does not guarantee a "
              "script_or_language_marker verdict -- that label also requires the reviewer to "
              "confirm activation tracks script/language regardless of content (B.4).",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "c3_script_language_score",
                "Script/language-dependence score per candidate, script_or_language_marker candidates highlighted.",
                [CAND_PATH, REVIEW_PATH], n=len(df))

    # C4: Discovery-vs-Confirmation confound divergence flags
    flag_rows = []
    for _, r in confound.iterrows():
        for metric, flag in r["divergence_flags"].items():
            flag_rows.append({"metric": metric, "flag": flag})
    flag_df = pd.DataFrame(flag_rows)
    ct = flag_df.groupby(["metric", "flag"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True, ax=ax, color=pc.PALETTE[:len(ct.columns)])
    ax.set_ylabel("Number of candidates (n=121)")
    ax.set_xlabel("Confound metric")
    ax.set_title("Discovery-vs-Confirmation confound-metric divergence flags, by metric")
    ax.legend(title="flag", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.text(0.5, -0.1,
              "Caption: 'divergent' means the metric changed by >50% relative magnitude "
              "between the two independent subsets for that candidate -- a direct, measured "
              "check of confound stability across Discovery and Confirmation.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "c4_confound_divergence_flags",
                "Count of Discovery-vs-Confirmation divergence flags per confound metric, across all 121 candidates.",
                [CONFOUND_PATH], n=len(confound))


if __name__ == "__main__":
    main()

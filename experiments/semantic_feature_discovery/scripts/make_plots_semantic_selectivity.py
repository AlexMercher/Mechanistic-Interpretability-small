"""
Phase 5, category B: semantic_selectivity. Primary-review classification and
confidence outcomes for the 121 candidates, and how domain concentration vs.
cross-domain entropy separates the reviewed classes. Reads only Phase 1 and
Phase 3 outputs (candidates.jsonl, primary_reviews.jsonl).
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "semantic_selectivity"
CAND_PATH = pc.DATA_DIR / "feature_analysis" / "candidates.jsonl"
REVIEW_PATH = pc.DATA_DIR / "validation_results" / "primary_reviews.jsonl"

CLASS_ORDER = ["coherent_semantic", "coherent_domain_specific", "polysemantic_or_mixed",
               "template_driven", "domain_marker", "script_or_language_marker",
               "rare_candidate", "uninterpretable", "insufficient_evidence"]
CONF_ORDER = ["high", "medium", "low", "unvalidated"]


def main():
    cand = pd.DataFrame(pc.load_jsonl(CAND_PATH))
    rev = pd.DataFrame(pc.load_jsonl(REVIEW_PATH))
    df = cand.merge(rev, on="feature_idx", how="inner")
    assert len(df) == 121, f"expected 121 merged candidates, got {len(df)}"

    # B1: final_classification counts (9-class taxonomy), primary review
    fig, ax = plt.subplots(figsize=(11, 6))
    counts = df["final_classification"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    ax.barh(CLASS_ORDER, counts.values, color=pc.PALETTE[2])
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=11)
    ax.set_xlabel("Number of candidates")
    ax.set_title(f"Primary-review final classification, 121-candidate cohort (n={len(df)})")
    fig.text(0.5, -0.05,
              "Caption: single primary-reviewer classification per candidate, 9-class taxonomy. "
              "insufficient_evidence is a first-class outcome here, never silently merged into "
              "not_reproduced or any other label.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "b1_classification_counts",
                "Primary-review final classification counts across the 121-candidate cohort.",
                [REVIEW_PATH], n=len(df))

    # B2: domain_concentration vs cross_domain_entropy_norm, colored by classification
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, cls in enumerate(CLASS_ORDER):
        sub = df[df["final_classification"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["cross_domain_entropy_norm"], sub["domain_concentration"],
                   label=f"{cls} (n={len(sub)})", color=pc.PALETTE[i % len(pc.PALETTE)], s=60, alpha=0.8)
    ax.set_xlabel("Cross-domain entropy (normalized, 0=single-domain, 1=uniform across 5 domains)")
    ax.set_ylabel("Domain concentration score (max_d p_d/b_d, 1.0=no concentration beyond base rate)")
    ax.set_yscale("log")
    ax.set_title("Candidate domain concentration vs. cross-domain entropy, by review classification")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.5, -0.07,
              f"Caption: n={len(df)} candidates, Discovery-subset metrics (B.1 formulas). High "
              "concentration + low entropy is expected and NOT penalized for domain-specific "
              "candidates per B.4's type-conditional confidence criteria.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "b2_concentration_vs_entropy_by_class",
                "Domain concentration vs. cross-domain entropy for all 121 candidates, colored by final classification.",
                [CAND_PATH, REVIEW_PATH], n=len(df))

    # B3: confidence level distribution
    fig, ax = plt.subplots(figsize=(9, 6))
    conf_counts = df["confidence"].value_counts().reindex(CONF_ORDER).fillna(0).astype(int)
    ax.bar(CONF_ORDER, conf_counts.values, color=pc.PALETTE[3])
    for i, v in enumerate(conf_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=12)
    ax.set_ylabel("Number of candidates")
    ax.set_title(f"Primary-review confidence levels, 121-candidate cohort (n={len(df)})")
    fig.text(0.5, -0.05,
              "Caption: 26 high + 35 medium = 61 candidates reach the >=medium threshold used "
              "to define the Phase 4 'eligible' relationship-analysis population.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "b3_confidence_level_counts",
                "Primary-review confidence level counts across the 121-candidate cohort.",
                [REVIEW_PATH], n=len(df))


if __name__ == "__main__":
    main()

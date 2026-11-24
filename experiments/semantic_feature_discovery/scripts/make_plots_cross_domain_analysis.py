"""
Phase 5, category E: cross_domain_analysis. (a) single-feature cross-domain
generalizers among the 61 eligible candidates, (b) the M3-seeded two-feature
parallel-concept hypotheses -- top-20 ranked, with the 1,107-raw-intersection
noise count shown explicitly as context, never as a discovery count. Reads
only Phase 4 output (cross_domain_analysis.json).
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "cross_domain_analysis"
CD_PATH = pc.SFD_DIR / "analysis" / "data" / "relationship_analysis" / "cross_domain_analysis.json"


def main():
    cd = pc.load_json(CD_PATH)
    single = pd.DataFrame(cd["single_feature_all_candidates_by_cross_domain_entropy"])
    top_h = pd.DataFrame(cd["m3_seeded_two_feature_hypotheses_top_ranked"])
    raw_count = cd["m3_seeded_two_feature_hypotheses_total_raw_count"]

    # E1: single-feature cross-domain entropy, all 61 eligible, highlight 5 coherent_semantic
    fig, ax = plt.subplots(figsize=(13, 6))
    is_gen = single["final_classification"] == "coherent_semantic"
    order = single.sort_values("cross_domain_entropy_norm_discovery", ascending=False)
    ax.bar(range(len(order)), order["cross_domain_entropy_norm_discovery"],
           color=[pc.PALETTE[3] if g else pc.PALETTE[8] for g in order["final_classification"] == "coherent_semantic"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["feature_idx"], rotation=90, fontsize=7)
    ax.set_xlabel("Feature index (eligible candidates, n=61)")
    ax.set_ylabel("Cross-domain entropy (normalized, Discovery subset)")
    ax.set_title(f"Single-feature cross-domain generalization, 61 eligible candidates\n"
                 f"({int(is_gen.sum())} independently reviewed coherent_semantic, highlighted)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=pc.PALETTE[3], label="coherent_semantic (n=5)"),
                        Patch(color=pc.PALETTE[8], label="other classification")], fontsize=10)
    fig.text(0.5, -0.18,
              "Caption: a single feature firing across multiple domains is NOT the same "
              "evidence as a two-feature cross-domain relationship (panel E2) -- this is a "
              "property of one feature, reviewed independently of entropy score.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "e1_single_feature_generalizers",
                "Cross-domain entropy for the 61 eligible candidates, with the 5 reviewer-confirmed coherent_semantic generalizers highlighted.",
                [CD_PATH], n=61)

    # E2: top-20 M3-seeded two-feature hypotheses, evidence_strength_score, with raw-count context
    fig, ax = plt.subplots(figsize=(12, 8))
    top_h_sorted = top_h.sort_values("evidence_strength_score", ascending=True)
    labels = [f"f{a}<->f{b}\n({ca}<->{cb})" for a, b, ca, cb in
              zip(top_h_sorted["feature_a"], top_h_sorted["feature_b"],
                  top_h_sorted["concept_a"], top_h_sorted["concept_b"])]
    ax.barh(range(len(top_h_sorted)), top_h_sorted["evidence_strength_score"], color=pc.PALETTE[4])
    ax.set_yticks(range(len(top_h_sorted)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Evidence-strength score (rate_a x rate_b x log(min concept-prompt count))")
    ax.set_title(f"Top {len(top_h)} M3-seeded two-feature parallel-concept hypotheses (of {raw_count} raw intersections)")
    fig.text(0.5, -0.12,
              f"Caption: {raw_count} raw (feature, feature, concept-pair) intersections passed "
              f"a lenient 5% firing-rate floor -- explicitly flagged as mostly noise from that "
              f"permissive floor, NOT {raw_count} discoveries. Only these top {len(top_h)}, every "
              "one labeled low_hypothesis_only confidence, are reported as candidate hypotheses; "
              "none is evidence the two features represent the same concept.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "e2_top_cross_domain_hypotheses",
                f"Top {len(top_h)} M3-seeded two-feature cross-domain hypotheses by evidence-strength score, out of {raw_count} raw (mostly noisy) intersections.",
                [CD_PATH], n=raw_count)


if __name__ == "__main__":
    main()

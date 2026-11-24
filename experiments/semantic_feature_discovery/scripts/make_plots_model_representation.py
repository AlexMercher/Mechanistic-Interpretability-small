"""
Phase 5, category G: model_representation. Overall funnel from the full
SAE-8192 dictionary down through candidate selection, review, and the
eligible relationship-analysis pool; the Confirmation-only coverage statistic
(B.9 limitation); and the corpus domain-token imbalance baseline the whole
Discovery subset is measured against. Reads only Phase 1-4 outputs.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "model_representation"
INV_PATH = pc.DATA_DIR / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl"
CAND_PATH = pc.DATA_DIR / "feature_analysis" / "candidates.jsonl"
REVIEW_PATH = pc.DATA_DIR / "validation_results" / "primary_reviews.jsonl"
ELIGIBLE_PATH = pc.SFD_DIR / "analysis" / "data" / "relationship_analysis" / "eligible_candidates.json"
COVERAGE_PATH = pc.DATA_DIR / "aggregate_statistics" / "confirmation_only_feature_coverage.json"
BASELINE_PATH = pc.DATA_DIR / "aggregate_statistics" / "discovery_domain_token_baseline.json"


def main():
    import pandas as pd
    inv = pd.DataFrame(pc.load_jsonl(INV_PATH))
    cand = pd.DataFrame(pc.load_jsonl(CAND_PATH))
    rev = pd.DataFrame(pc.load_jsonl(REVIEW_PATH))
    eligible = pc.load_json(ELIGIBLE_PATH)
    coverage = pc.load_json(COVERAGE_PATH)
    baseline = pc.load_json(BASELINE_PATH)

    n_total = len(inv)
    n_alive = int((~inv["is_dead_in_discovery"]).sum())
    n_candidates = len(cand)
    n_reviewed = len(rev)
    n_eligible = len(eligible["candidates"])
    n_coherent = int((rev["final_classification"].isin(["coherent_semantic", "coherent_domain_specific"])).sum())

    # G1: funnel from full dictionary to coherent, reviewed, validated features
    fig, ax = plt.subplots(figsize=(11, 7))
    stages = ["SAE-8192\nfull dictionary", "Alive in\nDiscovery", "Selected as\ncandidates",
              "Primary-reviewed\n(=candidates)", ">=medium confidence\n(eligible for Phase 4)",
              "coherent_semantic /\ncoherent_domain_specific"]
    values = [n_total, n_alive, n_candidates, n_reviewed, n_eligible, n_coherent]
    ax.barh(stages[::-1], values[::-1], color=pc.PALETTE[:len(stages)][::-1])
    for i, v in enumerate(values[::-1]):
        ax.text(v + 30, i, str(v), va="center", fontsize=11)
    ax.set_xlabel("Number of features")
    ax.set_title("Representation funnel: SAE-8192 seed 0, full dictionary -> reviewed-coherent features")
    fig.text(0.5, -0.08,
              f"Caption: only {n_candidates}/{n_total} features ({n_candidates/n_total:.1%}) were ever "
              f"reviewed at all -- this funnel describes the reviewed STRATIFIED SAMPLE, not an "
              f"exhaustive census; the {n_candidates/n_total:.1%} review rate must never be "
              f"extrapolated to claim {n_coherent/n_total:.1%} of all 8,192 features are coherent.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "g1_representation_funnel",
                "Funnel from the full SAE-8192 dictionary through candidate selection, review, and eligibility to reviewer-confirmed coherent features.",
                [INV_PATH, CAND_PATH, REVIEW_PATH, ELIGIBLE_PATH], n=n_total)

    # G2: Confirmation-only coverage statistic (B.9 limitation)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Dead in Discovery\n(ineligible for selection)", "Fires in Confirmation\nbut dead in Discovery\n(structurally missed)"]
    vals = [coverage["n_discovery_dead_features"] - coverage["n_confirmation_only_features"], coverage["n_confirmation_only_features"]]
    ax.bar(labels, vals, color=[pc.PALETTE[8], pc.PALETTE[3]])
    for i, v in enumerate(vals):
        ax.text(i, v + 20, str(v), ha="center", fontsize=12)
    ax.set_ylabel("Number of features")
    ax.set_title(f"Confirmation-only feature coverage (of {coverage['n_discovery_dead_features']} Discovery-dead features)")
    fig.text(0.5, -0.1,
              f"Caption: {coverage['n_confirmation_only_features']}/{coverage['n_discovery_dead_features']} "
              "Discovery-dead features fire at least once in Confirmation -- rare-feature "
              "discovery in this milestone is conditional on firing in Discovery; this is a "
              "documented coverage limitation (B.9), not a claim of exhaustive rare-feature coverage.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "g2_confirmation_only_coverage",
                "Confirmation-only feature coverage: Discovery-dead features that fire in the independent Confirmation subset.",
                [COVERAGE_PATH], n=coverage["n_discovery_dead_features"])

    # G3: corpus domain-token imbalance baseline (Discovery subset)
    fig, ax = plt.subplots(figsize=(9, 6))
    shares = baseline["domain_token_share"]
    doms = sorted(shares, key=lambda d: -shares[d])
    ax.bar(doms, [shares[d] for d in doms], color=pc.PALETTE[:len(doms)])
    for i, d in enumerate(doms):
        ax.text(i, shares[d] + 0.01, f"{shares[d]:.1%}", ha="center", fontsize=11)
    ax.axhline(0.2, color="gray", ls="--", lw=1, label="20% (even split across 5 domains)")
    ax.set_ylabel("Share of Discovery-subset tokens")
    ax.set_title(f"Discovery-subset domain-token baseline (n={baseline['n_discovery_tokens']} tokens, {baseline['n_discovery_prompts']} prompts)")
    ax.legend(fontsize=10)
    fig.text(0.5, -0.08,
              "Caption: this baseline (b_d in the B.1 domain-concentration formula) shows the "
              "corpus is not domain-balanced at the token level -- HINDI's longer native-script "
              "tokenization dominates token share despite equal per-domain prompt counts.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "g3_domain_token_baseline",
                "Discovery-subset domain-token share baseline used to normalize the domain-concentration score.",
                [BASELINE_PATH], n=baseline["n_discovery_tokens"])


if __name__ == "__main__":
    main()

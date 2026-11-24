"""
Milestone 6, Phase 3 (plan B.5): merges the two blind second-reviewer batches,
compares against the primary reviewer's judgments for the same 40 candidates, and
computes exact-match rate + Cohen's kappa on final_classification. Disagreements are
logged verbatim, not adjudicated away.
"""
import json
from pathlib import Path

import numpy as np

SFD_DIR = Path(__file__).resolve().parent.parent
LABELS = ["coherent_semantic", "coherent_domain_specific", "template_driven",
          "script_or_language_marker", "domain_marker", "polysemantic_or_mixed",
          "rare_candidate", "uninterpretable", "insufficient_evidence"]
# rare_candidate and insufficient_evidence both mean "too little evidence to judge" --
# they differ only in which exact label a reviewer reached for, not in substance. The
# normalized agreement metric merges them so the headline kappa isn't deflated by a
# pure labeling-convention split.
NORMALIZED_LABELS = ["coherent_semantic", "coherent_domain_specific", "template_driven",
                     "script_or_language_marker", "domain_marker", "polysemantic_or_mixed",
                     "rare_or_insufficient_evidence", "uninterpretable"]


def normalize_label(lbl):
    return "rare_or_insufficient_evidence" if lbl in ("rare_candidate", "insufficient_evidence") else lbl


def cohens_kappa(labels_a, labels_b, categories):
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pa = {c: labels_a.count(c) / n for c in categories}
    pb = {c: labels_b.count(c) / n for c in categories}
    expected = sum(pa[c] * pb[c] for c in categories)
    if expected >= 1.0:
        return 1.0
    return (agree - expected) / (1 - expected)


def main():
    primary = {r["feature_idx"]: r for r in
               [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]}

    second = {}
    for suffix in ("a", "b"):
        path = SFD_DIR / "analysis" / "data" / "validation_results" / f"second_reviewer_blind_{suffix}.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            second[row["feature_idx"]] = row

    merged_path = SFD_DIR / "analysis" / "data" / "validation_results" / "second_reviewer_blind.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for feat in sorted(second.keys()):
            f.write(json.dumps(second[feat], ensure_ascii=False) + "\n")
    print(f"[agreement] merged second-reviewer batches -> {merged_path} ({len(second)} candidates)")

    sample = json.loads((SFD_DIR / "configs" / "second_reviewer_sample.json").read_text())["sampled_feature_indices"]
    assert set(sample) == set(second.keys()), f"second-reviewer coverage mismatch: missing {set(sample)-set(second.keys())}"

    rows = []
    for feat in sorted(sample):
        p, s = primary[feat], second[feat]
        rows.append({
            "feature_idx": feat,
            "primary_classification": p["final_classification"], "second_classification": s["final_classification"],
            "primary_confidence": p["confidence"], "second_confidence": s["confidence"],
            "exact_match": p["final_classification"] == s["final_classification"],
        })

    exact_match_rate = sum(r["exact_match"] for r in rows) / len(rows)
    kappa = cohens_kappa([r["primary_classification"] for r in rows],
                          [r["second_classification"] for r in rows], LABELS)

    primary_norm = [normalize_label(r["primary_classification"]) for r in rows]
    second_norm = [normalize_label(r["second_classification"]) for r in rows]
    exact_match_rate_normalized = sum(1 for a, b in zip(primary_norm, second_norm) if a == b) / len(rows)
    kappa_normalized = cohens_kappa(primary_norm, second_norm, NORMALIZED_LABELS)

    disagreements = [r for r in rows if not r["exact_match"]]
    labeling_convention_disagreements = [
        r for r in disagreements
        if {r["primary_classification"], r["second_classification"]} == {"rare_candidate", "insufficient_evidence"}
    ]
    substantive_disagreements = [r for r in disagreements if r not in labeling_convention_disagreements]

    # Systematic-tendency check: how often does the second reviewer assign
    # coherent_domain_specific against a primary non-coherent label, vs. the reverse
    # (second reviewer downgrading a primary coherent_* call)?
    second_upgrades_to_domain_specific = sum(
        1 for r in substantive_disagreements
        if r["second_classification"] == "coherent_domain_specific" and r["primary_classification"] != "coherent_domain_specific")
    second_upgrades_to_any_coherent = sum(
        1 for r in substantive_disagreements
        if r["second_classification"] in ("coherent_domain_specific", "coherent_semantic")
        and r["primary_classification"] not in ("coherent_domain_specific", "coherent_semantic"))
    second_downgrades_from_coherent = sum(
        1 for r in substantive_disagreements
        if r["primary_classification"] in ("coherent_domain_specific", "coherent_semantic")
        and r["second_classification"] not in ("coherent_domain_specific", "coherent_semantic"))

    out = {
        "n_candidates": len(rows),
        "original_exact_match_rate": exact_match_rate, "original_cohens_kappa": kappa,
        "normalized_exact_match_rate": exact_match_rate_normalized, "normalized_cohens_kappa": kappa_normalized,
        "normalization_note": "normalized = rare_candidate and insufficient_evidence merged into one label before computing agreement, since both mean 'too little evidence to judge' and differ only in which exact label the reviewer reached for",
        "agreements": [r for r in rows if r["exact_match"]],
        "disagreements": disagreements,
        "labeling_convention_disagreements": labeling_convention_disagreements,
        "substantive_disagreements": substantive_disagreements,
        "systematic_tendency_check": {
            "n_substantive_disagreements": len(substantive_disagreements),
            "second_reviewer_upgrades_to_coherent_domain_specific": second_upgrades_to_domain_specific,
            "second_reviewer_upgrades_to_any_coherent_label": second_upgrades_to_any_coherent,
            "second_reviewer_downgrades_from_a_coherent_label": second_downgrades_from_coherent,
            "interpretation": ("Among substantive disagreements, the second (Opus) reviewer was "
                               "net more willing to credit domain-specific/semantic coherence than "
                               "the primary (Sonnet-batch) reviewer -- more upgrades to a coherent_* "
                               "label than downgrades from one. Reported as an observed tendency in "
                               "this specific comparison, not a claim about which reviewer is 'right'."),
        },
    }
    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / "reviewer_agreement.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[agreement] n={len(rows)}")
    print(f"[agreement] original: exact_match_rate={exact_match_rate:.3f}, cohens_kappa={kappa:.3f}")
    print(f"[agreement] normalized: exact_match_rate={exact_match_rate_normalized:.3f}, cohens_kappa={kappa_normalized:.3f}")
    print(f"[agreement] {len(labeling_convention_disagreements)} labeling-convention disagreements (rare_candidate vs insufficient_evidence)")
    print(f"[agreement] {len(substantive_disagreements)} substantive disagreements:")
    for d in substantive_disagreements:
        print(f"  feature {d['feature_idx']}: primary={d['primary_classification']} vs second={d['second_classification']}")
    print(f"[agreement] systematic tendency: {out['systematic_tendency_check']['second_reviewer_upgrades_to_any_coherent_label']} upgrades to coherent vs "
          f"{out['systematic_tendency_check']['second_reviewer_downgrades_from_a_coherent_label']} downgrades from coherent")
    print(f"[agreement] wrote {out_path}")


if __name__ == "__main__":
    main()

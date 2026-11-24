"""
Milestone 6, Phase 4 step 3 (plan B.3): cross-domain relationship analysis.

Two distinct kinds of cross-domain evidence, never conflated:
  (a) SINGLE-FEATURE cross-domain generalization: a candidate that itself fires across
      multiple domains (already surfaced in Phase 3 review + Phase 1 cross_domain_entropy).
      Summarized here from existing records, not recomputed.
  (b) TWO-FEATURE parallel-concept relationships: seeded from Milestone 3's
      teacher-reviewed cross-domain concept pairs (DNA<->SCIENCE, EMOTIONS<->HINDI,
      non-"no_relationship" only) -- for each such concept pair, check whether a
      domain-A-dominant candidate fires selectively on concept A's prompts AND a
      domain-B-dominant candidate fires selectively on concept B's prompts. This is
      NEVER proof that the two features represent "the same concept" -- it is reported
      as a hypothesis with explicit supporting evidence and a confidence level, using
      the Confirmation subset so the test is independent of whatever Discovery-subset
      pattern (if any) motivated selecting either candidate.
"""
import json
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
REL_DIR = SFD_DIR / "analysis" / "data" / "relationship_analysis"
M3_DIR = SFD_DIR.parent / "semantic_topology_25k"
DOMAIN_PAIRS = [("DNA", "SCIENCE"), ("EMOTIONS", "HINDI")]
MIN_CONCEPT_FIRING_RATE = 0.05  # candidate must fire on >=5% of a concept's own prompts
MIN_CONCEPT_PROMPTS = 10
TOP_N_HYPOTHESES_TO_REPORT = 20


def load_m3_cross_domain_pairs():
    hierarchy = json.loads((M3_DIR / "taxonomy" / "semantic_hierarchy_v1.json").read_text(encoding="utf-8"))
    concepts = hierarchy["concepts"]
    rels = [json.loads(l) for l in (M3_DIR / "taxonomy" / "concept_relationships.jsonl").open(encoding="utf-8")]
    out = []
    for r in rels:
        if r["relationship_type"] == "no_relationship":
            continue
        dp = tuple(sorted(r["domain_pair"]))
        if dp not in [tuple(sorted(p)) for p in DOMAIN_PAIRS]:
            continue
        ca, cb = concepts.get(r["cluster_a"]), concepts.get(r["cluster_b"])
        if ca is None or cb is None:
            continue
        out.append({
            "domain_pair": list(dp), "relationship_type": r["relationship_type"], "m3_confidence": r["confidence"],
            "cluster_a": r["cluster_a"], "cluster_b": r["cluster_b"],
            "prompt_ids_a": set(ca["prompt_ids"]), "prompt_ids_b": set(cb["prompt_ids"]),
            "evidence": r["evidence"][:200],
        })
    return out


def main():
    eligible = json.loads((REL_DIR / "eligible_candidates.json").read_text())["candidates"]
    reviews = {r["feature_idx"]: r for r in
               [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]}
    inventory = {r["feature_idx"]: r for r in
                 [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "sae_8192_seed0_inventory_discovery.jsonl").open(encoding="utf-8")]}

    # (a) single-feature cross-domain generalizers among the 61 eligible candidates.
    # cross_domain_entropy_norm was computed in select_candidates.py and lives in
    # candidates.jsonl, NOT in the raw per-feature inventory file -- using the wrong
    # source here previously made every row's entropy look missing and silently
    # emptied this entire section; fixed by reading candidates.jsonl directly.
    candidates_jsonl = {c["feature_idx"]: c for c in
                        [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]}
    single_feature = []
    for feat in eligible:
        rev = reviews.get(feat, {})
        cand = candidates_jsonl.get(feat, {})
        single_feature.append({
            "feature_idx": feat,
            "cross_domain_entropy_norm_discovery": cand.get("cross_domain_entropy_norm"),
            "final_classification": rev.get("final_classification"), "confidence": rev.get("confidence"),
            "candidate_interpretation": rev.get("candidate_interpretation"),
        })
    single_feature.sort(key=lambda r: -(r["cross_domain_entropy_norm_discovery"] or 0))
    single_feature_generalizers = [r for r in single_feature if r["final_classification"] == "coherent_semantic"]

    # (b) M3-seeded two-feature parallel-concept check, Confirmation subset only
    m3_pairs = load_m3_cross_domain_pairs()
    print(f"[cross_domain] {len(m3_pairs)} M3-reviewed non-no_relationship DNA<->SCIENCE / EMOTIONS<->HINDI concept pairs")

    prompt_df = pd.read_pickle(REL_DIR / "prompt_activation_matrix_confirmation.pkl")
    feat_cols = {c: f"f{c}" for c in eligible}

    two_feature_hypotheses = []
    for mp in m3_pairs:
        domain_a, domain_b = mp["domain_pair"]
        conf_prompts_a = set(prompt_df.index) & mp["prompt_ids_a"]
        conf_prompts_b = set(prompt_df.index) & mp["prompt_ids_b"]
        if len(conf_prompts_a) < MIN_CONCEPT_PROMPTS or len(conf_prompts_b) < MIN_CONCEPT_PROMPTS:
            continue

        cand_a_domain = [c for c in eligible if candidates_jsonl.get(c, {}).get("domain_concentration") and
                         max(inventory[c].get("domain_firing_counts", {}) or {}, key=lambda d: inventory[c]["domain_firing_counts"].get(d, 0), default=None) == domain_a]
        cand_b_domain = [c for c in eligible if candidates_jsonl.get(c, {}).get("domain_concentration") and
                         max(inventory[c].get("domain_firing_counts", {}) or {}, key=lambda d: inventory[c]["domain_firing_counts"].get(d, 0), default=None) == domain_b]

        for ca in cand_a_domain:
            col_a = prompt_df.loc[list(conf_prompts_a), feat_cols[ca]]
            rate_a = (col_a > 0).mean()
            if rate_a < MIN_CONCEPT_FIRING_RATE:
                continue
            for cb in cand_b_domain:
                col_b = prompt_df.loc[list(conf_prompts_b), feat_cols[cb]]
                rate_b = (col_b > 0).mean()
                if rate_b < MIN_CONCEPT_FIRING_RATE:
                    continue
                two_feature_hypotheses.append({
                    "feature_a": ca, "feature_a_domain": domain_a, "concept_a": mp["cluster_a"],
                    "feature_a_firing_rate_on_concept_a_confirmation": float(rate_a), "n_concept_a_confirmation_prompts": len(conf_prompts_a),
                    "feature_b": cb, "feature_b_domain": domain_b, "concept_b": mp["cluster_b"],
                    "feature_b_firing_rate_on_concept_b_confirmation": float(rate_b), "n_concept_b_confirmation_prompts": len(conf_prompts_b),
                    "m3_relationship_type": mp["relationship_type"], "m3_confidence": mp["m3_confidence"], "m3_evidence_excerpt": mp["evidence"],
                    "confidence": "low_hypothesis_only",
                    "caveat": ("This is NOT evidence the two features represent the same concept -- it only shows "
                               "each feature fires at above-floor rate on its OWN domain's half of an M3-reviewed "
                               "cross-domain concept pair. No claim of shared representation or causal link is made."),
                })

    # Rank by combined evidence strength rather than dump every weak intersection --
    # with a lenient per-feature firing-rate floor (5%) and ~26 concept pairs x dozens
    # of domain-matched candidates on each side, the raw combinatorial count (observed:
    # over a thousand) is mostly noise, not findings. Report the strongest N plus the
    # raw count as explicit context, not as a list of "discoveries."
    import math
    for h in two_feature_hypotheses:
        h["evidence_strength_score"] = (h["feature_a_firing_rate_on_concept_a_confirmation"] *
                                         h["feature_b_firing_rate_on_concept_b_confirmation"] *
                                         math.log(1 + min(h["n_concept_a_confirmation_prompts"], h["n_concept_b_confirmation_prompts"])))
    two_feature_hypotheses.sort(key=lambda h: -h["evidence_strength_score"])
    top_hypotheses = two_feature_hypotheses[:TOP_N_HYPOTHESES_TO_REPORT]

    out = {
        "eligible_candidates": eligible,
        "single_feature_cross_domain_generalizers_coherent_semantic": single_feature_generalizers,
        "single_feature_all_candidates_by_cross_domain_entropy": single_feature,
        "m3_seeded_two_feature_hypotheses_total_raw_count": len(two_feature_hypotheses),
        "m3_seeded_two_feature_hypotheses_top_ranked": top_hypotheses,
        "raw_count_caveat": (f"{len(two_feature_hypotheses)} raw (feature_a, feature_b, concept_pair) "
                              f"intersections passed the lenient {MIN_CONCEPT_FIRING_RATE:.0%} per-feature "
                              f"firing-rate floor -- this large count reflects how permissive that floor is "
                              f"combined with ~26 concept pairs x many domain-matched candidates on each "
                              f"side, NOT {len(two_feature_hypotheses)} genuine discoveries. Only the top "
                              f"{TOP_N_HYPOTHESES_TO_REPORT}, ranked by combined evidence strength "
                              f"(rate_a x rate_b x log(min concept-prompt count)), are reported as candidate "
                              f"hypotheses; even these are explicitly labeled low-confidence, hypothesis-only."),
        "methodology_note": ("(a) and (b) are different kinds of evidence and must not be "
                              "merged: (a) is about ONE feature firing across multiple domains; "
                              "(b) is about TWO DIFFERENT features, each domain-specific, "
                              "potentially covering 'parallel' content in an M3-reviewed "
                              "concept pair -- a weak, hypothesis-only signal, never upgraded "
                              "to a validated relationship without further causal evidence."),
    }
    out_path = REL_DIR / "cross_domain_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cross_domain] {len(single_feature_generalizers)} single-feature coherent_semantic cross-domain generalizers")
    print(f"[cross_domain] {len(two_feature_hypotheses)} raw two-feature intersections; reporting top {len(top_hypotheses)} by evidence strength")
    print(f"[cross_domain] wrote {out_path}")


if __name__ == "__main__":
    main()

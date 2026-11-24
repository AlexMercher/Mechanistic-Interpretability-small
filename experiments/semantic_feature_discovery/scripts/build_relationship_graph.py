"""
Milestone 6, Phase 4 step 4 (plan B.4): typed feature-relationship graph.

Nodes: the 61 eligible (>=medium confidence) candidates only -- not all 8192 features,
per the explicit "don't render an unreadable graph" constraint. Node metadata: feature
index, width/seed, primary review interpretation/classification/confidence, selection
stratum.

Edges are typed and evidence-backed; no edge is asserted without a stated rule, the
raw metric values behind it, and a confidence level. Every edge also carries a
provenance field naming the exact script/analysis that produced it, so a reader can
trace any claim back to raw data. No edge type here claims causation -- co-activation,
correlation, decoder similarity, and semantic similarity remain distinct evidence types
throughout (never collapsed into one score), per the explicit requirement.
"""
import json
from pathlib import Path

import networkx as nx
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent
REL_DIR = SFD_DIR / "analysis" / "data" / "relationship_analysis"

MERGE_DECODER_THR, MERGE_ACT_THR, MERGE_JACCARD_THR = 0.5, 0.5, 0.3
SIM_CORR_THR = 0.3
MUTUAL_EXCLUSION_THR = -0.3
HIER_CONTAINMENT_THR = 0.8
HIER_RATIO_THR = 2.0


def main():
    eligible = json.loads((REL_DIR / "eligible_candidates.json").read_text())["candidates"]
    reviews = {r["feature_idx"]: r for r in
               [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "primary_reviews.jsonl").open(encoding="utf-8")]}
    candidates_jsonl = {c["feature_idx"]: c for c in
                        [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "feature_analysis" / "candidates.jsonl").open(encoding="utf-8")]}
    decoder_rel = json.loads((REL_DIR / "decoder_relationship_analysis.json").read_text(encoding="utf-8"))
    coact = json.loads((REL_DIR / "coactivation_analysis.json").read_text(encoding="utf-8"))
    cross_dom = json.loads((REL_DIR / "cross_domain_analysis.json").read_text(encoding="utf-8"))

    G = nx.Graph()
    for feat in eligible:
        rev = reviews.get(feat, {})
        cand = candidates_jsonl.get(feat, {})
        G.add_node(feat, feature_idx=feat, dictionary_width=8192, seed=0,
                   final_classification=rev.get("final_classification"), confidence=rev.get("confidence"),
                   candidate_interpretation=rev.get("candidate_interpretation"),
                   primary_stratum=cand.get("primary_stratum"))

    pair_lookup = {(p["feature_a"], p["feature_b"]): p for p in decoder_rel["pairs"]}

    # possible_merge: strong agreement across decoder + activation-pattern + top-prompt overlap
    for (a, b), p in pair_lookup.items():
        if p["decoder_cosine"] >= MERGE_DECODER_THR and p["activation_pattern_cosine"] >= MERGE_ACT_THR and p["top_prompt_jaccard"] >= MERGE_JACCARD_THR:
            G.add_edge(a, b, edge_type="possible_merge", confidence="medium",
                       evidence={"decoder_cosine": p["decoder_cosine"], "activation_pattern_cosine": p["activation_pattern_cosine"],
                                 "top_prompt_jaccard": p["top_prompt_jaccard"]},
                       provenance="decoder_relationship_analysis.py",
                       note="Strong agreement across decoder direction, activation pattern, AND top-prompt overlap -- candidates may be redundant/near-duplicate directions in this dictionary, not necessarily two independent concepts.")

    # shared_surface_form: both template_driven/domain_marker AND decoder+activation agree
    for (a, b), p in pair_lookup.items():
        ca, cb = reviews.get(a, {}).get("final_classification"), reviews.get(b, {}).get("final_classification")
        if ca in ("template_driven", "domain_marker") and cb in ("template_driven", "domain_marker") and \
           p["decoder_cosine"] >= 0.3 and p["activation_pattern_cosine"] >= 0.3 and not G.has_edge(a, b):
            G.add_edge(a, b, edge_type="shared_surface_form", confidence="medium",
                       evidence={"decoder_cosine": p["decoder_cosine"], "activation_pattern_cosine": p["activation_pattern_cosine"],
                                 "classifications": [ca, cb]},
                       provenance="decoder_relationship_analysis.py + primary_reviews.jsonl",
                       note="Both candidates were independently classified as template/domain artifacts AND share decoder+activation similarity -- likely locked to overlapping surface patterns, not independent semantic features.")

    # semantic_similarity: both coherent (any flavor) and controlled correlation holds in >=1 domain stratum
    coact_pairs = {}
    for key, v in coact["controlled_results"].items():
        a, b = key.split("__")
        a, b = int(a[1:]), int(b[1:])
        coact_pairs[(a, b)] = v
    for (a, b), v in coact_pairs.items():
        ca, cb = reviews.get(a, {}).get("final_classification"), reviews.get(b, {}).get("final_classification")
        if ca not in ("coherent_semantic", "coherent_domain_specific") or cb not in ("coherent_semantic", "coherent_domain_specific"):
            continue
        strata = [s for s in v["confirmation_controlled"] if s.get("pearson") is not None and abs(s["pearson"]) >= SIM_CORR_THR]
        if strata and not G.has_edge(a, b):
            best = max(strata, key=lambda s: abs(s["pearson"]))
            G.add_edge(a, b, edge_type="semantic_similarity", confidence="low_to_medium",
                       evidence={"domain": best["domain"], "level": best["level"], "n_prompts": best["n_prompts"],
                                 "pearson": best["pearson"], "n_qualifying_domains": len(strata)},
                       provenance="coactivation_analysis.py (Confirmation subset, controlled)",
                       note="Both candidates independently reviewed as coherent; their Confirmation-subset, confound-controlled prompt-level activations correlate in at least one domain stratum. Correlational evidence only -- not a causal or 'same concept' claim.")
        elif not strata and not G.has_edge(a, b):
            pass  # correlation did not survive conditioning -- explicitly NOT asserted as a relationship

    # co_activation: generic statistical edge for any pooled-selected pair not already typed above
    for key in coact["selected_pairs_for_controlled_analysis"]:
        a, b = key.split("__")
        a, b = int(a[1:]), int(b[1:])
        if G.has_edge(a, b):
            continue
        pooled = coact["pooled_correlations"]["confirmation_prompt"].get(key, {})
        G.add_edge(a, b, edge_type="co_activation", confidence="low",
                   evidence={"pooled_confirmation_prompt_pearson": pooled.get("pearson"), "n": pooled.get("n")},
                   provenance="coactivation_analysis.py (pooled, Confirmation subset)",
                   note="Statistical co-activation above the screening threshold; no claim about why (shared template/domain/script confound is plausible and not ruled out) -- see confound-controlled breakdown in coactivation_analysis.json for this pair.")

    # mutual_exclusion: strongly negative controlled correlation in a majority of qualifying domain strata
    for (a, b), v in coact_pairs.items():
        strata = [s for s in v["confirmation_controlled"] if s.get("pearson") is not None]
        if not strata:
            continue
        neg = [s for s in strata if s["pearson"] <= MUTUAL_EXCLUSION_THR]
        if len(neg) > len(strata) / 2 and not G.has_edge(a, b):
            G.add_edge(a, b, edge_type="mutual_exclusion", confidence="low",
                       evidence={"n_negative_strata": len(neg), "n_total_strata": len(strata),
                                 "example_domain": neg[0]["domain"], "example_pearson": neg[0]["pearson"]},
                       provenance="coactivation_analysis.py (Confirmation subset, controlled)",
                       note="Negative correlation in a majority of qualifying domain strata -- candidates tend not to co-fire. Not evidence of a causal exclusion mechanism.")

    # cross_domain_connection: top-ranked M3-seeded hypotheses (both features must be graph nodes)
    for h in cross_dom["m3_seeded_two_feature_hypotheses_top_ranked"]:
        a, b = h["feature_a"], h["feature_b"]
        if a not in G or b not in G or G.has_edge(a, b):
            continue
        G.add_edge(a, b, edge_type="cross_domain_connection", confidence="low_hypothesis_only",
                   evidence={"concept_a": h["concept_a"], "concept_b": h["concept_b"],
                             "rate_a": h["feature_a_firing_rate_on_concept_a_confirmation"],
                             "rate_b": h["feature_b_firing_rate_on_concept_b_confirmation"],
                             "m3_relationship_type": h["m3_relationship_type"]},
                   provenance="cross_domain_analysis.py (M3-seeded, Confirmation subset)",
                   note=h["caveat"])

    # Export: node-link JSON (reproducible) + summary
    data = nx.node_link_data(G, edges="edges")
    out_path = REL_DIR / "relationship_graph.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    edge_types = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    summary = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "edge_type_counts": dict(edge_types)}
    (REL_DIR / "relationship_graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[graph] {summary}")
    print(f"[graph] wrote {out_path}")


if __name__ == "__main__":
    main()

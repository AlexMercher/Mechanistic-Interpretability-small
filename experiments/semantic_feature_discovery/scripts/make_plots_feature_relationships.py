"""
Phase 5, category F: feature_relationships. The typed relationship graph
(61 nodes / 139 edges), edge-type counts including the three legitimate
zero-count types, and the decoder-vs-functional/lexical disagreement scatter
(37/1830 pairs). Reads only Phase 4 outputs; renders the graph exported by
build_relationship_graph.py without recomputing anything.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402

CAT = "feature_relationships"
REL_DIR = pc.SFD_DIR / "analysis" / "data" / "relationship_analysis"
GRAPH_PATH = REL_DIR / "relationship_graph.json"
SUMMARY_PATH = REL_DIR / "relationship_graph_summary.json"
DECODER_PATH = REL_DIR / "decoder_relationship_analysis.json"

ALL_EDGE_TYPES = ["co_activation", "semantic_similarity", "cross_domain_connection",
                  "possible_merge", "shared_surface_form", "mutual_exclusion"]
EDGE_COLORS = dict(zip(ALL_EDGE_TYPES, pc.PALETTE[:len(ALL_EDGE_TYPES)]))
CLASS_COLORS = {
    "coherent_semantic": "#2ca02c", "coherent_domain_specific": "#1f77b4",
    "polysemantic_or_mixed": "#9467bd", "template_driven": "#ff7f0e",
    "domain_marker": "#8c564b", "script_or_language_marker": "#e377c2",
    "rare_candidate": "#7f7f7f", "uninterpretable": "#bcbd22", "insufficient_evidence": "#d62728",
}


def main():
    data = pc.load_json(GRAPH_PATH)
    summary = pc.load_json(SUMMARY_PATH)
    G = nx.node_link_graph(data, edges="edges")

    # F1: full typed graph layout
    fig, ax = plt.subplots(figsize=(15, 13))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    node_colors = [CLASS_COLORS.get(G.nodes[n].get("final_classification"), "#cccccc") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=220, ax=ax, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    for etype in ALL_EDGE_TYPES:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == etype]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[EDGE_COLORS[etype]] * len(edges),
                                    width=1.3, alpha=0.6, ax=ax, label=etype)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    edge_handles = [Line2D([0], [0], color=EDGE_COLORS[e], lw=2,
                            label=f"{e} (n={sum(1 for _,_,d in G.edges(data=True) if d['edge_type']==e)})")
                     for e in ALL_EDGE_TYPES]
    node_handles = [Patch(color=c, label=cls) for cls, c in CLASS_COLORS.items()
                    if cls in {G.nodes[n].get("final_classification") for n in G.nodes}]
    ax.legend(handles=edge_handles + node_handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Typed feature-relationship graph: {summary['n_nodes']} nodes (>=medium-confidence candidates), "
                 f"{summary['n_edges']} edges")
    ax.axis("off")
    fig.text(0.5, 0.0,
              "Caption: node color = primary-review final classification; edge color = relationship "
              "type. No edge implies causation. possible_merge / shared_surface_form / "
              "mutual_exclusion found zero qualifying pairs in this pool (not omitted from the schema).",
              ha="center", fontsize=10, wrap=True)
    pc.save_fig(fig, CAT, "f1_relationship_graph",
                "Typed feature-relationship graph: 61 eligible candidates, 139 evidence-backed edges across 3 edge types with activity (co_activation, semantic_similarity, cross_domain_connection).",
                [GRAPH_PATH], n=summary["n_nodes"])

    # F2: edge type counts, including the zero-count types explicitly
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = [summary["edge_type_counts"].get(e, 0) for e in ALL_EDGE_TYPES]
    ax.bar(ALL_EDGE_TYPES, counts, color=[EDGE_COLORS[e] for e in ALL_EDGE_TYPES])
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_ylabel("Number of edges")
    ax.set_title(f"Relationship-graph edge-type counts (total={summary['n_edges']} edges, {summary['n_nodes']} nodes)")
    fig.text(0.5, -0.1,
              "Caption: possible_merge, shared_surface_form, and mutual_exclusion are reported "
              "as explicit negative findings -- the strict pre-registered thresholds for these "
              "three types found zero qualifying pairs among the 61 eligible candidates.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "f2_edge_type_counts",
                "Edge-type counts in the relationship graph, including the three zero-count edge types as explicit negative findings.",
                [SUMMARY_PATH], n=summary["n_edges"])

    # F3: decoder cosine vs activation-pattern cosine scatter, 1830 pairs, disagreements highlighted
    dec = pc.load_json(DECODER_PATH)
    pairs = pd.DataFrame(dec["pairs"])
    disagree_keys = {(d["feature_a"], d["feature_b"]) for d in dec["disagreements"]}
    pairs["is_disagreement"] = list(zip(pairs["feature_a"], pairs["feature_b"]))
    pairs["is_disagreement"] = pairs["is_disagreement"].isin(disagree_keys)
    fig, ax = plt.subplots(figsize=(10, 8))
    normal = pairs[~pairs["is_disagreement"]]
    dis = pairs[pairs["is_disagreement"]]
    ax.scatter(normal["decoder_cosine"], normal["activation_pattern_cosine"],
               s=14, alpha=0.3, color=pc.PALETTE[8], label=f"agreement (n={len(normal)})")
    ax.scatter(dis["decoder_cosine"], dis["activation_pattern_cosine"],
               s=50, alpha=0.9, color=pc.PALETTE[3], marker="D", label=f"disagreement (n={len(dis)})")
    ax.axvline(dec["thresholds"]["decoder_high"], color="gray", ls="--", lw=1)
    ax.axhline(dec["thresholds"]["activation_high"], color="gray", ls="--", lw=1)
    ax.set_xlabel("Decoder-direction cosine similarity (geometric, no activation data)")
    ax.set_ylabel("Activation-pattern cosine similarity (functional, Confirmation subset)")
    ax.set_title(f"Decoder similarity vs. activation-pattern similarity, all {len(pairs)} eligible-candidate pairs")
    ax.legend(fontsize=10)
    fig.text(0.5, -0.07,
              f"Caption: {len(dis)}/{len(pairs)} pairs ({len(dis)/len(pairs):.1%}) disagree between "
              "decoder geometry and functional/lexical similarity -- decoder cosine is used only "
              "as a diagnostic signal, never a semantic verdict on its own.",
              ha="center", fontsize=11, wrap=True)
    pc.save_fig(fig, CAT, "f3_decoder_vs_activation_disagreements",
                "Decoder-direction cosine vs. activation-pattern cosine for all 1,830 eligible-candidate pairs, with the 37 decoder/functional disagreements highlighted.",
                [DECODER_PATH], n=len(pairs))


if __name__ == "__main__":
    main()

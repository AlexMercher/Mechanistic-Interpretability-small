"""
Phase 5 step 2: independent integrity check. Re-derives/cross-checks every
headline number against the raw output files rather than trusting prior
phase reports' prose. Writes analysis/data/aggregate_statistics/
integrity_report.json and prints a pass/fail summary. Reads only existing
outputs; modifies nothing (and verifies nothing outside this milestone's own
directory was modified either).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting_common as pc  # noqa: E402  (reused only for load_json/load_jsonl + paths)

SFD_DIR = pc.SFD_DIR
ROOT = SFD_DIR.parent.parent
DATA_DIR = pc.DATA_DIR

checks = []


def check(name, condition, detail):
    checks.append({"name": name, "passed": bool(condition), "detail": detail})
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")


def main():
    # 1. 121 candidates accounted for, across every artifact that should reference the cohort.
    cand = pc.load_jsonl(DATA_DIR / "feature_analysis" / "candidates.jsonl")
    cand_ids = {c["feature_idx"] for c in cand}
    check("candidates_count", len(cand) == 121 and len(cand_ids) == 121,
          f"candidates.jsonl has {len(cand)} rows, {len(cand_ids)} unique feature_idx")

    reviews = pc.load_jsonl(DATA_DIR / "validation_results" / "primary_reviews.jsonl")
    review_ids = {r["feature_idx"] for r in reviews}
    check("primary_reviews_cover_all_candidates", review_ids == cand_ids,
          f"primary_reviews.jsonl covers {len(review_ids)} features; "
          f"missing={sorted(cand_ids - review_ids)}, extra={sorted(review_ids - cand_ids)}")

    confound = pc.load_jsonl(DATA_DIR / "feature_analysis" / "confound_diagnostics.jsonl")
    check("confound_diagnostics_cover_all_candidates", {c["feature_idx"] for c in confound} == cand_ids,
          f"confound_diagnostics.jsonl covers {len(confound)} features")

    counterex = pc.load_jsonl(DATA_DIR / "feature_analysis" / "counterexamples.jsonl")
    counterex_ids = {c["feature_idx"] for c in counterex}
    check("counterexamples_cover_all_candidates", counterex_ids == cand_ids,
          f"counterexamples.jsonl covers {len(counterex_ids)} features")

    # 2. cross-seed reconciliation: every classification cell sums correctly, no double counting.
    recon = pc.load_json(SFD_DIR / "analysis" / "data" / "validation_results" / "cross_seed_capacity_reconciliation.json")
    csm = recon["cross_seed_matching"]
    for seed in ("seed_1", "seed_2"):
        classes = ["reproduced_strong", "numerically_matched_semantically_inconsistent", "possible_split",
                   "possible_merge", "not_reproduced", "insufficient_evidence"]
        total = sum(csm[seed][c] for c in classes)
        check(f"cross_seed_{seed}_sums_to_121", total == 121 == csm[seed]["sum_check"],
              f"{seed} six-way classification sums to {total} (recorded sum_check={csm[seed]['sum_check']})")
    check("cross_seed_denominator_is_121", csm["denominator"] == 121, f"denominator={csm['denominator']}")

    ccm = recon["cross_capacity_matching"]
    for width in ("width_4096", "width_16384"):
        classes = ["reproduced_strong", "numerically_matched_semantically_inconsistent", "possible_split",
                   "possible_merge", "not_reproduced", "insufficient_evidence"]
        total = sum(ccm[width][c] for c in classes)
        check(f"cross_capacity_{width}_sums_to_61", total == 61 == ccm[width]["sum_check"],
              f"{width} six-way classification sums to {total} (recorded sum_check={ccm[width]['sum_check']})")
    check("cross_capacity_denominator_is_61_not_121", ccm["denominator"] == 61,
          f"denominator={ccm['denominator']} (must be the >=medium-confidence subset, not the full 121)")

    # cross-check cross-seed raw results file directly against the 121-candidate cohort (independent of recon's own prose)
    cs_raw = pc.load_json(SFD_DIR / "analysis" / "data" / "validation_results" / "cross_seed_reproducibility.json")
    cs_feature_ids = {int(k) for k in cs_raw["results"].keys()}
    check("cross_seed_raw_results_cover_all_121", cs_feature_ids == cand_ids,
          f"cross_seed_reproducibility.json results cover {len(cs_feature_ids)} features")
    ins1 = {int(k) for k, v in cs_raw["results"].items() if v["seed_1"]["classification"] == "insufficient_evidence"}
    ins2 = {int(k) for k, v in cs_raw["results"].items() if v["seed_2"]["classification"] == "insufficient_evidence"}
    check("insufficient_evidence_identical_across_seeds", ins1 == ins2 and len(ins1) == 25,
          f"seed_1 insufficient_evidence set == seed_2 set: {ins1 == ins2}; |set|={len(ins1)}")

    # eligible candidates: exactly the >=medium-confidence subset of the 121, count 61.
    eligible = pc.load_json(SFD_DIR / "analysis" / "data" / "relationship_analysis" / "eligible_candidates.json")
    eligible_ids = set(eligible["candidates"])
    expected_eligible = {r["feature_idx"] for r in reviews if r["confidence"] in ("high", "medium")}
    check("eligible_candidates_is_exactly_medium_plus_high", eligible_ids == expected_eligible and len(eligible_ids) == 61,
          f"eligible_candidates.json has {len(eligible_ids)} features; "
          f"recomputed >=medium set from primary_reviews.jsonl has {len(expected_eligible)}; equal={eligible_ids == expected_eligible}")

    # 3. graph edges: 139 edges, all endpoints valid, all required fields present, types from the fixed set.
    graph = pc.load_json(SFD_DIR / "analysis" / "data" / "relationship_analysis" / "relationship_graph.json")
    node_ids = {n["id"] for n in graph["nodes"]}
    check("graph_has_61_nodes_matching_eligible", node_ids == eligible_ids,
          f"graph has {len(node_ids)} nodes; matches eligible_candidates set: {node_ids == eligible_ids}")

    valid_types = {"co_activation", "semantic_similarity", "cross_domain_connection",
                   "possible_merge", "shared_surface_form", "mutual_exclusion"}
    edges = graph["edges"]
    bad_endpoints = [e for e in edges if e["source"] not in node_ids or e["target"] not in node_ids]
    bad_types = [e for e in edges if e.get("edge_type") not in valid_types]
    missing_fields = [e for e in edges if not all(k in e for k in ("evidence", "confidence", "provenance"))]
    check("graph_has_139_edges", len(edges) == 139, f"graph has {len(edges)} edges")
    check("graph_edges_endpoints_valid", len(bad_endpoints) == 0, f"{len(bad_endpoints)} edges reference a non-node endpoint")
    check("graph_edges_types_valid", len(bad_types) == 0, f"{len(bad_types)} edges have an unrecognized edge_type")
    check("graph_edges_have_required_fields", len(missing_fields) == 0,
          f"{len(missing_fields)} edges missing evidence/confidence/provenance")

    from collections import Counter
    etype_counts = Counter(e["edge_type"] for e in edges)
    summary = pc.load_json(SFD_DIR / "analysis" / "data" / "relationship_analysis" / "relationship_graph_summary.json")
    check("graph_summary_matches_recount", dict(etype_counts) == summary["edge_type_counts"],
          f"recounted edge types {dict(etype_counts)} vs summary file {summary['edge_type_counts']}")

    # 4. figures trace to real, existing source files.
    manifest = pc.load_json(SFD_DIR / "analysis" / "plots" / "figure_manifest.json")
    missing_sources, missing_outputs = [], []
    for fig in manifest:
        for src in fig["source_files"]:
            if not Path(src).exists():
                missing_sources.append((fig["name"], src))
        for out_key in ("png", "svg"):
            if not (SFD_DIR / fig[out_key]).exists():
                missing_outputs.append((fig["name"], fig[out_key]))
    check("all_figures_reference_existing_source_files", len(missing_sources) == 0,
          f"{len(missing_sources)} missing source references: {missing_sources[:5]}")
    check("all_figure_output_files_exist", len(missing_outputs) == 0,
          f"{len(missing_outputs)} missing output files: {missing_outputs[:5]}")
    check("figure_manifest_has_21_entries", len(manifest) == 21, f"manifest has {len(manifest)} entries")

    # 5. Discovery/Confirmation separation throughout.
    split_cfg = pc.load_json(SFD_DIR / "configs" / "discovery_confirmation_split_config.json")
    assignment = pc.load_jsonl(SFD_DIR / "configs" / "discovery_confirmation_assignment.jsonl")
    subset_of = {r["prompt_id"]: r["subset"] for r in assignment}
    check("split_assignment_has_no_duplicate_prompt_ids", len(subset_of) == len(assignment),
          f"{len(assignment)} rows, {len(subset_of)} unique prompt_ids")
    check("split_assignment_only_two_subsets", set(subset_of.values()) <= {"discovery", "confirmation"},
          f"subset values observed: {set(subset_of.values())}")
    n_disc = sum(1 for v in subset_of.values() if v == "discovery")
    n_conf = sum(1 for v in subset_of.values() if v == "confirmation")
    check("split_counts_match_config", (n_disc, n_conf) == (1265, 1268),
          f"recounted discovery={n_disc}, confirmation={n_conf} (expected 1265/1268 per split_config)")
    check("split_config_records_same_counts",
          split_cfg.get("total_discovery_prompts") == 1265 and split_cfg.get("total_confirmation_prompts") == 1268,
          f"split_config total_discovery_prompts/total_confirmation_prompts = "
          f"{split_cfg.get('total_discovery_prompts')}/{split_cfg.get('total_confirmation_prompts')}")

    # evidence files tagged with the correct subset, never mixed
    ev_disc = pc.load_jsonl(DATA_DIR / "feature_analysis" / "candidate_evidence_discovery.jsonl")
    ev_conf = pc.load_jsonl(DATA_DIR / "feature_analysis" / "candidate_evidence_confirmation.jsonl")
    disc_prompt_ids = {pid for pid, s in subset_of.items() if s == "discovery"}
    conf_prompt_ids = {pid for pid, s in subset_of.items() if s == "confirmation"}

    def examples_of(rows):
        for r in rows:
            examples = r.get("examples", {})
            buckets = examples.values() if isinstance(examples, dict) else [examples]
            for bucket in buckets:
                for ex in bucket:
                    yield ex.get("prompt_id")

    disc_leak = [pid for pid in examples_of(ev_disc) if pid is not None and pid not in disc_prompt_ids]
    conf_leak = [pid for pid in examples_of(ev_conf) if pid is not None and pid not in conf_prompt_ids]
    check("discovery_evidence_contains_only_discovery_prompts", len(disc_leak) == 0,
          f"{len(disc_leak)} Discovery-evidence examples reference a non-Discovery prompt_id")
    check("confirmation_evidence_contains_only_confirmation_prompts", len(conf_leak) == 0,
          f"{len(conf_leak)} Confirmation-evidence examples reference a non-Confirmation prompt_id")

    counterex_leak = [ce["prompt_id"] for row in counterex for ce in row.get("counterexamples", [])
                       if ce.get("prompt_id") is not None and ce.get("prompt_id") not in conf_prompt_ids]
    check("counterexamples_contain_only_confirmation_prompts", len(counterex_leak) == 0,
          f"{len(counterex_leak)} counterexample prompt_ids are not in the Confirmation subset "
          "(seed queries may legitimately come from Discovery, but returned evidence must be Confirmation-only)")

    # 6. no source/taxonomy/activation/M5 artifact modified -- git-diff check against frozen dirs.
    frozen_paths = ["raw", "normalized", "processed",
                    "experiments/activation_extraction_qwen3_1_7b",
                    "experiments/sae_capacity_comparison",
                    "experiments/semantic_topology_25k",
                    "experiments/duplicate_language_embedding",
                    "experiments/corpus_validation_25k"]
    diff_out = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"] + frozen_paths,
                               capture_output=True, text=True)
    check("frozen_milestone_dirs_unmodified", diff_out.returncode == 0 and diff_out.stdout.strip() == "",
          f"git status --porcelain on frozen dirs: {diff_out.stdout.strip() or '(clean)'}")

    # 7. required Phase 1-5 outputs exist.
    required = [
        "configs/candidate_selection_criteria.json", "configs/template_clustering_config.json",
        "configs/discovery_confirmation_split_config.json", "configs/discovery_confirmation_assignment.jsonl",
        "configs/second_reviewer_sample.json",
        "analysis/data/feature_analysis/sae_8192_seed0_inventory_discovery.jsonl",
        "analysis/data/feature_analysis/candidates.jsonl",
        "analysis/data/feature_analysis/candidate_evidence_discovery.jsonl",
        "analysis/data/feature_analysis/candidate_evidence_confirmation.jsonl",
        "analysis/data/feature_analysis/confound_diagnostics.jsonl",
        "analysis/data/feature_analysis/counterexamples.jsonl",
        "analysis/data/aggregate_statistics/confirmation_only_feature_coverage.json",
        "analysis/data/validation_results/primary_reviews.jsonl",
        "analysis/data/validation_results/reviewer_agreement.json",
        "analysis/data/validation_results/cross_seed_reproducibility.json",
        "analysis/data/validation_results/cross_capacity_check.json",
        "analysis/data/validation_results/cross_seed_capacity_reconciliation.json",
        "analysis/data/validation_results/cross_seed_sensitivity_sweep.json",
        "analysis/data/relationship_analysis/eligible_candidates.json",
        "analysis/data/relationship_analysis/coactivation_analysis.json",
        "analysis/data/relationship_analysis/decoder_relationship_analysis.json",
        "analysis/data/relationship_analysis/cross_domain_analysis.json",
        "analysis/data/relationship_analysis/relationship_graph.json",
        "analysis/plots/figure_manifest.json",
        "reports/phase1_report.md", "reports/phase2_report.md", "reports/phase3_report.md",
        "reports/phase4_report.md",
    ]
    missing_required = [p for p in required if not (SFD_DIR / p).exists()]
    check("all_required_phase1_5_outputs_exist", len(missing_required) == 0,
          f"missing: {missing_required}")

    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = len(checks) - n_pass
    report = {"n_checks": len(checks), "n_passed": n_pass, "n_failed": n_fail, "checks": checks}
    out_path = DATA_DIR / "aggregate_statistics" / "integrity_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[integrity] {n_pass}/{len(checks)} checks passed, {n_fail} failed. Wrote {out_path}")
    return n_fail == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

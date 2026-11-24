"""
Milestone 6, Phase 3 (per CLAUDE.md "Opus agent usage" rule): condenses the 40
second-reviewer-sample candidates' full evidence bundles into a compact digest BEFORE
handing anything to an Opus reviewer. This is pure mechanical extraction (no
interpretation), so it is done directly in a script rather than delegated to an
agent -- the raw reading/digesting work that previously caused an Opus agent to run
long and hit a session limit is removed from the Opus call entirely.

Trims each candidate's bundle to: selection summary, top-5 Discovery examples, top-5
Confirmation examples, one-per-template examples (capped), confound diagnostics
(both subsets), and ALL counterexamples (already small). Still blind-compatible --
contains no primary-reviewer labels, confidence, or classification.
"""
import json
from pathlib import Path

SFD_DIR = Path(__file__).resolve().parent.parent
CAP_TOP = 5
CAP_TEMPLATE = 8


def trim_examples(examples_block, cap_top=CAP_TOP, cap_template=CAP_TEMPLATE):
    def trim_list(lst, cap):
        return [{"prompt_id": e["prompt_id"], "activation_value": e["activation_value"],
                 "domain": e["domain"], "subdomain": e["subdomain"],
                 "template_tier": e["template_tier"], "template_cluster_id": e["template_cluster_id"],
                 "prompt_text": e["prompt_text"][:250]} for e in lst[:cap]]
    return {
        "top": trim_list(examples_block.get("top", []), cap_top),
        "mid": trim_list(examples_block.get("mid", []), 3),
        "low": trim_list(examples_block.get("low", []), 3),
        "one_per_template": trim_list(examples_block.get("one_per_template", []), cap_template),
    }


def main():
    sample = json.loads((SFD_DIR / "configs" / "second_reviewer_sample.json").read_text())
    sample_feats = set(sample["sampled_feature_indices"])

    bundles = [json.loads(l) for l in (SFD_DIR / "analysis" / "data" / "validation_results" / "review_bundles.jsonl").open(encoding="utf-8")]
    bundles = [b for b in bundles if b["feature_idx"] in sample_feats]
    assert len(bundles) == 40, f"expected 40 bundles, got {len(bundles)}"

    digests = []
    for b in bundles:
        d_ev = b["discovery_evidence"]
        c_ev = b["confirmation_evidence"]
        digest = {
            "feature_idx": b["feature_idx"],
            "selection": b["selection"],
            "discovery_evidence": {
                "n_firing_tokens": d_ev.get("n_firing_tokens"), "n_unique_firing_prompts": d_ev.get("n_unique_firing_prompts"),
                "examples": trim_examples(d_ev.get("examples", {})),
                "template_diversity": d_ev.get("template_diversity"),
                "subdomain_diversity": d_ev.get("subdomain_diversity"), "lexical_diversity": d_ev.get("lexical_diversity"),
            },
            "confirmation_evidence": {
                "n_firing_tokens": c_ev.get("n_firing_tokens"), "n_unique_firing_prompts": c_ev.get("n_unique_firing_prompts"),
                "examples": trim_examples(c_ev.get("examples", {})),
                "template_diversity": c_ev.get("template_diversity"),
                "subdomain_diversity": c_ev.get("subdomain_diversity"), "lexical_diversity": c_ev.get("lexical_diversity"),
            },
            "confirmation_meets_minimum_evidence_rule": b["confirmation_meets_minimum_evidence_rule"],
            "confound_diagnostics": b["confound_diagnostics"],
            "counterexamples": b["counterexamples"],
        }
        digests.append(digest)

    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / "second_reviewer_digest.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in sorted(digests, key=lambda d: d["feature_idx"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    size_kb = out_path.stat().st_size / 1024
    print(f"[condense] wrote {out_path} ({len(digests)} candidates, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

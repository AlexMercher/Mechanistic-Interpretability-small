"""
Milestone 6, Phase 1 step 3 (plan B.6): tiered template-family detection.

No template-ID field exists anywhere in the corpus (checked during Phase 0 inspection).
Every prompt is tagged with exactly one of three tiers, which travels with every
downstream template-diversity statistic -- counts are never conflated across tiers:

  Tier 1 -- confirmed near-duplicate family: prompt participates in a connected
    component of >=2 prompts from M2's near_duplicate_candidates.csv (SequenceMatcher
    >=0.72, already blocked by (category, subdomain) when that file was built).
    Strongest evidence of a genuinely shared template.
  Tier 2 -- skeleton-fallback family: prompt has zero near-dup partners (not in any
    Tier-1 component of size >=2) but matches >=1 other prompt's masked "skeleton"
    (capitalized multi-word spans and digit runs replaced with <SLOT>, blanks
    collapsed), exact-matched within (domain, category). Weaker, heuristic evidence.
  Tier 3 -- unclustered/unknown: matches neither method. Its template relationship to
    any other prompt is UNKNOWN -- never counted as "an independent template" by
    default downstream.

This script does not alter any source record; it only reads normalized/*.jsonl and M2's
near-duplicate CSV (both read-only) and writes a new M6-owned tagging file.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfd_common as sf  # noqa: E402

SFD_DIR = Path(__file__).resolve().parent.parent

SLOT_RE_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*\b")
SLOT_RE_DIGITS = re.compile(r"\b\d+(\.\d+)?\b")
WHITESPACE_RE = re.compile(r"\s+")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_skeleton(prompt_text: str) -> str:
    """Masks capitalized multi-word spans and digit runs with <SLOT>, collapses
    consecutive slots, lowercases the remainder. A deliberately simple, deterministic
    heuristic -- not a general NLP template extractor (per plan B.6)."""
    masked = SLOT_RE_CAPITALIZED.sub("<SLOT>", prompt_text)
    masked = SLOT_RE_DIGITS.sub("<SLOT>", masked)
    masked = re.sub(r"(<SLOT>[\s,]*){2,}", "<SLOT> ", masked)
    masked = masked.lower()
    masked = WHITESPACE_RE.sub(" ", masked).strip()
    return masked


def main():
    records = sf.load_normalized_records()
    near_dup_pairs = sf.load_near_duplicate_candidates()

    # --- Tier 1: connected components over the near-duplicate graph ---
    uf = UnionFind()
    for pair in near_dup_pairs:
        uf.union(pair["prompt_id_a"], pair["prompt_id_b"])

    component_members = defaultdict(set)
    for pair in near_dup_pairs:
        root = uf.find(pair["prompt_id_a"])
        component_members[root].add(pair["prompt_id_a"])
        component_members[root].add(pair["prompt_id_b"])

    tier1_cluster_of = {}
    tier1_id_counter = 0
    for root, members in component_members.items():
        if len(members) < 2:
            continue
        cluster_id = f"tier1_{tier1_id_counter:06d}"
        tier1_id_counter += 1
        for pid in members:
            tier1_cluster_of[pid] = cluster_id

    print(f"[templates] Tier 1: {tier1_id_counter} confirmed near-dup families, "
          f"{len(tier1_cluster_of)} prompts covered")

    # --- Tier 2: skeleton fallback, only for prompts with zero Tier-1 coverage ---
    needs_tier2 = [pid for pid in records if pid not in tier1_cluster_of]
    skeleton_blocks = defaultdict(list)  # (domain, category) -> [(prompt_id, skeleton)]
    for pid in needs_tier2:
        rec = records[pid]
        skeleton = build_skeleton(rec["prompt"])
        skeleton_blocks[(rec["domain"], rec["category"])].append((pid, skeleton))

    tier2_cluster_of = {}
    tier2_id_counter = 0
    for (domain, category), items in skeleton_blocks.items():
        by_skeleton = defaultdict(list)
        for pid, skeleton in items:
            by_skeleton[skeleton].append(pid)
        for skeleton, pids in by_skeleton.items():
            if len(pids) < 2:
                continue
            cluster_id = f"tier2_{tier2_id_counter:06d}"
            tier2_id_counter += 1
            for pid in pids:
                tier2_cluster_of[pid] = cluster_id

    print(f"[templates] Tier 2: {tier2_id_counter} skeleton-fallback families, "
          f"{len(tier2_cluster_of)} prompts covered")

    # --- Tier 3: everything else ---
    n_tier3 = 0
    out_path = SFD_DIR / "analysis" / "data" / "aggregate_statistics" / "template_clusters.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pid in sorted(records.keys()):
            if pid in tier1_cluster_of:
                tier, cluster_id = 1, tier1_cluster_of[pid]
            elif pid in tier2_cluster_of:
                tier, cluster_id = 2, tier2_cluster_of[pid]
            else:
                tier, cluster_id = 3, None
                n_tier3 += 1
            f.write(json.dumps({
                "prompt_id": pid, "domain": records[pid]["domain"],
                "category": records[pid]["category"], "tier": tier,
                "template_cluster_id": cluster_id,
            }) + "\n")

    print(f"[templates] Tier 3 (unclustered/unknown): {n_tier3} prompts")
    print(f"[templates] wrote {out_path}")

    # --- Spot-check sample for Tier 2 (skeleton fallback) ---
    spotcheck = []
    seen_clusters = set()
    for (domain, category), items in skeleton_blocks.items():
        by_skeleton = defaultdict(list)
        for pid, skeleton in items:
            by_skeleton[skeleton].append(pid)
        for skeleton, pids in by_skeleton.items():
            if len(pids) < 2 or skeleton in seen_clusters:
                continue
            seen_clusters.add(skeleton)
            spotcheck.append({
                "domain": domain, "category": category, "skeleton": skeleton,
                "example_prompts": [{"prompt_id": p, "text": records[p]["prompt"]} for p in pids[:4]],
            })
            if len(spotcheck) >= 20:
                break
        if len(spotcheck) >= 20:
            break

    spotcheck_path = SFD_DIR / "reports" / "template_tier2_spotcheck.json"
    spotcheck_path.parent.mkdir(parents=True, exist_ok=True)
    spotcheck_path.write_text(json.dumps(spotcheck, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[templates] wrote {spotcheck_path} ({len(spotcheck)} examples for manual spot-check)")

    config = {
        "method_tier1": "connected_components_over_near_duplicate_candidates_csv (SequenceMatcher>=0.72, blocked by (category,subdomain) at source)",
        "method_tier2": "skeleton_fallback: mask capitalized multi-word spans + digit runs with <SLOT>, exact-match grouping within (domain, category), only for prompts with zero Tier-1 coverage",
        "method_tier3": "unclustered -- template relationship unknown, never counted as an independent template",
        "tier1_family_count": tier1_id_counter, "tier1_prompt_count": len(tier1_cluster_of),
        "tier2_family_count": tier2_id_counter, "tier2_prompt_count": len(tier2_cluster_of),
        "tier3_prompt_count": n_tier3,
        "total_prompts": len(records),
    }
    (SFD_DIR / "configs" / "template_clustering_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

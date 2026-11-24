#!/usr/bin/env python
"""
Objective A3 (part 1) -- generate a deterministic CANDIDATE pool of prompt
pairs for human/teacher-model review. This script does NOT assign relation
labels itself (cosine similarity is deliberately not used to generate
ground truth) -- it only selects which pairs are worth asking a reviewer
about, stratified across the required categories:

  same_concept_proxy        -- same domain+subdomain+category, likely same concept
  same_domain_diff_concept  -- same domain, different subdomain
  cross_domain_related_cand -- cross-domain, top-50 nearest-neighbour by embedding
                                (candidate only -- the reviewer decides if it's
                                actually related or a false neighbour)
  cross_domain_unrelated_cand -- cross-domain, random pair (expected unrelated)
  hindi_script_variant      -- HINDI domain, same subdomain+category, different
                                language_form (e.g. native_hindi vs romanized_hindi)
  template_variant          -- same (domain, subdomain, category), high text
                                overlap (difflib ratio > 0.6), proxy for template family

Writes evaluation/candidate_prompt_pairs.jsonl: one JSON object per pair with
both full prompt texts, both prompt_ids, domains/subdomains, the candidate
category, and (for cross_domain_related_cand only) the raw cosine similarity
that caused it to be selected -- shown to the reviewer for transparency, not
as a label.

Usage:
    python build_evaluation_candidates.py [--seed 42] [--per-category 30]
"""
import argparse
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    records = sc.load_normalized_records()
    annotations = sc.load_annotations()
    vectors, prompt_ids, _ = sc.load_embeddings()
    pid_to_row = {p: r for r, p in enumerate(prompt_ids)}
    n = len(prompt_ids)

    def mk_pair(pid_a, pid_b, category, extra=None):
        ra, rb = records[pid_a], records[pid_b]
        pair = {
            "pair_id": f"{category}_{pid_a}_{pid_b}",
            "candidate_category": category,
            "prompt_id_a": pid_a,
            "prompt_id_b": pid_b,
            "domain_a": ra["domain"], "subdomain_a": ra["subdomain"], "category_a": ra["category"],
            "domain_b": rb["domain"], "subdomain_b": rb["subdomain"], "category_b": rb["category"],
            "language_form_a": annotations[pid_a]["language_form"],
            "language_form_b": annotations[pid_b]["language_form"],
            "prompt_text_a": ra["prompt"],
            "prompt_text_b": rb["prompt"],
        }
        if extra:
            pair.update(extra)
        return pair

    pairs = []

    # same_concept_proxy
    by_fine = defaultdict(list)
    for p, r in records.items():
        by_fine[(r["domain"], r["subdomain"], r["category"])].append(p)
    fine_keys = [k for k, v in by_fine.items() if len(v) >= 2]
    rng.shuffle(fine_keys)
    for key in fine_keys[: args.per_category]:
        a, b = rng.sample(by_fine[key], 2)
        pairs.append(mk_pair(a, b, "same_concept_proxy"))

    # same_domain_diff_concept
    by_domain = defaultdict(list)
    for p, r in records.items():
        by_domain[r["domain"]].append(p)
    count = 0
    while count < args.per_category:
        dom = rng.choice(sc.DOMAINS)
        a, b = rng.sample(by_domain[dom], 2)
        if records[a]["subdomain"] != records[b]["subdomain"]:
            pairs.append(mk_pair(a, b, "same_domain_diff_concept"))
            count += 1

    # cross_domain_related_cand: for a random sample of query prompts, take their
    # single highest-similarity cross-domain neighbour
    query_sample = rng.sample(prompt_ids, args.per_category)
    for q in query_sample:
        qrow = pid_to_row[q]
        sims = vectors @ vectors[qrow]
        qdom = records[q]["domain"]
        dom_arr = np.array([records[p]["domain"] for p in prompt_ids])
        mask = dom_arr != qdom
        sims_masked = np.where(mask, sims, -1.0)
        best = int(np.argmax(sims_masked))
        pairs.append(mk_pair(q, prompt_ids[best], "cross_domain_related_cand",
                              {"embedding_cosine_similarity": float(sims[best])}))

    # cross_domain_unrelated_cand: fully random cross-domain pairs
    count = 0
    while count < args.per_category:
        a, b = rng.sample(prompt_ids, 2)
        if records[a]["domain"] != records[b]["domain"]:
            pairs.append(mk_pair(a, b, "cross_domain_unrelated_cand"))
            count += 1

    # hindi_script_variant: HINDI domain, same subdomain+category, different language_form
    hindi_by_fine = defaultdict(list)
    for p in by_domain["HINDI"]:
        hindi_by_fine[(records[p]["subdomain"], records[p]["category"])].append(p)
    hsv_pairs = []
    for key, pids in hindi_by_fine.items():
        forms = defaultdict(list)
        for p in pids:
            forms[annotations[p]["language_form"]].append(p)
        form_names = [f for f in forms if f != "uncertain"]
        for i in range(len(form_names)):
            for j in range(i + 1, len(form_names)):
                hsv_pairs.append((rng.choice(forms[form_names[i]]), rng.choice(forms[form_names[j]])))
    rng.shuffle(hsv_pairs)
    for a, b in hsv_pairs[: args.per_category]:
        pairs.append(mk_pair(a, b, "hindi_script_variant"))

    # template_variant: same fine key, high text-overlap, different text
    tv_pairs = []
    for key, pids in by_fine.items():
        sample_pids = pids if len(pids) <= 25 else rng.sample(pids, 25)
        for i in range(len(sample_pids)):
            for j in range(i + 1, len(sample_pids)):
                a, b = sample_pids[i], sample_pids[j]
                ta, tb = records[a]["prompt"], records[b]["prompt"]
                if ta == tb:
                    continue
                ratio = difflib.SequenceMatcher(None, ta, tb).quick_ratio()
                if ratio > 0.6:
                    tv_pairs.append((a, b))
        if len(tv_pairs) > args.per_category * 5:
            break
    rng.shuffle(tv_pairs)
    for a, b in tv_pairs[: args.per_category]:
        pairs.append(mk_pair(a, b, "template_variant"))

    eval_dir = args.out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "candidate_prompt_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for p in pairs:
        counts[p["candidate_category"]] += 1
    print(f"[build_evaluation_candidates] wrote {len(pairs)} candidate pairs to {out_path}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

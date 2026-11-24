#!/usr/bin/env python
"""
Objective B/D support -- derive a provisional concept name + description for
every discovered cluster from build_domain_clusters.py.

Per the task's explicit instruction not to "automatically name clusters
using their first few records," this uses TF-IDF distinctive-keyword
extraction over ALL of a cluster's prompt texts (not just a prefix),
scored against the rest of that cluster's domain as the background corpus
-- i.e. a term must be distinctive relative to the domain, not merely
frequent. This is still an automated heuristic, not a human judgment, so
every output row is explicitly marked review_status="auto_generated_unreviewed"
and is expected to be spot-checked (see run_metadata -- a sample is
reviewed by an Opus teacher model before the taxonomy is frozen).

Usage:
    python name_clusters.py [--top-k-terms 6]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import semantic_common as sc

STOPWORDS = set("""
a an the is are was were be been being of to in on for with and or but if then than
this that these those it its as at by from into about over under between out up down
not no nor so such what which who whom how why when where does do did doing have has
had can could may might must shall should will would you your yours they their them
one two three explain describe identify compare contrast analyze given following based
following scenario context question answer example provide following given
""".split())

WORD_RE = re.compile(r"[A-Za-zऀ-ॿ]{3,}")


def tokenize(text):
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k-terms", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=sc.EXP_DIR)
    args = ap.parse_args()

    records = sc.load_normalized_records()
    clustering_dir = args.out_dir / "clustering"

    named = {}
    for domain in sc.DOMAINS:
        detail = json.loads((clustering_dir / f"{domain.lower()}_cluster_details.json").read_text(encoding="utf-8"))
        clusters = detail["clusters"]

        # domain-level document frequency (background corpus = whole domain)
        domain_df = defaultdict(int)
        cluster_tokens = {}
        for c in clusters:
            toks = []
            for pid in c["prompt_ids"]:
                toks.extend(tokenize(records[pid]["prompt"]))
            cluster_tokens[c["cluster_id"]] = toks
        n_clusters = len(clusters)
        for cid, toks in cluster_tokens.items():
            for t in set(toks):
                domain_df[t] += 1

        for c in clusters:
            toks = cluster_tokens[c["cluster_id"]]
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            scored = []
            for t, f in tf.items():
                df = domain_df[t]
                # distinctiveness: high local frequency, low cross-cluster spread
                idf = (n_clusters + 1) / (df + 1)
                scored.append((f * idf, t))
            scored.sort(reverse=True)
            top_terms = [t for _, t in scored[: args.top_k_terms]]
            name = " / ".join(top_terms[:4]) if top_terms else "(no distinctive terms found)"
            named[c["cluster_id"]] = {
                "cluster_id": c["cluster_id"],
                "domain": domain,
                "subdomain": c["subdomain"],
                "provisional_name": name,
                "distinctive_terms": top_terms,
                "raw_record_count": c["raw_record_count"],
                "unique_prompt_count": c["unique_prompt_count"],
                "representative_prompt_ids": c["representative_prompt_ids"],
                "representative_prompt_texts": c["representative_prompt_texts"],
                "review_status": "auto_generated_unreviewed",
                "naming_method": (
                    "tfidf_distinctive_terms_over_full_cluster_text_v1 -- NOT a naive "
                    "first-record read; terms are chosen for being frequent within the "
                    "cluster AND rare across sibling clusters in the same domain."
                ),
            }

    out_path = clustering_dir / "cluster_provisional_names.json"
    out_path.write_text(json.dumps(named, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[name_clusters] wrote {len(named)} provisional names to {out_path}")


if __name__ == "__main__":
    main()

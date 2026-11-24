"""
Builds the /processed/ layer on top of /normalized/: semantic clusters
(by subdomain), derived chunks (fixed-size batches per category, for
downstream batched activation extraction), exact-duplicate detection, and
distribution reports. This layer only ever *adds* derived fields keyed by
prompt_id -- it never edits prompt text.

Run from the project root, after build_normalized.py: python scripts/build_processed.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"
PROCESSED_DIR = ROOT / "processed"
CHUNK_SIZE = 50


def snake_case(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return s.lower()


def load_domain_records(jsonl_path: Path):
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    for d in ["clusters", "chunks", "annotations", "reports"]:
        (PROCESSED_DIR / d).mkdir(parents=True, exist_ok=True)

    category_counts = defaultdict(int)       # (domain, category) -> count
    subdomain_counts = defaultdict(int)       # (domain, subdomain) -> count

    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        domain = jsonl_path.stem.upper()
        records = load_domain_records(jsonl_path)

        # --- clusters: group by subdomain ---
        clusters_out = []
        for rec in records:
            cluster_id = f"{domain}_{snake_case(rec['subdomain'])}"
            clusters_out.append({
                "prompt_id": rec["prompt_id"],
                "cluster_id": cluster_id,
                "subdomain": rec["subdomain"],
                "category": rec["category"],
            })
        with (PROCESSED_DIR / "clusters" / f"{jsonl_path.stem}_clusters.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for row in clusters_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # --- chunks: fixed-size batches per category, in file order ---
        by_category = defaultdict(list)
        for rec in records:
            by_category[rec["category"]].append(rec["prompt_id"])
        chunks_out = []
        for category, prompt_ids in by_category.items():
            for i in range(0, len(prompt_ids), CHUNK_SIZE):
                batch = prompt_ids[i : i + CHUNK_SIZE]
                chunk_id = f"{domain}_{category}_{i // CHUNK_SIZE:04d}"
                chunks_out.append({
                    "chunk_id": chunk_id,
                    "domain": domain,
                    "category": category,
                    "size": len(batch),
                    "prompt_ids": batch,
                })
        with (PROCESSED_DIR / "chunks" / f"{jsonl_path.stem}_chunks.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for row in chunks_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        chunk_id_by_prompt = {
            pid: row["chunk_id"] for row in chunks_out for pid in row["prompt_ids"]
        }
        cluster_id_by_prompt = {row["prompt_id"]: row["cluster_id"] for row in clusters_out}

        # --- exact-duplicate detection via prompt_sha256 ---
        by_hash = defaultdict(list)
        for rec in records:
            by_hash[rec["prompt_sha256"]].append(rec["prompt_id"])
        dup_report_rows = []
        is_duplicate = {}
        for sha, ids in by_hash.items():
            if len(ids) > 1:
                first = ids[0]
                for dup_id in ids[1:]:
                    dup_report_rows.append({"prompt_id": dup_id, "duplicate_of": first, "prompt_sha256": sha})
                    is_duplicate[dup_id] = True
                is_duplicate[first] = False
        with (PROCESSED_DIR / "reports" / f"{jsonl_path.stem}_exact_duplicates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["prompt_id", "duplicate_of", "prompt_sha256"])
            writer.writeheader()
            writer.writerows(dup_report_rows)

        # --- annotations: derived fields keyed by prompt_id, prompt text untouched ---
        with (PROCESSED_DIR / "annotations" / f"{jsonl_path.stem}_annotated.jsonl").open(
            "w", encoding="utf-8"
        ) as f:
            for rec in records:
                annotated = dict(rec)
                annotated["cluster_id"] = cluster_id_by_prompt[rec["prompt_id"]]
                annotated["chunk_id"] = chunk_id_by_prompt[rec["prompt_id"]]
                annotated["is_exact_duplicate"] = is_duplicate.get(rec["prompt_id"], False)
                f.write(json.dumps(annotated, ensure_ascii=False) + "\n")

        for rec in records:
            category_counts[(domain, rec["category"])] += 1
            subdomain_counts[(domain, rec["subdomain"])] += 1

        print(f"{domain}: {len(clusters_out)} clustered, {len(chunks_out)} chunks, "
              f"{len(dup_report_rows)} exact duplicates")

    # --- cross-domain distribution reports ---
    with (PROCESSED_DIR / "reports" / "category_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "category", "count"])
        for (domain, category), count in sorted(category_counts.items()):
            writer.writerow([domain, category, count])

    with (PROCESSED_DIR / "reports" / "subdomain_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "subdomain", "count"])
        for (domain, subdomain), count in sorted(subdomain_counts.items()):
            writer.writerow([domain, subdomain, count])

    print("wrote processed/reports/category_distribution.csv and subdomain_distribution.csv")


if __name__ == "__main__":
    main()

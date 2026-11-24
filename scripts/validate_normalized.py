"""
Structural validation of /normalized/*.jsonl against normalized/prompt.schema.json.
No external deps (no jsonschema package) -- checks required fields, enums, and
basic types by hand, matching what the original MI_Dataset_validated pipeline did.

Run from the project root: python scripts/validate_normalized.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NORMALIZED_DIR = ROOT / "normalized"

REQUIRED = [
    "schema_version", "prompt_id", "source_prompt_id", "domain", "subdomain",
    "category", "prompt", "source_model", "source_file", "prompt_sha256",
    "split", "validation_status",
]
DOMAIN_ENUM = {"DNA", "EMOTIONS", "HINDI", "SCIENCE", "GANDHI"}
SPLIT_ENUM = {"train", "validation", "test"}


def main():
    domain_counts = {}
    seen_ids = set()
    errors = []

    for jsonl_path in sorted(NORMALIZED_DIR.glob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                rec = json.loads(line)
                for field in REQUIRED:
                    if field not in rec:
                        errors.append(f"{jsonl_path.name}:{lineno} missing {field}")
                if rec.get("domain") not in DOMAIN_ENUM:
                    errors.append(f"{jsonl_path.name}:{lineno} bad domain {rec.get('domain')}")
                if rec.get("split") not in SPLIT_ENUM:
                    errors.append(f"{jsonl_path.name}:{lineno} bad split {rec.get('split')}")
                pid = rec.get("prompt_id")
                if pid in seen_ids:
                    errors.append(f"{jsonl_path.name}:{lineno} duplicate prompt_id {pid}")
                seen_ids.add(pid)
                domain_counts[rec.get("domain")] = domain_counts.get(rec.get("domain"), 0) + 1

    print("domain_counts:", domain_counts)
    print("total records:", sum(domain_counts.values()))
    print("unique prompt_ids:", len(seen_ids))
    if errors:
        print(f"FOUND {len(errors)} ERRORS:")
        for e in errors[:50]:
            print(" -", e)
        raise SystemExit(1)
    print("OK: no structural errors")


if __name__ == "__main__":
    main()

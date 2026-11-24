"""
Builds the canonical normalized JSONL layer for all five domains.

- DNA / EMOTIONS / HINDI already have validated normalized JSONL
  (MI_Dataset_validated/datasets/jsonl/*.jsonl). Those are copied verbatim
  (byte-identical records) into /normalized/ -- never re-parsed or rewritten.
- SCIENCE / GANDHI only exist as Markdown record blocks. This script parses
  those blocks and emits science.jsonl / gandhi.jsonl using the same schema,
  assigning a stable prompt_id per record. Prompt text itself is never
  rewritten -- only copied verbatim from the source Markdown.

Run from the project root, after build_raw_snapshot.py: python scripts/build_normalized.py
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
NORMALIZED_DIR = ROOT / "normalized"
LEGACY_JSONL_DIR = ROOT / "MI_Dataset_validated" / "datasets" / "jsonl"

SCHEMA_VERSION = "1.0"

# existing legacy domains: copied verbatim, record count already fixed
LEGACY_DOMAINS = {
    "DNA": LEGACY_JSONL_DIR / "dna.jsonl",
    "EMOTIONS": LEGACY_JSONL_DIR / "emotions.jsonl",
    "HINDI": LEGACY_JSONL_DIR / "hindi.jsonl",
}

# domains parsed fresh from raw Markdown in this script
MD_DOMAINS = {
    "SCIENCE": {
        "dir": RAW_DIR / "SCIENCE",
        "source_model": "Gemini",
    },
    "GANDHI": {
        "dir": RAW_DIR / "GANDHI",
        "source_model": "unspecified",
    },
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snake_case(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    return s.lower()


def deterministic_split(prompt_id: str) -> str:
    """80/10/10 split, stable by hash of the canonical prompt_id."""
    bucket = int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def parse_md_blocks(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    records = []
    for block in blocks:
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        if "PROMPT_ID" in fields and "PROMPT" in fields:
            records.append(fields)
    return records


def build_md_domain(domain: str, cfg: dict, counter_start: int):
    records = []
    seq = counter_start
    md_files = sorted(cfg["dir"].rglob("*.md"))
    # skip top-level report/reference files, keep only category record files
    md_files = [p for p in md_files if p.name not in {
        "GENERATION_REPORT.md", "REFERENCES.md", "VALIDATION_REPORT.md"
    }]
    for md_path in md_files:
        category_literal = md_path.parent.name  # e.g. "Contextual Understanding"
        for fields in parse_md_blocks(md_path):
            prompt_id = f"{domain}_{seq:06d}"
            seq += 1
            prompt_text = fields["PROMPT"]
            rec = {
                "schema_version": SCHEMA_VERSION,
                "prompt_id": prompt_id,
                "source_prompt_id": fields["PROMPT_ID"],
                "domain": domain,
                "subdomain": fields.get("SUBDOMAIN", ""),
                "domain_class": fields.get("DOMAIN_CLASS"),
                "category": snake_case(fields.get("CATEGORY", category_literal)),
                "source_category": fields.get("CATEGORY", category_literal),
                "prompt": prompt_text,
                "language": "English",
                "script": None,
                "source_script_label": None,
                "source_model": cfg["source_model"],
                "source_file": str(md_path.relative_to(ROOT).as_posix()),
                "prompt_sha256": sha256_text(prompt_text),
                "split": deterministic_split(prompt_id),
                "validation_status": "structural_pass_content_review_required",
            }
            if "REFERENCE_IDS" in fields:
                rec["reference_ids"] = fields["REFERENCE_IDS"].split()
            records.append(rec)
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    domain_counts = {}

    # legacy domains: copy verbatim, do not touch content
    for domain, src in LEGACY_DOMAINS.items():
        dst = NORMALIZED_DIR / f"{domain.lower()}.jsonl"
        shutil.copyfile(src, dst)
        with dst.open(encoding="utf-8") as f:
            domain_counts[domain] = sum(1 for _ in f)
        print(f"{domain}: copied {domain_counts[domain]} records verbatim -> {dst.relative_to(ROOT)}")

    # fresh domains: parse Markdown, assign stable prompt_id
    for domain, cfg in MD_DOMAINS.items():
        records = build_md_domain(domain, cfg, counter_start=1)
        dst = NORMALIZED_DIR / f"{domain.lower()}.jsonl"
        write_jsonl(dst, records)
        domain_counts[domain] = len(records)
        print(f"{domain}: parsed {len(records)} records -> {dst.relative_to(ROOT)}")

    print("domain_counts:", domain_counts)
    print("total:", sum(domain_counts.values()))


if __name__ == "__main__":
    main()

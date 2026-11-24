"""
Copies original, unmodified source Markdown files for all five domains into
a single top-level /raw/<DOMAIN>/ tree and records a sha256 manifest.

Sources:
  DNA / EMOTIONS / HINDI  -> MI_Dataset_validated/source_originals/<X>_by_category
  SCIENCE                 -> MONISHA_.../SCIENCE_5000_GEMINI
  GANDHI                  -> MONISHA_.../GANDHI_5000_PROMPTS/GANDHI_5000

Run from the project root: python scripts/build_raw_snapshot.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MONISHA = ROOT / "MONISHA_SCIENCE_AND_gANDHI_10000_PROMPTS (1)"

DOMAIN_SOURCES = {
    "DNA": ROOT / "MI_Dataset_validated" / "source_originals" / "DNA_by_category",
    "EMOTIONS": ROOT / "MI_Dataset_validated" / "source_originals" / "EMOTIONS_by_category",
    "HINDI": ROOT / "MI_Dataset_validated" / "source_originals" / "HINDI_by_category",
    "SCIENCE": MONISHA / "SCIENCE_5000_GEMINI",
    "GANDHI": MONISHA / "GANDHI_5000_PROMPTS" / "GANDHI_5000",
}

RAW_DIR = ROOT / "raw"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    manifest = {"domains": {}}
    for domain, src in DOMAIN_SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"missing source for {domain}: {src}")
        dst = RAW_DIR / domain
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

        files = []
        for p in sorted(dst.rglob("*")):
            if p.is_file():
                files.append({
                    "path": str(p.relative_to(RAW_DIR).as_posix()),
                    "sha256": sha256_of(p),
                    "bytes": p.stat().st_size,
                })
        manifest["domains"][domain] = {
            "source": str(src.relative_to(ROOT).as_posix()),
            "file_count": len(files),
            "files": files,
        }
        print(f"{domain}: copied {len(files)} files from {src}")

    (RAW_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote raw/MANIFEST.json")


if __name__ == "__main__":
    main()

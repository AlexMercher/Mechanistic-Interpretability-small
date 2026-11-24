"""
Shared plotting helpers for Milestone 6 Phase 5. Every plot script reads only
already-written Phase 1-4 JSON/JSONL/pickle outputs -- no new computation on
raw activations happens here. `save_fig` always writes a 300dpi PNG plus an
SVG, and records the exact source file(s) a plot was built from so
`validate_integrity.py` can check every figure traces back to real data.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SFD_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = SFD_DIR / "analysis" / "plots"
DATA_DIR = SFD_DIR / "analysis" / "data"
MANIFEST_PATH = PLOTS_DIR / "figure_manifest.json"

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("tab10")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_manifest = []
if MANIFEST_PATH.exists():
    try:
        _manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        _manifest = []


def save_fig(fig, category, name, caption, source_files, n=None):
    """Saves PNG (300dpi) + SVG under analysis/plots/<category>/<name>.{png,svg}
    and records an entry (category, name, caption, source_files, n) in
    figure_manifest.json, appended to across script runs."""
    out_dir = PLOTS_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    entry = {
        "category": category, "name": name, "caption": caption,
        "source_files": [str(Path(s)) for s in source_files],
        "n": n, "png": str(png_path.relative_to(SFD_DIR)), "svg": str(svg_path.relative_to(SFD_DIR)),
    }
    global _manifest
    _manifest = [e for e in _manifest if not (e["category"] == category and e["name"] == name)]
    _manifest.append(entry)
    MANIFEST_PATH.write_text(json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[plot] {category}/{name} <- {source_files}")

"""
Milestone 6, Phase 3 step 4 (plan B.2 step 5): threshold sensitivity sweep.

0.3 cosine / 0.1 margin are numerical-matching heuristics, not established
semantic-reproducibility criteria. Re-runs the full cross-seed matching procedure at
alternative thresholds and reports how classification counts shift -- so the headline
reproducibility numbers are never presented as if a single threshold were ground truth.
"""
import json
import subprocess
import sys
from pathlib import Path

SFD_DIR = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent / "cross_seed_reproducibility.py"

COSINE_VALUES = [0.25, 0.3, 0.4, 0.5]
MARGIN_VALUES = [0.05, 0.1, 0.2]


def main():
    rows = []
    for cosine in COSINE_VALUES:
        for margin in MARGIN_VALUES:
            suffix = f"_sens_c{cosine}_m{margin}"
            print(f"[sweep] cosine={cosine} margin={margin}")
            subprocess.run([sys.executable, str(SCRIPT), "--cosine-threshold", str(cosine),
                             "--margin-threshold", str(margin), "--out-suffix", suffix], check=True)
            data = json.loads((SFD_DIR / "analysis" / "data" / "validation_results" / f"cross_seed_reproducibility{suffix}.json").read_text())
            for seed_key in ("seed_1", "seed_2"):
                from collections import Counter
                counts = Counter(v[seed_key]["classification"] for v in data["results"].values())
                rows.append({"cosine_threshold": cosine, "margin_threshold": margin, "seed": seed_key,
                             "classification_counts": dict(counts)})

    out_path = SFD_DIR / "analysis" / "data" / "validation_results" / "cross_seed_sensitivity_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {out_path}")
    for r in rows:
        print(r["cosine_threshold"], r["margin_threshold"], r["seed"], r["classification_counts"])


if __name__ == "__main__":
    main()

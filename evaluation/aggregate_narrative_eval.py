"""R2.6 T4 — Aggregate LLM-as-Judge scores: per-axis mean + CI95 + breakdown.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Load scored samples
scores_path = RESULTS_DIR / "narrative_llm_scores_60.json"
with open(scores_path) as f:
    scored = json.load(f)

# Filter to successfully scored
valid = [s for s in scored if s.get("scores")]
n_total = len(scored)
n_valid = len(valid)
print(f"Loaded {n_total} entries, {n_valid} valid scores")

# ═════════════════════════════════════════════════════════════════════════
#  Overall per-axis
# ═════════════════════════════════════════════════════════════════════════
axes = ["Ax1", "Ax2", "Ax3", "Ax4"]
axis_labels = {
    "Ax1": "Tone Consistency",
    "Ax2": "Context Grounding",
    "Ax3": "Empathy",
    "Ax4": "Actionability",
}

overall = {}
print(f"\n{'='*50}")
print(f"  Overall (n={n_valid})")
print(f"{'='*50}")
for ax in axes:
    vals = [s["scores"][ax] for s in valid]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    se = std / np.sqrt(len(vals))
    ci95 = sp_stats.t.interval(0.95, df=len(vals)-1, loc=mean, scale=se)
    overall[ax] = {
        "label": axis_labels[ax],
        "mean": round(mean, 2),
        "std": round(std, 2),
        "ci95_low": round(ci95[0], 2),
        "ci95_high": round(ci95[1], 2),
        "n": len(vals),
    }
    print(f"  {ax} ({axis_labels[ax]:20s}): {mean:.2f} ± {std:.2f}  "
          f"CI95=[{ci95[0]:.2f}, {ci95[1]:.2f}]")

# ═════════════════════════════════════════════════════════════════════════
#  Per-stream-label breakdown
# ═════════════════════════════════════════════════════════════════════════
by_stream = defaultdict(list)
for s in valid:
    by_stream[s["stream_label"]].append(s["scores"])

stream_breakdown = {}
print(f"\n{'='*50}")
print(f"  Per-stream breakdown")
print(f"{'='*50}")
print(f"  {'Stream':<28} {'Ax1':>4} {'Ax2':>4} {'Ax3':>4} {'Ax4':>4} {'n':>3}")
print("-" * 50)

for stream in sorted(by_stream.keys()):
    entries = by_stream[stream]
    row = {}
    means = []
    for ax in axes:
        vals = [e[ax] for e in entries]
        m = float(np.mean(vals))
        row[ax] = round(m, 2)
        means.append(f"{m:.1f}")
    row["n"] = len(entries)
    stream_breakdown[stream] = row
    print(f"  {stream:<28} {means[0]:>4} {means[1]:>4} {means[2]:>4} {means[3]:>4} {len(entries):>3}")

# ═════════════════════════════════════════════════════════════════════════
#  Save
# ═════════════════════════════════════════════════════════════════════════
result = {
    "n_total_samples": n_total,
    "n_valid_scores": n_valid,
    "overall": overall,
    "per_stream": stream_breakdown,
}

out_path = RESULTS_DIR / "narrative_eval_summary.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✓ Saved: {out_path.name}")

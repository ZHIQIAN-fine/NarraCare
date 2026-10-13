"""Combine subagent LLM-judge score batches with narrative sample metadata."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Load sample metadata
with open(RESULTS_DIR / "narrative_samples_60.json") as f:
    samples = json.load(f)

# Load score batches
with open("/tmp/scores_batch1.json") as f:
    batch1 = json.load(f)
with open("/tmp/scores_batch2.json") as f:
    batch2 = json.load(f)

all_scores = batch1 + batch2
assert len(all_scores) == 60, f"Expected 60 scores, got {len(all_scores)}"
assert len(samples) == 60, f"Expected 60 samples, got {len(samples)}"

# Merge: each sample gets a "scores" dict
scored = []
for i, (sample, score_entry) in enumerate(zip(samples, all_scores)):
    expected_id = f"S{i:02d}"
    assert score_entry["id"] == expected_id, (
        f"ID mismatch at index {i}: expected {expected_id}, got {score_entry['id']}"
    )
    entry = dict(sample)
    entry["scores"] = {
        "Ax1": score_entry["Ax1"],
        "Ax2": score_entry["Ax2"],
        "Ax3": score_entry["Ax3"],
        "Ax4": score_entry["Ax4"],
    }
    entry["judge_model"] = "aime-subagent-llm"
    scored.append(entry)

out_path = RESULTS_DIR / "narrative_llm_scores_60.json"
with open(out_path, "w") as f:
    json.dump(scored, f, indent=2)

print(f"✓ Combined {len(scored)} scored samples → {out_path.name}")
print(f"  Stream distribution:")
from collections import Counter
dist = Counter(s["stream_label"] for s in scored)
for k in sorted(dist):
    print(f"    {k}: {dist[k]}")

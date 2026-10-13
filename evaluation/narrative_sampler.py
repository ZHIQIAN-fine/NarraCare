"""R2.6 T2 — Narrative Sampler: stratified 60-sample extraction.

Runs LinUCB 60-day (seed=42) and extracts prompt_level_log.
Stratifies into 5 layers × 12 samples = 60.

Layers:
  speak:doctor, speak:coach, speak:companion,
  speak:yellow-mixed (emotional red/yellow), closing:safety_followup
"""
import sys, json, time, random, logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

N_DAYS = 60
SAMPLES_PER_LAYER = 12
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 64)
print("[R2.6 T2] Narrative Sampler — stratified 60-sample extraction")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

# ── Run pipeline to get full prompt_level_log ────────────────────────
print("\n▶ Running LinUCB (safety ON) 60-day...")
pipe = NarraCarePipeline("configs/default.yaml")
logs = pipe.run_simulation(n_days=N_DAYS)
prompt_log = getattr(pipe, "_prompt_level_log", [])
print(f"  {len(logs)} interactions, {len(prompt_log)} prompt log entries")

# ── Build enriched entries ───────────────────────────────────────────
# Index interaction logs by (user_id, day) for context lookup
ix_index: Dict[tuple, dict] = {}
for ix in logs:
    key = (ix["user_id"], ix["day"])
    ix_index[key] = ix  # last turn if multiple

enriched = []
for entry in prompt_log:
    uid = entry.get("user_id")
    day = entry.get("day")
    tone = entry.get("tone_actually_used", "")
    scenario = entry.get("scenario", "")
    variant = entry.get("daily_closing_variant")
    scenario_key = entry.get("scenario_key_used", "")

    # Get matching interaction log for context
    ix_data = ix_index.get((uid, day), {})
    # emotional_level lives in the interaction log, not prompt metadata
    emotional = ix_data.get("emotional_level", "green")

    # Classify into layers
    if variant:
        # Daily closing
        if variant == "safety_followup":
            layer = "closing:safety_followup"
        else:
            continue  # we only need safety_followup closings
    else:
        # speak() entries
        if emotional in ("red_light", "yellow_light"):
            layer = "speak:yellow-mixed"
        elif tone == "doctor":
            layer = "speak:doctor"
        elif tone == "coach":
            layer = "speak:coach"
        elif tone == "companion":
            layer = "speak:companion"
        else:
            continue  # unknown tone

    # Build the sample record
    state = ix_data.get("state", {})
    chosen_task = ix_data.get("chosen_task", {})
    # For closings, use prompt_text from the prompt_level_log entry;
    # for speak() entries, use the interaction log's narrative field.
    if variant:
        narrative_text = entry.get("prompt_text", "")
    else:
        narrative_text = ix_data.get("narrative", "")

    enriched.append({
        "sample_id": f"{uid}_d{day}_{layer.replace(':', '_')}",
        "stream_label": layer,
        "user_id": uid,
        "day": day,
        "tone": tone,
        "scenario": scenario or variant or "",
        "scenario_key": scenario_key,
        "emotional_state_level": emotional if emotional != "green" else "normal",
        "user_pain": state.get("pain_level", 0),
        "user_fatigue": state.get("fatigue_level", 0),
        "task_name": chosen_task.get("task_id", ""),
        "task_difficulty": chosen_task.get("difficulty", 1),
        "task_category": chosen_task.get("category", ""),
        "prompt_text": narrative_text,
        "safety_blocked_today": bool(ix_data.get("infeasible_in_top_k")),
        "max_difficulty_at_decision": max(
            (t.get("difficulty", 1)
             for t in ix_data.get("recommended_tasks", [])),
            default=1,
        ),
    })

# ── Stratified sampling ─────────────────────────────────────────────
random.seed(42)
layers = defaultdict(list)
for e in enriched:
    layers[e["stream_label"]].append(e)

print(f"\n  Layer populations:")
for layer_name in sorted(layers.keys()):
    print(f"    {layer_name}: {len(layers[layer_name])}")

samples = []
for layer_name in ["speak:doctor", "speak:coach", "speak:companion",
                    "speak:yellow-mixed", "closing:safety_followup"]:
    pool = layers.get(layer_name, [])
    if len(pool) >= SAMPLES_PER_LAYER:
        selected = random.sample(pool, SAMPLES_PER_LAYER)
    else:
        selected = pool  # take all if < 12
        print(f"  ⚠ {layer_name}: only {len(pool)} available (need {SAMPLES_PER_LAYER})")
    samples.extend(selected)

# Re-index
for i, s in enumerate(samples):
    s["sample_index"] = i

print(f"\n  Total samples: {len(samples)}")
print(f"  Distribution: {dict(Counter(s['stream_label'] for s in samples))}"
      if samples else "  No samples!")

# ── Save ─────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "narrative_samples_60.json"
with open(out_path, "w") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)
print(f"\n✓ Saved: {out_path.name}")

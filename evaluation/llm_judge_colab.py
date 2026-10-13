#!/usr/bin/env python3
"""R2.6 T3 — LLM-as-Judge Narrative Evaluation (self-contained Colab / CLI script).

Run in Google Colab:
  1. Upload this file or paste into a notebook
  2. Set OPENAI_API_KEY in Colab secrets (userdata)
  3. Upload narrative_samples_60.json when prompted
  4. Run all cells

Run locally:
  export OPENAI_API_KEY="sk-..."
  python llm_judge_colab.py --samples narrative_samples_60.json --out narrative_eval_scores.json
"""

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 1 — Dependencies                                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝
# !pip install openai>=1.40.0 scipy numpy -q

import json, os, sys, time, argparse
from pathlib import Path
from typing import Optional

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 2 — API Key                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def setup_api_key():
    """Try Colab userdata first, then env var."""
    if "OPENAI_API_KEY" not in os.environ:
        try:
            from google.colab import userdata
            os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
            print("✓ API key loaded from Colab secrets")
        except (ImportError, Exception):
            pass
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it in Colab secrets or as an env var."
        )
    print(f"✓ API key present (last 4: ...{os.environ['OPENAI_API_KEY'][-4:]})")

setup_api_key()

from openai import OpenAI
import numpy as np

client = OpenAI()

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 3 — Load samples                                               ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def load_samples(path: Optional[str] = None) -> list:
    """Load narrative samples from file or Colab upload."""
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    # Try Colab upload
    try:
        from google.colab import files
        print("Upload narrative_samples_60.json:")
        uploaded = files.upload()
        fname = list(uploaded.keys())[0]
        return json.loads(uploaded[fname].decode("utf-8"))
    except ImportError:
        pass
    raise FileNotFoundError(
        "Cannot find samples file. Pass --samples <path> or upload in Colab."
    )

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 4 — Rubric (3-axis, per R2.6 spec)                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝
SYSTEM_PROMPT = """You are an evaluator for a physiotherapy rehabilitation chatbot. Each sample is one
chatbot message to a patient, together with the clinical context in which the
message was issued. Your job is to score the message on three independent axes, each
on a 1-5 Likert scale. You must base every score ONLY on the sample's prompt text
and the provided context — never speculate about information you do not see.

Return STRICT JSON with the following shape and no additional text:

{
  "coherence": {"score": <int 1-5>, "reason": "<one short sentence>"},
  "empathy":   {"score": <int 1-5>, "reason": "<one short sentence>"},
  "clinical":  {"score": <int 1-5>, "reason": "<one short sentence>"}
}

Axis definitions:

- COHERENCE: Does the message follow logically from the recent context
  (recent adherence, last task, current phase)? 1 = contradicts context, 5 = fully consistent.

- EMPATHY: Is the emotional register appropriate to the patient's current pain, fatigue,
  and emotional_state_level? 1 = tone mismatched (e.g., chipper to a red-light patient),
  5 = well calibrated to the state.

- CLINICAL FIDELITY: Does the message correctly describe the task (name, difficulty,
  category) and avoid clinically inappropriate claims given the condition and phase?
  1 = clinically wrong or dangerous, 5 = clinically accurate and phase-appropriate."""

USER_TEMPLATE = """=== CONTEXT ===
stream_label: {stream_label}
tone: {tone}
day: {day}  (of 60)
pain_level: {user_pain}  fatigue_level: {user_fatigue}
emotional_state_level: {emotional_state_level}
task_recommended: {task_name}
task_difficulty: {task_difficulty}
task_category: {task_category}
safety_blocked_today: {safety_blocked_today}
scenario_key: {scenario_key}

=== CHATBOT MESSAGE (to be scored) ===
{prompt_text}

Return the JSON object only."""

MODEL = "gpt-4o-mini"
TEMPERATURE_DEFAULT = 0
SEED = 42
AXES = ["coherence", "empathy", "clinical"]

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 5 — Scoring logic with retry                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def score_one(sample: dict, temperature: float = 0) -> dict:
    """Call OpenAI and parse the 3-axis JSON response."""
    user_msg = USER_TEMPLATE.format(**sample)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        seed=SEED,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    usage = resp.usage
    parsed = json.loads(raw)
    # Validate
    scores = {}
    for ax in AXES:
        entry = parsed[ax]
        s = int(entry["score"])
        assert 1 <= s <= 5, f"{ax} score {s} out of range"
        scores[ax] = {"score": s, "reason": entry["reason"]}
    return {
        "scores": scores,
        "raw": raw,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


def score_with_retry(sample: dict) -> dict:
    """Score with bucketed retry logic per R2.6 spec.

    API bucket cap = 2, format bucket cap = 1, total max = 3.
    """
    api_retries = 0
    fmt_retries = 0
    retries_log = []
    last_error = ""
    last_raw = ""

    attempts = [
        (TEMPERATURE_DEFAULT, "initial"),
    ]

    for attempt_idx in range(4):  # max 4 attempts (1 initial + 3 retries)
        temp = TEMPERATURE_DEFAULT if fmt_retries == 0 else 0.1
        try:
            result = score_one(sample, temperature=temp)
            return {
                "judge_failed": False,
                "retries": attempt_idx,
                "retries_log": retries_log,
                **result,
            }
        except (ConnectionError, TimeoutError, Exception) as e:
            err_str = str(e)
            is_api_error = any(kw in err_str.lower() for kw in
                               ["timeout", "500", "502", "503", "529",
                                "rate_limit", "ratelimit", "connection"])
            is_format_error = any(kw in err_str.lower() for kw in
                                  ["json", "key", "score", "assert", "parse"])

            if is_api_error:
                failure_type = "api"
                if api_retries >= 2:
                    last_error = f"api_cap_exceeded: {err_str}"
                    break
                api_retries += 1
                wait = 2 ** api_retries
            elif is_format_error:
                failure_type = "format"
                if fmt_retries >= 1:
                    last_error = f"format_cap_exceeded: {err_str}"
                    break
                fmt_retries += 1
                temp = 0.1
                wait = 1
            else:
                # Unknown error — classify as API
                failure_type = "api_unknown"
                if api_retries >= 2:
                    last_error = f"api_unknown_cap: {err_str}"
                    break
                api_retries += 1
                wait = 2 ** api_retries

            retries_log.append({
                "attempt": attempt_idx + 1,
                "failure_type": failure_type,
                "temperature": temp,
                "seed": SEED,
                "error_preview": err_str[:200],
            })
            print(f"    Retry {attempt_idx+1} ({failure_type}), wait {wait}s")
            time.sleep(wait)

    return {
        "judge_failed": True,
        "failure_type": last_error[:200],
        "retries": api_retries + fmt_retries,
        "retries_log": retries_log,
        "scores": None,
        "raw": last_raw,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 6 — Main scoring loop                                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def run_scoring(samples: list) -> dict:
    """Score all samples and return the full result dict."""
    scores_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    failed_count = 0

    print(f"\n{'='*60}")
    print(f"  Scoring {len(samples)} samples with {MODEL}")
    print(f"  temperature={TEMPERATURE_DEFAULT}, seed={SEED}")
    print(f"{'='*60}\n")

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id", f"S{i:02d}")
        stream = sample.get("stream_label", "?")
        print(f"  [{i+1:02d}/{len(samples)}] {sid} ({stream}) ...", end=" ")

        result = score_with_retry(sample)
        result["sample_id"] = sid
        result["stream_label"] = stream

        if result["judge_failed"]:
            failed_count += 1
            print(f"FAILED ({result.get('failure_type', '?')[:60]})")
        else:
            sc = result["scores"]
            coh = sc["coherence"]["score"]
            emp = sc["empathy"]["score"]
            cli = sc["clinical"]["score"]
            print(f"coh={coh} emp={emp} cli={cli}"
                  f"  (retries={result['retries']})")
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)

        scores_list.append(result)
        time.sleep(0.3)  # polite rate limit

    # Cost estimate (gpt-4o-mini pricing: $0.15/1M prompt, $0.60/1M completion)
    usd_estimate = (
        total_prompt_tokens * 0.15 / 1_000_000
        + total_completion_tokens * 0.60 / 1_000_000
    )

    output = {
        "config": {
            "model": MODEL,
            "temperature": TEMPERATURE_DEFAULT,
            "seed": SEED,
            "samples_evaluated": len(samples),
            "failed": failed_count,
        },
        "cost": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "usd_estimate": round(usd_estimate, 4),
            "usd_budget_cap": 0.03,
        },
        "scores": scores_list,
    }

    print(f"\n{'='*60}")
    print(f"  Done: {len(samples) - failed_count}/{len(samples)} scored")
    print(f"  Failed: {failed_count}")
    print(f"  Tokens: {total_prompt_tokens} prompt + "
          f"{total_completion_tokens} completion")
    print(f"  Cost: ~${usd_estimate:.4f} USD")
    if failed_count >= 6:
        print(f"  ⚠️  HOLD: failed ≥ 6, do NOT run aggregator")
    print(f"{'='*60}")

    return output


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 7 — Aggregation (inline, no separate file needed)              ║
# ╚═══════════════════════════════════════════════════════════════════════╝
from scipy import stats as sp_stats
from collections import defaultdict

def aggregate(scored: dict) -> dict:
    """Per-axis mean + CI95 + per-stream breakdown."""
    entries = scored["scores"]
    valid = [s for s in entries if not s.get("judge_failed")]
    n_valid = len(valid)
    failed_count = scored["config"]["failed"]

    # Overall
    overall = {}
    for ax in AXES:
        vals = [s["scores"][ax]["score"] for s in valid]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        se = std / np.sqrt(len(vals))
        ci = sp_stats.t.interval(0.95, df=len(vals)-1, loc=mean, scale=se)
        hw = (ci[1] - ci[0]) / 2
        overall[ax] = {
            "mean": round(mean, 2),
            "ci95_half_width": round(hw, 2),
            "n": len(vals),
        }

    # Per stream
    by_stream = defaultdict(list)
    for s in valid:
        by_stream[s["stream_label"]].append(s["scores"])
    stream_bd = {}
    for stream in sorted(by_stream.keys()):
        row = {"n": len(by_stream[stream])}
        for ax in AXES:
            vals = [e[ax]["score"] for e in by_stream[stream]]
            m = float(np.mean(vals))
            row[ax] = {"mean": round(m, 2)}
        stream_bd[stream] = row

    # Sanity
    all_scores = [(ax, s["sample_id"], s["scores"][ax]["score"])
                  for s in valid for ax in AXES]
    min_entry = min(all_scores, key=lambda x: x[2])
    max_entry = max(all_scores, key=lambda x: x[2])
    any_below_3 = any(v["mean"] < 3.0 for v in overall.values())

    # Missingness note
    miss_note = None
    if 3 <= failed_count <= 5:
        failed_ids = [s["sample_id"] for s in entries if s.get("judge_failed")]
        fail_types = defaultdict(int)
        for s in entries:
            if s.get("judge_failed"):
                ft = s.get("failure_type", "unknown")
                fail_types[ft.split(":")[0]] += 1
        layer_conc = defaultdict(int)
        for s in entries:
            if s.get("judge_failed"):
                layer_conc[s.get("stream_label", "?")] += 1
        miss_note = {
            "failed_count": failed_count,
            "failed_sample_ids": failed_ids,
            "failure_type_distribution": dict(fail_types),
            "layer_concentration": {k: v for k, v in layer_conc.items() if v >= 2},
        }

    summary = {
        "overall": overall,
        "by_stream_label": stream_bd,
        "sanity": {
            "min_score": {"axis": min_entry[0], "sample_id": min_entry[1],
                          "value": min_entry[2]},
            "max_score": {"axis": max_entry[0], "sample_id": max_entry[1],
                          "value": max_entry[2]},
            "any_axis_mean_below_3": any_below_3,
        },
        "missingness_note": miss_note,
    }

    # Print
    print(f"\n{'='*60}")
    print(f"  Aggregation (n={n_valid})")
    print(f"{'='*60}")
    for ax in AXES:
        o = overall[ax]
        print(f"  {ax:12s}: {o['mean']:.2f} ± {o['ci95_half_width']:.2f}")
    print(f"\n  Per-stream:")
    print(f"  {'Stream':<30} {'coh':>4} {'emp':>4} {'cli':>4}  n")
    for stream in sorted(stream_bd.keys()):
        r = stream_bd[stream]
        print(f"  {stream:<30} {r['coherence']['mean']:>4.1f}"
              f" {r['empathy']['mean']:>4.1f}"
              f" {r['clinical']['mean']:>4.1f}  {r['n']}")
    if any_below_3:
        print(f"\n  ⚠️  Some axis mean < 3.0 (signal, not failure)")
    print(f"{'='*60}")

    return summary


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Cell 8 — Run all                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge narrative eval")
    parser.add_argument("--samples", type=str, default=None,
                        help="Path to narrative_samples_60.json")
    parser.add_argument("--out-scores", type=str,
                        default="narrative_eval_scores.json",
                        help="Output scores file")
    parser.add_argument("--out-summary", type=str,
                        default="narrative_eval_summary.json",
                        help="Output summary file")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Skip aggregation (if failed >= 6)")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    print(f"✓ Loaded {len(samples)} samples")

    # Score
    scored = run_scoring(samples)

    # Save scores
    with open(args.out_scores, "w") as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Scores saved: {args.out_scores}")

    # Aggregate (unless failed >= 6 or --skip-aggregate)
    if scored["config"]["failed"] >= 6:
        print(f"\n⚠️  HOLD: {scored['config']['failed']} failures >= 6."
              f" Aggregation skipped. Review failing cases.")
    elif not args.skip_aggregate:
        summary = aggregate(scored)
        with open(args.out_summary, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Summary saved: {args.out_summary}")
    else:
        print("\n  Aggregation skipped (--skip-aggregate)")


if __name__ == "__main__":
    main()

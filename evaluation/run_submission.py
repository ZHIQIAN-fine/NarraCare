"""Final, versioned online evaluation. No LLM calls, timestamps or Git dependency."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import stats

from evaluation.metrics import safety_violation_rate
from src.pipeline import NarraCarePipeline

POLICIES = [
    "saps",
    "linucb",
    "neural_ucb",
    "random",
    "popularity",
    "rule_based",
    "epsilon_greedy",
]
DEVELOPMENT_SEEDS = list(range(42, 52))
FINAL_SEEDS = list(range(52, 62))
TRAIN = [f"U{i:03d}" for i in list(range(1, 11)) + list(range(16, 21))]
TEST = [f"U{i:03d}" for i in range(11, 16)]
PROTOCOL = {
    "version": "submission-v1",
    "days": 60,
    "primary_top_k": 1,
    "ier_k": 5,
    "diagnostic_seeds": DEVELOPMENT_SEEDS,
    "final_seeds": FINAL_SEEDS,
    "rationale": "Top-1 chosen from the one-task live interaction before inspecting diagnostic outcomes. No parameter search.",
    "reward": "Existing six-term reward on attempted tasks; skipped-task reward 0 is bookkeeping, not an observed physical outcome.",
    "independence": "One shared learner per run. Seed runs are repeated experimental units; personas are not independent replicates. Same seed does not align realized simulator noise.",
    "models": {
        "saps": {
            "alpha_ucb": 1.0,
            "tau_safe": 2.0,
            "alpha_prior": 1.0,
            "beta_prior": 1.0,
        },
        "linucb": {"alpha": 1.0},
        "neural_ucb": {
            "hidden_dim": 32,
            "training_epochs": 5,
            "alpha": 0.01,
            "safety_lambda": 3.0,
        },
    },
}
METRICS = [
    "reward",
    "IER",
    "SVR",
    "acceptance",
    "completion",
    "assigned",
    "unique_tasks_per_user",
]
logging.basicConfig(level=logging.ERROR)


def mean_user_svr(logs):
    by = defaultdict(list)
    for row in logs:
        by[row["user_id"]].append(row)
    return (
        float(np.mean([safety_violation_rate(rows) for rows in by.values()]))
        if by
        else 0.0
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def fingerprint():
    paths = (
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "configs").glob("*.yaml"))
        + list((ROOT / "data").glob("*.json"))
        + [Path(__file__), ROOT / "evaluation" / "metrics.py"]
    )
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest(), manifest


def summarize(logs, pipe, beta_before=(0, 0)):
    by = defaultdict(list)
    for row in logs:
        by[row["user_id"]].append(row)
    users = {}
    chosen_counts, recommended_counts, categories, difficulties = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    nonfirst = 0
    for uid, rows in sorted(by.items()):
        selected = [r["chosen_task_id"] for r in rows if r["chosen_task_id"]]
        users[uid] = {
            "reward": sum(r["reward"] for r in rows),
            "IER": float(np.mean([r["ier"] for r in rows])),
            "SVR": safety_violation_rate(rows),
            "acceptance": sum(r["action"] == "accept" for r in rows) / len(rows),
            "completion": sum(bool(r["feedback"].get("completed")) for r in rows)
            / len(rows),
            "assigned": sum(bool(r["recommended"]) for r in rows) / len(rows),
            "unique_tasks_per_user": len(set(selected)),
        }
    for r in logs:
        tid = r["chosen_task_id"]
        chosen_counts[tid] += 1
        recommended_counts.update(r["recommended"])
        task = pipe.task_lookup[tid]
        categories[task["category"]] += 1
        difficulties[str(task["difficulty"])] += 1
        nonfirst += bool(r["recommended"] and tid != r["recommended"][0])
    metrics = {k: float(np.mean([u[k] for u in users.values()])) for k in METRICS}
    metrics["SVR"] = mean_user_svr(logs)
    model = pipe.recommender.model
    correctness = {
        "interactions": len(logs),
        "skips": sum(r["action"] == "skip" for r in logs),
        "skips_with_physical_outcome": sum(
            r["action"] == "skip" and "pain_change" in r["feedback"] for r in logs
        ),
        "completed_counter": sum(pipe.simulator._user_tasks_completed.values()),
        "completed_feedback": sum(bool(r["feedback"].get("completed")) for r in logs),
        "attempted": sum(r["action"] == "accept" for r in logs),
        "violations": sum(bool(r["safety_violation"]) for r in logs),
    }
    if hasattr(model, "safe_alpha"):
        correctness["beta_positive"] = (
            sum(a - model._a0 for a in model.safe_alpha.values()) - beta_before[0]
        )
        correctness["beta_negative"] = (
            sum(b - model._b0 for b in model.safe_beta.values()) - beta_before[1]
        )
        assert (
            correctness["beta_positive"] + correctness["beta_negative"]
            == correctness["attempted"]
        )
    assert correctness["skips_with_physical_outcome"] == 0
    assert correctness["completed_counter"] == correctness["completed_feedback"]
    assert metrics["assigned"] == 1
    prompt_log = pipe._prompt_level_log
    narrative = {
        "render_calls": pipe.narrator.total_render_count,
        "fallbacks": dict(pipe.narrator.fallback_counter),
        "speak_calls": sum(p.get("scenario") != "daily_closing" for p in prompt_log),
        "closing_calls": sum(p.get("scenario") == "daily_closing" for p in prompt_log),
        "tones": dict(
            sorted(
                Counter(
                    p.get("tone_actually_used", "unknown") for p in prompt_log
                ).items()
            )
        ),
        "samples": [
            {
                "user_id": r["user_id"],
                "day": r["day"] + 1,
                "task": r["chosen_task_id"],
                "text": r["narrative"],
                "tone": r["narrator_tone"],
            }
            for r in logs
            if r["user_id"] == "U001" and r["day"] in [0, 14, 15, 29, 39, 40, 59]
        ],
    }
    return {
        "metrics": metrics,
        "per_user": users,
        "correctness": correctness,
        "diagnostics": {
            "chosen_task_counts": dict(sorted(chosen_counts.items())),
            "recommended_task_counts": dict(sorted(recommended_counts.items())),
            "chosen_categories": dict(sorted(categories.items())),
            "chosen_difficulties": dict(sorted(difficulties.items())),
            "nonfirst_choice_rate": nonfirst / len(logs),
            "global_unique_chosen": len(chosen_counts),
        },
        "narrative": narrative,
    }


def run(
    policy, seed, k=1, tau=2.0, safety=True, cohort=None, posterior=None, freeze=False
):
    override = {
        "system": {"seed": seed},
        "recommender": {"algorithm": policy, "top_k": k, "saps": {"tau_safe": tau}},
        "safety": {"enable": safety},
    }
    if policy == "neural_ucb":
        override["recommender"]["neural_ucb"] = {
            "hidden_dim": 32,
            "training_epochs": 5,
            "alpha": 0.01,
        }
        override["reward"] = {"safety_lambda": 3.0}
    pipe = NarraCarePipeline(config_overrides=override)
    if cohort is not None:
        pipe.simulator.profiles = {
            u: p for u, p in pipe.simulator.profiles.items() if u in cohort
        }
    model = pipe.recommender.model
    if posterior is not None:
        for attr, value in posterior.items():
            setattr(model, attr, copy.deepcopy(value))
    beta_before = (
        sum(
            a - getattr(model, "_a0", 1)
            for a in getattr(model, "safe_alpha", {}).values()
        ),
        sum(
            b - getattr(model, "_b0", 1)
            for b in getattr(model, "safe_beta", {}).values()
        ),
    )
    if freeze:
        pipe.recommender.update = lambda context, tid, reward: None
    with tempfile.TemporaryDirectory(prefix="narracare-final-") as tmp:
        old = os.getcwd()
        try:
            os.chdir(tmp)
            logs = pipe.run_simulation(n_days=60)
        finally:
            os.chdir(old)
    result = summarize(logs, pipe, beta_before)
    result.update(
        policy=policy,
        seed=seed,
        top_k=k,
        tau_safe=tau if policy == "saps" else None,
        safety_enabled=safety,
    )
    posterior = {
        attr: copy.deepcopy(getattr(model, attr))
        for attr in ["A", "b", "safe_alpha", "safe_beta"]
        if hasattr(model, attr)
    }
    return result, posterior


def aggregate(rows):
    return {
        m: {
            "mean": float(np.mean([r["metrics"][m] for r in rows])),
            "seed_std": float(np.std([r["metrics"][m] for r in rows], ddof=1))
            if len(rows) > 1
            else None,
        }
        for m in METRICS
    }


def paired(a, b, metric, family=1):
    x = np.array([r["metrics"][metric] for r in a])
    y = np.array([r["metrics"][metric] for r in b])
    d = x - y
    if len(d) < 2 or np.std(d) == 0:
        ci = [float(d.mean()), float(d.mean())]
        p = None
    else:
        ci = list(
            map(
                float,
                stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)),
            )
        )
        p = float(stats.ttest_rel(x, y).pvalue)
    return {
        "a_minus_b": float(d.mean()),
        "ci95": ci,
        "raw_p": p,
        "bonferroni_p": min(1, p * family) if p is not None else None,
        "n_seed_pairs": len(d),
    }


def suite(out, selection="all"):
    source, manifest = fingerprint()
    write(
        out / "protocol.json",
        {"protocol": PROTOCOL, "source_fingerprint": source, "source_files": manifest},
    )

    def cached(policy, seed, k=1, tau=2.0, safety=True):
        name = f"{policy}-s{seed}-k{k}-t{tau:g}-safe{int(safety)}.json"
        path = out / "runs" / name
        if path.exists():
            obj = json.loads(path.read_text())
            if obj.get("source_fingerprint") == source:
                return obj
        result, _ = run(policy, seed, k, tau, safety)
        result["source_fingerprint"] = source
        write(path, result)
        print(
            f"RUN {policy} seed={seed} k={k} tau={tau:g} safety={safety} reward={result['metrics']['reward']:.4f}",
            flush=True,
        )
        return result

    if selection in ["all", "diagnostics"]:
        diagnostic = {}
        for policy in ["saps", "linucb", "random"]:
            diagnostic[policy] = {}
            for k in [1, 3]:
                rows = [cached(policy, s, k) for s in DEVELOPMENT_SEEDS]
                diagnostic[policy][str(k)] = {
                    "aggregate": aggregate(rows),
                    "runs": rows,
                    "nonfirst_choice_rate": float(
                        np.mean(
                            [r["diagnostics"]["nonfirst_choice_rate"] for r in rows]
                        )
                    ),
                    "categories": dict(
                        sum(
                            (
                                Counter(r["diagnostics"]["chosen_categories"])
                                for r in rows
                            ),
                            Counter(),
                        )
                    ),
                    "difficulties": dict(
                        sum(
                            (
                                Counter(r["diagnostics"]["chosen_difficulties"])
                                for r in rows
                            ),
                            Counter(),
                        )
                    ),
                }
        write(
            out / "shortlist.json",
            {
                "protocol": "Exploratory development seeds 42–51; list setting changes stochastic draw consumption; coverage is descriptive, not a causal intervention.",
                "results": diagnostic,
            },
        )
    if selection in ["all", "final"]:
        primary = {p: [cached(p, s) for s in FINAL_SEEDS] for p in POLICIES}
        write(
            out / "main.json",
            {
                "setting": "Final top-1; ten previously uninspected seed runs 52–61; same 20 synthetic personas. Not held-out patient data.",
                "aggregate": {p: aggregate(v) for p, v in primary.items()},
                "runs": primary,
                "reward_comparisons": {
                    p: paired(primary["saps"], primary[p], "reward", 2)
                    for p in ["linucb", "random"]
                },
            },
        )
        off = [cached("saps", s, tau=0.0) for s in FINAL_SEEDS]
        write(
            out / "bonus.json",
            {
                "setting": "Fixed tau=2 versus tau=0; no grid or parameter selection. Top-1 final seeds.",
                "on": aggregate(primary["saps"]),
                "off": aggregate(off),
                "comparisons": {
                    m: paired(primary["saps"], off, m, 3)
                    for m in ["reward", "IER", "acceptance"]
                },
            },
        )
        no_safety = [cached("linucb", s, safety=False) for s in FINAL_SEEDS]
        write(
            out / "safety.json",
            {
                "setting": "LinUCB with/without runtime safety; emotional difficulty cap retained. SVR uses full five-rule audit; feasibility labels are authored, not clinical ground truth.",
                "on": aggregate(primary["linucb"]),
                "off": aggregate(no_safety),
                "pooled": {
                    "on_violations": sum(
                        r["correctness"]["violations"] for r in primary["linucb"]
                    ),
                    "on_attempted": sum(
                        r["correctness"]["attempted"] for r in primary["linucb"]
                    ),
                    "off_violations": sum(
                        r["correctness"]["violations"] for r in no_safety
                    ),
                    "off_attempted": sum(
                        r["correctness"]["attempted"] for r in no_safety
                    ),
                },
            },
        )
        cold = {}
        for p in ["saps", "linucb"]:
            train, post = run(p, 42, cohort=TRAIN)
            frozen, _ = run(p, 42, cohort=TEST, posterior=post, freeze=True)
            fresh, _ = run(p, 42, cohort=TEST)
            cold[p] = {
                "train": train,
                "reward_frozen_safety_adaptive": frozen,
                "fresh_online": fresh,
            }
        write(
            out / "coldstart.json",
            {
                "setting": "Exploratory seed42; train 15 ACL/RC users, evaluate five unseen BP personas. Reward updates frozen in transfer; SAPS Beta adapts. No validation cohort.",
                "train_ids": TRAIN,
                "rollout_ids": TEST,
                "results": cold,
            },
        )
        write(
            out / "narrator.json",
            {
                "setting": "Current top-1 primary runs. Template execution checks, not a narrative-quality or patient study.",
                "results": {
                    p: [{"seed": r["seed"], **r["narrative"]} for r in rows]
                    for p, rows in primary.items()
                },
            },
        )
    return source


def compare(reference, actual):
    failures = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                failures.append(path + " keys")
                return
            for k in a:
                walk(a[k], b[k], path + "/" + k)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                failures.append(path + " length")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + f"/{i}")
        elif (
            isinstance(a, (float, int))
            and not isinstance(a, bool)
            and isinstance(b, (float, int))
            and not isinstance(b, bool)
        ):
            if not np.isclose(a, b, rtol=1e-10, atol=1e-12):
                failures.append(path)
        elif a != b:
            failures.append(path)

    for name in [
        "protocol.json",
        "shortlist.json",
        "main.json",
        "bonus.json",
        "safety.json",
        "coldstart.json",
        "narrator.json",
    ]:
        walk(
            json.loads((reference / name).read_text()),
            json.loads((actual / name).read_text()),
            name,
        )
    if failures:
        raise RuntimeError("Reproduction mismatch: " + ", ".join(failures[:10]))
    print(
        "PASS: 7/7 final artifact comparisons (rtol=1e-10, atol=1e-12; no metrics excluded).",
        flush=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--suite", choices=["all", "diagnostics", "final"], default="all")
    p.add_argument("--reference", type=Path)
    a = p.parse_args(argv)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    suite(out, a.suite)
    if a.reference:
        compare(a.reference.resolve(), out)
    print("PASS: selected evaluation stages completed.", flush=True)


if __name__ == "__main__":
    main()

"""Offline system demonstration; the frozen pipeline owns all simulation logic.

Daily traces are generated in an isolated temporary subprocess. Comparison
metrics are read verbatim from frozen 60-day, 20-persona aggregate artifacts;
they do not describe the selected user's shorter demonstration trajectory.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MISSING = "N/A (not in frozen artifacts)"


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def policies() -> dict[str, str]:
    """Discover supported factory branches, exposing the CLI neuralucb name."""
    tree = ast.parse(source("src/agents/recommender.py"))
    factory = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_build_model")
    names = []
    for node in ast.walk(factory):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "algo"):
            names.extend(v.value for v in node.comparators
                         if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return {"neuralucb" if n == "neural_ucb" else n: n for n in names}


def inspect_system() -> None:
    """Read static source/configuration/data only; instantiate no components."""
    import yaml

    rules = read_json("data/safety_rules.json")
    total = sum(len(value) for value in rules.values())
    if total != 51:
        raise ValueError(f"Safety Layer B contains {total} rules; expected 51")
    saps_text = source("src/policies/saps.py")
    pipeline_text = source("src/pipeline.py")
    safety_text = source("src/agents/safety_agent.py")
    saps_cfg = yaml.safe_load(source("configs/saps.yaml"))
    formula = next(line.strip() for line in saps_text.splitlines()
                   if line.strip().startswith("composite ="))
    constant = next(line.strip() for line in pipeline_text.splitlines()
                    if line.startswith("SAPS_PAIN_SAFE_THRESHOLD ="))
    comparison = next(line.strip() for line in pipeline_text.splitlines()
                      if "_was_safe =" in line)

    def cite(path: str, fragment: str) -> None:
        lines = source(path).splitlines()
        number = next(i for i, line in enumerate(lines, 1) if fragment in line)
        print(f"  Source: {path}:{number}")

    print("=== NarraCare System Inspection ===")
    print(f"SAPS composite score: {formula}")
    cite("src/policies/saps.py", formula)
    print(f"tau_safe: {saps_cfg['recommender']['saps']['tau_safe']}")
    cite("configs/saps.yaml", "tau_safe:")
    print(f"Safety signal: {constant}; {comparison}")
    cite("src/pipeline.py", constant)
    cite("src/pipeline.py", comparison)
    print("\nSafety Layer A (runtime pre-filter, 5 rules PF_R1..PF_R5):")
    pre_filter = next(n for n in ast.walk(ast.parse(safety_text))
                      if isinstance(n, ast.FunctionDef) and n.name == "pre_filter")
    lines = safety_text.splitlines()[pre_filter.lineno - 1:pre_filter.end_lineno]
    for line in lines:
        match = re.search(r"# Rule ([1-5]):", line)
        if match:
            comment = line.strip()
            print(f"  PF_R{match.group(1)}  {comment}")
            cite("src/agents/safety_agent.py", comment)
    print("\nSafety Layer B (design-time clinical rules):")
    for key, value in rules.items():
        print(f"  {key}: {len(value)}")
    print(f"  Total: {total} rules")
    cite("data/safety_rules.json", '"contraindication_rules"')
    print("\nNarrator tones:")
    print("  " + ", ".join(read_json("data/narrator_tone_templates.json")["tones"]))
    cite("data/narrator_tone_templates.json", '"tones"')
    print("\nNarrator arcs (condition → act keys):")
    arcs = read_json("data/narrative_arcs.json")["arcs"]
    for condition, acts in arcs.items():
        print(f"  {condition}: " + " → ".join(acts))
    cite("data/narrative_arcs.json", '"arcs"')


class Delegate:
    """Transparent instance composition, without patching frozen classes."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return getattr(self.target, name)


def _worker() -> None:
    """Private subprocess entry: run once, write only temporary trace JSON."""
    from src.pipeline import NarraCarePipeline
    from evaluation.metrics import safety_violation_rate, infeasible_exploration_rate
    import yaml

    user, days, policy, seed, destination = sys.argv[1:]

    class SelectedSimulator(Delegate):
        def get_all_user_ids(self):
            return [user]

    class ObservedSafety(Delegate):
        def pre_filter(self, candidates, user_state, user_profile):
            result = self.target.pre_filter(candidates, user_state, user_profile)
            if pipe.current is not None:
                pipe.current["pre_filter"] = [len(candidates), len(result)]
            return result

        def post_check(self, ranked_tasks, user_state, consecutive_high_intensity=0):
            result = self.target.post_check(
                ranked_tasks, user_state, consecutive_high_intensity)
            if pipe.current is not None:
                pipe.current["ranked"] = ranked_tasks
                kept = {t["task_id"] for t in result}
                pipe.current["post_check_removed"] = [
                    t["task_id"] for t in ranked_tasks if t["task_id"] not in kept]
            return result

    class ObservedNarrator(Delegate):
        def say_daily_closing(self, day_log, user_profile, context):
            result = self.target.say_daily_closing(day_log, user_profile, context)
            if pipe.current is not None:
                pipe.current["closing"] = result
            return result

    class ObservedPipeline(NarraCarePipeline):
        def run_single_turn(self, user_id, user_input_or_state, **kwargs):
            self.current = {"state": dict(user_input_or_state), "day": kwargs["day"]}
            self.trace.append(self.current)
            result = super().run_single_turn(user_id, user_input_or_state, **kwargs)
            act, _ = self.narrator._get_arc_context(result["profile"], kwargs["day"])
            self.current.update({
                "act": act,
                "recommended": result["recommended"],
                "narrative": result["narrative"],
                "narrator_result": result["narrator_result"],
                "ier": result["ier"],
            })
            return result

    overrides = yaml.safe_load(source("configs/saps.yaml")) if policy == "saps" else {}
    overrides.setdefault("recommender", {})["algorithm"] = policy
    overrides["system"] = {"seed": int(seed)}
    # Disable legacy narrator; v1.3.2 speak/closing use the existing templates.
    overrides["narrator"] = {"enable": False}
    overrides["narrator_v2"] = {"enable": True}
    pipe = ObservedPipeline("configs/default.yaml", config_overrides=overrides)
    pipe.trace, pipe.current = [], None
    pipe.simulator = SelectedSimulator(pipe.simulator)
    pipe.safety_agent = ObservedSafety(pipe.safety_agent)
    pipe.narrator = ObservedNarrator(pipe.narrator)
    logs = pipe.run_simulation(n_days=int(days))
    by_day = {log["day"]: log for log in logs}
    for row in pipe.trace:
        log = by_day.get(row["day"])
        if log is None:
            row.update(action="no_interaction", reward=0.0, chosen_task_id="")
        else:
            row.update({key: log[key] for key in
                        ("action", "reward", "chosen_task_id", "feedback")})
        if "closing" not in row or not row["narrator_result"]:
            raise RuntimeError("Narrator did not return a complete daily trace")
    summary = {
        "reward": sum(log["reward"] for log in logs),
        "SVR": safety_violation_rate(logs, pipe.task_lookup),
        "IER": infeasible_exploration_rate(logs),
        "accepted": sum(log["action"] == "accept" for log in logs),
    }
    summary["violations"] = round(summary["SVR"] * summary["accepted"])
    Path(destination).write_text(json.dumps({"trace": pipe.trace, "summary": summary}),
                                 encoding="utf-8")


def trajectory(user: str, days: int, policy: str, seed: int) -> dict:
    """No result-path override exists; isolate execution using a temporary cwd."""
    with tempfile.TemporaryDirectory(prefix="narracare-demo-") as directory:
        output = Path(directory) / "trace.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        process = subprocess.run(
            [sys.executable, "-c", "from narracare.demo import _worker; _worker()",
             user, str(days), policy, str(seed), str(output)],
            cwd=directory, env=env, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(process.stderr.strip() or "Frozen pipeline subprocess failed")
        if process.stderr:
            print(process.stderr, file=sys.stderr, end="")
        return json.loads(output.read_text(encoding="utf-8"))


def quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def print_trajectory(data: dict, policy: str) -> None:
    for row in data["trace"]:
        state = row["state"]
        print(f"--- Day {row['day'] + 1} ---")
        print(f"[C1 UserSimulator] state: pain={state['pain_level']:.1f} "
              f"fatigue={state['fatigue_level']:.1f} mood={state['mood']:.1f} "
              f"act={row['act']} day={row['day']}")
        before, after = row["pre_filter"]
        print(f"[C3 SafetyAgent/pre_filter] {before} → {after} (filtered {before-after})")
        recommended = row["ranked"]
        if recommended:
            task = recommended[0]
            score = f"{task['score']:.2f}" if "score" in task else "N/A"
            print(f"[C2 Recommender/{policy}] top-1 = {task['task_id']} "
                  f"(score={score})")
        else:
            print(f"[C2 Recommender/{policy}] top-1 = none")
        removed = row["post_check_removed"]
        verdict = ("BLOCK; post_check removed " + ", ".join(removed)) if removed else "ALLOW"
        print(f"[C3 SafetyAgent/post_check] verdict = {verdict}")
        tone = row["narrator_result"]["tone_actually_used"]
        print(f"[C4 Narrator/tone={tone}] {quoted(row['narrative'])}")
        action = {"accept": "accept", "skip": "reject"}.get(row["action"], row["action"])
        print(f"[C1 UserSimulator] user = {action}; chosen_task_id={row['chosen_task_id']}")
        print(f"[C5 RewardComposer] reward={row['reward']:.4f}")
        closing, meta = row["closing"]
        print(f"[C4 Narrator/closing={meta['daily_closing_variant']}] {quoted(closing)}")
    summary = data["summary"]
    print("=== Trajectory Summary ===")
    print(f"Cumulative reward: {summary['reward']:.2f}")
    print(f"SVR: {summary['SVR']:.4f} ({summary['violations']} five-rule violations)")
    print(f"IER: {summary['IER']:.4f}")
    # Retain actual transitions, including a repeated act after another act.
    progression = []
    for row in data["trace"]:
        if not progression or progression[-1] != row["act"]:
            progression.append(row["act"])
    print("Narrator arc progression: " + "→".join(progression))


ARTIFACTS = {
    "saps": "saps_60day.json", "linucb": "linucb_60day.json",
    "neuralucb": "neuralucb_60day.json", "random": "random_60day.json",
    "popularity": "popularity_60day_seed42.json",
    "rule_based": "rule_based_60day_seed42.json",
    "epsilon_greedy": "epsilon_greedy_60day_seed42.json",
}


def print_comparison(selected: list[str], trajectories: dict) -> None:
    print("=== Demo trajectories (selected user and days) ===")
    print("Day | " + " | ".join(selected))
    for index in range(len(next(iter(trajectories.values()))["trace"])):
        cells = []
        for policy in selected:
            row = trajectories[policy]["trace"][index]
            top = row["ranked"][0]["task_id"] if row["ranked"] else "none"
            before, after = row["pre_filter"]
            verdict = "BLOCK" if row["post_check_removed"] else "ALLOW"
            cells.append(f"top-1={top}; filter={before}→{after}; {verdict}; action={row['action']}")
        print(f"{index + 1} | " + " | ".join(cells))
    print("=== Frozen artifact comparison (60 days, 20 personas; aggregate means) ===")
    print("Selected --user/--days/--seed do not change these frozen metrics.")
    print("Policy | Cumulative Reward | SVR | IER | Act distribution")
    for policy in selected:
        artifact = read_json("evaluation/results/" + ARTIFACTS[policy])
        metrics = artifact.get("aggregate_metrics", artifact)
        values = [metrics.get(key, MISSING) for key in
                  ("cum_reward_mean", "svr", "ier_mean", "act_distribution")]
        print(policy + " | " + " | ".join(str(value) for value in values))
        print(f"Source: evaluation/results/{ARTIFACTS[policy]}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        if "--inspect" in argv:
            if any(arg.split("=")[0] in {"--user", "--days", "--policy", "--seed"}
                   for arg in argv):
                print("warning: --inspect ignores --user/--days/--policy/--seed", file=sys.stderr)
            inspect_system()
            return 0
        import yaml
        from src.simulator import UserSimulator

        available = policies()
        config = yaml.safe_load(source("configs/default.yaml"))
        users = UserSimulator(config).get_all_user_ids()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--user", default="U001", choices=users)
        parser.add_argument("--days", type=int, default=60)
        parser.add_argument("--policy", default="saps", choices=available)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--compare", nargs="?", const=",".join(ARTIFACTS),
                            help="comma-separated policies; no value selects all seven")
        parser.add_argument("--inspect", action="store_true", help="inspect static system sources")
        args = parser.parse_args(argv)
        if not 1 <= args.days <= 60:
            parser.error("--days must be in [1, 60]")
        if args.seed < 0:
            parser.error("--seed must be a non-negative integer")
        if args.compare is not None:
            selected = args.compare.split(",")
            if any(p not in available for p in selected) or len(set(selected)) != len(selected):
                parser.error("--compare requires distinct policies from: " + ", ".join(available))
            traces = {p: trajectory(args.user, args.days, available[p], args.seed)
                      for p in selected}
            print_comparison(selected, traces)
        else:
            print_trajectory(trajectory(args.user, args.days, available[args.policy], args.seed),
                             args.policy)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

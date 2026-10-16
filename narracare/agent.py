"""CLI for the live conversational agent: python -m narracare.agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conversation.controller import ConversationAgent
from .conversation.llm import OpenAIPlanner, ProviderError


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NarraCare LLM recommender agent for synthetic rehabilitation demos"
    )
    root = Path(__file__).resolve().parents[1]
    users = [
        u["user_id"] for u in json.loads((root / "data/user_profiles.json").read_text())
    ]
    parser.add_argument("--user", default="U001", choices=users)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--day",
        type=int,
        default=1,
        help="Starting narrative day, 1–60; no clinical stage promotion",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Optional new JSON transcript outside protected repository paths",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.day <= 60:
        parser.error("--day must be in [1, 60]")
    if args.transcript:
        dest = args.transcript.resolve()
        if dest.is_relative_to(root):
            parser.error(
                "Write transcripts outside the repository to protect benchmark artifacts."
            )
        if dest.exists():
            parser.error("Transcript already exists; choose a new output file.")
    try:
        planner = OpenAIPlanner.from_env()
        agent = ConversationAgent(planner, args.user, args.seed, args.day)
    except ValueError as exc:
        parser.error(str(exc))
    except ProviderError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("NarraCare · Pip's live companion trail")
    print(
        "Synthetic profile. LLM tool decisions; hard Safety; template Narrator. Not patient-care advice."
    )
    print(
        "Messages and synthetic session context are sent to the configured LLM. Do not enter identifying patient data."
    )
    print("Example: Pain 3, fatigue 2, mood 7, ROM 110. I prefer flexibility.")
    print("Report actual completed/partial outcomes, or skip. Type quit to exit.")
    transcript = []
    exit_code = 0
    try:
        while not agent.tools.ended:
            try:
                text = input("You > ")
            except EOFError:
                break
            if text.strip().lower() in {"quit", "exit"}:
                print("Pip > Session ended.")
                break
            answer = agent.reply(text)
            print("Pip > " + answer)
            new_events = agent.tools.snapshot()["events"]
            print(
                "[Tool activity] "
                + ", ".join(f"{e['tool']}:{e['status']}" for e in new_events[-10:])
            )
            if agent.last_error:
                print(agent.last_error, file=sys.stderr)
                exit_code = 1
                break
            transcript.append(
                {"user": text, "assistant": answer, "state": agent.tools.snapshot()}
            )
    except KeyboardInterrupt:
        print("\nSession stopped.")
    if args.transcript:
        try:
            with args.transcript.open("x", encoding="utf-8") as out:
                json.dump(transcript, out, ensure_ascii=False, indent=2)
        except OSError:
            print("Could not write the optional transcript.", file=sys.stderr)
            return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

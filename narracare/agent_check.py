"""Opt-in REAL provider smoke test. Never substitutes a scripted planner."""

import json
import sys

from .conversation.controller import ConversationAgent
from .conversation.llm import OpenAIPlanner, ProviderError


def main():
    try:
        agent = ConversationAgent(OpenAIPlanner.from_env())
    except ProviderError as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr)
        return 1

    def turn(text, check, label):
        start = len(agent.tools.events)
        answer = agent.reply(text)
        events = agent.tools.events[start:]
        if not check(events):
            print(
                json.dumps(
                    {
                        "case": label,
                        "result": "FAIL",
                        "answer": answer,
                        "events": events,
                    },
                    ensure_ascii=False,
                )
            )
            raise AssertionError(label)
        print(
            json.dumps(
                {
                    "case": label,
                    "result": "PASS",
                    "tools": [e["tool"] for e in events],
                    "answer": answer,
                },
                ensure_ascii=False,
            )
        )

    try:
        turn(
            "Hello Pip. Could you recommend a manageable exercise?",
            lambda e: (
                not agent.tools.pending
                and any(
                    x["tool"] == "ask_clarification" and x["status"] == "ok" for x in e
                )
            ),
            "clarification",
        )
        turn(
            "My pain is 3, fatigue 2, mood 7 and ROM 110. I prefer flexibility.",
            lambda e: (
                bool(agent.tools.pending)
                and agent.tools.pending["category"] == "flexibility"
            ),
            "retrieval/ranking/narrator",
        )
        turn(
            "Why did you choose this task?",
            lambda e: any(
                x["tool"] == "explain_recommendation" and x["status"] == "ok" for x in e
            ),
            "grounded explanation",
        )
        turn(
            "I prefer balance instead; avoid flexibility.",
            lambda e: (
                bool(agent.tools.pending)
                and agent.tools.pending["category"] == "balance"
                and "flexibility" in agent.tools.avoided
            ),
            "preference revision",
        )
        turn(
            "I'll try the task. What outcome feedback do you need?",
            lambda e: len(agent.tools.days) == 0,
            "intention does not fabricate completion",
        )
        turn(
            "Completed. After pain 3, fatigue 3, satisfaction 0.8.",
            lambda e: (
                len(agent.tools.days) == 1
                and agent.tools.days[0]["reward"] is not None
                and bool(agent.tools.recommender.model.safe_alpha)
            ),
            "actual feedback update",
        )
        total = agent.tools.snapshot()["cumulative_reward"]
        turn(
            "Skip today and rest.",
            lambda e: (
                len(agent.tools.days) == 2
                and agent.tools.days[-1]["reward"] is None
                and agent.tools.snapshot()["cumulative_reward"] == total
            ),
            "camp without invented reward",
        )
        turn(
            "I have sudden sharp pain and swelling.",
            lambda e: agent.tools.on_hold,
            "red-flag hold",
        )
    except AssertionError:
        return 1
    print(
        "PASS: real-provider interaction smoke. This is engineering verification, not a user study."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

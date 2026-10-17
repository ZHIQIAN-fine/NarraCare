# Versions and entry points

**main is the final submission.** Its root README, `evaluation/results/final/`
and `submission/` describe one consistent top-1 protocol and current agent.

Earlier results/reports/replay/runners are under [archive/](../archive/README.md).
The historical JSON bytes are preserved with a SHA256 manifest; differences in
skip bookkeeping, day alignment and list size mean old and final reward tables
must not be interpreted as identical protocols. Archived runners are snapshots;
check out the documented historical commit to run them in their original layout.

| Location | Role |
|---|---|
| main | Final source, corrected online protocol, live agent, report and source package |
| agent branch | Earlier presentation delivery; old download URLs remain available |
| benchmark branch | Earlier pre-agent benchmark snapshot |
| archive/pre-submission/ | Historical artifacts and supporting material, secondary navigation |

No branch history is rewritten. The work branch is retained untouched.

| Entry point | Behavior |
|---|---|
| `python -m narracare.agent_app --port 7860` | Interactive LLM chat, companion trail and real state/tool panel |
| `python -m narracare.agent --user U001 --seed 42` | Same live guarded tool controller in a terminal |
| `python -m narracare.demo --user U001 --days 60 --policy saps` | Offline real seeded pipeline in temporary storage |
| `archive/pre-submission/demo/index.html` | Historical nine-trajectory browser replay, not a live learning agent |

Only live chat needs a provider key/network. Intention does not count as completion;
skips/pain pauses do not invent outcomes or update Beta. Actual reported feedback
updates the session-local learner. Neither the UI nor the seeded benchmark proves
patient recovery or adherence improvement.

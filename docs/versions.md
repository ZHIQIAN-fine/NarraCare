# NarraCare versions and demonstration entry points

Use **main** for current source. The conversational agent is implemented, not a
proposed extension. It uses an OpenAI-compatible LLM (default `gpt-4o-mini`) for
understanding and tool decisions, with the existing SAPS, Safety and template
Narrator behind guarded tools. `requirements-agent.txt` adds the optional Gradio
interface. A provider key and network are required for live interaction.

## Branch roles

| Branch | Contents | Intended use |
|---|---|---|
| `main` | Current agent source plus preserved benchmark | Development and assessment |
| `agent` | A reviewed main snapshot plus `deliverables/agent/` | Download report, deck, script and submission archive |
| `benchmark` | Pre-agent commit `da833fd084361deab3b80aaeb6c3aa0d2a773741` | Inspect the benchmark without the conversational addition |

The agent implementation first appeared at `9801bd4`. Documentation updates do
not change the experimental tables. The submission archive records its exact
source commit in `SUBMISSION_MANIFEST.json` and excludes `.git`, credentials,
environments, caches and old delivery bundles.

Earlier feature and delivery histories are preserved by immutable archive tags,
whose names include their original tip SHA. No old commits are rewritten.
Those tags are historical snapshots, not the recommended demo entry.

## Which demonstration is interactive

| Command or file | Behavior | Requirements |
|---|---|---|
| `python -m narracare.agent_app --port 7860` | Live Gradio chat, trail and state panel. Preference changes reach retrieval; explicit feedback updates reward/model/memory | Python, base and agent requirements, provider key, network |
| `python -m narracare.agent --user U001 --seed 42` | The same live tool controller in a terminal | Python, base requirements, provider key, network |
| `python -m narracare.demo --user U001 --days 60 --policy saps --seed 42` | Runs the original seeded simulation in temporary storage | Python and base requirements; offline after installation |
| `demo/index.html` | Nine embedded recorded trajectories. Replies alter presentation only | A browser; offline |

The live agent never silently falls back to recorded data or a scripted planner.
Saying "I'll try" does not create completion or reward. Actual reported outcomes
are required. Skip creates camp with no invented outcome, reward or trail step.
Narrative day never promotes the clinical stage stored in a synthetic profile.

## Evidence and interpretation

The unchanged benchmark has 20 synthetic personas, 60 days and seven policies.
The live agent has session-local learning, not pretrained benchmark statistics.
Its engineering checks and session rewards are separate from benchmark results.

Verified production-browser sequence: symptoms 3/2/7 with ROM 110; flexibility
task T017 from 10 retrieved items; balance preference revision to T029 from 4;
Safety 100 to 77 in both cases; no reward for intention; reported completion
reward 0.9401 and one trail step; subsequent camp retains those totals. Eight
real-provider smoke cases passed in one scripted synthetic sequence, and the
full regression suite passed 67 tests. None establishes clinical efficacy,
complete medical safety or improved adherence. Evidence is included with the
agent download bundle. Keep credentials outside source and use synthetic inputs.

# NarraCare — final coursework submission

A conversational rehabilitation recommender with a narrative companion: an LLM
interprets check-ins and chooses guarded tools; SAPS ranks catalogue tasks;
independent runtime constraints check delivery; a template Narrator presents the
result. A separate online benchmark evaluates recommendation behavior.

**Start here:** [final report and source archive](submission/README.md) ·
[agent architecture](docs/conversational_agent.md) ·
[data provenance](data/README.md) · [experiment protocol](evaluation/README.md).
Historical reports, results, replay and runners are under [archive/](archive/README.md).
They are not the final experimental evidence. No clinical effectiveness or
patient-safety claim follows from this synthetic coursework prototype.

## Objective and architecture

The practical aim is to make personalized task selection, feasibility constraints
and feedback adaptation inspectable together. ACL/rotator-cuff profiles describe
postoperative rehabilitation; BP describes chronic back pain, not back surgery.
There is no medical-record importer. Profiles are synthetic, current symptoms
are explicitly self-reported, and unknown ROM is clearly marked.

```text
Live: English check-in → LLM/tool controller → state and preference memory
      → catalogue retrieval + Safety → SAPS → final Safety checks
      → template Narrator → reported outcome → Reward Composer/model update
Benchmark: simulated state → Safety → policy → Safety → template Narrator
           → simulated choice/outcome → Reward Composer/update → daily closing
```

SAPS adapts disjoint LinUCB reward estimates with a sampled per-task Beta bonus:
`S(a) = r_ucb(a) + tau_safe * s`, `s ~ Beta(alpha_a, beta_a)`, `tau_safe = 2.0`.
The bonus affects ranking; independent predicates constrain delivered tasks.
It is not a clinical safety probability. Only attempted tasks produce physical
outcomes and Beta observations. Skips produce no invented recovery or satisfaction.

## Installation and data preparation

Python 3.10+ and CPU PyTorch are supported. The verification environment used
Python 3.12, NumPy 2.0.2, SciPy 1.14.1 and PyTorch 2.4.1+cpu.

```bash
git clone https://github.com/ZHIQIAN-fine/NarraCare.git
cd NarraCare
python -m pip install -r requirements.txt
# Optional live graphical interface:
python -m pip install -r requirements-agent.txt
# Regression checks:
python -m pip install pytest
```

All static inputs are included in `data/`: 20 synthetic personas, 100 authored
tasks, 51 design-rule definitions and narrative templates/arcs. There is no
public patient dataset, offline interaction-training set or pretrained download.
See the provenance and limitations in [data/README.md](data/README.md).

## Run the demonstration

### Interactive LLM agent — recommended system demonstration

Configure `OPENAI_API_KEY` or `NARRACARE_LLM_API_KEY` securely in your environment.
Do not put credentials in source. Default model: `gpt-4o-mini`; optional
`NARRACARE_LLM_MODEL` and `NARRACARE_LLM_BASE_URL` select a compatible HTTPS provider.

```bash
python -m narracare.agent_app --port 7860
# Terminal interface to the same agent:
python -m narracare.agent --user U001 --seed 42
```

The Gradio ChatInterface has a companion mountain trail and right-hand state/tool
panel. Guided forms prepare messages for review; **Send** still calls the LLM.
Try a check-in, a category preference, a preference revision and actual completion
feedback. “I'll try” does not earn reward. Skipping creates camp without a model
outcome update; explicitly reporting pain prevents all activity permits a pause.
Ordinary days aim to offer a suitable task, including recovery tasks.

The live agent requires a provider key and network, never substitutes a scripted
planner, and sends synthetic demo messages/context to the provider. It keeps
session learning in RAM and does not write benchmark results. Unknown ROM uses
an explicitly labelled conservative filtering proxy, not a measured value.
Satisfaction is recorded descriptively and does not enter the reward formula.

### Offline five-component benchmark CLI

```bash
python -m narracare.demo --user U001 --days 60 --policy saps --seed 42
python -m narracare.demo --inspect
python -m narracare.demo --user U001 --days 7 --compare saps,linucb,neuralucb
# No list selects all seven policies:
python -m narracare.demo --days 7 --compare
```

No API key/network is needed after installation. Defaults are U001 / 60 days /
SAPS / seed 42; days range 1–60. The CLI runs real components in temporary
storage and prints state, counts, task, tone, choice, reward and closing.
The comparison table reads **final top-1, 20-persona, 60-day, seeds 52–61 means**;
it does not describe the selected user's short trace. The historical standalone
HTML replay is in `archive/pre-submission/demo/`, not the interactive agent.

### Mapping to CA6124 Section 6

The live agent demonstrates natural-language understanding, clarification,
preference memory, retrieval, ranking, grounded presentation/explanation and
reported-feedback adaptation. The CLI separately exposes the five benchmark
components and their actual execution order. The trail/companion is an application
design: its effect on adherence or recovery has not been measured.

## Training, evaluation and reproduction

Learning is online, predicting before observed feedback, with one learner shared
across the 20 simulated users in each run. **Top-1** matches the one-task live
interface. Top-1 versus Top-3 is a separate development diagnostic (seeds 42–51).
Final seven-policy results use seeds **52–61**, the same personas, 60 days each.
These are independent seeded runs, not held-out real patients. The experiment
runner trains by online rollout; it needs no separate offline training command.
For exploratory settings, use `NarraCarePipeline(config_overrides=...)` or a
versioned runner override and a separate output directory. Formal overrides
are listed in evaluation/README.md and take precedence over YAML defaults;
changing YAML alone does not tune every formal policy. No tau search or
performance-based setting selection was performed.

```bash
# Regenerate all final evidence into temporary storage and compare references:
bash scripts/reproduce_all.sh
# Generate a separate experiment directory explicitly:
python -m evaluation.run_submission --output-dir /tmp/narracare-new-results
python -m pytest -q
# Optional real-provider engineering smoke (network/key/billing required):
python -m narracare.agent_check
```

The reproducer regenerates all seven top-level artifacts, compares every field
with numeric `rtol=1e-10, atol=1e-12`, changes no committed reference, and exits
nonzero on mismatch. The protocol records source/data/config SHA256 values.
See [evaluation/README.md](evaluation/README.md) for exact settings and metrics.

## Final results and interpretation

Mean per-person 60-day reward, then mean across ten final seeds:

| Policy | Cumulative reward | SVR | IER |
|---|---:|---:|---:|
| SAPS | 27.6085 | 0.0000 | 0.01535 |
| LinUCB | 27.3975 | 0.0000 | 0.01528 |
| Modified NeuralUCB | 22.1994 | 0.0000 | 0.00035 |
| Random | 32.2533 | 0.0000 | 0.41082 |
| Popularity | 22.9922 | 0.0000 | 0.01060 |
| RuleBased | 22.0204 | 0.0000 | 0.32597 |
| EpsilonGreedy | 24.7411 | 0.0000 | 0.05245 |

Reward is simulated utility, not recovery. SVR audits accepted tasks against five
implemented predicates; IER measures raw ungated top-5 proposals against the
current feasible set. Low IER does not prove clinical safety or bonus efficacy.
All seven filtered policies had zero recorded violations; LinUCB without runtime
filtering had 1,090/6,834 accepted-task violations (pooled 0.1595; macro SVR 0.1556).

SAPS minus LinUCB reward is +0.2110, 95% seed-paired CI [-0.2159, 0.6378],
adjusted p=0.5850. Random has higher reward and broader task coverage. The fixed
bonus on/off comparison does not establish benefit in reward, IER or acceptance.
Top-3 raises reward for all three diagnostic policies; coverage is descriptive
and has not been established as the cause of Random's advantage.

Modified NeuralUCB has a shared neural context term that cancels in arm ranking;
RuleBased assigns identical scores, with catalogue order breaking ties. These
are limited references. Popularity uses mean reward, not interaction counts.
The strongest evidence is end-to-end integration and inspectable execution of
specified constraints; clinical validation and narrative adherence studies remain
future work. [Archived versions](docs/versions.md) preserve earlier evidence.

## License and citation

MIT: [LICENSE](LICENSE). Project metadata: [CITATION.cff](CITATION.cff).

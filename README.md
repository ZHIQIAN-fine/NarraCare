# NarraCare

A safety-aware narrative recommender for simulated musculoskeletal rehabilitation.
CA6124 Group Project, 2026 Semester 1.

## Overview

NarraCare recommends rehabilitation exercises from patient context and feedback,
with a Safety Agent and a template-based Narrator. All data is synthetic:
20 personas, 100 tasks and a 60-day evaluation horizon. This is a coursework
prototype, not a clinically validated treatment system.

SAPS combines disjoint LinUCB reward UCB with a sampled Beta safety bonus.
Its recommendation output retains an internal safe-arm gate. IER uses a
separate ungated top-5 ranking, which also drives the existing negative reward
updates. The current SAPS IER is 0.0223, not a gate-guaranteed zero.

## System architecture

Actual daily execution in `src/pipeline.py`:

```text
User Simulator state -> patient features and memory
  -> Safety pre-filter and emotional difficulty cap
  -> Recommender -> Safety post-check -> template Narrator
  -> simulated user choice and feedback -> Reward Composer
  -> model updates -> Narrator daily closing
```

The five demonstrated components are User Simulator, Recommender, Safety Agent,
Narrator and Reward Composer. Reward calculation lives in `src/simulator.py`
and is invoked by the pipeline. Active recommendation models are defined in
`src/agents/recommender.py` and `src/policies/saps.py`. `src/models/` is an older,
separate implementation stack, not the factory used by the formal experiments.

## Installation and data preparation

Use Python 3.10 or later; CPU execution is supported. NeuralUCB uses PyTorch.
The review environment used Python 3.12, NumPy 2.0.2, SciPy 1.14.1 and
PyTorch 2.4.1+cpu. Dependency pins are in `requirements.txt`.

```bash
git clone https://github.com/ZHIQIAN-fine/NarraCare.git
cd NarraCare
python -m pip install -r requirements.txt
```

All inputs are included under `data/`; no download or preprocessing is needed.
See [data/README.md](data/README.md). These are authored static JSON files;
experiment scripts regenerate outputs, not the static catalogue or profiles.
The refreshed submission archive `narracare_submission.tar.gz` extracts into
`narracare_final/`. The earlier `narracare_1eb927a.tar.gz` predates this cleanup.

## Run the demonstration

Run from the repository root after installation:

```bash
# Real end-to-end pipeline, with component-labelled daily output
python -m narracare.demo --user U001 --days 60 --policy saps --seed 42

# Inspect source/config/data definitions without running a simulation
python -m narracare.demo --inspect

# Live traces followed by a separately labelled frozen aggregate table
python -m narracare.demo --user U001 --days 7 --compare saps,linucb,neuralucb

# No list means all seven policies
python -m narracare.demo --days 7 --compare
```

`python -m narracare` is equivalent. Defaults: U001, 60 days, saps, seed 42.
Days must be 1–60. Policies: `saps`, `linucb`, `neuralucb`, `random`,
`popularity`, `rule_based`, `epsilon_greedy`.

The CLI prints state, filtering counts, a proposed task and score, post-check
outcome, original narrator text/tone, simulated choice, reward and closing.
It runs the pipeline in a temporary-directory subprocess and does not overwrite
frozen results. **No OpenAI API key or network is needed after installation.**

With `--compare`, daily traces use the selected user/days/seed, but the final
table reads the stored **20-persona, 60-day aggregate artifacts**. The selected
arguments do not change those frozen metrics. Missing fields such as Act
distribution display `N/A (not in frozen artifacts)`. Demo defaults are separate
from policy-specific formal experiment settings.

For the visual demo, open **[demo/index.html](demo/index.html)** in a modern
browser. It is one self-contained file: no Python, server, installation, network
or API key. It embeds U001/U008/U014 x SAPS/LinUCB/NeuralUCB x 60 recorded days.
Replies change the trail presentation, not the saved choices/rewards. It is an
interactive replay; use the CLI to verify runtime component integration.

### Mapping to CA6124 Section 6

One CLI invocation integrates simulation, recommendation, safety, narrative
rendering and feedback. `--inspect` exposes source definitions. The HTML replay
makes the recorded state, safety counts, task, tone and reward accessible without
installing the backend.

## Methods and implementation boundaries

| Policy | Implementation used by the pipeline |
|---|---|
| SAPS | LinUCB reward UCB + sampled Beta safety bonus, with internal safe-arm gate |
| LinUCB | Disjoint linear UCB on 15-dimensional patient context |
| NeuralUCB | Modified arm-mean/noise heuristic plus a shared neural context prediction; that shared term cancels in arm ranking |
| Random | Random candidate scores |
| Popularity | Historical mean reward per task, not acceptance count |
| RuleBased | Identical context-dependent score for every task; stable candidate order resolves ties |
| EpsilonGreedy | Epsilon 0.1 exploration; otherwise historical mean reward |

NeuralUCB is not Zhou et al.'s original gradient-UCB algorithm. RuleBased is an
order-based reference, not a strong task-aware clinical rule ranker.

SAPS: `S(a) = r_ucb(a) + tau_safe * s`, where
`s ~ Beta(alpha_a, beta_a)` and `tau_safe = 2.0`. Only safety is Thompson sampled.
The chosen arm receives a Beta update from `pain_change <= 1.0`; blocked arms
have no post-hoc outcome. Beta parameters are indexed by task and shared across
users. LinUCB reward parameters are per task, with patient context as input.

Safety Layer A has five runtime pre-filter checks: contraindication overlap,
stage mismatch, high pain/high difficulty, severe fatigue/non-gentle category,
and low ROM/demanding or cardio task. Layer B has 51 design definitions:
16 contraindication, 9 dosage, 6 fatigue, 10 crisis, 10 progression. These are
not 51 additional predicates all executed on each recommendation.

## Training and hyperparameter settings

Learning occurs online inside rollouts; no pretrained download is required.
One learner is shared across personas. Formal runner overrides take precedence
over `configs/default.yaml` and policy YAML defaults.

| Setting | Formal experiment value |
|---|---|
| Seed / days / personas | 42 / 60 / 20 |
| Context dimension / shortlist | 15 / 3 |
| SAPS alpha_ucb / tau_safe / Beta priors | 1.0 / 2.0 / (1, 1) |
| LinUCB alpha | 1.0 |
| NeuralUCB hidden width / epochs / alpha | 32 / 5 / 0.01 |
| NeuralUCB reward safety_lambda | 3.0 |
| Other policies' base safety_lambda | 1.0 |
| EpsilonGreedy epsilon | 0.1 |
| Reward completion / safety / difficulty / pain / dropout / category weights | 0.35 / 0.25 / 0.20 / 0.10 / 0.05 / 0.05 |

Exploratory configurations can use `config_overrides`:

```python
from src.pipeline import NarraCarePipeline

pipe = NarraCarePipeline("configs/default.yaml", config_overrides={
    "recommender": {"algorithm": "saps", "saps": {"tau_safe": 2.0}},
    "system": {"seed": 42},
})
logs = pipe.run_simulation(n_days=60)
```

Keep exploratory runs separate from frozen results. No completed tau grid
search or extra seeds 43/44 are claimed by this submission.

## Reproduce experiments

Experiments intentionally write `evaluation/results/`; use a disposable clone
if preserving the original artifacts byte-for-byte is required.

```bash
bash scripts/reproduce_all.sh
bash scripts/reproduce_coldstart.sh
```

The first script executes SAPS/LinUCB, four baseline/NoSafety runs, NeuralUCB
and its historical variant, safety diagnostics, robustness and narrative
sampling. **Random and cold-start appear in its comparison list but are not
regenerated by that script.** Cold-start has the separate command above.
Regenerate Random explicitly with `python evaluation/run_60day.py --method random`.
Regenerate the separate LinUCB summary with `python evaluation/run_60day.py --method linucb`.
Paired-test commands
are in `evaluation/scripts/paired_t_test.py`.

Verification uses exact comparison for five files and removes only
`wall_time_s` before comparing eight others. There is no numeric tolerance.
The repaired version passed **13/13 comparisons, 0 FAIL, 0 SKIP** on two
successive runs in the review environment. This is a same-environment result:
cross-machine BLAS differences around 1e-14 can still fail exact comparison.
Inspect the printed summary because comparison failures do not currently force
a nonzero script exit. Random and cold-start PASS lines compare existing files;
those policies were also regenerated separately for this release.

See [artifact index](evaluation/results/README.md) and
[implementation notes](docs/implementation_notes.md) for coverage and caveats.

### Optional narrative scoring

The Narrator is template-based and offline. Only optional external LLM judging
uses a key. The supported scoring script is `evaluation/llm_judge_colab.py`;
its header describes local/Colab use. Export `OPENAI_API_KEY` or use Colab
secrets; copying `.env.example` does not automatically load environment variables.
Later remote model responses are not guaranteed byte-identical. Save new scores
separately from committed scores. Offline aggregation instructions are in
[evaluation/results/README_narrative_eval.md](evaluation/results/README_narrative_eval.md).

## Recorded results and interpretation

Current results after the ungated-IER and five-rule SVR repair, seed 42,
20 personas and 60 days. Reward is mean per-person cumulative reward.

| Policy | Cumulative reward | Ungated IER |
|---|---|---|
| SAPS | 42.0182 +/- 3.0136 | 0.0223 |
| LinUCB | 41.9395 +/- 3.0015 | 0.0233 |
| NeuralUCB modified | 38.31 | 0.0022 |
| Random | 45.94 | 0.4082 |
| Popularity | 39.03 | 0.0127 |
| RuleBased | 43.33 | 0.3257 |
| EpsilonGreedy | 40.95 | 0.0503 |

IER measures the second, ungated all-candidate top-5 against the current feasible
set, including the emotional difficulty cap. The SAPS measurement bypasses its
output gate. Its infeasible proposals also receive the existing negative reward
updates, so repairing the path changes learning and reward, not just a column.

SAPS minus LinUCB reward is +0.0787, t(19)=0.3328, p=0.7429,
95% CI [-0.4163, 0.5737]. The IER difference is -0.0010, p=0.6295.
Neither supports a significant advantage over LinUCB. Random has higher reward.
Against the limited modified NeuralUCB reference, reward difference is +3.7046
(p=1.01e-6), but SAPS has higher IER by 0.02017 (p=1.88e-6).
All tests describe a fixed cohort with shared online learning and one seed,
not independent repeated-run or clinical evidence. The two current paired
comparison artifacts supersede the old gated-IER comparisons.

Every interaction now carries `safety_violation`. Only accepted tasks count,
audited against the full five pre-filter predicates using configured thresholds,
even in NoSafety/one-drop runs. Legacy logs are replayed through SafetyAgent,
including ROM. The seven filtered policies have zero accepted-task violations.
Dedicated pooled replay gives LinUCB WithSafety 0/743 versus NoSafety 171/772
(0.2215). Main formal summaries retain per-person macro averaging, giving
NoSafety 0.2181. Both divide by accepted tasks but weight users differently.
None establishes enforcement of all 51 design rules or clinical safety.

Cold-start SAPS reward is 41.55 on training users, 42.01 on BP reward-frozen
rollout and 43.35 when retrained on BP. Frozen minus Retrained is -1.3420
(raw p=0.0350, n=5). Their IERs are 0.0267 / 0.0080 / 0.0627 respectively.
The lower frozen IER accompanies lower reward, not a reward transfer advantage.
Beta still adapts in the reward-frozen variant, so it is not fully frozen transfer.
These are small, exploratory comparisons without a clinical interpretation.
Other limits include Narrator condition/day fallback, simulator bookkeeping
and weak reference policies; see the implementation notes.

## License and citation

MIT; see [LICENSE](LICENSE). Software metadata: [CITATION.cff](CITATION.cff).

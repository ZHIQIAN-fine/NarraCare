# Final online evaluation

Run `bash scripts/reproduce_all.sh` from the repository root. It uses a temporary
output directory, regenerates every cited suite and compares all seven JSON
artifacts in `results/final/`, including per-run/per-user values. No metric is
excluded. Numeric tolerance is rtol=1e-10, atol=1e-12 for floating-point portability;
strings, keys and list lengths must match. Failure returns nonzero.

`python -m evaluation.run_submission --output-dir /tmp/new-results` generates
new outputs. Set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1
for the documented CPU protocol (the shell reproducer does this). Outputs contain
no timing fields, timestamps or Git dependency. `protocol.json` fingerprints
active source, metrics, runner, configuration and input JSON.

## Protocol and evidence

| Artifact | Experiment |
|---|---|
| protocol.json | Settings, seed ranges and SHA256 source/input manifest |
| main.json | Seven policies, top-1, 20 users × 60 days × seeds 52–61; seed-level comparisons |
| shortlist.json | SAPS/LinUCB/Random top-1 vs top-3, development seeds 42–51; task/category/difficulty coverage |
| bonus.json | SAPS fixed tau=2 vs tau=0 on final seeds; no parameter search |
| safety.json | LinUCB runtime safety on/off, final seeds, macro and pooled violation counts |
| coldstart.json | Exploratory seed42: 15 ACL/RC training users → five BP users; reward-frozen transfer vs fresh online |
| narrator.json | All 70 main runs: render/fallback counts, tones and selected original samples |

Top-1 was chosen to match the live one-task interface before inspecting these
diagnostic outcomes. Final seeds were not used for that decision; the users and
simulator are unchanged, so this is not held-out patient validation. Seed pairing
provides repeated-run uncertainty; different policies consume randomness differently,
so it does not provide matched patient outcome noise. One learner is shared among
users; users are not treated as independent repetitions in main inference.

## Metrics and aggregation

- Reward: sum per user across 60 days, then average users and seeds. Existing
  six-term synthetic utility is scored only for attempted tasks. Skip zero is
  aggregate bookkeeping; no physical outcome or reward update is fabricated.
- Acceptance: attempted tasks / scheduled user-days; completion: completed tasks /
  scheduled user-days. Neither is measured patient adherence.
- SVR: per-user violations / accepted tasks, then mean users and seeds; zero
  when a user has no accepted task. A violation means at least one of the five
  pre-filter predicates fails for that accepted task. Pooled counts are separate.
- IER: mean fraction of a second, ungated all-catalogue top-5 ranking outside the
  current feasible set, including emotional difficulty caps. This probe is
  stochastic for some policies and also drives existing infeasible-arm penalties;
  it is not passive instrumentation. SAPS bypasses its internal output gate for
  this probe. With safety disabled the feasible set changes, so NoSafety IER is
  not comparable to filtered IER as a five-rule safety outcome.
- Coverage: number of distinct selected catalogue IDs per user, including a task
  offered but declined. Descriptive only, not a diversity intervention.
- Assigned fraction checks that each simulated day has a final feasible task.
  A catalogue coverage gap fails explicitly rather than bypassing constraints.

Main reward comparisons use two-sided paired t intervals/tests over ten seed
means; Bonferroni correction covers SAPS versus LinUCB and Random (family of 2).
The bonus family contains reward, IER and acceptance (family of 3). These are
conditional simulator comparisons, not evidence of clinical generalization.

## Fixed settings and limitations

Context dimension 15; SAPS alpha_ucb=1, tau=2, Beta(1,1); LinUCB alpha=1;
epsilon=0.1. Modified NeuralUCB uses hidden width32, 5 epochs, alpha0.01 and
reward safety_lambda3; other policies use safety_lambda1. This inherited reward
setting means NeuralUCB reward is not a perfectly identical-utility comparison.
The original task-aware gradient-UCB algorithm is not implemented. RuleBased
ties every candidate; Popularity means mean reward. See the final report.

No conventional offline train/validation/test split exists. The optional cold
transfer freezes reward updates but leaves SAPS Beta adaptive; it is exploratory
and small, not a fully frozen transfer or clinical validation study. Narrative
checks verify rendering and condition/day alignment, not empathy quality.
Historical LLM-judge scores are archived and are not final-run results.

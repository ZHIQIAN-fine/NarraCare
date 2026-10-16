# Evaluation artifact index

Current numerical results use ungated SAPS IER and explicit accepted-task
five-rule SVR observations. They supersede the earlier gated-IER comparison.
Static input data and narrator templates are unchanged. All current result
claims in README and the report refer to the regenerated files below.

## Measurement

IER is the ungated all-candidate top-5 infeasible fraction. The feasible set
also includes the emotional difficulty cap. The same infeasible set drives
existing negative reward updates, so the repair changes SAPS learning.
SVR checks accepted tasks against PF_R1-PF_R5. Main summary runners retain
per-person macro averaging; pooled post-hoc replay uses all accepted tasks.
NoSafety is therefore 0.2181 in its formal macro summary and 171/772 = 0.2215
in pooled replay. These are different weights, not conflicting rule decisions.

## Current policy comparisons

| File | Generator or source | Scope |
|---|---|---|
| `saps_60day.json` | `evaluation/run_saps_60day.py` | Current ungated IER / five-rule SVR |
| `saps_60day_per_user.json` | `evaluation/run_saps_60day.py` | Current SAPS and LinUCB per-person arrays |
| `linucb_60day.json` | `evaluation/run_60day.py --method linucb` | Separate LinUCB summary |
| `random_60day.json` | `evaluation/run_60day.py --method random` | Separately regenerated Random summary |
| `neuralucb_60day.json` | `evaluation/run_neuralucb_60day.py` | Modified neural reference, including observed SVR |
| `neural_ucb_prefix_60day_seed42.json` | `evaluation/run_neuralucb_prefix_60day.py` | Historical architecture rerun, not current NeuralUCB |
| `neural_ucb_prefix_60day_seed42_metrics.csv` | `evaluation/run_neuralucb_prefix_60day.py` | Historical architecture per-person data |
| `popularity_60day_seed42.json` | `evaluation/run_baselines_60day.py` | Mean-reward popularity |
| `popularity_60day_seed42.csv` | `evaluation/run_baselines_60day.py` | Per-person data |
| `rule_based_60day_seed42.json` | `evaluation/run_baselines_60day.py` | Stable input-order reference |
| `rule_based_60day_seed42.csv` | `evaluation/run_baselines_60day.py` | Per-person data |
| `epsilon_greedy_60day_seed42.json` | `evaluation/run_baselines_60day.py` | Epsilon 0.1 reference |
| `epsilon_greedy_60day_seed42.csv` | `evaluation/run_baselines_60day.py` | Per-person data |
| `linucb_nosafety_60day_seed42.json` | `evaluation/run_baselines_60day.py` | Per-person mean SVR, 0.2181 |
| `linucb_nosafety_60day_seed42.csv` | `evaluation/run_baselines_60day.py` | Per-person SVR uses accepted denominator |
| `paired_t_saps_vs_linucb.json` | `evaluation/scripts/paired_t_test.py` | Regenerated fixed-cohort comparison |
| `paired_t_saps_vs_neuralucb.json` | `evaluation/scripts/paired_t_test.py` | Regenerated fixed-cohort comparison |

## Safety and narrative diagnostics

| File | Generator or source | Scope |
|---|---|---|
| `posthoc_svr_60day_seed42.json` | `evaluation/svr_narrator_invariants.py` | Pooled accepted-task replay |
| `nosafety_svr_posthoc_full.json` | `evaluation/nosafety_svr_full.py` | 171/772 unique accepted-task violations |
| `svr_posthoc_rule_full.json` | `evaluation/q1q4_factcheck.py` | Rule inventory and replay |
| `robustness_grid_5row.json` | `evaluation/robustness_grid.py` | One-rule-drop grid |
| `robustness_grid_5row.csv` | `evaluation/robustness_grid.py` | Grid table data |
| `robustness_grid_summary.json` | `retained derived summary` | Same LinUCB grid, unchanged numeric results |
| `narrator_invariants_60day_seed42.json` | `evaluation/svr_narrator_invariants.py` | Initial diagnostic, superseded by v2 for I3 |
| `narrator_invariants_v2_60day_seed42.json` | `evaluation/q1q4_factcheck.py` | Corrected slot/closing wiring audit |
| `narrative_samples_60.json` | `evaluation/narrative_sampler.py` | Regenerated, identical LinUCB sample texts |
| `narrative_eval_scores_run2_colab_3axis.json` | `retained external judge run` | No new API call; scores remain attached to sample texts |
| `narrative_eval_summary.json` | `offline aggregation of retained judge scores` | Descriptive ratings, no patient efficacy claim |

## Cold start and measurement provenance

| File | Generator or source | Scope |
|---|---|---|
| `coldstart_60day_seed42.json` | `evaluation/run_coldstart.py` | Regenerated repaired SAPS and unchanged LinUCB transfer |
| `coldstart_posteriors_saps.json` | `evaluation/run_coldstart.py` | Regenerated training posterior snapshot |
| `coldstart_posteriors_linucb.json` | `evaluation/run_coldstart.py` | Regenerated training posterior snapshot |
| `measurement_provenance.json` | `release validation` | Semantics, source/input hashes and validation scope |
| `saps_60day_pre_wiring.json` | `retained historical run` | Excluded from current comparison; old implementation |

## Reproduction commands

```bash
bash scripts/reproduce_all.sh
bash scripts/reproduce_coldstart.sh
python evaluation/run_60day.py --method random
python evaluation/run_60day.py --method linucb
```

The core script regenerates its experiment stages and compares 13 listed files,
but Random/cold-start require the separate commands above. Two current paired
comparisons use `evaluation/scripts/paired_t_test.py`; executable examples are
in its module header. The NeuralUCB runner creates
`evaluation/results/logs/NeuralUCB_60day_logs.json`, required by the second
comparison. This generated log is excluded from Git and from the submission
archive; run the producer before running that comparison.

A repeated core run passed 13/13 listed comparisons against regenerated references.
Five comparisons are exact; eight strip `wall_time_s` only. This is not a
cross-machine byte identity guarantee. A printed FAIL currently does not force
a nonzero script exit, so inspect the summary. Random/cold-start comparison PASS
alone does not mean regeneration. External judge scores are preserved rather
than rerun; optional judging requires an API key, unlike the demos.

The retained `saps_60day_pre_wiring.json` and initial narrator diagnostic are
explicit historical references, not evidence for current SAPS performance.
See ../../docs/implementation_notes.md for remaining implementation limits.

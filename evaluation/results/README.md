# evaluation/results/ — Artifact Index

> Auto-generated reference for all reproducible evaluation artefacts.
> Each file is force-committed (bypassing `.gitignore`) when its producing
> experiment is frozen.

## Main Method Comparison (§5.2)

| File | Commit | Description |
|------|--------|-------------|
| `saps_60day.json` | 469fd31 | SAPS vs LinUCB 60-day formal verification |
| `neuralucb_60day.json` | 260508a | Post-fix NeuralUCB 60-day results |
| `neural_ucb_prefix_60day_seed42.json` | 4703c37 | Pre-fix NeuralUCB ablation baseline |
| `neural_ucb_prefix_60day_seed42_metrics.csv` | 4703c37 | Pre-fix per-user CSV |

## Baseline Reruns (§5.2 Table 5.1)

| File | Commit | Description |
|------|--------|-------------|
| `popularity_60day_seed42.json` | 7c17d4c | Popularity baseline |
| `rule_based_60day_seed42.json` | 7c17d4c | RuleBased baseline |
| `epsilon_greedy_60day_seed42.json` | 7c17d4c | ε-Greedy baseline |
| `linucb_nosafety_60day_seed42.json` | 7c17d4c | LinUCB with safety disabled |

## Safety Analysis (§5.4)

| File | Commit | Description |
|------|--------|-------------|
| `posthoc_svr_60day_seed42.json` | 965802c | Post-hoc SVR replay (per-rule breakdown) |
| `nosafety_svr_posthoc_full.json` | 897f4e9 | Full breakdown with overlap matrix |
| `svr_posthoc_rule_full.json` | 37cd39f | Q1–Q4 fact-check comprehensive |
| `robustness_grid_5row.json` | aa8ada2 | One-drop robustness grid (with solo_hit) |
| `robustness_grid_5row.csv` | aa8ada2 | Grid CSV for Table import |
| `robustness_grid_summary.json` | dd052f8 | Grid + solo-hit summary |

## Narrator Analysis (§5.5)

| File | Commit | Description |
|------|--------|-------------|
| `narrator_invariants_60day_seed42.json` | 965802c | Initial invariant check |
| `narrator_invariants_v2_60day_seed42.json` | 37cd39f | Corrected I1/I3 invariants |
| `narrative_samples_60.json` | aa8ada2 | 60 stratified narrative samples |
| `narrative_llm_scores_60.json` | aa8ada2 | LLM-as-Judge scores (subagent) |
| `narrative_eval_summary.json` | aa8ada2 | Aggregated narrative eval + CI95 |

## Cold-Start Transfer (§7.4)

| File | Commit | Description |
|------|--------|-------------|
| `coldstart_60day_seed42.json` | (R2.7) | Main cold-start result |
| `coldstart_posteriors_saps.json` | (R2.7) | SAPS frozen posteriors (100 arms) |
| `coldstart_posteriors_linucb.json` | (R2.7) | LinUCB frozen posteriors (100 arms) |

## Reproduction

```bash
# Full reproduction (~5 min)
bash scripts/reproduce_all.sh

# Cold-start only (~30 sec)
bash scripts/reproduce_coldstart.sh

# LLM-as-Judge (requires OpenAI key, run in Colab)
# See evaluation/llm_judge_colab.ipynb
```

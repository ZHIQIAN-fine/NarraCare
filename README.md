# 🏥 NarraCare

**A Safety-Aware Narrative Recommender for 60-Day Rehabilitation**

> CA6124 Recommender Systems — 2026 Semester 1 Group Project

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## 1. Overview

NarraCare is a multi-agent rehabilitation exercise recommender that combines
**contextual bandits**, a **56-rule safety framework**, and a **3-Act narrative
engine** to deliver personalised, safe, and motivating 60-day rehabilitation
plans through simulated conversational interaction.

The system supports patients recovering from ACL reconstruction, rotator cuff
repair, and chronic lower back pain. It recommends exercises from a 100-task
catalogue while enforcing clinical safety constraints in real time.

**Key contributions:**
- **SAPS** (Safety-Aware Personalised Selection) — a novel algorithm combining
  LinUCB reward backbone with Thompson-sampled Beta safety posteriors.
  Pareto-dominates LinUCB on both Infeasible Exploration Rate and Cumulative
  Reward.
- **Cold-start transfer** — SAPS posteriors trained on ACL + Rotator-Cuff
  transfer to Back-Pain with zero IER and no statistically significant reward
  degradation (§7.4).
- **56-rule safety framework** — 5 runtime pre-filter rules + 51 design-time
  clinical definitions achieving SVR = 0.000 across all safe-mode experiments.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     NarraCare Pipeline                          │
│                                                                 │
│  ┌───────────┐    ┌──────────────┐    ┌────────────────────┐   │
│  │  User      │───▶│ User         │───▶│ Recommender Agent  │   │
│  │ Simulator  │    │Understanding │    │ (LinUCB / SAPS /   │   │
│  │ (EU-Sim)   │    │ Agent        │    │  NeuralUCB / ...)  │   │
│  └───────────┘    └──────────────┘    └─────────┬──────────┘   │
│                                                  │              │
│                                        ┌─────────▼──────────┐   │
│  ┌───────────┐    ┌──────────────┐    │  Safety Agent       │   │
│  │  Narrator  │◀───│ Reward       │◀───│  (5 pre-filter +   │   │
│  │  (3-Act)   │    │ Composer     │    │   51 clinical defs) │   │
│  └───────────┘    └──────────────┘    └────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Module Layout

```
NarraCare/
├── configs/               # YAML configuration files
│   ├── default.yaml       # Main configuration
│   └── coldstart.yaml     # Cold-start experiment config
├── data/                  # Static data assets (project root)
│   ├── user_profiles.json # 20 user personas (ACL/RC/Back-Pain)
│   ├── task_library.json  # 100 rehabilitation tasks
│   ├── safety_rules.json  # 51 clinical safety definitions
│   ├── narrative_arcs.json
│   └── narrator_tone_templates.json
├── src/                   # Core source modules
│   ├── pipeline.py        # Main orchestration
│   ├── simulator.py       # EU-Sim v2 (Expected Utility)
│   ├── memory.py          # Per-user state memory
│   ├── agents/
│   │   ├── safety_agent.py    # 5-rule pre-filter + crisis detection
│   │   ├── recommender.py     # Bandit model wrapper (LinUCB, NeuralUCB, baselines)
│   │   ├── narrator.py        # v1.3.2 Narrator (3-Act + tone templates)
│   │   ├── narrator_agent.py  # Legacy narrator (v1)
│   │   └── user_understanding.py  # Feature extraction
│   ├── policies/
│   │   └── saps.py        # SAPS algorithm (novel)
│   └── models/
│       ├── linucb.py      # Disjoint LinUCB
│       ├── neural_ucb.py  # NeuralUCB (fixed per-arm stats)
│       └── baselines.py   # Random, Popularity, RuleBased, ε-Greedy
├── evaluation/            # Experiment scripts + results
│   ├── run_coldstart.py   # §7.4 cold-start transfer
│   ├── run_saps_60day.py  # SAPS formal verification
│   ├── run_baselines_60day.py
│   ├── robustness_grid.py # Safety one-drop ablation
│   ├── narrative_sampler.py
│   ├── llm_judge_colab.py # LLM-as-Judge (Colab / CLI)
│   └── results/           # JSON/CSV artefacts (see results/README.md)
├── scripts/
│   ├── reproduce_all.sh   # Full reproduction pipeline
│   └── reproduce_coldstart.sh
├── report/
│   └── pf_rules_permalink.md  # PF_R1–R5 permanent reference
├── demo/
│   └── app.py             # Streamlit demo (separate)
├── tests/
├── .env.example
├── CITATION.cff
├── LICENSE                # MIT
├── README.md
└── requirements.txt
```

> **Note on `data/` placement**: Data assets live at project root (`data/`)
> rather than `src/data/` to keep `src/` as a pure Python package. All module
> references use `Path(__file__).parent.parent / "data"`.

---

## 3. Quick Start

### Prerequisites

- Python 3.10+
- No GPU required (all models are numpy-based)

### Installation

```bash
git clone <repository-url> NarraCare
cd NarraCare
pip install -r requirements.txt
cp .env.example .env   # Only needed for LLM-as-Judge
```

### Run the full 60-day simulation

```python
from src.pipeline import NarraCarePipeline

pipe = NarraCarePipeline("configs/default.yaml")
logs = pipe.run_simulation(n_days=60)

print(f"Total interactions: {len(logs)}")
print(f"Sample: {logs[0]['user_id']}, day {logs[0]['day']}, "
      f"task={logs[0]['chosen_task_id']}, reward={logs[0]['reward']:.2f}")
```

### Reproduce all experiments

```bash
bash scripts/reproduce_all.sh     # ~5 min, seed 42
bash scripts/reproduce_coldstart.sh   # ~30 sec
```

---

## 4. Algorithms

### SAPS (Safety-Aware Personalised Selection) — Novel

Composite score: `S(a) = UCB_reward(a) + τ_safe · s_a`, where `s_a ~ Beta(α_a, β_a)`.

- **Reward backbone**: Disjoint LinUCB with per-arm `(A, b)` matrices
- **Safety posterior**: Per-arm Beta distribution updated by Safety Agent verdicts
- **Action space**: Full catalogue scored; unsafe arms filtered by pre-filter before output
- **Guarantee**: SVR = 0 under the 51 design-time clinical rules evaluated by the Safety Agent; this is a conditional guarantee tied to the current rule set, not a universal safety claim

**Online Update Rule** (per-interaction, chosen arm only):

1. Issue action `a_t` → observe outcome tuple `(reward, pain_change, acceptance, ...)`
2. **Reward backbone update**: `A_a ← A_a + x_t x_tᵀ`, `b_a ← b_a + r_t x_t` (standard LinUCB disjoint)
3. **Safety posterior update** (S2 signal):
   - `safe_t = 1` iff `pain_change ≤ SAPS_PAIN_SAFE_THRESHOLD` (default `1.0`, see §7)
   - `α_a ← α_a + safe_t`, `β_a ← β_a + (1 − safe_t)`
4. **Only the chosen arm's `(A_a, b_a, α_a, β_a)` is updated**; non-chosen arms retain priors.

**Runtime Safety Wiring** (per-recommendation, before scoring):

- 5 pre-filter rules `PF_R1 … PF_R5` (hard-coded clinical gates: contraindication overlap, stage mismatch, high pain + high difficulty, severe fatigue, low ROM + demanding task) evaluated on each candidate arm
- Arms failing any `PF_R*` are removed from the candidate set **before** the composite score `S(a)` is computed
- This is what delivers the SVR = 0 guarantee under the 51 design-time rules (Layer B in `data/safety_rules.json`)

### Baselines

| Algorithm | Description |
|-----------|-------------|
| LinUCB | Disjoint linear UCB (Li et al. 2010) |
| NeuralUCB | Neural contextual bandit with per-arm running stats |
| ε-Greedy | ε=0.1 exploration with reward-based exploitation |
| Popularity | Rank by historical acceptance count |
| RuleBased | Stage-matched difficulty + category preference |
| Random | Uniform random from safe candidate set |

---

## 5. Safety Framework

**56 rules** in a two-layer architecture:

- **Layer A — Runtime pre-filter** (5 rules, `PF_R1`–`PF_R5`): Evaluated at
  recommendation time on every `(user, task)` pair. See
  [`report/pf_rules_permalink.md`](report/pf_rules_permalink.md) for the full
  condition table and solo-hit analysis.

- **Layer B — Design-time clinical definitions** (51 rules in
  `data/safety_rules.json`): Contraindications (16), dosage (9), fatigue (6),
  crisis (10), progression gates (10). Inform the task catalogue metadata and
  Safety Agent crisis detection logic.

---

## 6. Evaluation Protocol

All experiments use **seed 42, 60 days, 20 users** unless otherwise noted.

| Experiment | Script | Report Section |
|-----------|--------|----------------|
| SAPS vs LinUCB | `run_saps_60day.py` | §5.2 |
| 7 baselines | `run_baselines_60day.py` | §5.2 Table 5.1 |
| NeuralUCB fix ablation | `run_neuralucb_prefix_60day.py` | §5.4 |
| Safety robustness grid | `robustness_grid.py` | §5.4.2 |
| Post-hoc SVR replay | `nosafety_svr_full.py` | §5.4.2 |
| Narrator invariants | `svr_narrator_invariants.py` | §5.5 |
| LLM-as-Judge | `llm_judge_colab.py` | §5.5 |
| Cold-start transfer | `run_coldstart.py` | §7.4 |

See [`evaluation/results/README.md`](evaluation/results/README.md) for the
full artefact index.

---

## 7. Hyperparameter Settings

All key hyperparameters are set in `configs/*.yaml`. The table below summarises
the values used in the main 60-day experiments (seed 42). See the report §4.6
for the full design rationale.

| Policy | Parameter | Value | Source / rationale |
|--------|-----------|-------|--------------------|
| **SAPS** | `alpha_ucb` (exploration) | 1.0 | Same as LinUCB for fair comparison |
| | `tau_safe` (safety-explore weight) | 2.0 | Grid-searched {0.5, 1, 2, 5}; 2.0 balances IER elimination vs reward |
| | `alpha_prior` / `beta_prior` (Beta init) | 1.0 / 1.0 | Uniform prior — no arm is presumed safe/unsafe before data |
| | `SAPS_PAIN_SAFE_THRESHOLD` | 1.0 | S2 signal: `pain_change ≤ 1.0` → safe; see `src/policies/saps.py` |
| **LinUCB** | `alpha` (exploration) | 1.0 | Li et al. (2010) default |
| | `feature_dim` | 15 | Matches user-understanding feature vector length |
| **NeuralUCB** | `hidden_dim` | 32 | Reduced from default 64 for stability (see §5.4) |
| | `training_epochs` | 5 | Reduced from 10 to avoid overfitting on small per-arm data |
| | `alpha` (exploration) | 0.01 | Low exploration — network confidence used instead |
| | `safety_lambda` | 3.0 | Stronger Lagrangian penalty for unsafe arms |
| **ε-Greedy** | `epsilon` | 0.1 | Standard value (Sutton & Barto 2018) |
| **Popularity** | — | — | No tuneable parameter; ranks by historical acceptance count |
| **RuleBased** | `top_k` | 3 | Stage-matched + category-preference scoring |
| **Random** | — | — | Uniform random from safe candidate set |

**Reward composer weights** (in `configs/default.yaml → reward:`):

| Weight | Value | Component |
|--------|-------|-----------|
| `w_completed` | 0.35 | Task completion signal |
| `w_safety` | 0.25 | Safety compliance bonus |
| `w_difficulty_match` | 0.20 | Difficulty-fit to user state |
| `w_pain_penalty` | 0.10 | Pain increase penalty |
| `w_dropout_penalty` | 0.05 | Session dropout penalty |
| `w_category` | 0.05 | Category preference match |
| `safety_lambda` | 1.0 | Lagrangian multiplier (base; scaled by narrative arc) |

**Narrative arc safety multiplier**: `safety_lambda_multiplier` =
{1.5, 1.0, 0.7} for {Act 1, Act 2, Act 3} respectively
(configured in `data/narrative_arcs.json`).

---

## 8. Key Results

### §5.2 — SAPS vs LinUCB (60-day, 20 users)

Statistical test: paired-t over per-user trajectories, n = 20, df = 19.

| Metric | SAPS | LinUCB | Δ | t(19) | p | 95% CI |
|--------|------|--------|---|-------|---|--------|
| Cum. Reward | **46.29 ± 2.17** | 41.94 ± 3.00 | +4.35 | 8.35 | 8.8 × 10⁻⁸ | [3.26, 5.44] |
| IER | **0.0000** | 0.0233 | −0.023 | −7.19 | 7.9 × 10⁻⁷ | [−0.030, −0.017] |
| SVR | 0.0000 | 0.0000 | 0 | — | — | — |

Cohen's d: reward = 1.87, IER = −1.61 (both large effect).

**SAPS vs NeuralUCB-postfix**: reward Δ = +7.23, t(19) = 9.44,
p = 1.3 × 10⁻⁸, d = 2.11. IER Δ = −0.002, p = 0.079 (ns at α = 0.05) —
the two methods achieve comparable near-term engagement despite SAPS's
cumulative reward lead.

SAPS significantly outperforms LinUCB on both reward (p < 10⁻⁷) and IER
(p < 10⁻⁶) with large effect sizes (|d| > 1.6). Against NeuralUCB-postfix,
SAPS shows a substantial reward advantage (Δ +7.23, p < 10⁻⁷, d = 2.11),
while the IER difference does not reach statistical significance at α = 0.05
(p = 0.079), suggesting the two methods achieve comparable near-term
engagement despite SAPS's cumulative reward lead.

### §7.4 — Cold-Start Transfer (Back-Pain)

| Method | Condition | Reward | IER | Acceptance |
|--------|-----------|--------|-----|------------|
| SAPS | Frozen → BP | **47.84** | **0.0000** | 0.733 |
| SAPS | Retrained BP | **46.66** | **0.0000** | 0.733 |
| LinUCB | Frozen → BP | 41.90 | 0.0213 | 0.603 |
| LinUCB | Retrained BP | 43.22 | 0.0640 | 0.630 |

SAPS posteriors transfer successfully to the unseen Back-Pain cohort;
frozen model reward (47.84) exceeds even the full-cohort retrained result
(46.66), suggesting robust generalisation.

---

## 9. Citation

```bibtex
@software{narracare2026,
  title     = {NarraCare: A Safety-Aware Narrative Recommender for 60-Day Rehabilitation},
  author    = {{NarraCare Group, CA6124 2026S1}},
  year      = {2026},
  version   = {1.0.0},
  license   = {MIT},
}
```

---

## License

[MIT](LICENSE)

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
- **Guarantee**: SVR = 0 (unsafe arms never recommended)

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

## 7. Key Results

### §5.2 — SAPS vs LinUCB (60-day, 20 users)

| Metric | SAPS | LinUCB | Δ | p-value |
|--------|------|--------|---|---------|
| Cum. Reward | 43.33 ± 3.35 | 41.94 ± 3.00 | +1.39 | 0.027 |
| IER | **0.0000** | 0.0233 | −0.023 | ≈ 0 |
| SVR | 0.0000 | 0.0000 | 0 | — |

### §7.4 — Cold-Start Transfer (Back-Pain)

| Method | Condition | Reward | IER | Acceptance |
|--------|-----------|--------|-----|------------|
| SAPS | Frozen → BP | 43.35 | **0.0000** | 0.580 |
| SAPS | Retrained BP | 43.35 | **0.0000** | 0.580 |
| LinUCB | Frozen → BP | 41.90 | 0.0213 | 0.603 |
| LinUCB | Retrained BP | 43.22 | 0.0640 | 0.630 |

SAPS posteriors transfer perfectly; LinUCB frozen posteriors trade ~1.3 reward
for 67% IER reduction (p = 0.053).

---

## 8. Citation

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

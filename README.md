# 🏥 NarraCare: Multi-Agent Conversational Rehabilitation Recommender System

> A contextual-bandit-based rehabilitation exercise recommender that uses a multi-agent architecture (Recommender + Safety + Narrator) to deliver personalized, safe, and narrative-enhanced rehabilitation plans through conversational interaction.

---

## 📋 Overview

NarraCare is an intelligent rehabilitation recommender system designed to help patients recovering from surgeries (ACL reconstruction, rotator cuff repair) or managing chronic conditions (lower back pain). The system goes beyond simple exercise recommendation by:

1. **Understanding** the patient's current state (pain, fatigue, ROM, mood)
2. **Recommending** personalized exercises using contextual bandits (LinUCB / Neural UCB)
3. **Enforcing safety** through rule-based constraint checking and crisis detection
4. **Narrating** recommendations in warm, story-like rehabilitation camp briefings
5. **Learning** from patient feedback to improve future recommendations

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NarraCare Pipeline                          │
│                                                                     │
│  User Input ──► User Understanding ──► Feature Vector              │
│                      Agent                    │                     │
│                                               ▼                     │
│  Task Library ──► Safety Agent ──► Recommender Agent ──► Top-K     │
│                  (Pre-filter)      (LinUCB / Neural)    Tasks      │
│                                               │                     │
│                                               ▼                     │
│                  Safety Agent ──► Narrator Agent ──► Narrative      │
│                  (Post-check)                        Output         │
│                                               │                     │
│                                               ▼                     │
│                  User Feedback ──► Reward Computation ──► Model     │
│                                                          Update    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **User Understanding Agent** — Parses natural language or structured input into a 15-dimensional feature vector
- **Recommender Agent** — Contextual bandit (LinUCB / Neural UCB) with multiple baselines
- **Safety Agent** — Rule-based pre-filtering, post-checking, crisis detection, and stage progression gating
- **Narrator Agent** — Template-based or LLM-powered narrative wrapping of recommendations

### 🎰 Contextual Bandit Recommendation
- **LinUCB** (Li et al., 2010) — Disjoint linear upper confidence bound
- **Neural UCB** — Feed-forward neural network with exploration bonus (PyTorch)
- **Baselines** — Random, Popularity, ε-Greedy, Rule-Based

### 📖 Narrative-Enhanced Task Presentation
- Rehabilitation camp-style motivational briefings
- Three rotating templates: coach-like, empathetic, progress-focused
- Per-task explanations (why this exercise fits your current state)
- Optional LLM mode for richer, more natural narratives

### 🛡️ Healthcare Safety Constraints
- **16 contraindication rules** — Condition-specific, stage-aware exercise restrictions
- **9 dosage rules** — Pain/fatigue-adaptive intensity limiting
- **6 fatigue rules** — Volume control and mandatory rest enforcement
- **10 crisis detection rules** — Keyword-based red-flag detection (EN + ZH)
- **10 progression gates** — Evidence-based stage advancement criteria

### 📊 Comprehensive Evaluation (3-Layer)
- **Offline Metrics** — Precision, Recall, NDCG, Hit Rate, Cumulative Reward
- **Beyond-Accuracy Metrics** — Safety Violation Rate, Constraint Satisfaction, Diversity, Novelty, Difficulty Adaptation
- **Interactive Metrics** — Task Success Rate, Acceptance Rate, Avg Turns, Adherence Rate
- **LLM Judge** — GPT-based evaluation of explanation quality and narrative consistency

---

## 📁 Project Structure

```
NarraCare/
├── config/
│   └── default.yaml                # System configuration (algorithms, thresholds, weights)
├── data/
│   ├── user_profiles.json          # 15 synthetic patients (3 conditions × 5 personalities)
│   ├── task_library.json           # 100 rehabilitation exercises with metadata
│   └── safety_rules.json           # 51 safety rules (contraindications, dosage, crisis, progression)
├── demo/
│   └── app.py                      # 🎯 Streamlit interactive demo
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── user_understanding.py   # NL parsing + feature extraction (15-dim vector)
│   │   ├── recommender.py          # Bandit models (LinUCB, NeuralUCB, baselines)
│   │   ├── safety_agent.py         # Pre-filter, post-check, crisis detection, progression
│   │   └── narrator_agent.py       # Template / LLM narrative generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linucb.py               # Standalone LinUCB implementation
│   │   ├── neural_ucb.py           # Standalone Neural UCB implementation
│   │   └── baselines.py            # Random, Popularity, ε-Greedy, Rule-Based
│   ├── memory.py                   # Per-user in-memory state & interaction storage
│   └── simulator.py                # Synthetic user behavior simulation (3 personality types)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip / conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd NarraCare

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive Demo

```bash
# Template mode (no API key needed)
streamlit run demo/app.py

# With LLM mode (optional — toggle in the UI)
export OPENAI_API_KEY="your-key-here"
streamlit run demo/app.py
```

The demo provides:
- **Patient selection** from 15 synthetic profiles (3 conditions × 5 personalities)
- **Manual state control** via sliders (pain, fatigue, ROM)
- **Chat-style interface** showing narrative recommendations
- **One-click exercise selection** with reward feedback
- **Safety dashboard** with real-time alert monitoring
- **Progress tracking** with completion rates and category distribution charts
- **Demo Mode** for auto-playing a 5-day simulation

### Run Experiments (Offline Simulation)

```python
import yaml
from src.simulator import UserSimulator
from src.agents.recommender import RecommenderAgent
from src.agents.safety_agent import SafetyAgent
from src.agents.narrator_agent import NarratorAgent
from src.agents.user_understanding import UserUnderstandingAgent
from src.memory import UserMemory

# Load config
with open("config/default.yaml") as f:
    config = yaml.safe_load(f)

# Initialize components
sim = UserSimulator(config)
rec = RecommenderAgent(config)
safety = SafetyAgent(config)
narrator = NarratorAgent(config, mode="simulation")
uu = UserUnderstandingAgent(config, mode="simulation")
memory = UserMemory(config)

# Run simulation for one user
user_id = "U001"
profile = sim.get_profile(user_id)

for day in range(30):
    state = sim.simulate_daily_state(user_id, day)
    # ... (full pipeline)
```

---

## 🏗️ System Architecture

### Pipeline Flow

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User Input   │────►│ User             │────►│ Feature Vector  │
│ (NL / struct)│     │ Understanding    │     │ (15-dim)        │
└──────────────┘     │ Agent            │     └────────┬────────┘
                     └──────────────────┘              │
                                                       ▼
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Task Library │────►│ Safety Agent     │────►│ Safe Candidates │
│ (100 tasks)  │     │ (Pre-filter)     │     └────────┬────────┘
└──────────────┘     └──────────────────┘              │
                                                       ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │ Recommender      │────►│ Top-K Tasks     │
                     │ Agent (LinUCB)   │     └────────┬────────┘
                     └──────────────────┘              │
                                                       ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │ Safety Agent     │────►│ Checked Tasks   │
                     │ (Post-check)     │     └────────┬────────┘
                     └──────────────────┘              │
                                                       ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │ Narrator Agent   │────►│ Narrative       │
                     │ (Template/LLM)   │     │ Output          │
                     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                     ┌──────────────────┐     ┌─────────────────┐
                     │ User Feedback    │────►│ Reward → Model  │
                     │ (choice/skip)    │     │ Update          │
                     └──────────────────┘     └─────────────────┘
```

### Feature Vector (15 dimensions)

| Dim | Feature | Source | Normalization |
|-----|---------|--------|---------------|
| 0 | `pain_level` | Daily state | /10 |
| 1 | `fatigue_level` | Daily state | /10 |
| 2 | `rom` | Daily state | /180 |
| 3 | `mood` | Daily state | /10 |
| 4 | `age` | Profile | /100 |
| 5 | `days_since_surgery` | Profile | /365 |
| 6 | `stage` | Profile | /4 |
| 7 | `completion_rate_7d` | Memory | [0, 1] |
| 8 | `consecutive_high_intensity` | Memory | /7 |
| 9 | `has_specific_complaints` | Daily state | {0, 1} |
| 10 | `feedback_positive` | Daily state | {0, 1} |
| 11 | `bmi` | Profile | (x-15)/25 |
| 12 | `baseline_pain` | Profile | /10 |
| 13 | `baseline_fatigue` | Profile | /10 |
| 14 | `motivation_score` | Profile | personality map |

### Reward Function

```
R = 0.40 × R_completed + 0.30 × R_safety + 0.15 × R_difficulty_match
  + 0.10 × R_pain_penalty + 0.05 × R_dropout_penalty
```

### Safety Rule Categories

| Category | Count | Description |
|----------|-------|-------------|
| Contraindication Rules | 16 | Condition-specific, stage-aware exercise restrictions |
| Dosage Rules | 9 | Pain/fatigue-adaptive intensity limiting |
| Fatigue Rules | 6 | Volume control and mandatory rest enforcement |
| Crisis Detection | 10 | Keyword-based red-flag detection (EN + ZH bilingual) |
| Progression Gates | 10 | Evidence-based stage advancement criteria |

---

## 📊 Evaluation

### Three-Layer Evaluation Framework

#### Layer 1: Offline Metrics
Standard recommendation metrics evaluated on simulated interaction logs:
- **Precision@K** — Fraction of recommended tasks that were "relevant" (completed)
- **Recall@K** — Fraction of completable tasks that were recommended
- **NDCG@K** — Normalized discounted cumulative gain
- **Hit Rate@K** — Whether at least one recommended task was completed
- **Cumulative Reward** — Total reward accumulated over the simulation

#### Layer 2: Beyond-Accuracy Metrics
Healthcare-specific quality metrics:
- **Safety Violation Rate** — % of recommendations that violated safety rules
- **Constraint Satisfaction** — % of recommendations respecting all patient constraints
- **Diversity** — Intra-list diversity of recommended exercise categories
- **Novelty** — % of recommendations the user hasn't seen before
- **Difficulty Adaptation** — Correlation between user state and recommended intensity

#### Layer 3: Interactive Metrics
Conversational and behavioral metrics:
- **Task Success Rate** — % of recommended exercises successfully completed
- **Acceptance Rate** — % of recommendations accepted (not skipped)
- **Average Turns** — Mean conversation turns per recommendation session
- **Adherence Rate** — 7-day rolling exercise adherence

### LLM Judge (Optional)
GPT-based evaluation of narrative quality:
- **Explanation Quality** — Are task explanations clear, relevant, and personalized?
- **Narrative Consistency** — Does the narrative match the patient's state and history?

---

## 📈 Results

### Experiment Configuration
- **Users**: 15 synthetic patients (5 ACL, 5 rotator cuff, 5 chronic back pain)
- **Tasks**: 100 rehabilitation exercises
- **Simulation**: 30 days × 1 interaction/day
- **Algorithms**: LinUCB, Neural UCB, Random, Popularity, ε-Greedy, Rule-Based

> _Results tables and figures will be populated after running the full experiment suite._

| Algorithm | Precision@3 | NDCG@3 | Cumulative Reward | Safety Violation Rate |
|-----------|-------------|--------|-------------------|-----------------------|
| LinUCB | — | — | — | — |
| Neural UCB | — | — | — | — |
| ε-Greedy | — | — | — | — |
| Random | — | — | — | — |
| Popularity | — | — | — | — |
| Rule-Based | — | — | — | — |

---

## 🔧 Configuration

All system parameters are configured in `config/default.yaml`:

```yaml
# Key configuration options
recommender:
  algorithm: "linucb"    # linucb | neural_ucb | random | popularity | rule_based | epsilon_greedy
  top_k: 3               # Number of tasks to recommend per interaction

safety:
  enable: true
  pain_threshold_high: 7  # Pain >= 7 → restrict to low intensity
  fatigue_threshold_high: 8  # Fatigue >= 8 → only stretching/breathing

narrator:
  style: "rehabilitation_camp"  # rehabilitation_camp | clinical | casual
  language: "en"               # en | zh

simulation:
  n_users: 15
  n_days: 30
```

---

## 🩺 Patient Profiles

The system includes 15 diverse synthetic patients across 3 conditions:

| Condition | Patients | Stages |
|-----------|----------|--------|
| ACL Reconstruction | U001–U005 | Early → Return to Sport |
| Rotator Cuff Repair | U006–U010 | Early → Return to Activity |
| Chronic Lower Back Pain | U011–U015 | Early → Maintenance |

Each patient has a unique personality type affecting simulated behavior:
- **Motivated** (3 patients) — High acceptance, prefers challenges
- **Cautious** (4 patients) — Low acceptance, prefers gentle exercises
- **Inconsistent** (4 patients) — Variable adherence, prone to skipping

---

## 📖 Citation

```bibtex
@misc{narracare2026,
  title={NarraCare: A Multi-Agent Conversational Rehabilitation Recommender System
         with Contextual Bandits and Narrative-Enhanced Interaction},
  author={[Student Name]},
  year={2026},
  note={Course project}
}
```

---

## 👥 Team

- **[Student Name]** — System design, implementation, and evaluation

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 NarraCare

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# data/ — Static Data Assets

All files are **synthetic** and created for the CA6124 group project.
No real patient data is used. No external data licence is required.

---

## Files

### `task_library.json` — 100 rehabilitation exercises

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Unique ID (`T001`–`T100`) |
| `name` | str | English exercise name |
| `name_zh` | str | Chinese translation |
| `category` | str | One of 5 categories (see below) |
| `body_part` | str | Target body part |
| `difficulty` | int | 1 (easy) – 5 (hard) |
| `duration_min` | int | Estimated minutes per session |
| `reps` / `sets` | int | Repetition and set counts |
| `contraindications` | list[str] | Conditions where this task is unsafe |
| `applicable_stages` | list[str] | Rehab stages (`early`, `mid`, `late`, `maintenance`) |
| `equipment` | str | Required equipment (or `"none"`) |
| `description` | str | Plain-language exercise instruction |
| `safety_notes` | str | Clinician safety guidance |

**Category breakdown** (100 tasks total):

| Category | Count |
|----------|-------|
| strength | 41 |
| flexibility | 29 |
| balance | 12 |
| cardio | 10 |
| rest | 8 |

### `user_profiles.json` — 20 simulated patients

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | str | `U001`–`U020` |
| `condition` | str | `ACL_reconstruction` / `rotator_cuff_repair` / `chronic_lower_back_pain` |
| `current_stage` | str | Starting rehab stage |
| `pain_baseline` / `fatigue_baseline` | float | Initial patient state |
| `rom_current` / `rom_target` | float | Range-of-motion (degrees) |
| `preference` | dict | Category preferences |
| `contraindications` | list[str] | Patient-specific contraindications |

**Condition split**: U001–U005 and U016–U020 ACL (10), U006–U010 Rotator-Cuff (5), U011–U015 Back-Pain (5).

### `safety_rules.json` — 51 design-time clinical rule definitions

| Rule category | Count | Purpose |
|---------------|-------|---------|
| `contraindication_rules` | 16 | Block tasks contraindicated for the patient's condition |
| `dosage_rules` | 9 | Enforce rep/set/duration limits by stage |
| `fatigue_rules` | 6 | Reduce intensity under fatigue |
| `crisis_rules` | 10 | Halt or escalate when pain/fatigue spike |
| `progression_gates` | 10 | Gate stage transitions on ROM/pain criteria |

These 51 authored definitions form Layer B. They are not all executed as
runtime predicates; see `docs/implementation_notes.md`. Layer A = 5 runtime
pre-filter rules `PF_R1`–`PF_R5` in `src/agents/safety_agent.py`.

### `narrative_arcs.json` — 3-Act narrative metadata

Maps `(condition, rehab_stage, day)` → narrative act + tone/recommender biases.
Three condition arcs: `ACL`, `RotatorCuff`, `BackPain`, each with `act1`/`act2`/`act3`.
Includes `safety_lambda_multiplier` per act (1.5 / 1.0 / 0.7).

### `narrator_tone_templates.json` — Narrator tone and scenario templates

Defines tone registers (doctor, coach, companion), scenario triggers, and
transition rules used by `src/agents/narrator.py`.

---

## Reproduction

From a clean clone, the following reproduces the main 60-day experiments:

```bash
pip install -r requirements.txt
bash scripts/reproduce_all.sh          # SAPS, LinUCB, baselines (~5 min)
bash scripts/reproduce_coldstart.sh    # Cold-start transfer (~30 sec)
```

Static assets are read by the pipeline and its components from root `data/`.
No preprocessing is required. Reproduction generates experiment outputs, not
these static assets.

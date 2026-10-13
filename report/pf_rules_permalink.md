# Layer A Pre-Filter Rules — Permanent Reference

> **Source of truth**: `src/agents/safety_agent.py`, lines 144–173
> **Commit frozen at**: dd052f8

These 5 hardcoded rules run at recommendation time on every (user, task) pair.
A task that triggers **any** rule is removed from the candidate set before the
bandit sees it.

| ID | Condition Expression | Semantic | NoSafety SVR Hits |
|----|---------------------|----------|-------------------|
| **PF_R1** | `user_contras & task_contras` | Contraindication tag overlap — user and task share a contra tag | 26 / 772 (3.4%) |
| **PF_R2** | `task_stages and user_stage not in task_stages` | Stage mismatch — task not applicable to user's rehab stage | 154 / 772 (19.9%) |
| **PF_R3** | `pain >= 7 and difficulty >= 4` | High pain + high difficulty — structural zero in current sim (yellow override caps diff) | 0 / 772 (0.0%) |
| **PF_R4** | `fatigue >= 8 and cat ∉ {rest, flexibility}` | Severe fatigue + non-gentle category | 14 / 772 (1.8%) |
| **PF_R5** | `rom < 60 and (difficulty >= 3 or cat == "cardio")` | Low ROM + demanding task — structural zero (sim ROM ≥ 83.9°) | 0 / 772 (0.0%) |

## Solo-Hit Analysis (from overlap matrix)

| Rule | Total Catches | Overlap | Solo Catches | Note |
|------|--------------|---------|-------------|------|
| PF_R1 | 26 | 23 w/ PF_R2 | **3** | Redundantly protected |
| PF_R2 | 154 | 23 w/ PF_R1 | **131** | Dominant unique source |
| PF_R3 | 0 | 0 | 0 | Structural zero |
| PF_R4 | 14 | 0 | **14** | Independent |
| PF_R5 | 0 | 0 | 0 | Structural zero |

**Verification**: 3 + 131 + 14 + 23 (overlap) = 171 = `total_violating` ✓

## Design-Time Layer B

51 clinical rule definitions in `data/safety_rules.json`:
- 16 contraindication rules (CR001–CR016)
- 9 dosage rules (DR001–DR009)
- 6 fatigue rules (FR001–FR006)
- 10 crisis rules (XR001–XR010)
- 10 progression gates (PG001–PG010)

**Total framework**: 5 runtime pre-filter + 51 design-time definitions = **56 rules**

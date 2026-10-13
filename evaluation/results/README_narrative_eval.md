# Narrative Evaluation Artifacts

This project ran LLM-as-Judge narrative evaluation TWICE under different
rubrics. The report (§5.5, §7.3, Table 7.2) exclusively cites Run 2.

## Run 2 — authoritative (report §5.5 / §7.3 / Table 7.2)
- Platform: Google Colab
- Model: gpt-4o-mini (temperature=0, seed=42)
- Rubric: 3 axes — Coherence / Empathy / Clinical Fidelity (1–5 scale)
- Samples: 60, aligned with narrative_samples_60.json
- Cost: $0.0069 (22,547 prompt + 5,937 completion tokens)
- Raw per-sample scores: narrative_eval_scores_run2_colab_3axis.json
- Aggregated summary: narrative_eval_summary.json
- Aggregated values: Coherence 4.82 ± 0.11; Empathy 4.15 ± 0.14; Clinical Fidelity 4.72 ± 0.15

## Run 1 — archived (NOT cited by the report)
- Platform: local subagent fallback (gpt-4o-mini via subagent API returned 403)
- Model: aime-subagent-llm
- Rubric: 4 axes — Tone Consistency / Context Grounding / Empathy / Actionability
- File: narrative_eval_summary_run1_subagent_4axis.json
- Not used in the report; retained for provenance only.

## Reproduction
Run 2 can be reproduced in evaluation/llm_judge_colab.ipynb with a valid
OPENAI_API_KEY. The pipeline is deterministic given temperature=0, seed=42.
Aggregation can be independently recomputed from
narrative_eval_scores_run2_colab_3axis.json via:
    python -c "import json, statistics as s; ..."  (see notebook final cell)

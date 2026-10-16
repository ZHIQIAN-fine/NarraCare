# Narrative Evaluation Artifacts

LLM-as-Judge narrative evaluation (report §5.5, §7.3, Table 7.2).

## Configuration
- Platform: Google Colab
- Model: gpt-4o-mini (temperature=0, seed=42)
- Rubric: 3 axes — Coherence / Empathy / Clinical Fidelity (1–5 scale)
- Samples: 60, aligned with narrative_samples_60.json
- Cost: $0.0069 (22,547 prompt + 5,937 completion tokens)
- Raw per-sample scores: narrative_eval_scores_run2_colab_3axis.json
- Aggregated summary: narrative_eval_summary.json
- Aggregated values: Coherence 4.82 ± 0.11; Empathy 4.15 ± 0.14; Clinical Fidelity 4.72 ± 0.15

## Reproduction
Reproducible in evaluation/llm_judge_colab.ipynb with a valid
OPENAI_API_KEY. The pipeline is deterministic given temperature=0, seed=42.
Aggregation can be independently recomputed from
narrative_eval_scores_run2_colab_3axis.json via:
    python -c "import json, statistics as s; ..."  (see notebook final cell)

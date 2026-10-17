# Narrative evaluation artifacts

The stored run uses gpt-4o-mini, temperature 0 and seed 42, with 60 samples,
three axes and zero judge failures. Requested temperature/seed do not guarantee
byte-identical responses from a later remote service run.

- Raw scores: narrative_eval_scores_run2_colab_3axis.json
- Input samples: narrative_samples_60.json
- Aggregate: narrative_eval_summary.json
- Cost: $0.0069, 22,547 prompt and 5,937 completion tokens
- Coherence 4.82 +/- 0.11; Empathy 4.15 +/- 0.14;
  Clinical Fidelity 4.72 +/- 0.15 (95% CI half-widths)

## Offline verification

From the repository root, recompute aggregate statistics without an API key:

```bash
python - <<'PY'
import json
import numpy as np
from scipy import stats
from pathlib import Path
root = Path('evaluation/results')
raw = json.loads((root / 'narrative_eval_scores_run2_colab_3axis.json').read_text())
valid = [entry for entry in raw['scores'] if not entry.get('judge_failed')]
for axis in ('coherence', 'empathy', 'clinical'):
    values = [entry['scores'][axis]['score'] for entry in valid]
    mean = float(np.mean(values))
    low, high = stats.t.interval(.95, len(values)-1, loc=mean, scale=stats.sem(values))
    print(f'{axis}: {mean:.2f} +/- {(high-low)/2:.2f}; n={len(values)}')
PY
```

## Optional new external scoring

Use evaluation/llm_judge_colab.py, following its header instructions and exporting
OPENAI_API_KEY (or using Colab secrets). The malformed old notebook was removed.
A new remote scoring run is a new evaluation, not a deterministic replacement
for the stored raw scores. Save it outside the frozen result filenames.

# LLM-as-Judge Evaluation Rubric

## Axes (1–5 Likert)

### Coherence
Does the message follow logically from the recent context (adherence, last task, current phase)?
- **1**: Contradicts context or references wrong task/phase
- **2**: Vaguely related but misses key context
- **3**: Acceptable but generic — doesn't leverage specific state
- **4**: References relevant context, minor omissions
- **5**: Fully consistent with all provided context

### Empathy
Is the emotional register appropriate to the patient's current pain, fatigue, and emotional state?
- **1**: Tone severely mismatched (e.g., chipper to a red-light patient)
- **2**: Mostly insensitive to current state
- **3**: Neutral — neither warm nor cold
- **4**: Shows awareness of patient's state
- **5**: Well calibrated, genuinely responsive to emotional/physical state

### Clinical Fidelity
Does the message correctly describe the task and avoid clinically inappropriate claims?
- **1**: Clinically wrong or potentially dangerous
- **2**: Correct task name but wrong rationale or difficulty
- **3**: Correct description, generic rationale
- **4**: Accurate with phase-appropriate framing
- **5**: Clinically precise, well-reasoned for the patient's condition and stage

## Scoring Protocol

- **Model**: gpt-4o-mini
- **Temperature**: 0 (requested setting; remote responses are not guaranteed byte-identical)
- **Seed**: 42
- **Response format**: JSON object with `{axis: {score, reason}}`
- **Retry**: API bucket cap = 2, format bucket cap = 1
- **Fail-gate**: If ≥ 6 samples fail, do NOT aggregate

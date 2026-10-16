# NarraCare agent submission and presentation

The current deliverables describe the **implemented LLM conversational agent**,
with the original synthetic benchmark preserved. Source snapshot:
`660e85aa2929ffa1c8a1a7fcbd05506a571945b9`. Core agent implementation: `9801bd4`.

| Download | Contents |
|---|---|
| [PowerPoint deck](NarraCare_Agent_Presentation.pptx?raw=true) | Supplied NTU template, six main slides for three minutes and three question backups. Includes editable results chart/table and a real Gradio screenshot |
| [Editable report](NarraCare_Agent_Report.docx?raw=true) | Agent architecture, practical value, companion design, run instructions and separate engineering evidence. Original eight experimental tables unchanged. A4, Arial 11, single spaced; 11 rendered pages including references |
| [English speaking script and Q&A](NarraCare_Agent_Script_and_QA.docx?raw=true) | 305-word main talk, one-minute primary Q&A, technical answers and a live-demo rehearsal sequence |
| [Submission archive](narracare_agent_submission.tar.gz?raw=true) | Runnable source, frozen benchmark assets and the three documents above under `submission/`. Extracts into `narracare_agent/` |

The report remains editable for final review and PDF export. No provider key is
included. [SHA256SUMS](SHA256SUMS) identifies all four exact downloads.

## Run the interactive demonstration

```bash
tar xzf narracare_agent_submission.tar.gz
cd narracare_agent
python -m pip install -r requirements.txt -r requirements-agent.txt
# Set OPENAI_API_KEY or NARRACARE_LLM_API_KEY securely in the environment.
python -m narracare.agent_app --port 7860
```

Open the local address printed by Gradio. The terminal version is
`python -m narracare.agent --user U001 --seed 42`. A provider key and network
are required, default model `gpt-4o-mini`. Use synthetic inputs only.

The Gradio app is **live interaction**. It does not replay the HTML trajectories.
In the verified production session:

1. An incomplete check-in prompted clarification.
2. Pain 3, fatigue 2, mood 7 and ROM 110 plus a flexibility preference yielded
   T017 from ten retrieved items. Safety retained 77 of 100 catalogue tasks.
3. Changing preference to balance and avoiding flexibility selected T029 from
   four retrieved items, with the same hard safety filtering.
4. "I'll try" created neither a reward nor a completed step.
5. Explicit reported completion yielded reward 0.9401 and one completed step.
6. Skip created camp and advanced the narrative day while retaining reward and
   completed steps.

The separate `demo/index.html` is an offline recorded replay. Its replies affect
presentation only. The original `python -m narracare.demo` runs the benchmark
pipeline in temporary storage and needs no key after dependency installation.

## Verification

The final archive was extracted and tested with the validated isolated agent
dependency environment. All **67 tests passed** with five dependency deprecation
warnings. The 60-day offline CLI ran twice with byte-identical stdout. The real
provider CLI clarified and recommended successfully. A real-provider smoke run
passed all eight engineering cases in one synthetic scripted sequence. Production
browser checks passed preference changes, actual feedback and camp with no page
errors. See [evidence](evidence/).

All 91 protected files, including 36 frozen result files, matched their original
checksums in the checkout and extracted archive. One pre-existing ignored
synthetic NeuralUCB log is bundled as supplemental evidence; the manifest
identifies it separately from Git-tracked source. Three committed relative data
symlinks are preserved with their original targets; no compatibility symlink is
added. No new benchmark experiment is claimed by this documentation release.

These are engineering checks, not eight patient sessions, a conversation-quality
study, clinical safety validation or evidence of improved adherence.

## Version guide

- [main](https://github.com/ZHIQIAN-fine/NarraCare/tree/main): current source.
- [agent](https://github.com/ZHIQIAN-fine/NarraCare/tree/agent): reviewed source
  snapshot and this delivery bundle.
- [benchmark](https://github.com/ZHIQIAN-fine/NarraCare/tree/benchmark): preserved
  pre-agent benchmark at `da833fd`.
- Older feature histories remain available under `archive/` tags. The earlier
  delivery branch is retained so existing download links remain usable.

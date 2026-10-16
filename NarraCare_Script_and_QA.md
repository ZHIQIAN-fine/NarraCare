# NarraCare Three Minute Presentation

Main talk: slides 1–6. Backup slides 7–9. Rehearse aloud for a three-minute limit.

## Slide 1 0:00–0:12 NarraCare

NarraCare recommends daily rehabilitation exercises with safety checks and a narrative companion. We combine an inspectable recommendation pipeline with a gentle adventure experience.

## Slide 2 0:12–0:35 Personalised rehabilitation

Rehabilitation can feel repetitive, while pain and fatigue change daily. A useful recommender must respect those constraints and make progress understandable. We explore this with twenty synthetic personas, one hundred exercises and sixty-day rollouts, covering knee, shoulder and back rehabilitation.

## Slide 3 0:35–1:20 Five components in one daily loop

Every day, the User Simulator supplies symptoms. Safety filters the catalogue before SAPS ranks candidates and checks the shortlist afterwards. The template Narrator presents a task. Simulated feedback drives the Reward Composer and online learning. SAPS adds a sampled Beta safety bonus to LinUCB's reward estimate. We inspect safety twice: IER measures infeasible ungated proposals, while SVR checks accepted tasks. This separates the learner's proposals from what the filter allows.

## Slide 4 1:20–2:00 Reward and proposal feasibility

SAPS scores forty-two point zero two, close to LinUCB, without a significant advantage. Random earns more reward, but its infeasible proposal rate is forty point eight two percent, versus SAPS's two point two three percent. More directly, removing Safety from LinUCB produces one hundred seventy-one violating accepted tasks. These results support runtime filtering within our simulator.

## Slide 5 2:00–2:40 A companion for the rehabilitation journey

Our demo turns this workflow into a mountain trail with Pip, a supportive companion. Forest, Canyon and Summit mark sixty days of small achievements. Camping makes rest part of the journey, without pressure to push harder. Real task details and safety counts remain visible. The CLI runs the backend; the HTML replays recorded trajectories. Replies change only the presentation. Engagement and adherence benefits still need user evaluation.

## Slide 6 2:40–3:00 Conclusion

Our contribution is an inspectable safety-aware architecture, complementary safety metrics and a companion-based application design. Next, we plan a bounded agent that clarifies symptoms and uses existing tools. That extension remains future work. Thank you.

# One Minute Q&A

## Is this already an agent?

The evaluated version is a sequential recommender. It adapts to simulated feedback, but we have not validated conversational clarification or autonomous tool planning. Our next prototype will clarify symptoms, select existing tools and remember preferences while preserving Safety.

## Why use SAPS if Random earns more?

Random earns more simulator reward, but its ungated IER is 0.4082 versus SAPS's 0.0223. Shared filtering protects both. SAPS is close to LinUCB, without a significant advantage. Our contribution is the inspectable architecture and complementary safety diagnostics.

## Does the companion improve recovery or adherence?

That is our design intention, not an established result. The trail offers small achievements and rest without pressure. The HTML replays recorded outcomes. Patient benefit needs a user study and clinical evaluation.

# Additional Questions

## What does IER measure, and what does SVR measure?

IER is the fraction of an ungated top-five proposal list outside the current feasible set. SVR checks accepted tasks against all five runtime predicates. Our separate pooled LinUCB replay finds 0/743 violations with Safety and 171/772 without it. These metrics describe implemented constraints in a simulator, not a clinical safety guarantee. See backup slide 8.

## What is original about the project?

The contribution is the rehabilitation-specific architecture and integration, the joint use of proposal-level IER and execution-level SVR, and the companion-trail application design. We do not claim to have invented IER, bandits or game-based rehabilitation. The distinctive combination makes restrictions, learning behavior and presentation inspectable together.

## What is SAPS technically?

SAPS adds a sampled per-task Beta safety bonus, weighted by 2.0, to a LinUCB UCB reward score. Chosen-task pain_change at most 1.0 gives a positive safety observation. Ungated infeasible proposals also receive existing negative reward updates. The Beta and penalty effects have not yet been isolated by ablation.

## Why is the Day 1 screenshot in the Forest but its narrator tone is companion?

Forest/Canyon/Summit are presentation stages. The backend retains a day-zero arc fallback and condition-name mismatch, disclosed in the report. The first trace may therefore use act3 and companion. The HTML displays the actual narrator tone and does not repair or conceal that behavior.

## Is the demo live, and can my click change the recommendation?

The CLI runs the five-component backend. The single-file HTML replays nine embedded trajectories. Its Try/Skip replies change the trail presentation only, preserving recorded tasks, choices and rewards. A live dialogue agent is proposed separately.

## Are all 56 safety rules executed?

No. Five runtime pre-filter predicates execute in the evaluated path. The JSON holds 51 design-time definitions. Together they are 56 authored rules, not 56 runtime checks. Clinical coverage requires a separate review.

## Why use synthetic users, and what are the main limits?

They allow controlled inspection without identifiable patient records. The study uses twenty personas, one seed and a shared learner. The rule-based and neural references have implementation limitations. We do not establish clinical efficacy, engagement gains or a significant advantage over LinUCB.

## Can the agent be finished in three days?

A narrow prototype is feasible if we reuse the current tools and restrict the dialogue to symptoms, preferences and task feedback. The deliverable is a tested tool-selection and clarification loop, not an open-ended medical assistant. It remains proposed until the new interaction tests pass. See backup slide 9.

# Proposed Three Day Agent Prototype

Proposed follow-up only. No implementation or completion claim.

## Day 1 Confirmed state and dialogue decisions

Add a small controller with explicit session state. Extract a restricted set of symptom/preference fields from forms or a documented English phrase grammar. Ask about missing, contradictory or out-of-range values instead of silently defaulting. Choose between clarify, recommend, request feedback and stop. Keep the current demo and formal results unchanged.

## Day 2 Tool calls and actual feedback

Wrap the existing candidate filtering/ranking path and UserMemory behind narrow tool interfaces. Call run_single_turn with confirmed structured state, rather than the legacy interactive_session that synthesizes feedback. Collect task completion and before/after symptom feedback explicitly. Validate the reward input contract before process_feedback. Retain the hard safety checks, template narrator and traceable tool results. A preference change must reach the context or candidate selection, rather than only changing reply wording.

## Day 3 Scripted interaction validation

Test complete check-ins, missing/contradictory symptoms, high pain/fatigue, skip, changed preference, empty feasible sets and unsafe requests. Assert the selected action/tool, actual state and memory updates, and all five predicates on delivered tasks. Demonstrate appropriate response changes without requiring a different top-1 on every turn. Record separate interaction traces and regression results. Only then describe the extension as a bounded dialogue/tool-orchestration agent prototype.
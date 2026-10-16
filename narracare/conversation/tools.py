"""Auditable component tools. All state is session-local; no experiment writes."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import yaml

from src.agents.narrator import Narrator
from src.agents.recommender import RecommenderAgent
from src.agents.safety_agent import SafetyAgent
from src.agents.user_understanding import UserUnderstandingAgent
from src.memory import UserMemory
from src.pipeline import SAPS_PAIN_SAFE_THRESHOLD
from src.simulator import UserSimulator

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "ACL_reconstruction": ("ACL", "knee"),
    "rotator_cuff_repair": ("RotatorCuff", "shoulder"),
    "chronic_lower_back_pain": ("BackPain", "back"),
}
TERMINAL = {
    "ask_clarification",
    "present_recommendation",
    "explain_recommendation",
    "record_feedback",
    "rest_today",
    "show_state",
    "end_session",
}


class ToolError(ValueError):
    pass


def number(value, lo, hi, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not lo <= value <= hi
    ):
        raise ToolError(
            f"{name} must be a finite number in [{lo}, {hi}]. Ask the user to clarify."
        )
    return float(value)


def schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def tool_schemas():
    cats = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["strength", "flexibility", "balance", "cardio", "rest"],
        },
    }
    evidence = {
        "type": "string",
        "description": "Exact quote copied from the current user message, never invented.",
    }
    return [
        schema(
            "update_checkin",
            "Extract ONLY explicitly reported current symptoms. Never fill omitted fields from a profile. ROM may be null only if explicitly unknown. Updates invalidate old recommendations.",
            {
                "fields": {
                    "type": "object",
                    "properties": {
                        k: {"type": ["number", "null"] if k == "rom" else "number"}
                        for k in ["pain_level", "fatigue_level", "mood", "rom"]
                    },
                    "additionalProperties": False,
                },
                "evidence": evidence,
            },
            ["fields", "evidence"],
        ),
        schema(
            "set_preferences",
            "Maintain explicitly stated exercise category preferences/avoidances. Use gentle/stretching = flexibility, cardio/aerobic = cardio. Do not change medical profile or contraindications.",
            {
                "preferred_categories": cats,
                "avoid_categories": cats,
                "evidence": evidence,
            },
            ["preferred_categories", "avoid_categories", "evidence"],
        ),
        schema(
            "revise_plan",
            "User requests a different unperformed plan, reports changed symptoms, or rejects the current task. Discard the pending proposal without outcome/learning, preserving the day. Then update check-in/preferences and retrieve again.",
            {"evidence": evidence},
            ["evidence"],
        ),
        schema(
            "retrieve_candidates",
            "After a complete check-in, retrieve catalogue items by body part and preference, with hard safety pre-filter. Never substitute invented tasks.",
        ),
        schema(
            "rank_candidates",
            "After retrieval, call SAPS ranking and Safety post_check. A fresh retrieve is required after any update.",
        ),
        schema(
            "present_recommendation",
            "After ranking, render the validated top task with the template Narrator, real score and grounded explanation. Ends this user turn. Never call twice for the same plan.",
        ),
        schema(
            "explain_recommendation",
            "Explain the current recommendation using actual filters, category, task properties and composite S. No hidden reasoning or clinical claims. Ends turn.",
        ),
        schema(
            "ask_clarification",
            "Ask for missing check-in fields or outcome feedback. Use conflict for contradictory numbers. Ends turn. Never guess a symptom.",
            {
                "topic": {
                    "type": "string",
                    "enum": ["checkin", "conflict", "feedback", "preference", "intent"],
                }
            },
            ["topic"],
        ),
        schema(
            "record_feedback",
            "Record an explicitly completed/partial task, NOT an intention to try. Requires actual AFTER pain/fatigue and satisfaction (0-1). Updates reward, SAPS and memory once. For skip use rest_today. Ends turn.",
            {
                "action": {"type": "string", "enum": ["completed", "partial"]},
                "post_pain": {"type": "number"},
                "post_fatigue": {"type": "number"},
                "satisfaction": {"type": "number"},
                "evidence": evidence,
            },
            ["action", "post_pain", "post_fatigue", "satisfaction", "evidence"],
        ),
        schema(
            "rest_today",
            "User explicitly skips or requests rest. Close the day as a campsite, without fabricated outcome, reward or Beta update. Ends turn.",
            {"evidence": evidence},
            ["evidence"],
        ),
        schema(
            "show_state",
            "Show current state and supported usage, or acknowledge intent to try and ask for actual completion feedback. Ends turn.",
        ),
        schema("end_session", "User requests ending the session. Ends turn."),
    ]


class SessionTools:
    def __init__(self, user="U001", seed=42, day=1):
        if not 1 <= day <= 60:
            raise ValueError("day must be in [1, 60]")
        self.config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        self.config["recommender"] = yaml.safe_load(
            (ROOT / "configs/saps.yaml").read_text()
        )["recommender"]
        self.config["system"]["seed"] = seed
        self.reward_composer = UserSimulator(
            self.config
        )  # compute_reward only, no simulated behavior.
        self.profile = copy.deepcopy(self.reward_composer.get_profile(user))
        if not self.profile:
            raise ValueError(
                "Unknown user. Legal values: "
                + ", ".join(self.reward_composer.get_all_user_ids())
            )
        self.user, self.day = user, day
        self.memory = UserMemory(self.config)
        self.memory.set_profile(user, self.profile)
        self.understanding = UserUnderstandingAgent(self.config, mode="simulation")
        self.safety = SafetyAgent(self.config)
        self.recommender = RecommenderAgent(self.config)
        self.narrator = Narrator(
            ROOT / "data/narrative_arcs.json",
            ROOT / "data/narrator_tone_templates.json",
            seed,
        )
        self.tasks = json.loads((ROOT / "data/task_library.json").read_text())
        for task in self.tasks:
            task["intensity"] = (
                "low"
                if task["difficulty"] <= 2
                else "medium"
                if task["difficulty"] == 3
                else "high"
            )
        self.categories = {t["category"] for t in self.tasks}
        self.preferred = list(self.profile.get("preference", []))
        self.avoided = []  # Profile dislikes are preferences, not medical contraindications.
        self.state = {}
        self.last_reported_state = {}
        self.candidates, self.ranked, self.pending = [], [], None
        self.revision, self.retrieved_revision, self.ranked_revision = 0, -1, -1
        self.events, self.days = [], []
        self.current_message = ""
        self.tone, self.narrative, self.closing = "not rendered", "", ""
        self.last_counts = {}
        self.on_hold, self.ended = False, False
        self.streak = 0
        self.last_question = ""
        self.features = None
        self.presented = False

    @property
    def arc_profile(self):
        # Input adapter for NEW agent only: frozen benchmark mappings stay unchanged.
        return {**self.profile, "condition": CONDITIONS[self.profile["condition"]][0]}

    def missing(self):
        return [
            key
            for key in ["pain_level", "fatigue_level", "mood", "rom"]
            if key not in self.state
        ]

    def safe_state(self):
        # Missing ROM cannot silently become a reassuring measured 90/180 degrees.
        return {
            **self.state,
            "rom": self.state.get("rom") if self.state.get("rom") is not None else 0,
            "day": self.day,
        }

    def quote(self, evidence):
        if (
            not isinstance(evidence, str)
            or not evidence.strip()
            or evidence not in self.current_message
        ):
            raise ToolError(
                "Evidence must be an exact nonempty quote from this user message."
            )
        return evidence.lower()

    def invalidate(self):
        self.revision += 1
        self.candidates, self.ranked, self.pending = [], [], None
        self.retrieved_revision = self.ranked_revision = -1
        self.features = None
        self.presented = False

    def update_checkin(self, fields, evidence):
        text = self.quote(evidence)
        if (
            not isinstance(fields, dict)
            or not fields
            or not set(fields)
            <= {
                "pain_level",
                "fatigue_level",
                "mood",
                "rom",
            }
        ):
            raise ToolError(
                "Only pain_level, fatigue_level, mood and rom can be updated."
            )
        parsed = {}
        aliases = {
            "pain_level": r"pain|sore(?:ness)?",
            "fatigue_level": r"fatigue|tired(?:ness)?|energy",
            "mood": r"mood",
            "rom": r"rom|range of motion",
        }
        for key, val in fields.items():
            if key == "rom" and val is None:
                if not re.search(r"rom|range of motion", text) or not re.search(
                    r"unknown|not (?:know|measured)|don't know|unavailable", text
                ):
                    raise ToolError(
                        "Unknown ROM must be explicitly reported as unknown."
                    )
                parsed[key] = None
                continue
            val = number(val, 0, 180 if key == "rom" else 10, key)
            matches = re.findall(
                r"(?:" + aliases[key] + r")[^\d\n,;]{0,24}(-?\d+(?:\.\d+)?)", text
            )
            observed = {float(x) for x in matches}
            if len(observed) > 1:
                raise ToolError(
                    f"Conflicting {key} values. Ask which value is current."
                )
            if val not in observed:
                raise ToolError(
                    f"{key} needs a quoted numeric rating next to its label. Do not infer a rating from 'sore' or 'tired'."
                )
            parsed[key] = val
        if self.pending and self.presented:
            raise ToolError(
                "A task is awaiting outcome. Record its feedback or explicitly skip before a new check-in."
            )
        self.invalidate()
        self.state.update(parsed)
        self.last_reported_state = dict(self.state)
        return {"state": self.state, "missing": self.missing(), "source": "user_report"}

    def set_preferences(self, preferred_categories, avoid_categories, evidence):
        text = self.quote(evidence)
        if not isinstance(preferred_categories, list) or not isinstance(
            avoid_categories, list
        ):
            raise ToolError("Preference fields must be category lists.")
        new = preferred_categories + avoid_categories
        if not all(isinstance(x, str) and x in self.categories for x in new):
            raise ToolError("Unknown exercise category.")
        aliases = {
            "flexibility": r"flexibility|stretch|gentle",
            "strength": r"strength|strong|weights",
            "cardio": r"cardio|aerobic",
            "balance": r"balance",
            "rest": r"rest|camp",
        }
        if any(not re.search(aliases[x], text) for x in new):
            raise ToolError("Each preference must be supported by the quoted message.")
        if set(preferred_categories) & set(avoid_categories):
            raise ToolError(
                "A category cannot be both preferred and avoided. Clarify first."
            )
        if self.pending and self.presented:
            raise ToolError(
                "For an explicitly rejected or unperformed plan, first call revise_plan with the user's revision evidence, then set_preferences and retrieve/rank/present again. If the task was performed, collect actual feedback instead."
            )
        self.invalidate()
        self.preferred, self.avoided = (
            list(dict.fromkeys(preferred_categories)),
            list(dict.fromkeys(avoid_categories)),
        )
        self.profile["preference"] = self.preferred
        self.memory.set_profile(self.user, self.profile)
        return {
            "preferred_categories": self.preferred,
            "avoid_categories": self.avoided,
            "source": "user_report",
        }

    def revise_plan(self, evidence):
        text = self.quote(evidence)
        if not self.pending:
            raise ToolError("There is no pending plan to revise.")
        if not re.search(
            r"instead|alternative|different|sore|pain|fatigue|tired|reject|too hard|prefer|avoid|don't like",
            text,
        ):
            raise ToolError(
                "An explicit revision request or changed symptom is required."
            )
        self.invalidate()
        return {"revised": True, "learning_update": False, "day": self.day}

    def retrieve_candidates(self):
        if self.missing():
            raise ToolError("Missing current check-in: " + ", ".join(self.missing()))
        if self.pending and self.presented:
            raise ToolError("A presented task awaits feedback or skip.")
        feasible = self.safety.pre_filter(self.tasks, self.safe_state(), self.profile)
        target = CONDITIONS[self.profile["condition"]][1]
        relevant = [
            t
            for t in feasible
            if t["body_part"] in {target, "full_body"}
            and t["category"] not in self.avoided
        ]
        preferred = [t for t in relevant if t["category"] in self.preferred]
        self.candidates = (
            preferred or relevant
        )  # Soft preference only; safety/body-part/avoidance never relaxed.
        self.retrieved_revision = self.revision
        self.last_counts = {
            "catalogue": len(self.tasks),
            "safety_passed": len(feasible),
            "safety_blocked": len(self.tasks) - len(feasible),
            "relevant": len(relevant),
            "retrieved": len(self.candidates),
            "preferred_match": bool(preferred),
            "rom_unknown_conservative": self.state.get("rom") is None,
        }
        return {
            **self.last_counts,
            "items": [
                {
                    k: t[k]
                    for k in ["task_id", "name", "category", "body_part", "difficulty"]
                }
                for t in self.candidates
            ],
        }

    def rank_candidates(self):
        if self.retrieved_revision != self.revision:
            raise ToolError(
                "Call retrieve_candidates after the latest check-in/preferences."
            )
        if not self.candidates:
            raise ToolError(
                "No feasible relevant candidates. Clarify preferences or offer rest; never invent a fallback."
            )
        state = self.safe_state()
        self.features = self.understanding.parse_structured(
            state,
            self.profile,
            completion_rate_7d=self.memory.get_completion_rate(self.user, 7),
            consecutive_high_intensity=self.memory.get_consecutive_high_intensity(
                self.user
            ),
        )
        self.recommender.model.set_safe_ids({t["task_id"] for t in self.candidates})
        ranked = self.recommender.recommend(self.features.to_numpy(), self.candidates)
        by_id = {t["task_id"]: t for t in self.candidates}
        if any(t.get("task_id") not in by_id for t in ranked):
            self.ranked = []
            self.ranked_revision = -1
            raise ToolError("Ranker returned an item outside the retrieved set.")
        canonical = [
            {**by_id[t["task_id"]], "score": number(t["score"], -1e12, 1e12, "S")}
            for t in ranked
        ]
        checked = self.safety.post_check(
            canonical, state, self.memory.get_consecutive_high_intensity(self.user)
        )
        # Validate canonical tasks again, with no fallback after rejection.
        self.ranked = self.safety.pre_filter(checked, state, self.profile)
        self.ranked_revision = self.revision
        return {
            "ranked": [
                {
                    "task_id": t["task_id"],
                    "name": t["name"],
                    "S": t["score"],
                    "category": t["category"],
                    "difficulty": t["difficulty"],
                }
                for t in self.ranked
            ],
            "post_check": "PASS" if self.ranked else "ABSTAIN",
        }

    def explanation(self):
        task = self.pending
        if not task:
            raise ToolError("No presented recommendation to explain.")
        c = self.last_counts
        rom_note = (
            " ROM is unknown; the filter used a conservative 0-degree proxy, not a measurement."
            if self.state.get("rom") is None
            else ""
        )
        return (
            f"{task['name']} ({task['task_id']}) is a {task['category']} task for {task['body_part']}, difficulty {task['difficulty']}/5. "
            f"Safety retained {c['safety_passed']}/{c['catalogue']} catalogue tasks; retrieval retained {c['retrieved']} matching items. "
            f"Your preferred categories are {', '.join(self.preferred) or 'not specified'}; explicit avoidances remain excluded. "
            f"SAPS ranked this task first with composite S = {task['score']:.3f}. "
            "S combines a LinUCB UCB reward term with a sampled Beta safety bonus. It is not a clinical success probability. "
            "All five pre-filter predicates and the shortlist post-check passed. "
            "This explains the actual selection path, not a causal attribution of individual score terms."
            + rom_note
        )

    def present_recommendation(self):
        if self.ranked_revision != self.revision or not self.ranked:
            raise ToolError(
                "Retrieve and rank a feasible plan first; otherwise abstain."
            )
        if self.presented:
            raise ToolError(
                "This plan was already presented. Request actual feedback or explain it."
            )
        task = self.ranked[0]
        if not self.safety.pre_filter(
            [task], self.safe_state(), self.profile
        ) or not self.safety.post_check(
            [task],
            self.safe_state(),
            self.memory.get_consecutive_high_intensity(self.user),
        ):
            raise ToolError("Final safety validation failed. No task can be presented.")
        self.pending = copy.deepcopy(task)
        self.presented = True
        self.memory.update_state(self.user, self.safe_state())
        context = {
            "name": self.profile["name"],
            "day": self.day,
            "task_name": task["name"],
            "difficulty": task["difficulty"],
            "body_part": task["body_part"],
            "rehab_stage": self.profile["current_stage"],
            "reason_from_bandit": "ranked within the feasible, relevant candidate set",
        }
        self.narrative, meta = self.narrator.speak(
            "recommendation_framing",
            context,
            self.arc_profile,
            self.day,
            self.safe_state(),
            {"consecutive_skips": 0},
        )
        self.tone = meta["tone_actually_used"]
        c = self.last_counts
        summary = (
            f"Safety: {c['catalogue']} → {c['safety_passed']} candidates; retrieval: {c['retrieved']}. "
            f"SAPS ranked {task['task_id']} first (S = {task['score']:.3f}). "
            f"Category: {task['category']}, difficulty {task['difficulty']}/5. Ask why for selection details."
        )
        if self.state.get("rom") is None:
            summary += " ROM is unknown; filtering used a conservative 0-degree proxy, not a measurement."
        return {
            "message": f"**Pip · {self.tone}**\n\n{self.narrative}\n\n{summary}\n\nIf you choose to do this demo task, report completed/partial, AFTER pain, AFTER fatigue and satisfaction (0–1). Or say skip. No outcome is assumed.",
            "task_id": task["task_id"],
            "S": task["score"],
            "tone": self.tone,
        }

    def explain_recommendation(self):
        return {"message": self.explanation()}

    def ask_clarification(self, topic):
        self.last_question = topic
        if topic == "checkin":
            missing = self.missing()
            text = (
                "Please report today's "
                + ", ".join(missing or ["pain_level", "fatigue_level", "mood", "rom"])
                + ". Pain, fatigue and mood use 0–10. ROM uses degrees; say 'ROM unknown' if not measured."
            )
        elif topic == "feedback":
            text = "Did you complete, partially complete, or skip the proposed task? For completed/partial, report AFTER pain (0–10), AFTER fatigue (0–10) and satisfaction (0–1). 'I'll try' does not mean completed."
        elif topic == "conflict":
            text = "I found conflicting or invalid information. Please restate one current numeric rating for each symptom, or clearly label your after-task feedback."
        elif topic == "preference":
            text = "Which categories would you prefer or avoid: strength, flexibility, balance, cardio or rest? Preferences cannot override safety."
        elif topic == "intent":
            text = "Would you like a recommendation, an explanation, to record task feedback, or to rest today?"
        else:
            raise ToolError("Unknown clarification topic.")
        return {"message": text, "missing": self.missing()}

    def _close_day(self, action, reward, pain_delta=0):
        old_day = self.day
        context = {
            "name": self.profile["name"],
            "adherence": f"{self.memory.get_completion_rate(self.user, 7):.0%}",
            "pain_delta": pain_delta,
            "pain_level": self.state.get("pain_level", 0),
            "fatigue_level": self.state.get("fatigue_level", 0),
            "next_focus": "a fresh check-in",
            "streak_count": self.streak,
            "personal_moment_from_today": "You made space to check in with yourself",
        }
        self.closing, meta = self.narrator.say_daily_closing(
            {
                "day": old_day,
                "user_skipped": action == "skipped",
                "user_accepted": action != "skipped",
                "pain_delta": pain_delta,
                "fatigue_level": self.state.get("fatigue_level", 0),
            },
            self.arc_profile,
            context,
        )
        self.days.append(
            {
                "day": old_day,
                "action": action,
                "reward": reward,
                "task_id": self.pending["task_id"] if self.pending else None,
                "closing": self.closing,
                "variant": meta["daily_closing_variant"],
                "source": "user_report",
            }
        )
        self.day += 1
        self.state = {}
        self.invalidate()
        if self.day > 60:
            self.ended = True
        return old_day

    def record_feedback(self, action, post_pain, post_fatigue, satisfaction, evidence):
        text = self.quote(evidence)
        if not self.pending or not self.presented:
            raise ToolError(
                "No presented task awaits feedback, or its feedback was already recorded."
            )
        if action not in {"completed", "partial"}:
            raise ToolError("Action must be completed or partial.")
        if action == "completed" and not re.search(
            r"\b(?:completed|finished|done)\b", text
        ):
            raise ToolError(
                "Explicit completion is required. 'Accept' or 'I'll try' is only intention."
            )
        if re.search(
            r"\b(?:not|haven't|didn't|never)\b.{0,12}\b(?:completed|finished|done)\b",
            text,
        ):
            raise ToolError("Completion is negated; clarify the outcome.")
        if action == "partial" and not re.search(
            r"partial|stopped|couldn't finish", text
        ):
            raise ToolError("Explicit partial completion is required.")
        values = {
            "post_pain": number(post_pain, 0, 10, "post_pain"),
            "post_fatigue": number(post_fatigue, 0, 10, "post_fatigue"),
            "satisfaction": number(satisfaction, 0, 1, "satisfaction"),
        }
        for field, val in values.items():
            label = {
                "post_pain": r"(?:after[ -]?(?:task)?\s*)?pain",
                "post_fatigue": r"(?:after[ -]?(?:task)?\s*)?(?:fatigue|tiredness)",
                "satisfaction": "satisfaction",
            }[field]
            found = {
                float(x)
                for x in re.findall(label + r"[^\d\n,;]{0,18}(-?\d+(?:\.\d+)?)", text)
            }
            if found != {val}:
                raise ToolError(
                    f"{field} must match a single numeric value in the quoted outcome report."
                )
        state = self.safe_state()
        delta = values["post_pain"] - state["pain_level"]
        feedback = {
            "completed": action == "completed",
            "action": action,
            "pain_change": delta,
            "fatigue_change": values["post_fatigue"] - state["fatigue_level"],
            "satisfaction": values["satisfaction"],
        }
        reward = self.reward_composer.compute_reward(feedback, state, self.pending)
        vec = self.features.to_numpy()
        tid = self.pending["task_id"]
        self.recommender.update(vec, tid, reward)
        self.recommender.model.update_safety(tid, delta <= SAPS_PAIN_SAFE_THRESHOLD)
        self.memory.record_interaction(
            self.user,
            tid,
            action,
            reward,
            {
                **state,
                "pain_level": post_pain,
                "fatigue_level": post_fatigue,
                "task_intensity": self.pending["intensity"],
            },
            day=self.day,
            context_state=state,
        )
        self.state.update(
            pain_level=values["post_pain"], fatigue_level=values["post_fatigue"]
        )
        self.last_reported_state = dict(self.state)
        self.streak = self.streak + 1 if action == "completed" else 0
        closed_day = self._close_day(action, reward, delta)
        return {
            "message": f"Day {closed_day} recorded from your report. Reward Composer = {reward:.4f}. SAPS reward and Beta safety posteriors updated.\n\n{self.closing}\n\nNext day starts with a fresh check-in.",
            "reward": reward,
            "pain_change": delta,
            "was_safe": delta <= SAPS_PAIN_SAFE_THRESHOLD,
            "source": "user_report",
        }

    def rest_today(self, evidence):
        text = self.quote(evidence)
        if not re.search(r"\b(?:skip|rest|camp|too tired)\b", text):
            raise ToolError("An explicit skip/rest request is required.")
        if self.pending:
            self.memory.record_interaction(
                self.user,
                self.pending["task_id"],
                "skipped",
                0.0,
                {**self.safe_state(), "task_intensity": "none"},
                day=self.day,
                context_state=self.safe_state(),
            )
        self.streak = 0
        closed_day = self._close_day("skipped", None)
        return {
            "message": f"Day {closed_day}: camp together. The trail does not advance. No symptom outcome, reward or Beta observation was invented.\n\n{self.closing}",
            "reward": None,
            "source": "user_report",
        }

    def show_state(self):
        return {
            "message": "Pip is here for a manageable step, or a rest. "
            + (
                "A task awaits your actual outcome. Tell me completed/partial with after pain, fatigue and satisfaction, or skip."
                if self.pending
                else "Tell me today's pain, fatigue and mood (0–10), ROM in degrees or 'ROM unknown', and exercise preferences. You can ask why, or say skip."
            ),
            "snapshot": self.snapshot(),
        }

    def end_session(self):
        self.ended = True
        return {
            "message": "Session ended. Your current benchmark files are unchanged. Take care."
        }

    def snapshot(self):
        day = min(self.day, 60)
        act = "act1" if day <= 15 else "act2" if day <= 40 else "act3"
        return {
            "user": self.user,
            "name": self.profile["name"],
            "condition": self.profile["condition"],
            "day": day,
            "act": act,
            "stage": self.profile["current_stage"],
            "state": copy.deepcopy(self.state),
            "last_reported_state": dict(self.last_reported_state),
            "missing": self.missing(),
            "preferred_categories": list(self.preferred),
            "avoid_categories": list(self.avoided),
            "pending_task": copy.deepcopy(self.pending),
            "counts": dict(self.last_counts),
            "tone": self.tone,
            "narrative": self.narrative,
            "closing": self.closing,
            "on_hold": self.on_hold,
            "ended": self.ended,
            "days_recorded": len(self.days),
            "trail_steps": sum(r["action"] == "completed" for r in self.days),
            "streak": self.streak,
            "cumulative_reward": sum(r["reward"] or 0 for r in self.days),
            "reward_history": copy.deepcopy(self.days),
            "events": copy.deepcopy(self.events[-20:]),
        }

    def dispatch(self, name, args):
        allowed = {tool["function"]["name"] for tool in tool_schemas()}
        if self.on_hold or self.ended:
            raise ToolError(
                "Session is on safety hold or ended. No further tools may execute."
            )
        if name not in allowed or not isinstance(args, dict):
            raise ToolError("Unknown tool or invalid arguments.")
        if self.day > 60:
            raise ToolError("The 60-day demo is complete.")
        try:
            result = getattr(self, name)(**args)
        except TypeError:
            raise ToolError(
                "Tool arguments do not match its declared schema."
            ) from None
        self.events.append(
            {"tool": name, "status": "ok", "result": copy.deepcopy(result)}
        )
        return result

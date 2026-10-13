"""
NarraCare — Streamlit Interactive Demo
=======================================
A multi-agent conversational rehabilitation recommender system.

Run:
    cd NarraCare
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import streamlit as st

# ── Path setup so we can import from src ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml

from agents.narrator_agent import NarratorAgent
from agents.safety_agent import SafetyAgent
from agents.recommender import RecommenderAgent
from agents.user_understanding import UserUnderstandingAgent
from memory import UserMemory

# ── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

STAGE_LABELS = {
    "early_rehab": "🟡 Early Rehab",
    "mid_rehab": "🟠 Mid Rehab",
    "late_rehab": "🔵 Late Rehab",
    "return_to_sport": "🟢 Return to Sport",
    "return_to_activity": "🟢 Return to Activity",
    "maintenance": "🟢 Maintenance",
}
INTENSITY_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}
CATEGORY_EMOJI = {
    "strength": "💪", "flexibility": "🧘", "balance": "⚖️",
    "cardio": "🏃", "breathing": "🌬️", "relaxation": "🧘‍♂️",
    "plyometrics": "🦘", "rest": "😴",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_user_profiles() -> list[dict]:
    with open(DATA_DIR / "user_profiles.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_task_library() -> list[dict]:
    with open(DATA_DIR / "task_library.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: derive intensity from difficulty level
# ═══════════════════════════════════════════════════════════════════════════

def _intensity(task: dict) -> str:
    """Map task difficulty (1-7) to intensity label."""
    d = task.get("difficulty", 1)
    if task.get("intensity"):
        return task["intensity"]
    if d <= 2:
        return "low"
    elif d <= 4:
        return "medium"
    else:
        return "high"


def _enrich_tasks(tasks: list[dict]) -> list[dict]:
    """Ensure every task has an 'intensity' field."""
    for t in tasks:
        if "intensity" not in t:
            t["intensity"] = _intensity(t)
    return tasks


# ═══════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """Initialise all session state variables on first run."""
    defaults = {
        "current_day": 1,
        "messages": [],             # chat history [{role, content, extra}]
        "safety_alerts": [],        # list of triggered rules
        "completed_tasks": [],      # list of task dicts
        "skipped_tasks": [],
        "selected_user_id": None,
        "agents_ready": False,
        "demo_mode": False,
        "demo_running": False,
        "use_llm": False,
        "pain_override": None,
        "fatigue_override": None,
        "rom_override": None,
        "pending_options": None,    # current recommendation options
        "interaction_count": 0,
        "total_reward": 0.0,
        "consecutive_high": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session() -> None:
    """Reset all session state (new session)."""
    keys_to_keep = {"selected_user_id", "use_llm", "demo_mode"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    init_session_state()


# ═══════════════════════════════════════════════════════════════════════════
# Agent Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def get_agents(config: dict) -> tuple:
    """Build and cache agents in session state."""
    mode = "interactive" if st.session_state.get("use_llm") else "simulation"
    rec = RecommenderAgent(config)
    safety = SafetyAgent(config)
    narrator = NarratorAgent(config, mode=mode)
    uu = UserUnderstandingAgent(config, mode=mode)
    memory = UserMemory(config)
    return rec, safety, narrator, uu, memory


def filter_candidates(task_lib: list[dict], profile: dict) -> list[dict]:
    """Coarse pre-filter: stage-applicable + not contra-indicated."""
    stage = profile.get("current_stage", "early_rehab")
    contras = set(profile.get("contraindications", []))
    filtered = []
    for t in task_lib:
        if stage not in t.get("applicable_stages", []):
            continue
        task_contras = set(t.get("contraindications", []))
        if task_contras & contras:
            continue
        filtered.append(t)
    return filtered


def run_recommendation_pipeline(
    config: dict,
    profile: dict,
    user_state: dict,
    task_lib: list[dict],
    day: int,
    memory: UserMemory,
    rec: RecommenderAgent,
    safety: SafetyAgent,
    narrator: NarratorAgent,
    uu: UserUnderstandingAgent,
) -> tuple[str, list[dict], list[str]]:
    """Execute the full NarraCare pipeline for one day.

    Returns (narrative_text, recommended_tasks, safety_alerts).
    """
    user_id = profile["user_id"]

    # 1. Feature extraction
    completion_rate = memory.get_completion_rate(user_id, 7) if memory else 0.0
    consec_high = st.session_state.get("consecutive_high", 0)
    feat = uu.parse_structured(user_state, profile, completion_rate, consec_high)

    # 2. Candidate pre-filter (stage + contraindication)
    candidates = filter_candidates(task_lib, profile)
    candidates = _enrich_tasks(candidates)

    # 3. Safety pre-filter
    safe_candidates = safety.pre_filter(candidates, user_state, profile)

    # Fallback: if safety removed everything, keep at least stretching/breathing
    if not safe_candidates:
        safe_candidates = [
            t for t in candidates
            if t.get("category") in ("flexibility", "breathing", "relaxation")
        ]
    if not safe_candidates:
        safe_candidates = candidates[:3]  # absolute fallback

    # 4. Recommend
    recommended = rec.recommend(feat.to_numpy(), safe_candidates, top_k=config["recommender"]["top_k"])

    # 5. Safety post-check
    recommended = safety.post_check(recommended, user_state, consec_high)

    # 6. Collect safety alerts for this round
    alerts: list[str] = []
    pain = user_state.get("pain_level", 0)
    fatigue = user_state.get("fatigue_level", 0)
    if pain >= 7:
        alerts.append(f"🔴 High pain detected ({pain}/10) — restricting to low intensity")
    if fatigue >= 8:
        alerts.append(f"🟠 Severe fatigue ({fatigue}/10) — gentle exercises only")
    if user_state.get("swelling"):
        alerts.append("🔴 Swelling reported — blocking high-impact tasks")
    if consec_high >= 3:
        alerts.append(f"⚠️ {consec_high} consecutive high-intensity days — forcing rest")

    # 7. Narrative wrapping
    narrative = narrator.wrap(recommended, user_state, profile, day, completion_rate)

    return narrative, recommended, alerts


# ═══════════════════════════════════════════════════════════════════════════
# UI Components
# ═══════════════════════════════════════════════════════════════════════════

def render_left_sidebar(profiles: list[dict]) -> dict | None:
    """Left sidebar: user selector + profile display + manual state."""
    st.sidebar.markdown("## 👤 User Selection")

    user_map = {f"{p['user_id']} — {p['name']}": p for p in profiles}
    options = list(user_map.keys())

    selected_label = st.sidebar.selectbox("Select Patient", options, index=0)
    profile = user_map[selected_label]

    # Detect user change
    if st.session_state.selected_user_id != profile["user_id"]:
        st.session_state.selected_user_id = profile["user_id"]
        reset_session()
        st.session_state.selected_user_id = profile["user_id"]

    # ── Profile Card ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Patient Profile")
    stage_label = STAGE_LABELS.get(profile.get("current_stage", ""), profile.get("current_stage", ""))
    st.sidebar.markdown(f"""
| Field | Value |
|-------|-------|
| **Name** | {profile['name']} |
| **Age** | {profile['age']} |
| **Condition** | `{profile['condition']}` |
| **Stage** | {stage_label} |
| **Personality** | {profile.get('personality', 'N/A')} |
| **Pain Baseline** | {profile.get('pain_baseline', 'N/A')}/10 |
| **Fatigue Baseline** | {profile.get('fatigue_baseline', 'N/A')}/10 |
| **ROM Current** | {profile.get('rom_current', 'N/A')}° |
| **Adherence** | {profile.get('adherence_history', 0):.0%} |
""")

    # ── Manual State Input ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Today's State")

    pain_default = profile.get("pain_baseline", 3)
    fatigue_default = profile.get("fatigue_baseline", 4)
    rom_default = profile.get("rom_current", 90) or 90

    pain = st.sidebar.slider("Pain Level", 0, 10, int(pain_default), key="pain_slider")
    fatigue = st.sidebar.slider("Fatigue Level", 0, 10, int(fatigue_default), key="fatigue_slider")
    rom = st.sidebar.slider("ROM (degrees)", 0, 180, int(rom_default), key="rom_slider")

    st.session_state.pain_override = pain
    st.session_state.fatigue_override = fatigue
    st.session_state.rom_override = rom

    # ── New Day Button ──
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.sidebar.button("🌅 New Day", use_container_width=True):
            st.session_state.current_day += 1
            st.session_state.pending_options = None
            add_message("system", f"━━━ **Day {st.session_state.current_day}** ━━━")
    with col2:
        if st.sidebar.button("🔄 Reset", use_container_width=True):
            reset_session()
            st.rerun()

    # ── Mode toggles ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    st.session_state.demo_mode = st.sidebar.toggle("🤖 Demo Mode (Auto-play)", value=st.session_state.demo_mode)
    st.session_state.use_llm = st.sidebar.toggle("🧠 LLM Mode (requires API key)", value=st.session_state.use_llm)

    if st.session_state.use_llm:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    return profile


def render_right_sidebar() -> None:
    """Right sidebar: safety dashboard + progress stats + chart."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛡️ Safety Dashboard")

    alerts = st.session_state.safety_alerts
    if alerts:
        for alert in alerts[-8:]:  # show last 8
            st.sidebar.warning(alert)
    else:
        st.sidebar.success("✅ No safety alerts this session")

    # ── Progress Stats ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Progress Stats")
    completed = len(st.session_state.completed_tasks)
    skipped = len(st.session_state.skipped_tasks)
    total = completed + skipped
    comp_rate = completed / total if total > 0 else 0.0
    adherence = comp_rate

    day = st.session_state.current_day

    col1, col2 = st.sidebar.columns(2)
    col1.metric("📅 Day", day)
    col2.metric("🏋️ Tasks Done", completed)

    col3, col4 = st.sidebar.columns(2)
    col3.metric("✅ Completion", f"{comp_rate:.0%}")
    col4.metric("📈 Adherence", f"{adherence:.0%}")

    if st.session_state.interaction_count > 0:
        avg_reward = st.session_state.total_reward / st.session_state.interaction_count
        st.sidebar.metric("⭐ Avg Reward", f"{avg_reward:.3f}")

    # ── Category distribution chart ──
    if st.session_state.completed_tasks:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Completed Task Categories")
        cats = [t.get("category", "other") for t in st.session_state.completed_tasks]
        counter = Counter(cats)

        import pandas as pd
        df = pd.DataFrame(
            {"Category": list(counter.keys()), "Count": list(counter.values())}
        )
        df = df.sort_values("Count", ascending=True)
        st.sidebar.bar_chart(df.set_index("Category"))


def add_message(role: str, content: str, extra: dict | None = None) -> None:
    """Append a message to the chat history."""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "extra": extra or {},
    })


def render_chat() -> None:
    """Render the chat history in the main area."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            st.markdown(
                f'<div style="background:#1e293b; color:#e2e8f0; padding:16px; '
                f'border-radius:12px; margin:8px 0; border-left:4px solid #3b82f6;">'
                f'{content}</div>',
                unsafe_allow_html=True,
            )
        elif role == "narrator":
            st.markdown(
                f'<div style="background:#0f172a; color:#f1f5f9; padding:16px; '
                f'border-radius:12px; margin:8px 0; border-left:4px solid #8b5cf6;">'
                f'🤖 <b>NarraCare</b><br><br>{content}</div>',
                unsafe_allow_html=True,
            )
        elif role == "safety":
            st.markdown(
                f'<div style="background:#7f1d1d; color:#fecaca; padding:14px; '
                f'border-radius:12px; margin:8px 0; border-left:4px solid #ef4444;">'
                f'🛡️ <b>Safety Agent</b><br><br>{content}</div>',
                unsafe_allow_html=True,
            )
        elif role == "user":
            st.markdown(
                f'<div style="background:#1e3a5f; color:#bfdbfe; padding:14px; '
                f'border-radius:12px; margin:8px 0; border-left:4px solid #22d3ee; '
                f'text-align:right;">'
                f'💬 <b>Patient</b><br><br>{content}</div>',
                unsafe_allow_html=True,
            )
        elif role == "feedback":
            st.markdown(
                f'<div style="background:#064e3b; color:#a7f3d0; padding:14px; '
                f'border-radius:12px; margin:8px 0; border-left:4px solid #10b981;">'
                f'📝 <b>Feedback</b><br><br>{content}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="NarraCare — Rehab Recommender Demo",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown("""
    <style>
    .stApp { background-color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    h1, h2, h3 { color: #e6edf3; }
    .stMarkdown { color: #c9d1d9; }
    div[data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.3rem; }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Load data
    config = load_config()
    profiles = load_user_profiles()
    task_lib = _enrich_tasks(load_task_library())

    # ── Layout: use sidebar for left + right panels, main for chat ──
    # Streamlit has one sidebar; we split the main area into columns.

    # ── Left Sidebar ──
    profile = render_left_sidebar(profiles)
    if profile is None:
        st.warning("Please select a user from the sidebar.")
        return

    # ── Build agents ──
    rec, safety, narrator, uu, memory = get_agents(config)

    # Store profile in memory
    memory.set_profile(profile["user_id"], profile)

    # ── Main Area: 3-column layout ──
    main_col, right_col = st.columns([3, 1])

    with main_col:
        st.markdown("# 🏥 NarraCare — Rehabilitation Recommender")
        st.markdown(
            f"**Patient:** {profile['name']}  |  "
            f"**Day:** {st.session_state.current_day}  |  "
            f"**Stage:** {STAGE_LABELS.get(profile.get('current_stage', ''), '')}  |  "
            f"**Mode:** {'🧠 LLM' if st.session_state.use_llm else '📝 Template'}"
        )
        st.markdown("---")

        # Render existing chat
        render_chat()

        # ── Generate recommendations button ──
        if st.session_state.pending_options is None:
            if st.button("🎯 Get Today's Recommendations", type="primary", use_container_width=True):
                with st.spinner("Running NarraCare pipeline..."):
                    user_state = {
                        "day": st.session_state.current_day,
                        "pain_level": st.session_state.pain_override or profile.get("pain_baseline", 3),
                        "fatigue_level": st.session_state.fatigue_override or profile.get("fatigue_baseline", 4),
                        "rom": st.session_state.rom_override or profile.get("rom_current", 90) or 90,
                        "mood": max(0, 10 - (
                            (st.session_state.pain_override or 3) +
                            (st.session_state.fatigue_override or 4)
                        ) / 2),
                        "specific_complaints": [],
                        "feedback_on_last_task": {},
                        "swelling": False,
                    }

                    memory.update_state(profile["user_id"], user_state)

                    narrative, recommended, alerts = run_recommendation_pipeline(
                        config, profile, user_state, task_lib,
                        st.session_state.current_day, memory,
                        rec, safety, narrator, uu,
                    )

                    # Store alerts
                    st.session_state.safety_alerts.extend(alerts)

                    # Add narrative to chat
                    add_message("narrator", narrative)

                    if alerts:
                        alert_text = "<br>".join(alerts)
                        add_message("safety", alert_text)

                    st.session_state.pending_options = recommended
                    st.rerun()

        # ── Option selection ──
        if st.session_state.pending_options:
            st.markdown("### 🎯 Choose an Exercise")
            options = st.session_state.pending_options

            cols = st.columns(len(options))
            for i, (col, task) in enumerate(zip(cols, options)):
                with col:
                    cat = task.get("category", "general")
                    emoji = CATEGORY_EMOJI.get(cat, "📋")
                    intensity = task.get("intensity", "low")
                    int_emoji = INTENSITY_EMOJI.get(intensity, "⚪")
                    score = task.get("score", 0)

                    st.markdown(
                        f"""
                        <div style="background:#1e293b; padding:14px; border-radius:10px;
                                    border:1px solid #334155; text-align:center;">
                            <div style="font-size:2em;">{emoji}</div>
                            <div style="color:#e2e8f0; font-weight:bold; margin:8px 0;">
                                {task.get('name', task.get('task_id'))}
                            </div>
                            <div style="color:#94a3b8; font-size:0.85em;">
                                {int_emoji} {intensity.title()} · {cat}<br>
                                ⏱️ {task.get('duration_min', '?')} min · {task.get('reps', '?')} reps × {task.get('sets', '?')} sets<br>
                                Score: {score:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(f"✅ Select", key=f"select_{i}", use_container_width=True):
                        handle_task_selection(task, config, profile, memory, rec, uu)

            # Skip option
            st.markdown("")
            if st.button("⏭️ Skip Today's Exercises", use_container_width=True):
                handle_skip(options[0] if options else {}, config, profile, memory)

            # Free text input
            st.markdown("---")
            user_text = st.text_input(
                "💬 Or type a message (e.g., describe how you feel, report issues):",
                key="user_text_input",
                placeholder="e.g., 'My knee feels stiff today' or 'I had sharp pain yesterday'"
            )
            if user_text:
                handle_user_text(user_text, config, profile, safety)

        # ── Demo Mode auto-play ──
        if st.session_state.demo_mode and not st.session_state.demo_running:
            st.markdown("---")
            if st.button("▶️ Run Demo Simulation (5 days)", type="primary", use_container_width=True):
                run_demo_mode(config, profile, task_lib, memory, rec, safety, narrator, uu)

    # ── Right Column: Safety + Stats ──
    with right_col:
        st.markdown("## 🛡️ Safety Dashboard")

        alerts = st.session_state.safety_alerts
        if alerts:
            for alert in alerts[-6:]:
                st.markdown(
                    f'<div style="background:#7f1d1d; color:#fecaca; padding:8px; '
                    f'border-radius:8px; margin:4px 0; font-size:0.85em;">{alert}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="background:#064e3b; color:#a7f3d0; padding:10px; '
                'border-radius:8px;">✅ No safety alerts</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## 📊 Progress")

        completed = len(st.session_state.completed_tasks)
        skipped = len(st.session_state.skipped_tasks)
        total = completed + skipped
        comp_rate = completed / total if total > 0 else 0.0

        st.metric("📅 Current Day", st.session_state.current_day)
        st.metric("🏋️ Tasks Completed", completed)
        st.metric("⏭️ Tasks Skipped", skipped)

        # Progress bar
        st.markdown("**Completion Rate**")
        st.progress(comp_rate)
        st.markdown(f"<center><b>{comp_rate:.0%}</b></center>", unsafe_allow_html=True)

        if st.session_state.interaction_count > 0:
            avg_reward = st.session_state.total_reward / st.session_state.interaction_count
            st.metric("⭐ Avg Reward", f"{avg_reward:.3f}")

        # Category chart
        if st.session_state.completed_tasks:
            st.markdown("---")
            st.markdown("### 📊 Category Mix")
            cats = [t.get("category", "other") for t in st.session_state.completed_tasks]
            counter = Counter(cats)
            import pandas as pd
            df = pd.DataFrame({
                "Category": list(counter.keys()),
                "Count": list(counter.values()),
            })
            st.bar_chart(df.set_index("Category"))


# ═══════════════════════════════════════════════════════════════════════════
# Event Handlers
# ═══════════════════════════════════════════════════════════════════════════

def handle_task_selection(
    task: dict, config: dict, profile: dict,
    memory: UserMemory, rec: RecommenderAgent, uu: UserUnderstandingAgent,
) -> None:
    """Process a user selecting a specific task."""
    user_id = profile["user_id"]

    # Build user message
    task_name = task.get("name", task.get("task_id"))
    add_message("user", f"I'll do **{task_name}** today! 💪")

    # Simulate feedback
    pain = st.session_state.pain_override or profile.get("pain_baseline", 3)
    fatigue = st.session_state.fatigue_override or profile.get("fatigue_baseline", 4)
    rom = st.session_state.rom_override or profile.get("rom_current", 90) or 90

    user_state = {
        "pain_level": pain, "fatigue_level": fatigue,
        "rom": rom, "mood": 5, "day": st.session_state.current_day,
    }

    # Compute reward
    feat = uu.parse_structured(user_state, profile)
    intensity = task.get("intensity", "low")
    intensity_val = {"low": 1, "medium": 2, "high": 3}.get(intensity, 1)
    stress = (pain + fatigue) / 20.0
    ideal = 1 + (1 - stress) * 2
    diff_match = max(0, 1 - abs(intensity_val - ideal) / 2.0)
    reward = 0.4 * 1.0 + 0.3 * 1.0 + 0.15 * diff_match + 0.10 * 1.0 + 0.05 * 1.0

    # Update recommender
    rec.update(feat.to_numpy(), task["task_id"], reward)

    # Record in memory
    memory.record_interaction(
        user_id, task["task_id"], "completed", reward,
        user_state, st.session_state.current_day, user_state,
    )

    # Track stats
    st.session_state.completed_tasks.append(task)
    st.session_state.interaction_count += 1
    st.session_state.total_reward += reward

    if intensity == "high":
        st.session_state.consecutive_high += 1
    else:
        st.session_state.consecutive_high = 0

    # Generate feedback message
    feedback_text = (
        f"**Great work completing {task_name}!** 🎉\n\n"
        f"- Category: {CATEGORY_EMOJI.get(task.get('category', ''), '📋')} {task.get('category', 'N/A')}\n"
        f"- Intensity: {INTENSITY_EMOJI.get(intensity, '⚪')} {intensity}\n"
        f"- Duration: {task.get('duration_min', '?')} min\n"
        f"- Reward score: **{reward:.3f}**\n"
        f"- Difficulty match: {diff_match:.0%}\n\n"
        f"_Keep it up — consistency is key to recovery!_ 🌟"
    )
    add_message("feedback", feedback_text)

    st.session_state.pending_options = None
    st.rerun()


def handle_skip(task: dict, config: dict, profile: dict, memory: UserMemory) -> None:
    """Handle user skipping today's exercises."""
    user_id = profile["user_id"]

    add_message("user", "I'll skip today's exercises. ⏭️")

    user_state = {
        "pain_level": st.session_state.pain_override or 5,
        "fatigue_level": st.session_state.fatigue_override or 5,
        "rom": st.session_state.rom_override or 90,
        "day": st.session_state.current_day,
    }

    memory.record_interaction(
        user_id, task.get("task_id", "none"), "skipped", 0.0,
        user_state, st.session_state.current_day, user_state,
    )
    st.session_state.skipped_tasks.append(task)
    st.session_state.interaction_count += 1
    st.session_state.consecutive_high = 0

    add_message("feedback",
        "That's okay — rest is part of recovery too. "
        "We'll be here when you're ready! 🌿"
    )
    st.session_state.pending_options = None
    st.rerun()


def handle_user_text(
    text: str, config: dict, profile: dict, safety: SafetyAgent,
) -> None:
    """Handle free-text user input (crisis check + state parsing)."""
    add_message("user", text)

    # Crisis detection
    is_crisis, crisis_msg = safety.check_crisis(text)
    if is_crisis:
        add_message("safety", crisis_msg)
        st.session_state.safety_alerts.append(f"🚨 Crisis: {crisis_msg[:80]}...")
        st.session_state.pending_options = None
        st.rerun()
    else:
        add_message("narrator",
            "Thanks for sharing! I've noted your feedback. "
            "This will help me adjust your next recommendations. 📝"
        )
        st.rerun()


def run_demo_mode(
    config, profile, task_lib, memory, rec, safety, narrator, uu,
) -> None:
    """Auto-run a 5-day simulation for demo purposes."""
    st.session_state.demo_running = True

    from simulator import UserSimulator
    sim = UserSimulator(config)

    for day_offset in range(5):
        day = st.session_state.current_day + day_offset

        add_message("system", f"━━━ **Day {day} (Demo)** ━━━")

        # Simulate state
        user_state = sim.simulate_daily_state(profile["user_id"], day)
        user_state["day"] = day
        memory.update_state(profile["user_id"], user_state)

        # Run pipeline
        narrative, recommended, alerts = run_recommendation_pipeline(
            config, profile, user_state, task_lib, day,
            memory, rec, safety, narrator, uu,
        )

        st.session_state.safety_alerts.extend(alerts)
        add_message("narrator", narrative)
        if alerts:
            add_message("safety", "<br>".join(alerts))

        # Simulate choice
        if recommended:
            chosen_id, choice_meta = sim.simulate_choice(
                profile["user_id"], recommended, user_state
            )
            action = choice_meta.get("action", "completed")
            chosen_task = choice_meta.get("chosen_task", recommended[0])
            task_name = chosen_task.get("name", chosen_id)

            if action == "completed":
                add_message("user", f"I'll do **{task_name}**! 💪")
                feedback = sim.simulate_feedback(
                    profile["user_id"], chosen_task, user_state
                )
                reward = sim.compute_reward(feedback, user_state, chosen_task)
                feat = uu.parse_structured(user_state, profile)
                rec.update(feat.to_numpy(), chosen_id, reward)

                memory.record_interaction(
                    profile["user_id"], chosen_id, "completed", reward,
                    user_state, day, user_state,
                )
                st.session_state.completed_tasks.append(chosen_task)
                st.session_state.total_reward += reward

                add_message("feedback",
                    f"✅ Completed **{task_name}** — "
                    f"Reward: {reward:.3f} | "
                    f"Satisfaction: {feedback.get('satisfaction', 0):.0%}"
                )
            else:
                add_message("user", f"Skipping today... ⏭️")
                memory.record_interaction(
                    profile["user_id"], chosen_id, "skipped", 0.0,
                    user_state, day, user_state,
                )
                st.session_state.skipped_tasks.append(chosen_task)
                add_message("feedback", "Rest day — that's fine! 🌿")

            st.session_state.interaction_count += 1

    st.session_state.current_day += 5
    st.session_state.demo_running = False
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

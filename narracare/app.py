"""An offline visual journal. Run: streamlit run narracare/app.py.

Requires Streamlit >= 1.28 (install it separately if missing; requirements.txt
is intentionally unchanged). Navigation replays one session-cached trajectory;
it never advances or retrains the underlying pipeline.
"""

from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path
import sys
import time

import streamlit as st

# Streamlit runs this file as a script rather than as a package module.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narracare.demo import trajectory


st.set_page_config(page_title="NarraCare · A recovery story",
                   page_icon=str(Path(__file__).with_name("assets") / "favicon.svg"),
                   layout="wide")

CSS = """
<style>
:root { --ink:#203e35; --muted:#788278; --paper:#f5f3eb; --sage:#dce4d5; }
.stApp { background:var(--paper); color:var(--ink); }
html,body,[class*="css"],.stMarkdown,p { font-family:Arial, sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"],#MainMenu,footer { display:none; }
.block-container { max-width:1320px; padding:2rem 2.8rem 2rem; }
h1,h2,h3 { color:var(--ink); }
.masthead { display:flex; align-items:center; justify-content:space-between; padding-bottom:20px;
 border-bottom:1px solid #d8ded1; margin-bottom:26px; }
.brand { font-family:Georgia,serif; font-size:29px; letter-spacing:-1px; }
.brand i { display:inline-grid; place-items:center; width:36px; height:36px; background:#294d3d;
 color:#f5f3eb; border-radius:12px 12px 12px 3px; margin-right:10px; font-style:normal; font-size:24px; }
.edition,.eyebrow { font-size:10px; letter-spacing:2px; font-weight:700; color:#76826f; text-transform:uppercase; }
.edition span { background:#e4e9db; padding:8px 12px; border-radius:20px; letter-spacing:1px; margin-left:14px; }
.intro { display:flex; justify-content:space-between; align-items:flex-end; margin:6px 0 18px; }
.intro h1 { font-family:Georgia,serif; font-size:48px; font-weight:400; line-height:1.09; letter-spacing:-2px; margin:6px 0 10px; padding:0; }
.intro p { font-size:13px; color:#778274; margin:0; line-height:1.7; }
.chapter { text-align:right; min-width:175px; padding-bottom:5px; }
.chapter b { display:block; font-family:Georgia,serif; font-weight:400; font-size:35px; }
.chapter span { font-size:11px; color:#778274; }
.paper { background:#fffcf6; border:1px solid #e5e7dc; border-radius:24px; padding:26px; margin-bottom:16px; }
.narrator { display:flex; align-items:center; gap:11px; margin:18px 0 12px; }
.avatar { width:40px; height:40px; display:grid; place-items:center; background:#e1e9d9;
 border-radius:50%; font-family:Georgia,serif; font-size:24px; color:#4b6e49; }
.narrator strong { font-size:13px; font-weight:600; }
.narrator small { display:block; font-size:10px; color:#8a927f; margin-top:4px; }
blockquote { border:0; padding:0; margin:16px 0 22px; font-family:Georgia,serif;
 font-size:24px; line-height:1.46; letter-spacing:-.4px; color:#294437; min-height:128px; }
.task { background:#eef1e5; border-radius:16px; padding:18px; display:flex; gap:14px; align-items:center; }
.task-icon { font-size:27px; background:#fffcf6; width:49px; height:49px; border-radius:13px; display:grid; place-items:center; flex-shrink:0; }
.task h3 { font-family:Georgia,serif; font-size:20px; font-weight:400; padding:0; margin:4px 0; }
.task small { color:#7a856e; font-size:11px; }
.task .eyebrow { font-size:9px; }
.outcome { display:flex; justify-content:space-between; align-items:center; padding:17px 1px 0; font-size:12px; color:#71826d; }
.pill { display:inline-block; padding:5px 10px; border-radius:20px; background:#e5ebdc; color:#47673e; font-size:10px; }
.pill.skip { background:#f4e6d4; color:#9b7545; }
.closing { border-top:1px solid #e4e7da; margin-top:19px; padding-top:16px; }
.closing p { font-family:Georgia,serif; color:#76836d; font-size:14px; line-height:1.6; margin:7px 0 0; }
.world { position:relative; border:1px solid #dbe3d2; border-radius:24px; overflow:hidden;
 background:#e8eddf; padding:22px 22px 16px; margin-bottom:16px; }
.world img { width:100%; display:block; margin:0; }
.world-head { display:flex; justify-content:space-between; align-items:center; }
.world h2 { font-family:Georgia,serif; font-weight:400; font-size:25px; padding:0; margin:8px 0 0; }
.world-caption { text-align:center; font-family:Georgia,serif; color:#6b8062; font-size:14px; margin-top:-6px; }
.milestones { display:flex; justify-content:space-between; padding:17px 6px 2px; border-top:1px solid #cfd9c5; margin-top:15px; }
.milestones div { font-size:9px; color:#8e9984; }
.milestones b { display:block; color:#5c7452; font-size:11px; margin-bottom:4px; font-weight:500; }
.milestones .future b { color:#96a08f; }
.vitals { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }
.vital { background:#eeeee5; border:1px solid #e1e4d6; border-radius:16px; padding:15px; }
.vital label { font-size:10px; letter-spacing:1px; color:#7c8774; text-transform:uppercase; }
.vital strong { display:block; font-family:Georgia,serif; font-size:27px; font-weight:400; margin:7px 0; }
.vital strong small { font-family:Arial,sans-serif; font-size:11px; color:#8d9585; }
.meter { height:4px; background:#dedfce; border-radius:5px; overflow:hidden; }
.meter i { display:block; height:100%; border-radius:5px; background:#94ab80; transition:width .5s; }
.safety { display:flex; justify-content:space-between; align-items:center; padding:14px 17px;
 background:#e5ebda; border:1px solid #d9e2cd; border-radius:15px; margin:0 0 16px; }
.safety strong { font-size:12px; font-weight:500; }
.safety small { display:block; font-size:10px; color:#7d8b71; margin-top:4px; }
.safety .seal { border:1px solid #afc099; border-radius:50%; width:30px; height:30px; display:grid; place-items:center; color:#6c8b54; }
.progress-card { background:#fffcf6; border:1px solid #e5e7dc; border-radius:18px; padding:17px 20px; }
.progress-card .numbers { display:flex; justify-content:space-between; margin:12px 0 4px; }
.numbers b { font-family:Georgia,serif; font-size:24px; font-weight:400; }
.numbers small { display:block; font-size:9px; color:#8a9480; margin-top:4px; }
.sparkline { width:100%; height:55px; }
.nav-label { font-size:10px; letter-spacing:1.5px; color:#8a9480; text-transform:uppercase; margin:14px 0 0; }
[data-testid="stSlider"] { padding-top:0; }
[data-testid="stSlider"] p { font-size:11px; color:#76826f; }
.stButton button { border-radius:12px; border:1px solid #d7decb; background:#fffcf6; color:#4a6544; min-height:42px; font-size:12px; }
.stButton button:hover { border-color:#718c60; color:#294d3d; background:#e7eddd; }
.stButton button[kind="primary"] { background:#294d3d; border-color:#294d3d; color:#fffaf0; }
.stButton button:disabled { opacity:.35; }
[data-testid="stSelectbox"] label p { font-size:10px; color:#7b8672; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius:12px; border-color:#dbe1d1; background:#eeeee5; font-size:12px; }
.footer { display:flex; justify-content:space-between; border-top:1px solid #dbe0d1; padding-top:17px;
 margin-top:25px; color:#8c9682; font-size:10px; }
.pipeline { letter-spacing:.6px; font-size:9px; color:#8a9580; }
.finish { background:#e9e3cd; border:1px solid #d9cfad; border-radius:16px; padding:16px 20px;
 margin-bottom:16px; font-family:Georgia,serif; font-size:18px; color:#806a34; }
.paper,.world,.vitals { animation:arrive .45s ease-out; }
@keyframes arrive { from {opacity:.3;transform:translateY(5px)} to {opacity:1;transform:translateY(0)} }
@media(prefers-reduced-motion:reduce) { * {animation:none!important;transition:none!important;} }
@media(max-width:750px) { .block-container {padding:1.5rem 1rem;} .intro h1 {font-size:35px;} .edition {display:none;} .chapter {min-width:100px;} blockquote {font-size:21px;} .footer {display:block;} }
</style>
"""


def html(value: str) -> None:
    st.markdown(value, unsafe_allow_html=True)


def move(day: int) -> None:
    st.session_state.day_cursor = max(1, min(60, day))


def patient_changed() -> None:
    st.session_state.day_cursor = 1


def landscape(day: int) -> str:
    """An original, embedded SVG landscape; no image or font network requests."""
    light = "#edcf8b" if day == 60 else "#ead9a7"
    x = 66 + (day - 1) / 59 * 268
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 290">
    <defs><linearGradient id="sky" x2="0" y2="1"><stop stop-color="#dfe7d5"/><stop offset="1" stop-color="#f0edcf"/></linearGradient>
    <linearGradient id="hill" x2="0" y2="1"><stop stop-color="#b4c59d"/><stop offset="1" stop-color="#d8dfbd"/></linearGradient></defs>
    <circle cx="214" cy="137" r="121" fill="url(#sky)"/>
    <circle cx="258" cy="67" r="29" fill="{light}"/>
    <circle cx="258" cy="67" r="39" fill="none" stroke="{light}" opacity=".35"/>
    <path d="M73 183 149 67 216 176 271 106 353 191Z" fill="#c1ceb1"/>
    <path d="m149 67-22 34 24-11 17 12Z" fill="#edf0df"/>
    <path d="m271 106-20 28 21-8 15 9Z" fill="#e5ead6"/>
    <path d="M41 210Q131 135 224 191T388 192L390 265H36Z" fill="url(#hill)"/>
    <path d="M35 237Q136 171 230 217T390 217V273H35Z" fill="#93ad84"/>
    <path d="M45 265Q151 233 205 251T380 244L382 280H44Z" fill="#688d6a"/>
    <path d="M82 277Q206 254 197 233T262 210Q309 199 330 183" stroke="#e5dfb9" stroke-width="13" fill="none"/>
    <path d="M82 277Q206 254 197 233T262 210Q309 199 330 183" stroke="#f2edce" stroke-width="2" stroke-dasharray="2 10" fill="none"/>
    <g fill="#4f7559"><path d="m67 145-16 38h32Z"/><path d="m67 163-21 38h42Z"/><rect x="65" y="193" width="4" height="25"/>
    <path d="m349 150-13 30h26Z"/><path d="m349 167-18 32h36Z"/><rect x="347" y="194" width="4" height="20"/></g>
    <g fill="#dfd5a6"><circle cx="128" cy="237" r="3"/><circle cx="302" cy="243" r="3"/><circle cx="287" cy="257" r="2"/></g>
    <path d="m184 66 7-3 7 3m7-8 5-2 5 2" stroke="#8da281" fill="none" stroke-width="1.5"/>
    <circle cx="{x:.1f}" cy="279" r="6" fill="#f6efd4" stroke="#446a4c" stroke-width="2"/>
    <path d="M60 279H344" stroke="#446a4c" opacity=".2"/>
    <circle cx="330" cy="183" r="5" fill="#f3e5af" stroke="#7f955c" stroke-width="2"/>
    </svg>'''


def sparkline(values: list[float]) -> str:
    maximum = max(max(values), 1)
    points = " ".join(f"{i * 360 / 59:.1f},{49 - v / maximum * 42:.1f}"
                      for i, v in enumerate(values))
    return (f'<svg class="sparkline" viewBox="0 0 360 55" role="img" aria-label="Cumulative reward up to this day">'
            '<path d="M0 49H360" stroke="#e4e7da"/>'
            f'<polyline points="{points}" fill="none" stroke="#7a9970" stroke-width="2.5" stroke-linecap="round"/></svg>')


def main() -> None:
    html(CSS)
    html('<div class="masthead"><div class="brand"><i>✳</i>NarraCare</div>'
         '<div class="edition">THE RECOVERY JOURNAL <span>● OFFLINE & PERSONAL</span></div></div>')
    profiles = json.loads((ROOT / "data/user_profiles.json").read_text(encoding="utf-8"))
    by_id = {p["user_id"]: p for p in profiles}
    st.session_state.setdefault("day_cursor", 1)
    st.session_state.setdefault("journeys", {})
    title, selector = st.columns([3, 1])
    with title:
        html('<div class="eyebrow">60 DAYS · ONE SMALL STEP AT A TIME</div>')
    with selector:
        user = st.selectbox("Whose story are we following?", list(by_id), key="patient",
                            format_func=lambda uid: f"{by_id[uid].get('name', uid)} · {uid}",
                            on_change=patient_changed)
    if user not in st.session_state.journeys:
        with st.spinner("Preparing your 60-day story…"):
            started = time.perf_counter()
            result = trajectory(user, 60, 'saps', 42)
            st.session_state.journeys[user] = result
            st.session_state.setdefault("load_seconds", {})[user] = time.perf_counter() - started
    data = st.session_state.journeys[user]
    day = st.session_state.day_cursor
    row = data["trace"][day - 1]
    profile = by_id[user]
    state = row["state"]
    name = escape(profile.get("name", user).split()[0])
    title_text = "Look how far you've come." if day == 60 else "A little stronger.<br>A little more you."
    html(f'<div class="intro"><div><h1>{title_text}</h1>'
         f'<p>{name}’s recovery story. Every small step belongs here.</p></div>'
         f'<div class="chapter"><div class="eyebrow">YOUR JOURNEY</div><b>Day {day:02d} <span>/ 60</span></b>'
         '<span>A simulated story, one day at a time</span></div></div>')
    if day == 60:
        html('<div class="finish">✧ &nbsp; Sixty days. A story worth celebrating. Here’s to everything that comes next.</div>')
        if not st.session_state.get("celebrated"):
            st.balloons()
            st.session_state.celebrated = True
    left, right = st.columns([1.22, 1], gap="large")
    tone = row["narrator_result"].get("tone_actually_used", "N/A")
    role = {"doctor": "A steady hand", "coach": "In your corner", "companion": "Walking with you"}.get(tone, "By your side")
    task = row["ranked"][0] if row["ranked"] else {}
    completed = row["action"] == "accept"
    with left:
        closing, closing_meta = row["closing"]
        html(f'<section class="paper"><div class="eyebrow">A NOTE FOR TODAY</div>'
             f'<div class="narrator"><div class="avatar">✧</div><div><strong>{role}</strong>'
             f'<small>{escape(tone)} · template narrator</small></div></div>'
             f'<blockquote>“{escape(row["narrative"])}”</blockquote>'
             '<div class="task"><div class="task-icon">↟</div><div><div class="eyebrow">TODAY’S FIRST RECOMMENDATION</div>'
             f'<h3>{escape(task.get("name", "No recommendation"))}</h3>'
             f'<small>{escape(str(task.get("category", "N/A")))} &nbsp; · &nbsp; difficulty {escape(str(task.get("difficulty", "N/A")))}/5'
             f' &nbsp; · &nbsp; {escape(str(task.get("duration_min", "N/A")))} min</small></div></div>'
             f'<div class="outcome"><span class="pill {"" if completed else "skip"}">'
             f'{"✓ Accepted" if completed else "↪ A pause today"}</span>'
             f'<span>Daily reward &nbsp; <b>{row["reward"]:.3f}</b></span></div>'
             '<div class="closing"><div class="eyebrow">BEFORE YOU GO</div>'
             f'<p>“{escape(closing)}”</p></div></section>')
        vitals = []
        for key, label in [("pain_level", "Pain"), ("fatigue_level", "Fatigue"), ("mood", "Mood")]:
            value = state.get(key)
            text = f"{value:.1f}" if isinstance(value, (int, float)) else "N/A"
            width = max(0, min(100, value * 10)) if isinstance(value, (int, float)) else 0
            vitals.append(f'<div class="vital"><label>{label}</label><strong>{text} <small>/ 10</small></strong>'
                          f'<div class="meter"><i style="width:{width}%"></i></div></div>')
        html('<div class="vitals">' + "".join(vitals) + '</div>')
    with right:
        encoded = base64.b64encode(landscape(day).encode()).decode()
        arc_names = {"act1": "Protection & trust", "act2": "Plateau & negotiation", "act3": "Autonomy & farewell"}
        html('<section class="world"><div class="world-head"><div class="eyebrow">YOUR RECOVERY WORLD</div>'
             f'<span class="pill">{escape(row["act"].upper())}</span></div>'
             f'<h2>{arc_names.get(row["act"], escape(row["act"]))}</h2>'
             f'<img alt="A quiet mountain landscape with a winding recovery path" src="data:image/svg+xml;base64,{encoded}">'
             '<div class="world-caption">Progress isn’t a straight line. Neither is this path.</div>'
             '<div class="milestones"><div><b>01 · Protect</b>Find your footing</div>'
             f'<div class="{"future" if day < 16 else ""}"><b>02 · Persist</b>Build your rhythm</div>'
             f'<div class="{"future" if day < 41 else ""}"><b>03 · Flourish</b>Make it your own</div></div></section>')
        before, after = row["pre_filter"]
        removed = row["post_check_removed"]
        html('<div class="safety"><div>'
             f'<strong>{"Safety gate adjusted the shortlist" if removed else "Safety gate passed"}</strong>'
             f'<small>{after} options passed pre-filter · {before-after} filtered before ranking</small></div>'
             '<div class="seal">✓</div></div>')
        visited = data["trace"][:day]
        values, total = [], 0.0
        for item in visited:
            total += item["reward"]
            values.append(total)
        accepted = sum(item["action"] == "accept" for item in visited)
        html('<section class="progress-card"><div class="eyebrow">THE LITTLE THINGS ADD UP</div>'
             f'<div class="numbers"><div><b>{total:.2f}</b><small>CUMULATIVE REWARD</small></div>'
             f'<div><b>{accepted}<small>ACCEPTED DAYS / {day}</small></b></div>'
             f'<div><b>{day/60:.0%}</b><small>STORY EXPLORED</small></div></div>'
             + sparkline(values) + '</section>')
    html('<div class="nav-label">TURN THE PAGE &nbsp; / &nbsp; your story is already here</div>')
    st.slider("Journey day", 1, 60, key="day_cursor", label_visibility="collapsed")
    previous, today, mid, finale, space, following = st.columns([1, .85, .85, .85, 1, 1.6])
    previous.button("← Previous", on_click=move, args=(day - 1,), disabled=day == 1, use_container_width=True)
    today.button("Day 1", on_click=move, args=(1,), use_container_width=True)
    mid.button("Day 15", on_click=move, args=(15,), use_container_width=True)
    finale.button("Day 60", on_click=move, args=(60,), use_container_width=True)
    following.button("Next day →" if day < 60 else "Read it again ↺", type="primary",
                     on_click=move, args=(day + 1 if day < 60 else 1,), use_container_width=True)
    with st.expander("Behind this moment · the integrated system"):
        st.caption("User Simulator → Safety pre-filter → SAPS → Safety post-check → template Narrator → user feedback → Reward Composer → daily closing")
        st.write({"user_id": user, "pipeline_day": row["day"], "act": row["act"],
                  "top_1": task.get("task_id"), "chosen_task_id": row["chosen_task_id"],
                  "action": row["action"], "reward": row["reward"], "ROM": state.get("rom", "N/A"),
                  "closing_variant": closing_meta.get("daily_closing_variant"),
                  "trajectory_load_seconds": round(st.session_state.load_seconds[user], 3)})
        st.caption("The simulated user may choose another task from the shortlist. Acceptance refers to that recorded choice. All 60 days are generated once per patient per session; navigation only changes the cursor.")
        st.caption("The frozen pipeline begins at index 0; arc ranges begin at 1, so the first recorded act falls back to act3. The displayed act is preserved from the model.")
    html('<div class="footer"><span>NarraCare &nbsp; · &nbsp; A recovery story, not just a recommendation.</span>'
         '<span class="pipeline">SAPS · SEED 42 · TEMPLATE NARRATOR · FULLY OFFLINE</span></div>')


if __name__ == "__main__":
    main()

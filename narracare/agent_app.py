"""Gradio ChatInterface and a live state panel for Pip's companion trail."""

from __future__ import annotations

import argparse
import html
import math
import os

from .conversation.controller import ConversationAgent
from .conversation.llm import OpenAIPlanner, ProviderError

CSS = """
body,.gradio-container{background:#0d1821!important;color:#e4ede8!important;font-family:Arial,sans-serif!important}
.gradio-container{max-width:1500px!important;margin:auto!important}
#brand{padding:22px 10px 6px}#brand h1{font-size:34px;color:#dcece4;letter-spacing:-1px;margin-bottom:5px}
#brand p{color:#9bb5b0;margin:0}#chat{border-radius:22px;background:#172631;border:1px solid #30434b}
#chat .message{font-size:15px;line-height:1.65}.scene{background:linear-gradient(130deg,#183332,#192635);border:1px solid #354b4e;border-radius:22px;padding:24px;margin:12px 0;color:#dfebe6}
.scene h2{color:#b6dfbe;font-weight:500;font-size:27px;margin:3px 0 10px}.scene small{color:#91aaa7;letter-spacing:1.3px}.scene .trail{display:flex;gap:4px;flex-wrap:wrap;margin:14px 0}.node{width:10px;height:10px;border-radius:50%;background:#364954}.node.walked{background:#aad785}.node.now{outline:2px solid #dcd5a5;outline-offset:2px}
.scene .scene-top{display:flex;justify-content:space-between;align-items:center}.companions{font-size:54px;letter-spacing:7px;white-space:nowrap}.scene p{margin:8px 0;color:#b7cbc7}.scene .task{border-top:1px solid #46615b;padding-top:12px;margin-top:16px;font-size:14px}
.state-panel{background:#172631;border:1px solid #32464c;border-radius:20px;padding:22px;color:#dbe9e2;margin-top:12px}.state-panel h3{font-weight:500;font-size:21px;color:#cce3ca;margin:0 0 12px}.state-panel p{color:#aabfba;font-size:13px;line-height:1.7}.vital{margin:15px 0}.vital-label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:7px}.vital progress{width:100%;height:8px;accent-color:#b3d58f}.stat-pair{display:flex;justify-content:space-between;border-bottom:1px solid #30434b;padding:8px 0;font-size:13px}.call{margin:8px 0;font-size:12px}.tag{color:#c5dca6}.note{color:#91a6a1;font-size:12px;line-height:1.7}.connection{padding:10px 15px;border-radius:12px;background:#24352e;color:#c9deb3;font-size:12px}.legend{font-size:12px;color:#9bb5b0;display:flex;justify-content:space-between}
.scene svg[viewBox="0 0 640 170"]{width:100%;height:110px;display:block}
.scene .task,.scene .task b,.state-panel .vital-label,.state-panel .vital-label span,.state-panel .stat-pair,.state-panel .stat-pair span,.state-panel .call{color:#dbe8e1!important}
.gradio-container .prose p,.gradio-container .prose strong{color:#b7cbc4!important}
.state-panel progress{appearance:none;border:0;border-radius:8px;overflow:hidden;background:#344950}
.state-panel progress::-webkit-progress-bar{background:#344950;border-radius:8px}.state-panel progress::-webkit-progress-value{background:#b0d392;border-radius:8px}
#chat .prose p,#chat .prose strong,#chat .prose code{color:#2b4048!important}
footer{display:none!important}button{border-radius:12px!important}#controls{padding:4px 0 14px}
"""


def escape(value):
    return html.escape(str(value), quote=True)


def trail_art(steps, act, camp=False):
    # Presentation only. The node count uses observed completed tasks, never a clinical estimate.
    colors = {
        "act1": ("#254c48", "#365858"),
        "act2": ("#725b4b", "#957357"),
        "act3": ("#526f82", "#aec7ce"),
    }
    back, front = colors[act]
    points = [(20 + i * 10.1, 126 - i * 1.25 + math.sin(i / 5) * 14) for i in range(60)]
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{3 if i != min(steps, 59) else 5}" fill="{("#b6dc91" if i < steps else "#e2d39d" if i == min(steps, 59) else "#4e666c")}"/>'
        for i, (x, y) in enumerate(points)
    )
    x, y = points[min(steps, 59)]
    position = f'<text x="{max(45, min(595, x)):.1f}" y="{y - 13:.1f}" fill="#d9e9d4" font-size="12" text-anchor="middle">{"CAMP" if camp else "YOU + PIP"}</text>'
    return f'<svg viewBox="0 0 640 170" role="img" aria-label="Sixty-node companion trail, {steps} observed completed steps"><path d="M0 150 70 25 160 120 245 5 340 115 440 20 520 98 610 0 640 70V170H0Z" fill="{back}"/><path d="M0 165 110 75 220 150 330 55 440 150 560 40 640 95V170H0Z" fill="{front}" opacity=".48"/>{circles}{position}</svg>'


def companion_art(camp=False):
    if camp:
        return '<svg width="130" height="100" viewBox="0 0 130 100" aria-label="A shared camp"><path d="m10 85 40-65 40 65Z" fill="#d8b990"/><path d="m50 20-8 65h48Z" fill="#9b7d5b"/><path d="m50 56-9 29h22Z" fill="#1b2e32"/><path d="M106 87q-14-15 0-33 0 11 7 8 8 16-7 25Z" fill="#edbb75"/></svg>'
    return '<svg width="135" height="115" viewBox="0 0 135 115" aria-label="Patient and Pip walking together"><ellipse cx="66" cy="108" rx="57" ry="5" fill="#0d1c24"/><path d="m29 81-6 25m22-25 6 25" stroke="#acc4bd" stroke-width="8" stroke-linecap="round"/><path d="M15 83V59q19-22 40 0v24Z" fill="#bd946e"/><circle cx="35" cy="41" r="16" fill="#dfb89a"/><path d="M19 39q-2-30 31-12l3 15-15-14Z" fill="#38515a"/><path d="m86 81-4 25m22-25 6 25" stroke="#a1d6b5" stroke-width="8" stroke-linecap="round"/><path d="M72 88V44Q88-12 111 22q16 18 7 66Z" fill="#8cc8a6"/><ellipse cx="95" cy="45" rx="18" ry="19" fill="#d7ebd8"/><path d="m84 44 5-3 4 3m5 0 5-3 4 3m-16 10q5 5 10 0" fill="none" stroke="#224440" stroke-width="2.5" stroke-linecap="round"/><path d="m81 69 29 5" stroke="#e5ce8e" stroke-width="9"/></svg>'


def reward_art(rows):
    values = []
    total = 0
    for row in rows:
        if row["reward"] is not None:
            total += row["reward"]
        values.append(total)
    if not values:
        return '<p class="note">No reported outcomes yet. The curve will use actual Reward Composer outputs.</p>'
    top = max(max(values), 1)
    pts = " ".join(
        f"{10 + i * 260 / max(1, len(values) - 1):.1f},{80 - v / top * 65:.1f}"
        for i, v in enumerate(values)
    )
    return f'<svg viewBox="0 0 280 95" aria-label="Cumulative reported reward"><path d="M10 82H275" stroke="#3e5759"/><polyline points="{pts}" fill="none" stroke="#b6dc91" stroke-width="3"/></svg>'


def render_panels(agent=None):
    s = agent.tools.snapshot() if agent else None
    if s is None:
        return (
            f'<div class="scene"><small>A LIVE CONVERSATIONAL JOURNEY</small><div class="scene-top"><h2>Misty Forest</h2><div class="companions">{companion_art()}</div></div><p>Meet Pip, your trail companion. A small step, or a quiet camp. Your check-in decides what comes next.</p>{trail_art(0, "act1")}<p class="note">60 narrative days. No recorded simulation is replayed here.</p></div>',
            '<div class="state-panel"><h3>Your check-in</h3><p>Pain, fatigue, mood and ROM start unknown. Pip will ask before recommending.</p><p>Choose a synthetic profile, then say hello or describe how you feel.</p></div>',
            {
                "connection": "Not contacted. The first message requires a configured LLM."
            },
        )
    names = {
        "act1": ("Misty Forest", "Pip scouts the path ahead"),
        "act2": ("Rocky Canyon", "Pip walks beside you"),
        "act3": ("Snowy Summit", "Pip makes room for your choices"),
    }
    place, formation = names[s["act"]]
    current = s["state"] or s["last_reported_state"]
    reaction = "A manageable step is enough."
    if s["on_hold"]:
        reaction = "Safety hold. The journey pauses for clinical support."
    elif s["reward_history"] and s["reward_history"][-1]["action"] == "skipped":
        reaction = "Camp together. Rest belongs on this journey."
    elif current.get("fatigue_level", 0) >= 8:
        reaction = "Pip signals a pause and stays close."
    elif current.get("pain_level", 0) >= 7:
        reaction = "Pip moves closer. Tell us how you feel before another step."
    elif current.get("mood", 0) >= 8:
        reaction = "Pip hums a little trail song."
    task = s["pending_task"]
    task_html = ""
    if task:
        task_html = f'<div class="task">Recommended: <b>{escape(task["name"])}</b> ({escape(task["task_id"])})<br>{escape(task["category"])} · Difficulty {task["difficulty"]}/5 · SAPS S = {task["score"]:.3f}</div>'
    c = s["counts"]
    safety_line = (
        f"Safety: {c['catalogue']} → {c['safety_passed']} candidates. {c['safety_blocked']} blocked by runtime predicates. Retrieval: {c['retrieved']} relevant/preferred items."
        if c
        else "Safety waits for a complete check-in."
    )
    if not task and c:
        safety_line = "Last computed plan: " + safety_line
    scene = f"""<div class="scene"><small>DAY {s["day"]} / 60 · {escape(s["act"].upper())} · {s["streak"]}-DAY COMPLETION STREAK</small>
    <div class="scene-top"><div><h2>{place}</h2><p>{formation}</p></div><div class="companions">{companion_art(bool(s["reward_history"] and s["reward_history"][-1]["action"] == "skipped"))}</div></div>
    <p>{escape(reaction)}</p>{trail_art(s["trail_steps"], s["act"], bool(s["reward_history"] and s["reward_history"][-1]["action"] == "skipped"))}<div class="legend"><span>Forest 1–15</span><span>Canyon 16–40</span><span>Summit 41–60</span></div>
    <p class="note">{s["trail_steps"]} completed trail steps. Skips close a day without moving the trail. Narrative day never promotes clinical stage.</p>
    {task_html}<p>🛡 {escape(safety_line)}</p></div>"""
    vitals = ""
    for key, name, maximum in [
        ("pain_level", "Pain", 10),
        ("fatigue_level", "Fatigue", 10),
        ("mood", "Mood", 10),
        ("rom", "ROM", 180),
    ]:
        value = current.get(key)
        label = "Unknown" if value is None else f"{value:g} / {maximum}"
        vitals += f'<div class="vital"><div class="vital-label"><span>{name}</span><span>{label}</span></div><progress max="{maximum}" value="{value or 0}"></progress></div>'
    rows = [
        ("Profile", s["name"]),
        ("Condition", s["condition"]),
        ("Clinical stage", s["stage"]),
        ("Narrator tone", s["tone"]),
        ("Preferred", ", ".join(s["preferred_categories"]) or "Any"),
        ("Avoid", ", ".join(s["avoid_categories"]) or "None"),
        ("Reward total", f"{s['cumulative_reward']:.4f}"),
        ("Days recorded", s["days_recorded"]),
    ]
    stats = "".join(
        f'<div class="stat-pair"><span>{escape(k)}</span><span>{escape(v)}</span></div>'
        for k, v in rows
    )
    calls = "".join(
        f'<div class="call">{"✓" if e["status"] == "ok" else "·"} <span class="tag">{escape(e["tool"])}</span> {escape(e["status"])}</div>'
        for e in s["events"][-8:]
    )
    freshness = (
        "Current check-in"
        if s["state"]
        else "Last report. A fresh check-in is required."
    )
    collectibles = "First step ✓" if s["trail_steps"] else "First step pending"
    if any(r["action"] == "skipped" for r in s["reward_history"]):
        collectibles += " · Rest counts ✓"
    if s["streak"] >= 3:
        collectibles += " · Three-day rhythm ✓"
    panel = f'<div class="state-panel"><h3>Live patient state</h3><p>{freshness}</p>{vitals}{stats}<h3 style="margin-top:22px">Reward trail</h3>{reward_art(s["reward_history"])}<p>{collectibles}</p><h3 style="margin-top:22px">Recent system activity</h3>{calls or "Awaiting your first message"}<p class="note">Numbers come from your reports and tool outputs. Unknown is never a simulated measurement. Reward is a benchmark proxy, not recovery.</p></div>'
    return (
        scene,
        panel,
        {"tool_activity": s["events"], "observed_days": s["reward_history"]},
    )


def build_demo(planner_factory=OpenAIPlanner.from_env):
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit(
            "Install optional UI dependencies: python -m pip install -r requirements-agent.txt"
        ) from None
    import json

    from .conversation.tools import ROOT

    users = json.loads((ROOT / "data/user_profiles.json").read_text())
    scene0, panel0, activity0 = render_panels()

    def respond(message, history, session, user, start_day):
        if session is None:
            session = {"agent": None, "generation": 0}
        generation = session["generation"]
        if session["agent"] is None:
            try:
                session["agent"] = ConversationAgent(
                    planner_factory(), user, 42, int(start_day)
                )
            except (ProviderError, ValueError) as exc:
                return (
                    str(exc),
                    session,
                    scene0,
                    panel0,
                    {"connection": "LLM unavailable; no offline fallback"},
                )
        agent = session["agent"]
        answer = agent.reply(message)
        if generation != session["generation"]:
            scene, panel, activity = render_panels(session["agent"])
            return (
                "The previous journey was reset. Please send your check-in for the current journey.",
                session,
                scene,
                panel,
                activity,
            )
        scene, panel, activity = render_panels(agent)
        return answer, session, scene, panel, activity

    theme = gr.themes.Base(
        font=["Arial", "sans-serif"],
        font_mono=["monospace"],
        primary_hue="emerald",
        neutral_hue="slate",
    )
    with gr.Blocks(
        title="NarraCare · Pip's Live Trail",
        css=CSS,
        theme=theme,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            '<div id="brand"><h1>NarraCare <span style="color:#9bd7bb">Walk with Pip</span></h1><p>A gentle rehabilitation adventure, guided by your check-in.</p></div>'
        )
        gr.HTML(
            '<div class="connection">LIVE LLM TOOL CALLING · SAPS · HARD SAFETY · TEMPLATE NARRATOR</div>'
        )
        gr.Markdown(
            "**Coursework prototype using synthetic profiles. Not a clinical service.** Your messages and synthetic session context go to the configured LLM provider. Do not enter identifiable patient information. No remote sharing by default."
        )
        with gr.Row(elem_id="controls"):
            user = gr.Dropdown(
                [(f"{u['user_id']} · {u['name']}", u["user_id"]) for u in users],
                value="U001",
                label="Synthetic journey",
            )
            start_day = gr.Number(
                value=1,
                minimum=1,
                maximum=60,
                precision=0,
                label="Start narrative day (clinical stage stays fixed)",
            )
            reset = gr.Button("Start a fresh journey", variant="secondary")
        session = gr.State({"agent": None, "generation": 0}, time_to_live=3600)
        panel = gr.HTML(panel0, render=False)
        activity = gr.JSON(activity0, render=False)
        with gr.Row():
            with gr.Column(scale=3):
                scene = gr.HTML(scene0)
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Pip, your trail companion",
                    height=360,
                    elem_id="chat",
                    sanitize_html=True,
                )
                message = gr.Textbox(
                    placeholder="My pain is 3, fatigue 2, mood 7 and ROM 110. I prefer flexibility.",
                    label="Your check-in or question",
                    submit_btn="Send",
                    stop_btn=True,
                    max_length=4000,
                )
                gr.ChatInterface(
                    fn=respond,
                    chatbot=chatbot,
                    textbox=message,
                    type="messages",
                    additional_inputs=[session, user, start_day],
                    additional_outputs=[session, scene, panel, activity],
                    cache_examples=False,
                )
                with gr.Row():
                    for label, text in [
                        (
                            "I'll try",
                            "I'll try the proposed task. What feedback do you need?",
                        ),
                        (
                            "A bit sore",
                            "I feel sore. Please ask my current pain before another recommendation.",
                        ),
                        ("Too tired", "I am too tired; I want to rest today."),
                        ("Skip today", "Skip today; let's camp."),
                    ]:
                        gr.Button(label, size="sm").click(
                            lambda value=text: value,
                            outputs=message,
                            queue=False,
                            api_name=False,
                        )
                gr.Examples(
                    examples=[
                        "Hello Pip, can you help me plan a manageable session?",
                        "Pain 3, fatigue 2, mood 7, ROM 110. I prefer flexibility.",
                        "Why did you choose that exercise?",
                        "I would like to skip and rest today.",
                    ],
                    inputs=message,
                    cache_examples=False,
                )

            with gr.Column(scale=1, min_width=320):
                panel.render()
                with gr.Accordion("Observed tool calls and day records", open=False):
                    activity.render()

        def fresh(session):
            if session is None:
                session = {"agent": None, "generation": 0}
            if session["agent"] is not None:
                session["agent"].tools.ended = True
            session["generation"] += 1
            session["agent"] = None
            return session, [], scene0, panel0, activity0

        for trigger in [reset.click, user.change, start_day.change]:
            trigger(
                fresh,
                inputs=session,
                outputs=[session, chatbot, scene, panel, activity],
                queue=False,
            )
        gr.Markdown(
            "**How to play:** check in, read the recommendation and its explanation, then report actual completion or choose camp. Accepting a task does not fabricate its outcome. All five runtime predicates remain authoritative. BP profiles represent chronic back pain, not back surgery."
        )
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NarraCare live Gradio companion agent"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    demo = build_demo()
    demo.queue(default_concurrency_limit=8).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, show_error=False
    )


if __name__ == "__main__":
    main()

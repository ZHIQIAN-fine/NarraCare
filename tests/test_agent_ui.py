"""Real Gradio HTTP/session regression using an explicitly injected planner."""

import importlib.util
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None,
    reason="optional Gradio dependencies are not installed",
)


def test_chatinterface_live_state_and_session_isolation():
    from gradio_client import Client
    from test_conversation import CHECKIN, ScriptedPlanner, plan_actions

    from narracare.agent_app import build_demo

    feedback = "Completed. After pain 3, fatigue 3, satisfaction 0.8."

    def factory():
        return ScriptedPlanner(
            *plan_actions(),
            (
                "record_feedback",
                {
                    "action": "completed",
                    "post_pain": 3,
                    "post_fatigue": 3,
                    "satisfaction": 0.8,
                    "evidence": feedback,
                },
            ),
            ("rest_today", {"evidence": "Skip today and rest"}),
        )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = build_demo(factory)
    try:
        app.queue().launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
        )
        a, b = (
            Client(f"http://127.0.0.1:{port}", verbose=False),
            Client(f"http://127.0.0.1:{port}", verbose=False),
        )
        r = a.predict(CHECKIN, "U001", 1, api_name="/chat")
        assert "SAPS ranked" in r[0] and "IT Band" in r[0]
        assert "Live patient state" in r[2] and "77" in r[1]
        r = a.predict(feedback, "U001", 1, api_name="/chat")
        assert "Day 1 recorded" in r[0] and "0.8622" in r[2]
        r = a.predict("Skip today and rest", "U001", 1, api_name="/chat")
        assert "Day 2: camp together" in r[0]
        assert r[3]["observed_days"][-1]["reward"] is None
        other = b.predict(CHECKIN, "U001", 1, api_name="/chat")
        assert "DAY 1 / 60" in other[1] and other[3]["observed_days"] == []
        reset = a.predict(api_name="/fresh")
        assert reset[0] == [] and "start unknown" in reset[2]
    finally:
        app.close()


def test_panel_escapes_data_and_marks_unknown_values():
    from test_conversation import ScriptedPlanner

    from narracare.agent_app import render_panels
    from narracare.conversation.controller import ConversationAgent

    a = ConversationAgent(ScriptedPlanner())
    a.tools.profile["name"] = "<script>alert(1)</script>"
    _, panel, _ = render_panels(a)
    assert "<script>" not in panel and "&lt;script&gt;" in panel
    assert "Unknown" in panel

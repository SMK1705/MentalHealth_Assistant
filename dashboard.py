import streamlit as st
import ui


def _assistant_analyses(conversation):
    """Per-message analytics dicts from any analyzed turn of the conversation."""
    return [m["analysis"] for m in (conversation or []) if m.get("analysis")]


def trajectory(conversation):
    """Build the sentiment sparkline (points string + signed delta) from the
    per-message sentiment scores. Mirrors the cockpit metric geometry."""
    analyses = _assistant_analyses(conversation)
    scores = [a.get("sentiment_score") for a in analyses
              if isinstance(a.get("sentiment_score"), (int, float))]
    if len(scores) < 2:
        return "", "—", "#8a93a8", False
    n = len(scores)
    pts = []
    for i, v in enumerate(scores):
        x = (i / (n - 1)) * 240
        y = max(4.0, min(56.0, 30 - (v * 26)))
        pts.append(f"{x:.1f},{y:.1f}")
    delta = scores[-1] - scores[0]
    if delta > 0.02:
        txt, color = f"▲ +{delta:.2f}", "#2f8f6b"
    elif delta < -0.02:
        txt, color = f"▼ {delta:.2f}", "#b23a31"
    else:
        txt, color = "→ flat", "#8a93a8"
    return " ".join(pts), txt, color, True


def render_dashboard(conversation, patient_profile=None):
    """Right-column session metrics: current emotion, topic, sentiment trajectory.

    Reads only data already produced per message — no model calls here.
    """
    analyses = _assistant_analyses(conversation)
    emotion, topic = "—", "—"
    if analyses:
        latest = analyses[-1]
        urgency = latest.get("urgency") or {}
        if urgency.get("label"):
            emotion = str(urgency["label"]).title()
        topic = latest.get("topic") or "—"

    points, delta_text, delta_color, has = trajectory(conversation)
    st.markdown(
        ui.metrics_panel(emotion, topic, points, delta_text, delta_color, has),
        unsafe_allow_html=True,
    )

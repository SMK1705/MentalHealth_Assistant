import streamlit as st


def _assistant_analyses(conversation):
    """Per-message analytics dicts from any analyzed turn of the conversation."""
    return [m["analysis"] for m in (conversation or []) if m.get("analysis")]


def render_dashboard(conversation, patient_profile):
    """Render the live counselor dashboard from the conversation's analytics.

    Reads only data already produced per message (sentiment, urgency/emotion,
    topic, safety) — no model calls here.
    """
    st.subheader("Live session dashboard")
    analyses = _assistant_analyses(conversation)

    if not analyses:
        st.caption("Emotional and risk metrics appear here as the conversation progresses.")
        _render_profile(patient_profile)
        return

    latest = analyses[-1]
    flagged = [a.get("safety_protocol") for a in analyses if a.get("safety_protocol")]

    # Current risk status
    sp = latest.get("safety_protocol")
    if sp:
        label = f"Risk: {sp.get('action', 'URGENT')} — {sp.get('flag_type')}"
        (st.error if sp.get("action") == "CRITICAL" else st.warning)(label)
    elif flagged:
        st.warning(f"Risk: {len(flagged)} flag(s) earlier this session")
    else:
        st.success("Risk: none detected")

    # Metric cards
    c1, c2, c3 = st.columns(3)
    score = latest.get("sentiment_score")
    c1.metric("Sentiment", latest.get("sentiment") or "—",
              f"{score:+.2f}" if isinstance(score, (int, float)) else None)
    urgency = latest.get("urgency") or {}
    emotion = urgency.get("label")
    c2.metric("Emotion", emotion.title() if emotion else "—")
    c3.metric("Topic", latest.get("topic") or "—")

    bits = []
    if isinstance(urgency.get("score"), (int, float)):
        bits.append(f"emotion {urgency['score']:.0%}")
    if isinstance(latest.get("topic_confidence"), (int, float)):
        bits.append(f"topic {latest['topic_confidence']:.0%}")
    if bits:
        st.caption(" · ".join(bits) + " confidence")

    # Emotional trajectory
    scores = [a.get("sentiment_score") for a in analyses if isinstance(a.get("sentiment_score"), (int, float))]
    if len(scores) >= 2:
        st.caption("Emotional trajectory · sentiment per message")
        st.line_chart(scores, height=160)

    # Safety timeline
    if flagged:
        st.caption("Safety timeline")
        for i, a in enumerate(analyses, start=1):
            f = a.get("safety_protocol")
            if f:
                st.markdown(f"- Message {i}: **{f.get('action')}** — {f.get('flag_type')}")

    # Topics detected this session
    topics = []
    for a in analyses:
        topic = a.get("topic")
        if topic and topic not in topics:
            topics.append(topic)
    if topics:
        st.caption("Topics detected")
        st.markdown(" ".join(f"`{t}`" for t in topics))

    _render_profile(patient_profile)


def _render_profile(patient_profile):
    profile = patient_profile or {}
    mh = profile.get("medical_history") or []
    tg = profile.get("therapy_goals") or []
    if not (mh or tg):
        return
    st.caption("Patient profile")
    if mh:
        st.markdown("**History:** " + ", ".join(mh))
    if tg:
        st.markdown("**Goals:** " + ", ".join(tg))

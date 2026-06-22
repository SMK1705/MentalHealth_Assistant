def _fmt_date(value):
    if not value:
        return "—"
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10]
    except Exception:
        return str(value)[:10]


def build_patient_summary(profile):
    """Compact text summary of the patient record for the LLM prompt."""
    profile = profile or {}
    parts = [f"patient_id: {profile.get('patient_id', '')}"]
    mh = profile.get("medical_history") or []
    tg = profile.get("therapy_goals") or []
    if mh:
        parts.append("medical history: " + "; ".join(mh))
    if tg:
        parts.append("therapy goals: " + "; ".join(tg))
    return "\n".join(parts)


def build_history_summary(sessions, limit=5):
    """Compact text summary of prior sessions for the LLM prompt."""
    if not sessions:
        return "(no prior sessions)"
    lines = []
    for s in sessions[:limit]:
        date = _fmt_date(s.get("created_at"))
        topics = ", ".join(s.get("detected_topics") or []) or "—"
        flags = ", ".join(s.get("risk_flags") or [])
        flag_txt = f"; risk flags: {flags}" if flags else ""
        lines.append(f"{date}: {topics}{flag_txt}")
    return "\n".join(lines)


def render_patient_overview(patient_id, profile=None, exclude_session_id=None):
    """Top patient-context card + collapsible history timeline for the doctor.

    Reads the profile (passed in to avoid a re-fetch) and the patient's prior
    session logs. ``exclude_session_id`` drops the in-progress session so the
    counts/timeline reflect *prior* context.
    """
    import streamlit as st
    from patient_profile import get_patient_sessions

    profile = profile or {}
    medical_history = profile.get("medical_history") or []
    therapy_goals = profile.get("therapy_goals") or []

    try:
        sessions = get_patient_sessions(patient_id)
    except Exception:
        sessions = []
    if exclude_session_id:
        sessions = [s for s in sessions if s.get("session_id") != exclude_session_id]

    last = sessions[0] if sessions else None

    cols = st.columns([2, 1, 1])
    cols[0].markdown(f"**Patient** `{patient_id}`")
    cols[1].metric("Past sessions", len(sessions))
    cols[2].metric("Last seen", _fmt_date(last.get("created_at")) if last else "—")

    if medical_history:
        st.markdown("**Clinical history:** " + " · ".join(f"`{x}`" for x in medical_history))
    if therapy_goals:
        st.markdown("**Therapy goals:** " + ", ".join(therapy_goals))

    if last:
        bits = []
        topics = last.get("detected_topics") or []
        flags = last.get("risk_flags") or []
        score = last.get("sentiment_score")
        if topics:
            bits.append("topics: " + ", ".join(topics))
        if flags:
            bits.append("risk flags: " + ", ".join(flags))
        if isinstance(score, (int, float)):
            bits.append(f"sentiment {score:+.2f}")
        if bits:
            st.caption("Most recent session — " + " · ".join(bits))

    if sessions:
        with st.expander(f"Session history ({len(sessions)})"):
            for s in sessions[:20]:
                date = _fmt_date(s.get("created_at"))
                topics = ", ".join(s.get("detected_topics") or []) or "—"
                flags = ", ".join(s.get("risk_flags") or [])
                flag_txt = f" · risk: {flags}" if flags else ""
                score = s.get("sentiment_score")
                score_txt = f" · sentiment {score:+.2f}" if isinstance(score, (int, float)) else ""
                st.markdown(f"- **{date}** — {topics}{flag_txt}{score_txt}")
    else:
        st.caption("No prior sessions on record for this patient.")

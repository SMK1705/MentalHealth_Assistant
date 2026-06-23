def _fmt_date(value):
    if not value:
        return "—"
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)[:10]
    except Exception:
        return str(value)[:10]


def _days_ago(value):
    """Best-effort '· Nd ago' suffix from a date/datetime/ISO string."""
    from datetime import datetime, date
    try:
        if hasattr(value, "year") and not isinstance(value, datetime):
            d = datetime(value.year, value.month, value.day)
        elif isinstance(value, datetime):
            d = value
        else:
            d = datetime.fromisoformat(str(value)[:19])
        days = (datetime.now() - d).days
        if days <= 0:
            return "today"
        return f"{days}d ago"
    except Exception:
        return ""


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
    """Left-column patient context: overview card + session-history timeline.

    Reads the profile (passed in to avoid a re-fetch) and the patient's prior
    session logs. ``exclude_session_id`` drops the in-progress session so the
    counts/timeline reflect *prior* context.
    """
    import streamlit as st
    import ui

    profile = profile or {}
    medical_history = profile.get("medical_history") or []
    therapy_goals = profile.get("therapy_goals") or []

    try:
        from patient_profile import get_patient_sessions
        sessions = get_patient_sessions(patient_id)
    except Exception:
        sessions = []
    if exclude_session_id:
        sessions = [s for s in sessions if s.get("session_id") != exclude_session_id]

    last = sessions[0] if sessions else None
    subtitle = "returning patient" if sessions else "new patient"
    if last:
        ago = _days_ago(last.get("created_at"))
        last_seen = _fmt_date(last.get("created_at")) + (f" · {ago}" if ago else "")
    else:
        last_seen = "first session"

    st.markdown(
        ui.patient_overview_card(patient_id, subtitle, len(sessions),
                                 medical_history, therapy_goals, last_seen),
        unsafe_allow_html=True,
    )

    items = []
    for s in sessions[:8]:
        flags = s.get("risk_flags") or []
        items.append({
            "date": _fmt_date(s.get("created_at")),
            "topics": s.get("detected_topics") or [],
            "score": s.get("sentiment_score"),
            "flag": flags[0] if flags else None,
        })
    st.markdown(ui.history_timeline(items), unsafe_allow_html=True)

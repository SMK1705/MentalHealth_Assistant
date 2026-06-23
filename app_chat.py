"""Streamlit entry point — the Live Session Cockpit, wired to the real backend.

The pixel-perfect design (``cockpit_component/``) is mounted as a bidirectional
Streamlit component and driven by the real application:

- **Open a patient** (any ID, a persona card, or a recent) loads their real
  profile + session history from MongoDB into the overview and timeline.
- **Patient turns** run the real pipeline — deterministic crisis screen, emotion
  (DistilRoBERTa), sentiment (RoBERTa), topic (BART), semantic retrieval
  (MiniLM + Pinecone over the corpus), and Groq decision-support.
- **Sessions persist** — each turn archives the conversation + session log
  (topics, risk flags, sentiment, doctor notes) to MongoDB.
- **Recent patients** on the launch screen come from the real ``patients``
  collection.

``PT-0042`` remains the built-in scripted demo. Heavy ML imports are lazy, so
the launch screen loads instantly. The Streamlit-native UI is preserved in
``app_live.py``.
"""
import hmac
import logging
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config import settings
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Live Session Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
#MainMenu, header, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] { background: #eef1f6; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe { display: block; border: none; width: 100%; height: 100vh !important; }
[data-testid="stElementContainer"]:has(iframe),
[data-testid="element-container"]:has(iframe) { height: 100vh !important; }
</style>
""",
    unsafe_allow_html=True,
)

if not settings.app_password:
    logger.warning("APP_PASSWORD is not set — the app is running without authentication.")


def check_authentication() -> bool:
    """Gate the app behind a shared password when settings.app_password is set."""
    if not settings.app_password:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("🔒 Sign in")
    password = st.text_input("Password", type="password", key="auth_password")
    if st.button("Sign in"):
        if hmac.compare_digest(password, settings.app_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_authentication():
    st.stop()

_COMPONENT_DIR = Path(__file__).parent / "cockpit_component"
_lsa_cockpit = components.declare_component("lsa_cockpit", path=str(_COMPONENT_DIR))

# Profile for the built-in PT-0042 scripted demo (matches its overview card).
_DEMO_PROFILE = {
    "patient_id": "PT-0042",
    "medical_history": ["generalized anxiety disorder", "panic attacks"],
    "therapy_goals": ["reduce daily anxiety", "manage panic episodes"],
}


def _fmt_score(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0.00"
    return ("+%.2f" % n) if n >= 0 else ("%.2f" % n)


# ----------------------------------------------------------------- patient load
def _roster():
    """A few real patients from MongoDB for the launch-screen 'recent patients'."""
    try:
        from db import get_db
        from patient_profile import get_patient_sessions
        from patient_overview import _fmt_date
        docs = list(get_db()["patients"].find({}, {"patient_id": 1, "_id": 0}).limit(6))
    except Exception:
        logger.exception("Failed to load the patient roster")
        return None
    out = []
    for d in docs:
        pid = d.get("patient_id")
        if not pid:
            continue
        try:
            sessions = get_patient_sessions(pid)
        except Exception:
            sessions = []
        last = sessions[0] if sessions else None
        if last:
            topic = (last.get("detected_topics") or ["session"])[0]
            out.append({"id": pid, "last": f"{_fmt_date(last.get('created_at'))} · {topic}",
                        "flag": bool(last.get("risk_flags"))})
        else:
            out.append({"id": pid, "last": "no prior sessions", "flag": False})
    return out or None


def _load_patient(pid: str) -> dict:
    """Load a patient's real profile + history from MongoDB for the cockpit."""
    from patient_profile import get_patient_profile, get_patient_sessions
    from patient_overview import _fmt_date, _days_ago

    try:
        profile = get_patient_profile(pid)
    except Exception:
        logger.exception("Patient profile load failed")
        return {"found": False, "patientId": pid, "error": "Could not reach the patient database."}
    if profile is None:
        return {"found": False, "patientId": pid,
                "error": f"No record found for {pid}. Check the ID, or seed the patient first."}

    prof = profile.dict()
    try:
        sessions = get_patient_sessions(pid)
    except Exception:
        sessions = []

    last = sessions[0] if sessions else None
    if last:
        ago = _days_ago(last.get("created_at"))
        last_seen = _fmt_date(last.get("created_at")) + (f" · {ago}" if ago else "")
    else:
        last_seen = "first session"

    timeline = []
    for sdoc in sessions[:8]:
        flags = sdoc.get("risk_flags") or []
        timeline.append({
            "date": _fmt_date(sdoc.get("created_at")),
            "topics": sdoc.get("detected_topics") or [],
            "score": float(sdoc.get("sentiment_score") or 0.0),
            "flag": flags[0] if flags else None,
        })

    patient = {
        "id": pid,
        "subtitle": "returning patient" if sessions else "new patient",
        "sessionCount": len(sessions),
        "history": prof.get("medical_history") or [],
        "goals": prof.get("therapy_goals") or [],
        "lastSeen": last_seen,
    }
    return {"found": True, "patient": patient, "timeline": timeline,
            "session_id": str(uuid.uuid4()), "profile": prof}


# ------------------------------------------------------------------ turn + archive
def _run_real_turn(text: str, transcript: str, profile: dict) -> dict:
    """Run the real pipeline for one patient turn and shape it for the cockpit."""
    from unified_guidance import analyze_message
    from session_assistant import generate_session_suggestions
    from patient_overview import build_patient_summary

    analysis = analyze_message(text, profile, transcript)
    urgency = analysis.get("urgency") or {}
    sp = analysis.get("safety_protocol")

    emotion = urgency.get("label") or "neutral"
    e_score = urgency.get("score") or 0.0
    sentiment = (analysis.get("sentiment") or "neutral").lower()
    s_score = analysis.get("sentiment_score") or 0.0
    topic = analysis.get("predicted_topic") or "general"
    t_conf = analysis.get("topic_confidence") or 0.0

    signals = (
        f"topic: {topic} ({t_conf}); sentiment: {sentiment} ({s_score}); "
        f"emotion: {emotion} ({e_score}); "
        f"crisis flag: {(sp.get('flag_type') + '/' + sp.get('action')) if sp else 'none'}"
    )
    suggestions = generate_session_suggestions(
        transcript=transcript,
        patient_summary=build_patient_summary(profile),
        history_summary="(loaded patient record)",
        signals=signals,
        examples=analysis.get("historical_examples"),
        doctor_questions="(see transcript)",
    )

    why_signals = [
        {"k": "Emotion (DistilRoBERTa)", "v": f"{emotion} · {round(float(e_score) * 100)}%"},
        {"k": "Sentiment (RoBERTa)", "v": f"{sentiment} · {_fmt_score(s_score)}"},
        {"k": "Topic (BART zero-shot)", "v": f"{topic} · {round(float(t_conf) * 100)}%"},
        {"k": "Crisis screen", "v": (f"{sp['flag_type']} · {sp['action']}") if sp else "clear"},
    ]
    cases = [
        {"id": str(ex.get("questionID") or ex.get("id") or "—"), "sim": "",
         "q": (ex.get("questionText") or "")[:200], "a": (ex.get("answerText") or "")[:200]}
        for ex in (analysis.get("historical_examples") or [])[:3]
    ]

    return {
        "analysis": {"emotion": emotion, "emotionScore": float(e_score), "sentiment": sentiment,
                     "sentimentScore": float(s_score), "topic": topic, "topicConf": float(t_conf)},
        "suggestions": {k: suggestions.get(k) for k in
                        ("emotional_state", "state_confidence", "red_flags",
                         "missing_info", "next_questions", "follow_ups", "caveat")},
        "why": {"signals": why_signals, "cases": cases,
                "context": analysis.get("analysis_context") or ""},
        "crisis": ({"flag_type": sp["flag_type"], "action": sp["action"], "response": sp["response"]}
                   if sp else None),
        "errors": analysis.get("errors") or [],
    }


def _archive(active: dict, messages, notes: str, result: dict) -> None:
    """Persist the conversation + session log for the active patient session."""
    from archiver import archive_conversation, archive_session
    from schemas import Conversation, Message, SessionLog
    try:
        conv = Conversation(
            session_id=active["session_id"], patient_id=active["patient_id"],
            messages=[Message(content=m.get("text", ""),
                              is_user=(m.get("speaker") == "patient"),
                              speaker=m.get("speaker", "patient")) for m in (messages or [])],
        )
        archive_conversation(conv)

        topic = (result.get("analysis") or {}).get("topic")
        if topic and topic not in active["topics"]:
            active["topics"].append(topic)
        crisis = result.get("crisis")
        if crisis and crisis.get("flag_type") and crisis["flag_type"] not in active["risk_flags"]:
            active["risk_flags"].append(crisis["flag_type"])

        archive_session(SessionLog(
            session_id=active["session_id"], patient_id=active["patient_id"],
            detected_topics=active["topics"], risk_flags=active["risk_flags"],
            sentiment_score=float((result.get("analysis") or {}).get("sentimentScore") or 0.0),
            doctor_notes=notes or "", suggestions=[],
        ))
    except Exception:
        logger.exception("Session archival failed")
        st.session_state["lsa_archive_failed"] = True


_ERROR_RESULT = {
    "analysis": {"emotion": "neutral", "emotionScore": 0.0, "sentiment": "neutral",
                 "sentimentScore": 0.0, "topic": "general", "topicConf": 0.0},
    "suggestions": {"emotional_state": "", "state_confidence": "low", "red_flags": [],
                    "missing_info": [], "next_questions": [], "follow_ups": [],
                    "caveat": "The analysis pipeline errored for this turn."},
    "why": {"signals": [], "cases": [], "context": ""},
    "crisis": None,
}

# --------------------------------------------------------------------- main loop
if "lsa_roster" not in st.session_state:
    st.session_state["lsa_roster"] = _roster()

payload = _lsa_cockpit(
    result=st.session_state.get("lsa_result"),
    roster=st.session_state.get("lsa_roster"),
    key="lsa_cockpit",
    default=None,
)

if isinstance(payload, dict):
    nonce = payload.get("nonce")
    kind = payload.get("kind")
    if nonce is not None and nonce != st.session_state.get("lsa_handled_nonce"):
        st.session_state["lsa_handled_nonce"] = nonce

        if kind == "open":
            loaded = _load_patient((payload.get("patientId") or "").strip().upper())
            result = {"nonce": nonce, "kind": "open", "found": loaded["found"]}
            if loaded["found"]:
                result["patient"] = loaded["patient"]
                result["timeline"] = loaded["timeline"]
                st.session_state["lsa_active"] = {
                    "patient_id": loaded["patient"]["id"],
                    "session_id": loaded["session_id"],
                    "profile": loaded["profile"],
                    "risk_flags": [], "topics": [],
                }
            else:
                result["patientId"] = loaded.get("patientId")
                result["error"] = loaded.get("error")
            st.session_state["lsa_result"] = result
            st.rerun()

        elif kind == "patient_turn":
            active = st.session_state.get("lsa_active")
            profile = active["profile"] if active else _DEMO_PROFILE
            try:
                result = _run_real_turn(payload.get("text", ""), payload.get("transcript", ""), profile)
            except Exception:
                logger.exception("Real pipeline failed for a patient turn")
                result = dict(_ERROR_RESULT)
            result["nonce"] = nonce
            result["kind"] = "turn"
            if active:  # real patient (not the in-memory demo) -> persist
                _archive(active, payload.get("messages"), payload.get("notes", ""), result)
            st.session_state["lsa_result"] = result
            st.rerun()

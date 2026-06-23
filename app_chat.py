import asyncio
import hmac
import streamlit as st
import logging
import uuid
from datetime import datetime
import ui
from logging_config import setup_logging
from unified_guidance import analyze_message
from archiver import archive_conversation, archive_session
from schemas import Conversation, Message, SessionLog
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import analyze_sentiment
from llm_rag import generate_advice
from patient_profile import (
    get_patient_profile, create_patient_profile, update_patient_fields, get_patient_sessions,
)
from safety import SafetyChecker
from config import settings
from dashboard import render_dashboard
from session_assistant import generate_session_suggestions, render_suggestions
from patient_overview import render_patient_overview, build_patient_summary, build_history_summary

# Ensure an event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# Stateless crisis screener for the guaranteed fallback.
_safety_checker = SafetyChecker()

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
        # Constant-time comparison to avoid leaking the password via timing.
        if hmac.compare_digest(password, settings.app_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

# Page configuration
st.set_page_config(
    page_title="Live Session Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cockpit theme: hide default chrome, fonts, widget styling, segmented toggle.
st.markdown(ui.css(), unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "conversation_model" not in st.session_state:
    st.session_state.conversation_model = Conversation(
        session_id=str(uuid.uuid4()), patient_id="test_patient", messages=[]
    )
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = {}
if "session_risk_flags" not in st.session_state:
    st.session_state.session_risk_flags = []
if "session_topics" not in st.session_state:
    st.session_state.session_topics = []
if "session_suggestions" not in st.session_state:
    st.session_state.session_suggestions = []


def _parse_lines(text):
    """Split a textarea value into a clean list of non-empty, stripped lines."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


# Seeded demo patients so the tool can be tried without setup.
DEMO_PERSONAS = {
    "Alex — anxiety": {
        "patient_id": "demo_alex",
        "medical_history": ["generalized anxiety disorder", "chronic insomnia"],
        "therapy_goals": ["improve sleep", "reduce daily anxiety"],
    },
    "Jordan — grief": {
        "patient_id": "demo_jordan",
        "medical_history": ["recent bereavement", "low mood"],
        "therapy_goals": ["process grief", "re-engage with daily life"],
    },
    "Sam — work stress": {
        "patient_id": "demo_sam",
        "medical_history": ["burnout", "work-related stress"],
        "therapy_goals": ["set boundaries", "manage workload stress"],
    },
}

# One-click PATIENT turns that walk a distress -> crisis -> recovery arc in a demo.
SUGGESTED_PROMPTS = [
    {"label": "Share distress", "text": "I've been really anxious lately and I can't sleep at all"},
    {"label": "Express a crisis", "text": "Honestly, sometimes I feel like I want to end it all"},
    {"label": "Show improvement", "text": "Talking this through actually helps — thank you"},
]

# Session-scoped keys cleared between sessions.
_SESSION_KEYS = (
    "conversation", "conversation_model", "patient_profile", "session_risk_flags",
    "session_topics", "session_suggestions", "latest_suggestions", "latest_analysis",
    "history_summary", "doctor_notes_input", "demo_mode", "session_started", "last_report",
)


def _new_live_session(patient_id):
    """Reset per-session state and start a fresh transcript for a patient."""
    st.session_state.conversation = []
    st.session_state.conversation_model = Conversation(
        session_id=str(uuid.uuid4()), patient_id=patient_id, messages=[]
    )
    st.session_state.session_risk_flags = []
    st.session_state.session_topics = []
    st.session_state.session_suggestions = []
    st.session_state.session_started = datetime.now()
    for k in ("latest_suggestions", "latest_analysis", "history_summary",
              "doctor_notes_input", "last_report"):
        st.session_state.pop(k, None)


def start_demo(persona):
    """Begin a self-serve demo with a seeded persona (in-memory, no DB writes)."""
    st.session_state.demo_mode = True
    st.session_state.patient_profile = {
        "patient_id": persona["patient_id"],
        "medical_history": list(persona["medical_history"]),
        "therapy_goals": list(persona["therapy_goals"]),
    }
    _new_live_session(persona["patient_id"])
    st.session_state.page = "chat"
    st.rerun()


def reset_session():
    """Clear the session and return to the landing page."""
    for key in _SESSION_KEYS + ("page",):
        st.session_state.pop(key, None)
    st.rerun()


def _load_history_summary():
    try:
        pid = st.session_state.conversation_model.patient_id
        sid = st.session_state.conversation_model.session_id
        sessions = [s for s in get_patient_sessions(pid) if s.get("session_id") != sid]
        return build_history_summary(sessions)
    except Exception:
        logger.exception("Failed to load patient history summary")
        return "(no prior sessions)"


def _format_signals(analysis):
    a = analysis or {}
    urgency = a.get("urgency") or {}
    sp = a.get("safety_protocol")
    return "; ".join([
        f"topic: {a.get('predicted_topic')} ({a.get('topic_confidence')})",
        f"sentiment: {a.get('sentiment')} ({a.get('sentiment_score')})",
        f"emotion: {urgency.get('label')} ({urgency.get('score')})",
        f"crisis flag: {(sp.get('flag_type') + '/' + sp.get('action')) if sp else 'none'}",
    ])


def _persist_session():
    """Archive the transcript + session log unless this is an in-memory demo.

    Persistence failures are recorded in session state (not silently dropped)
    so the clinician is warned that the turn may not have been saved — this
    runs right before ``st.rerun()``, so a warning shown here would be lost.
    """
    if st.session_state.get("demo_mode"):
        return
    failed = False
    try:
        archive_conversation(st.session_state.conversation_model)
    except Exception:
        logger.exception("Failed to archive conversation")
        failed = True
    try:
        latest = st.session_state.get("latest_analysis") or {}
        archive_session(SessionLog(
            session_id=st.session_state.conversation_model.session_id,
            patient_id=st.session_state.conversation_model.patient_id,
            detected_topics=st.session_state.session_topics,
            risk_flags=st.session_state.session_risk_flags,
            sentiment_score=float(latest.get("sentiment_score") or 0.0),
            doctor_notes=st.session_state.get("doctor_notes_input", ""),
            suggestions=st.session_state.get("session_suggestions", []),
        ))
    except Exception:
        logger.exception("Failed to archive session log")
        failed = True
    st.session_state["persist_error"] = failed


def _run_assistant(patient_text):
    """Analyze a patient turn and generate decision-support for the doctor."""
    transcript = "\n".join(
        f"{m.get('speaker', 'patient').capitalize()}: {m['content']}"
        for m in st.session_state.conversation
    )
    doctor_questions = "\n".join(
        m["content"] for m in st.session_state.conversation if m.get("speaker") == "doctor"
    ) or "(none yet)"

    if "history_summary" not in st.session_state:
        st.session_state.history_summary = _load_history_summary()

    analysis = {"safety_protocol": _safety_checker.check_input(patient_text)}
    suggestions = {}
    try:
        with st.status("Analyzing the patient turn…", expanded=True) as status:
            analysis = analyze_message(patient_text, st.session_state.patient_profile, transcript)
            sp = analysis.get("safety_protocol")
            st.write(f"Safety: {sp['action'] + ' — ' + sp['flag_type'] if sp else 'no crisis indicators'}")
            urgency = analysis.get("urgency") or {}
            st.write(
                f"Emotion: {urgency.get('label') or 'neutral'} · "
                f"Topic: {analysis.get('predicted_topic') or 'n/a'} · "
                f"Sentiment: {analysis.get('sentiment') or 'n/a'}"
            )
            st.write(f"Retrieved {len(analysis.get('historical_examples') or [])} similar case(s).")
            degraded = analysis.get("errors") or []
            if degraded:
                st.write(f"⚠️ Degraded — unavailable: {', '.join(degraded)}")
            status.update(label="Generating decision support…")
            suggestions = generate_session_suggestions(
                transcript=transcript,
                patient_summary=build_patient_summary(st.session_state.patient_profile),
                history_summary=st.session_state.history_summary,
                signals=_format_signals(analysis),
                examples=analysis.get("historical_examples"),
                doctor_questions=doctor_questions,
            )
            if suggestions.get("_error"):
                status.update(label="Decision support unavailable — model error",
                              state="error", expanded=False)
            elif degraded:
                status.update(label="Decision support ready (degraded analysis)",
                              state="complete", expanded=False)
            else:
                status.update(label="Decision support ready", state="complete", expanded=False)
    except Exception:
        logger.exception("Assistant generation error")
        st.error("The assistant hit an unexpected error analyzing this turn. "
                 "The deterministic safety check (shown above, if any) still applies.")

    st.session_state.latest_analysis = analysis
    st.session_state.latest_suggestions = suggestions

    analytics = {
        "topic": analysis.get("predicted_topic"),
        "topic_confidence": analysis.get("topic_confidence"),
        "sentiment": analysis.get("sentiment"),
        "sentiment_score": analysis.get("sentiment_score"),
        "safety_protocol": analysis.get("safety_protocol"),
        "urgency": analysis.get("urgency"),
        # Session-state only (not archived) for the "why" panel.
        "historical_examples": analysis.get("historical_examples"),
        "analysis_context": analysis.get("analysis_context"),
    }
    # Attach analytics to the patient turn (feeds the metrics dashboard).
    if st.session_state.conversation:
        st.session_state.conversation[-1]["analysis"] = analytics
    lean = {k: analytics.get(k) for k in
            ("topic", "topic_confidence", "sentiment", "sentiment_score", "safety_protocol", "urgency")}
    if st.session_state.conversation_model.messages:
        st.session_state.conversation_model.messages[-1].metadata = lean

    # Audit trail entry.
    st.session_state.session_suggestions.append({
        "turn": patient_text,
        "suggestions": {k: suggestions.get(k) for k in
                        ("emotional_state", "next_questions", "red_flags", "missing_info", "follow_ups")},
        "safety": analytics.get("safety_protocol"),
    })

    # Roll up distinct risk flags + topics for the session.
    protocol = analytics.get("safety_protocol")
    flag_type = protocol.get("flag_type") if protocol else None
    if flag_type and flag_type not in st.session_state.session_risk_flags:
        st.session_state.session_risk_flags.append(flag_type)
    topic = analytics.get("topic")
    if topic and topic not in st.session_state.session_topics:
        st.session_state.session_topics.append(topic)


def handle_turn(content, speaker):
    """Append a transcript turn. Patient turns trigger the assistant; doctor
    turns are logged as context only."""
    is_patient = speaker == "patient"
    st.session_state.conversation.append({
        "speaker": speaker, "content": content,
        "time": datetime.now().strftime("%H:%M"),
    })
    st.session_state.conversation_model.add_message(
        Message(content=content, is_user=is_patient, speaker=speaker)
    )
    if is_patient:
        _run_assistant(content)
    _persist_session()
    st.rerun()


# --------------------------------------------------------------- cockpit glue
def _elapsed_clock():
    """mm:ss since the session started (updates each rerun, not live-ticking)."""
    start = st.session_state.get("session_started")
    if not start:
        return "00:00"
    secs = max(0, int((datetime.now() - start).total_seconds()))
    return f"{secs // 60:02d}:{secs % 60:02d}"


# Per-message transcript chips: (text, background, color).
_AMBER = ("#fbf3e4", "#8a6a1f")
_TOPIC = ("#eaf0fd", "#3052b8")


def _transcript_messages():
    """Adapt the live conversation into ui.transcript's message dicts, attaching
    emotion/sentiment/topic chips and the crisis flag to analyzed patient turns."""
    out = []
    for m in st.session_state.conversation:
        speaker = m.get("speaker") or "patient"
        entry = {"speaker": speaker, "text": m.get("content"),
                 "time": m.get("time", ""), "chips": [], "crisis": False}
        a = m.get("analysis")
        if a and speaker == "patient":
            chips = []
            urgency = a.get("urgency") or {}
            if urgency.get("label"):
                score = urgency.get("score")
                label = str(urgency["label"]).title()
                txt = f"{label} · {score:.0%}" if isinstance(score, (int, float)) else label
                chips.append((txt, *_AMBER))
            ss = a.get("sentiment_score")
            if isinstance(ss, (int, float)):
                bg, col = (("#fdeceb", "#b23a31") if ss < 0
                           else ("#e8f5ef", "#2f8f6b") if ss > 0 else ("#eef1f6", "#647089"))
                chips.append((f"{ss:+.2f}", bg, col))
            if a.get("topic"):
                chips.append((str(a["topic"]), *_TOPIC))
            entry["chips"] = chips
            entry["crisis"] = bool(a.get("safety_protocol"))
        out.append(entry)
    return out


_PIPELINE_DEFS = [
    ("Crisis screen", "regex · deterministic"),
    ("Emotion / urgency", "DistilRoBERTa"),
    ("Sentiment", "RoBERTa 3-class"),
    ("Topic", "BART zero-shot"),
    ("Semantic retrieval", "MiniLM · Pinecone"),
    ("Generating guidance", "Groq · llama-3.3-70b"),
]


def _pipeline_stages():
    """Build the analysis-pipeline panel state from the latest turn's outcome."""
    a = st.session_state.get("latest_analysis") or {}
    sug = st.session_state.get("latest_suggestions") or {}
    if not a.get("analysis_context") and not a.get("safety_protocol") and not st.session_state.conversation:
        return ([{"label": l, "sub": s, "status": "pending"} for l, s in _PIPELINE_DEFS],
                False, "IDLE")
    errors = a.get("errors") or []
    sp = a.get("safety_protocol")
    status_for = lambda key: "pending" if key in errors else "done"
    stages = [
        {"label": "Crisis screen", "sub": "regex · deterministic",
         "status": "alert" if sp else "done"},
        {"label": "Emotion / urgency", "sub": "DistilRoBERTa", "status": status_for("urgency")},
        {"label": "Sentiment", "sub": "RoBERTa 3-class", "status": status_for("analysis")},
        {"label": "Topic", "sub": "BART zero-shot", "status": status_for("analysis")},
        {"label": "Semantic retrieval", "sub": "MiniLM · Pinecone", "status": status_for("retrieval")},
        {"label": "Generating guidance", "sub": "Groq · llama-3.3-70b",
         "status": "alert" if sug.get("_error") else "done"},
    ]
    degraded = bool(errors) or bool(sug.get("_error"))
    return stages, False, ("DEGRADED" if degraded else "COMPLETE")


def _open_session(patient_id, medical_history, therapy_goals):
    """Load/create the patient profile and switch to the live session."""
    try:
        profile = get_patient_profile(patient_id)
        if profile is None:
            profile = create_patient_profile(
                patient_id, medical_history=medical_history, therapy_goals=therapy_goals,
            )
        elif (medical_history and not profile.medical_history) or (
            therapy_goals and not profile.therapy_goals
        ):
            profile = update_patient_fields(
                patient_id,
                medical_history=medical_history or profile.medical_history,
                therapy_goals=therapy_goals or profile.therapy_goals,
            )
    except Exception:
        logger.exception("Failed to load or create patient profile")
        st.error("Could not load the patient profile. Please try again later.")
        return

    st.session_state.demo_mode = False
    st.session_state.patient_profile = profile.dict()
    _new_live_session(patient_id)
    st.session_state.page = "chat"
    st.rerun()


# Landing / launch screen
def landing_page():
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown(ui.launch_hero(), unsafe_allow_html=True)

        patient_id = st.text_input("Patient ID", key="patient_id_input", placeholder="e.g. PT-0042")
        with st.expander("Add clinical context (optional)"):
            medical_history_input = st.text_area(
                "Clinical history (one item per line)", key="medical_history_input"
            )
            therapy_goals_input = st.text_area(
                "Therapy goals (one item per line)", key="therapy_goals_input"
            )

        if st.button("Open session", type="primary", use_container_width=True):
            patient_id = (patient_id or "").strip()
            if not patient_id:
                st.error("Please enter a patient ID before starting.")
            else:
                _open_session(
                    patient_id,
                    _parse_lines(st.session_state.get("medical_history_input")),
                    _parse_lines(st.session_state.get("therapy_goals_input")),
                )

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin:18px 0 12px;">'
            f'<span style="flex:1;height:1px;background:#dde3ee;"></span>'
            f'<span style="font:500 10.5px/1 {ui.MONO};color:#aab2c4;letter-spacing:.04em;">'
            f'OR START WITH A DEMO PERSONA</span>'
            f'<span style="flex:1;height:1px;background:#dde3ee;"></span></div>',
            unsafe_allow_html=True,
        )
        demo_cols = st.columns(len(DEMO_PERSONAS))
        for i, (name, persona) in enumerate(DEMO_PERSONAS.items()):
            if demo_cols[i].button(name, key=f"demo_persona_{i}", use_container_width=True):
                start_demo(persona)

        st.markdown(ui.launch_footer(), unsafe_allow_html=True)


def _generate_report():
    """Compute the end-of-session report from accumulated session signals."""
    conv = st.session_state.conversation
    # Classify over the patient's own turns (doctor questions skew topic/sentiment).
    patient_text = "\n".join(m["content"] for m in conv if m.get("speaker") == "patient") \
        or "\n".join(m["content"] for m in conv)
    classifier = load_topic_classifier()
    predicted_topic, topic_confidence = predict_topic(patient_text, classifier)
    sentiment, sentiment_score = analyze_sentiment(patient_text)

    risk_flags = st.session_state.get("session_risk_flags") or []
    topics = st.session_state.get("session_topics") or []
    notes = st.session_state.get("doctor_notes_input", "")
    prompt = (
        f"Patient Session Report:\n"
        f"Topic: {predicted_topic} (Confidence: {topic_confidence:.2f})\n"
        f"Topics observed this session: {', '.join(topics) or 'n/a'}\n"
        f"Overall Sentiment: {sentiment} (Score: {sentiment_score})\n"
        f"Risk flags raised this session: {', '.join(risk_flags) or 'none'}\n"
        f"Clinician notes: {notes or '(none)'}\n\n"
        "Provide a structured clinician-facing session summary with sections:\n"
        "1. **Presentation & key themes**\n"
        "2. **Areas of concern / risk** — explicitly address every risk flag listed above.\n"
        "3. **Suggested follow-up** (for the clinician to consider; not prescriptive).\n"
        "Use clear headings and bullet points. Do not diagnose or prescribe."
    )
    recommendations_obj = generate_advice(prompt)
    recommendations = (recommendations_obj if isinstance(recommendations_obj, str)
                       else str(recommendations_obj))
    return {
        "topic": predicted_topic, "topic_conf": topic_confidence,
        "sentiment": sentiment, "score": sentiment_score,
        "risk_flags": risk_flags, "topics": topics, "recommendations": recommendations,
    }


@st.dialog("End-of-session report", width="large")
def _report_dialog():
    pid = st.session_state.conversation_model.patient_id
    st.caption(f"{pid} · {_elapsed_clock()} · {len(st.session_state.conversation)} turns logged")
    if st.button("Generate / regenerate", type="primary") or "last_report" not in st.session_state:
        with st.spinner("Generating the session report…"):
            try:
                st.session_state["last_report"] = _generate_report()
            except Exception:
                logger.exception("Error generating report")
                st.session_state.pop("last_report", None)
                st.error("An error occurred while generating the report. Please try again later.")
                return
    rep = st.session_state.get("last_report")
    if not rep:
        return
    if rep["risk_flags"]:
        st.error("**Risk flags raised this session:** " + ", ".join(rep["risk_flags"]))
    st.markdown(f"**Predicted topic:** {rep['topic']} · **Confidence:** {rep['topic_conf']:.2f}")
    st.markdown(f"**Overall sentiment:** {rep['sentiment']} ({rep['score']})")
    if rep["topics"]:
        st.caption("Topics detected: " + ", ".join(rep["topics"]))
    st.markdown(rep["recommendations"])
    st.caption("Decision support generated from session signals — not a diagnosis or medical "
               "record. Review, edit, and sign before filing.")


# Chat / cockpit page
def chat_page():
    # Surface a persistence failure from the previous turn (set just before the
    # rerun that brought us here, so it could not be shown inline).
    if st.session_state.pop("persist_error", False):
        st.warning("⚠️ The last turn may not have been saved to the database. "
                   "Check the patient profile exists and the database is reachable.")

    pid = st.session_state.conversation_model.patient_id
    sid = st.session_state.conversation_model.session_id
    conv = st.session_state.conversation
    profile = st.session_state.get("patient_profile") or {}
    latest_analysis = st.session_state.get("latest_analysis") or {}
    latest_suggestions = st.session_state.get("latest_suggestions") or {}

    st.markdown(ui.header_bar(pid, _elapsed_clock(), len(conv)), unsafe_allow_html=True)

    c_info, c_report, c_new = st.columns([6, 1.6, 1.2])
    if st.session_state.get("demo_mode"):
        c_info.caption("Demo mode — sample patient, nothing is saved.")
    if c_report.button("End session & report", type="primary", use_container_width=True):
        _report_dialog()
    if c_new.button("New session", use_container_width=True):
        reset_session()

    left, center, right = st.columns([316, 480, 404], gap="medium")

    # LEFT — patient context + clinician notes
    with left:
        render_patient_overview(pid, profile, exclude_session_id=sid)
        st.markdown('<div class="lsa-h" style="margin-top:4px;"><span class="bar"></span>'
                    '<span class="t">CLINICIAN NOTES</span></div>', unsafe_allow_html=True)
        st.text_area("Clinician notes", key="doctor_notes_input", height=110,
                     label_visibility="collapsed",
                     placeholder="Private notes for this session — saved with the session…")

    # CENTER — crisis banner + transcript + composer
    with center:
        sp = latest_analysis.get("safety_protocol")
        if sp:
            st.markdown(ui.crisis_banner(sp.get("action"), sp.get("flag_type"), sp.get("response")),
                        unsafe_allow_html=True)
        st.markdown(ui.transcript(_transcript_messages(), len(conv)), unsafe_allow_html=True)

        speaker_label = st.radio("Log turn as", ["Patient", "Doctor"], horizontal=True,
                                 key="speaker_select", label_visibility="collapsed")
        if st.session_state.get("demo_mode"):
            st.caption("Quick demo patient turns")
            prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
            for i, p in enumerate(SUGGESTED_PROMPTS):
                if prompt_cols[i].button(p["label"], key=f"suggested_{i}", use_container_width=True):
                    handle_turn(p["text"], "patient")

    # RIGHT — analysis pipeline + decision support + session metrics
    with right:
        stages, running, status_label = _pipeline_stages()
        st.markdown(ui.pipeline_panel(stages, running, status_label), unsafe_allow_html=True)
        render_suggestions(latest_suggestions, latest_analysis)
        render_dashboard(conv, profile)

    # Pinned two-channel input.
    turn = st.chat_input(f"Log what the {speaker_label.lower()} said…")
    if turn:
        handle_turn(turn, "patient" if speaker_label == "Patient" else "doctor")


# Main Navigation
if not check_authentication():
    st.stop()

if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chat":
    chat_page()

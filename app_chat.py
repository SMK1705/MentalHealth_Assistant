import asyncio
import streamlit as st
import logging
import uuid
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
        if password == settings.app_password:
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

# Hide default Streamlit chrome and apply light layout polish.
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1200px;}
[data-testid="stChatMessage"] {padding: 0.2rem 0;}
[data-testid="stMetric"] {background: var(--secondary-background-color, #F4F7F6);
    border-radius: 10px; padding: 10px 14px;}
</style>
""", unsafe_allow_html=True)

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
    "history_summary", "doctor_notes_input", "demo_mode",
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
    for k in ("latest_suggestions", "latest_analysis", "history_summary", "doctor_notes_input"):
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
    """Archive the transcript + session log unless this is an in-memory demo."""
    if st.session_state.get("demo_mode"):
        return
    try:
        archive_conversation(st.session_state.conversation_model)
    except Exception:
        logger.exception("Failed to archive conversation")
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
            status.update(label="Generating decision support…")
            suggestions = generate_session_suggestions(
                transcript=transcript,
                patient_summary=build_patient_summary(st.session_state.patient_profile),
                history_summary=st.session_state.history_summary,
                signals=_format_signals(analysis),
                examples=analysis.get("historical_examples"),
                doctor_questions=doctor_questions,
            )
            status.update(label="Decision support ready", state="complete", expanded=False)
    except Exception:
        logger.exception("Assistant generation error")

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
    st.session_state.conversation.append({"speaker": speaker, "content": content})
    st.session_state.conversation_model.add_message(
        Message(content=content, is_user=is_patient, speaker=speaker)
    )
    if is_patient:
        _run_assistant(content)
    _persist_session()
    st.rerun()


# Landing Page
def landing_page():
    st.markdown("""
        <div style='text-align:center; padding:16px;'>
            <h1>🩺 Live Session Assistant</h1>
            <p style='font-size:17px;'>
                A real-time co-pilot for mental-health clinicians. Enter a patient ID to load their
                history, then log the live session — the assistant suggests the patient's likely state,
                next questions to ask, follow-ups, and red flags.
                <br><br><b>Decision support only — not a diagnosis. The clinician stays in control.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    patient_id = st.text_input("Patient ID", key="patient_id_input")
    medical_history_input = st.text_area(
        "Clinical history (optional, one item per line)", key="medical_history_input"
    )
    therapy_goals_input = st.text_area(
        "Therapy goals (optional, one item per line)", key="therapy_goals_input"
    )

    if st.button("Open session"):
        patient_id = patient_id.strip()
        if not patient_id:
            st.error("Please enter a patient ID before starting.")
            return

        medical_history = _parse_lines(medical_history_input)
        therapy_goals = _parse_lines(therapy_goals_input)

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

    st.divider()
    st.markdown("#### Or try a live demo")
    st.caption("Explore the assistant with a sample patient — no patient ID or setup needed.")
    demo_cols = st.columns(len(DEMO_PERSONAS))
    for i, (name, persona) in enumerate(DEMO_PERSONAS.items()):
        if demo_cols[i].button(name, key=f"demo_persona_{i}", use_container_width=True):
            start_demo(persona)


# Chat Page
def chat_page():
    st.title("🩺 Live Session Assistant")
    st.caption("Decision support for the clinician — not a diagnosis. You stay in control.")

    ctrl_left, ctrl_right = st.columns([3, 1])
    if st.session_state.get("demo_mode"):
        ctrl_left.caption("Demo mode — sample patient, nothing is saved.")
    if ctrl_right.button("New session", use_container_width=True):
        reset_session()

    render_patient_overview(
        st.session_state.conversation_model.patient_id,
        st.session_state.get("patient_profile") or {},
        exclude_session_id=st.session_state.conversation_model.session_id,
    )
    st.divider()

    col_chat, col_dash = st.columns([1.3, 1], gap="large")

    with col_chat:
        st.markdown("##### Live transcript")
        for msg in st.session_state.conversation:
            speaker = msg.get("speaker") or ("patient" if msg.get("role") == "user" else "doctor")
            avatar = "🩺" if speaker == "doctor" else "🧑"
            with st.chat_message(speaker, avatar=avatar):
                st.markdown(f"**{speaker.capitalize()}:** {msg['content']}")

        if st.session_state.get("demo_mode"):
            st.caption("Quick demo patient turns")
            prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
            for i, prompt in enumerate(SUGGESTED_PROMPTS):
                if prompt_cols[i].button(prompt["label"], key=f"suggested_{i}", use_container_width=True):
                    handle_turn(prompt["text"], "patient")

    with col_dash:
        st.markdown("##### Session assistant")
        render_suggestions(
            st.session_state.get("latest_suggestions") or {},
            st.session_state.get("latest_analysis") or {},
        )
        with st.expander("Session metrics", expanded=False):
            render_dashboard(st.session_state.conversation, st.session_state.get("patient_profile") or {})

        st.text_area("Doctor notes", key="doctor_notes_input", height=100,
                     placeholder="Your observations, overrides, plan…")

        audit = st.session_state.get("session_suggestions") or []
        if audit:
            with st.expander(f"Assistant audit trail ({len(audit)})"):
                for i, entry in enumerate(audit, 1):
                    s = entry.get("suggestions") or {}
                    st.markdown(f"**Turn {i}** — _{(entry.get('turn') or '')[:60]}_")
                    if s.get("emotional_state"):
                        st.caption(f"state: {s['emotional_state']}")
                    if s.get("next_questions"):
                        st.caption("asked to consider: " + " | ".join(s["next_questions"][:3]))
                    if entry.get("safety"):
                        st.caption(f"safety: {entry['safety'].get('flag_type')}")

    # Two-channel input: choose the speaker, then log the turn.
    speaker_label = st.radio("Log turn as", ["Patient", "Doctor"], horizontal=True, key="speaker_select")
    turn = st.chat_input(f"Log what the {speaker_label.lower()} said…")
    if turn:
        handle_turn(turn, "patient" if speaker_label == "Patient" else "doctor")

    with st.expander("Generate end-of-session report"):
        if st.button("Generate report", key="exit_button"):
            with st.spinner("Generating the session report…"):
                try:
                    conversation_text = "\n".join(
                        f"{m.get('speaker', 'patient').capitalize()}: {m['content']}"
                        for m in st.session_state.conversation
                    )
                    classifier = load_topic_classifier()
                    predicted_topic, topic_confidence = predict_topic(conversation_text, classifier)
                    sentiment, sentiment_score = analyze_sentiment(conversation_text)
                    prompt = (
                        f"Patient Session Report:\n"
                        f"Topic: {predicted_topic} (Confidence: {topic_confidence:.2f})\n"
                        f"Overall Sentiment: {sentiment} (Score: {sentiment_score})\n\n"
                        "Provide a structured clinician-facing session summary with sections:\n"
                        "1. **Presentation & key themes**\n"
                        "2. **Areas of concern / risk**\n"
                        "3. **Suggested follow-up** (for the clinician to consider; not prescriptive).\n"
                        "Use clear headings and bullet points. Do not diagnose or prescribe."
                    )
                    recommendations_obj = generate_advice(prompt)
                    recommendations = (recommendations_obj if isinstance(recommendations_obj, str)
                                       else str(recommendations_obj))
                    st.divider()
                    st.subheader("📝 Session report")
                    st.markdown(f"**Predicted topic:** {predicted_topic} · **Confidence:** {topic_confidence:.2f}")
                    st.markdown(f"**Overall sentiment:** {sentiment} ({sentiment_score})")
                    st.markdown(recommendations)
                except Exception:
                    logger.exception("Error generating report")
                    st.error("An error occurred while generating the report. Please try again later.")


# Main Navigation
if not check_authentication():
    st.stop()

if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chat":
    chat_page()

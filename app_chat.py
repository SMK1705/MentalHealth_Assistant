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
from llm_rag import generate_advice, stream_advice
from patient_profile import get_patient_profile, create_patient_profile, update_patient_fields
from safety import SafetyChecker
from config import settings
from dashboard import render_dashboard
from explain import render_advice, render_why

# Ensure an event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# Stateless crisis screener for the guaranteed UI fallback (see chat_page).
_safety_checker = SafetyChecker()

if not settings.app_password:
    logger.warning("APP_PASSWORD is not set — the app is running without authentication.")


def check_authentication() -> bool:
    """Gate the app behind a shared password when settings.app_password is set.

    Returns True when access is allowed. When no password is configured access
    is allowed (a warning is logged at startup); otherwise the user must enter
    the matching password once per session.
    """
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
    page_title="Mental Health Counselor Guidance",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit chrome and apply light layout polish.
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1180px;}
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
        session_id=str(uuid.uuid4()),
        patient_id="test_patient",
        messages=[]
    )
if "patient_profile" not in st.session_state:
    st.session_state.patient_profile = {}
if "session_risk_flags" not in st.session_state:
    st.session_state.session_risk_flags = []
if "session_topics" not in st.session_state:
    st.session_state.session_topics = []

def _parse_lines(text):
    """Split a textarea value into a clean list of non-empty, stripped lines."""
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


# Seeded demo patients so investors can try the tool without setup.
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

# One-click messages that walk a distress -> crisis -> recovery arc in a demo.
SUGGESTED_PROMPTS = [
    {"label": "Share distress", "text": "I've been really anxious lately and I can't sleep at all"},
    {"label": "Express a crisis", "text": "Honestly, sometimes I feel like I want to end it all"},
    {"label": "Show improvement", "text": "Talking this through actually helps — thank you"},
]


def start_demo(persona):
    """Begin a self-serve demo with a seeded persona (in-memory, no DB writes)."""
    st.session_state.demo_mode = True
    st.session_state.patient_profile = {
        "patient_id": persona["patient_id"],
        "medical_history": list(persona["medical_history"]),
        "therapy_goals": list(persona["therapy_goals"]),
    }
    st.session_state.conversation = []
    st.session_state.conversation_model = Conversation(
        session_id=str(uuid.uuid4()), patient_id=persona["patient_id"], messages=[]
    )
    st.session_state.session_risk_flags = []
    st.session_state.session_topics = []
    st.session_state.page = "chat"
    st.rerun()


def reset_session():
    """Clear the session and return to the landing page."""
    for key in (
        "conversation", "conversation_model", "patient_profile",
        "session_risk_flags", "session_topics", "demo_mode", "page",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def handle_user_message(message):
    """Run one patient message through the guidance pipeline and update state.

    Shared by the chat input and the demo one-click prompts. Skips database
    archiving in demo mode (the demo runs purely in-memory).
    """
    st.session_state.conversation.append({"role": "user", "content": message})
    st.session_state.conversation_model.add_message(Message(content=message, is_user=True))

    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.conversation
    )

    assistant_reply = "I'm sorry, something went wrong."
    analytics = {"safety_protocol": _safety_checker.check_input(message)}

    try:
        with st.status("Analyzing the patient message…", expanded=True) as status:
            analysis = analyze_message(message, st.session_state.patient_profile, conversation_text)

            sp = analysis.get("safety_protocol")
            st.write(f"Safety: {sp['action'] + ' — ' + sp['flag_type'] if sp else 'no crisis indicators'}")
            urgency = analysis.get("urgency") or {}
            st.write(
                f"Emotion: {urgency.get('label') or 'neutral'} · "
                f"Topic: {analysis.get('predicted_topic') or 'n/a'} · "
                f"Sentiment: {analysis.get('sentiment') or 'n/a'}"
            )
            st.write(f"Retrieved {len(analysis.get('historical_examples') or [])} similar case(s).")

            status.update(label="Generating guidance…")
            assistant_reply = st.write_stream(
                stream_advice(analysis.get("analysis_context", ""), analysis.get("historical_examples"))
            ) or "No advice available."
            status.update(label="Guidance ready", state="complete", expanded=False)

        analytics = {
            "topic": analysis.get("predicted_topic"),
            "topic_confidence": analysis.get("topic_confidence"),
            "sentiment": analysis.get("sentiment"),
            "sentiment_score": analysis.get("sentiment_score"),
            "safety_protocol": analysis.get("safety_protocol"),
            "urgency": analysis.get("urgency"),
            # Kept in session state only (not archived) for the "why this guidance" panel.
            "historical_examples": analysis.get("historical_examples"),
            "analysis_context": analysis.get("analysis_context"),
        }
    except Exception:
        logger.exception("Guidance generation error")
        analytics = {"safety_protocol": _safety_checker.check_input(message)}

    st.session_state.conversation.append({
        "role": "assistant",
        "content": assistant_reply,
        "analysis": analytics,
    })
    # Archived metadata stays lean (no retrieved examples / raw context).
    lean_metadata = {k: analytics.get(k) for k in
                     ("topic", "topic_confidence", "sentiment", "sentiment_score", "safety_protocol", "urgency")}
    st.session_state.conversation_model.add_message(
        Message(content=assistant_reply, is_user=False, metadata=lean_metadata)
    )

    # Roll up distinct risk flags + topics for the session.
    protocol = analytics.get("safety_protocol")
    flag_type = protocol.get("flag_type") if protocol else None
    if flag_type and flag_type not in st.session_state.session_risk_flags:
        st.session_state.session_risk_flags.append(flag_type)
    topic = analytics.get("topic")
    if topic and topic not in st.session_state.session_topics:
        st.session_state.session_topics.append(topic)

    # Persist to the database unless this is an in-memory demo session.
    if not st.session_state.get("demo_mode"):
        archive_conversation(st.session_state.conversation_model)
        try:
            archive_session(SessionLog(
                session_id=st.session_state.conversation_model.session_id,
                patient_id=st.session_state.conversation_model.patient_id,
                detected_topics=st.session_state.session_topics,
                risk_flags=st.session_state.session_risk_flags,
                sentiment_score=float(analytics.get("sentiment_score") or 0.0),
            ))
        except Exception:
            logger.exception("Failed to archive session log")

    st.rerun()


# Landing Page
def landing_page():
    st.markdown("""
        <div style='text-align:center; padding:20px;'>
            <h1>Welcome to the Mental Health Counselor Guidance System</h1>
            <p style='font-size:18px;'>
                This AI-driven application provides tailored counseling advice based on patient conversations.
                <br><br><b>Capabilities:</b>
                <br>- Personalized recommendations
                <br>- Conversation analysis (topics & sentiment)
                <br>- Comprehensive reporting
                <br><br><b>Limitations:</b>
                <br>- Advisory only; professional judgment required
                <br>- Use anonymized data only
                <br>- Operations may be resource-intensive
                <br><br><b>Instructions:</b>
                <br>1. Click "Get Started" to begin chatting.
                <br>2. Interact naturally with the assistant.
                <br>3. Generate detailed reports when finished.
            </p>
        </div>
        """, unsafe_allow_html=True)

    patient_id = st.text_input("Patient ID", key="patient_id_input")
    medical_history_input = st.text_area(
        "Medical history (optional, one item per line)", key="medical_history_input"
    )
    therapy_goals_input = st.text_area(
        "Therapy goals (optional, one item per line)", key="therapy_goals_input"
    )

    if st.button("Get Started"):
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
                    patient_id,
                    medical_history=medical_history,
                    therapy_goals=therapy_goals,
                )
            elif (medical_history and not profile.medical_history) or (
                therapy_goals and not profile.therapy_goals
            ):
                # Enrich a previously-empty profile with the entered details.
                profile = update_patient_fields(
                    patient_id,
                    medical_history=medical_history or profile.medical_history,
                    therapy_goals=therapy_goals or profile.therapy_goals,
                )
        except Exception:
            logger.exception("Failed to load or create patient profile")
            st.error("Could not load the patient profile. Please try again later.")
            return

        st.session_state.patient_profile = profile.dict()
        st.session_state.conversation_model.patient_id = patient_id
        st.session_state.page = "chat"
        st.rerun()

    st.divider()
    st.markdown("#### Or try a live demo")
    st.caption("Explore the tool with a sample patient — no patient ID or setup needed.")
    demo_cols = st.columns(len(DEMO_PERSONAS))
    for i, (name, persona) in enumerate(DEMO_PERSONAS.items()):
        if demo_cols[i].button(name, key=f"demo_persona_{i}", use_container_width=True):
            start_demo(persona)

# Chat Page
def chat_page():
    st.title("🧠 Mental Health Counselor Guidance")

    ctrl_left, ctrl_right = st.columns([3, 1])
    if st.session_state.get("demo_mode"):
        ctrl_left.caption("Demo mode — sample patient, nothing is saved.")
    if ctrl_right.button("New session", use_container_width=True):
        reset_session()

    col_chat, col_dash = st.columns([1.4, 1], gap="large")

    with col_chat:
        # Display messages freshly each time using Streamlit's native chat components
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    render_advice(msg["content"])
                    analysis = msg.get("analysis")
                    if analysis:
                        st.caption(
                            f"Topic: {analysis.get('topic')} (Confidence: {analysis.get('topic_confidence', 0.0):.2f}) | "
                            f"Sentiment: {analysis.get('sentiment')} ({analysis.get('sentiment_score')})"
                        )

                        safety_protocol = analysis.get("safety_protocol")
                        if safety_protocol:
                            action = safety_protocol.get("action", "URGENT")
                            banner = (
                                f"⚠️ Crisis indicator detected ({action}). "
                                f"Suggested protocol: {safety_protocol.get('response', '')}"
                            )
                            if action == "CRITICAL":
                                st.error(banner)
                            else:
                                st.warning(banner)

                        # Only show the urgency banner when there is no crisis banner
                        # already (the crisis banner takes priority and conveys urgency).
                        urgency = analysis.get("urgency")
                        if urgency and urgency.get("is_urgent") and not safety_protocol:
                            score = urgency.get("score")
                            score_text = f" ({score:.2f})" if isinstance(score, (int, float)) else ""
                            st.warning(
                                f"Elevated emotional urgency: {urgency.get('label')}{score_text}"
                            )

                        render_why(analysis)

    with col_dash:
        render_dashboard(
            st.session_state.conversation,
            st.session_state.get("patient_profile") or {},
        )

    # Demo: one-click messages that walk a distress -> crisis -> recovery arc.
    if st.session_state.get("demo_mode"):
        st.caption("Quick demo messages")
        prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
        for i, prompt in enumerate(SUGGESTED_PROMPTS):
            if prompt_cols[i].button(prompt["label"], key=f"suggested_{i}", use_container_width=True):
                handle_user_message(prompt["text"])

    # Chat input
    user_message = st.chat_input("Type your message here...")
    if user_message:
        handle_user_message(user_message)

    # Report generation section with spinner
    if st.button("Exit Conversation and Show Report", key="exit_button", help="Generate detailed patient report"):
        with st.spinner("Generating your detailed patient report..."):
            try:
                conversation_text = "\n".join(
                    f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
                )
                
                # Predict primary topic and confidence
                classifier = load_topic_classifier()
                predicted_topic, topic_confidence = predict_topic(conversation_text, classifier)
                
                # Evaluate overall sentiment
                sentiment, sentiment_score = analyze_sentiment(conversation_text)
                
                # Prepare a structured prompt for generating actionable recommendations
                prompt = (
                    f"Patient Report Analysis:\n"
                    f"Topic: {predicted_topic} (Confidence: {topic_confidence:.2f})\n"
                    f"Overall Sentiment: {sentiment} (Score: {sentiment_score})\n\n"
                    "Please provide a well-structured patient report with the following sections:\n"
                    "1. **Strengths & Positive Aspects**: Highlight any positive observations.\n"
                    "2. **Areas of Concern**: Identify aspects that require further attention.\n"
                    "3. **Actionable Recommendations**: Offer clear, step-by-step advice for addressing challenges.\n"
                    "Ensure that the report is organized with clear headings and bullet points."
                )
                
                recommendations_obj = generate_advice(prompt)
                if isinstance(recommendations_obj, list) and hasattr(recommendations_obj[0], "content"):
                    recommendations = recommendations_obj[0].content
                else:
                    recommendations = str(recommendations_obj)
                
                st.divider()
                st.subheader("📝 Patient Report")
                st.markdown(f"**Predicted Topic:** {predicted_topic}  \n**Confidence:** {topic_confidence:.2f}")
                st.markdown(f"**Overall Sentiment:** {sentiment}  \n**Sentiment Score:** {sentiment_score}")
                st.markdown("### Recommendations")
                st.markdown(recommendations)
                
            except Exception as e:
                logger.exception("Error generating report")
                st.error("An error occurred while generating the patient report. Please try again later.")

# Main Navigation
if not check_authentication():
    st.stop()

if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chat":
    chat_page()

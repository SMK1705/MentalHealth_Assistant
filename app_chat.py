import asyncio
import streamlit as st
import logging
import uuid
from logging_config import setup_logging
from unified_guidance import generate_counselor_guidance
from archiver import archive_conversation, archive_session
from schemas import Conversation, Message, SessionLog
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import analyze_sentiment
from llm_rag import generate_advice
from patient_profile import get_patient_profile, create_patient_profile, update_patient_fields
from safety import SafetyChecker

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

# Page configuration
st.set_page_config(
    page_title="Mental Health Counselor Guidance",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit menus and headers
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
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

# Chat Page
def chat_page():
    st.title("🧠 Mental Health Counselor Guidance")

    # Show the loaded patient profile so the personalization context is visible.
    profile = st.session_state.get("patient_profile") or {}
    medical_history = profile.get("medical_history") or []
    therapy_goals = profile.get("therapy_goals") or []
    if medical_history or therapy_goals:
        with st.expander("Patient profile", expanded=False):
            if medical_history:
                st.markdown("**Medical history**")
                for item in medical_history:
                    st.markdown(f"- {item}")
            if therapy_goals:
                st.markdown("**Therapy goals**")
                for item in therapy_goals:
                    st.markdown(f"- {item}")

    # Display messages freshly each time using Streamlit's native chat components
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
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

                    urgency = analysis.get("urgency")
                    if urgency and urgency.get("is_urgent"):
                        score = urgency.get("score")
                        score_text = f" ({score:.2f})" if isinstance(score, (int, float)) else ""
                        st.warning(
                            f"Elevated emotional urgency: {urgency.get('label')}{score_text}"
                        )

    # Chat input with spinner for each conversation message generation
    user_message = st.chat_input("Type your message here...")
    if user_message:
        # Append user message
        st.session_state.conversation.append({"role": "user", "content": user_message})
        st.session_state.conversation_model.add_message(Message(content=user_message, is_user=True))

        # Build conversation context
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
        )

        with st.spinner("Processing your message..."):
            try:
                guidance = generate_counselor_guidance(
                    user_message,
                    st.session_state.patient_profile,
                    conversation_text,
                )
                assistant_reply = guidance.get("generated_advice", "No advice available.")
                analytics = {
                    "topic": guidance.get("predicted_topic"),
                    "topic_confidence": guidance.get("topic_confidence"),
                    "sentiment": guidance.get("sentiment"),
                    "sentiment_score": guidance.get("sentiment_score"),
                    "safety_protocol": guidance.get("safety_protocol"),
                    "urgency": guidance.get("urgency"),
                }
            except Exception as e:
                logger.exception("Guidance generation error")
                assistant_reply = "I'm sorry, something went wrong."
                # Even on total failure, never drop the crisis screen for the
                # latest patient message — re-run the cheap regex directly.
                analytics = {"safety_protocol": _safety_checker.check_input(user_message)}

        # Append assistant's reply along with analytics
        st.session_state.conversation.append({
            "role": "assistant",
            "content": assistant_reply,
            "analysis": analytics,
        })
        st.session_state.conversation_model.add_message(
            Message(content=assistant_reply, is_user=False, metadata=analytics)
        )

        # Archive conversation
        archive_conversation(st.session_state.conversation_model)

        # Roll up session-level analytics (distinct risk flags + topics) and
        # persist a SessionLog so crisis events are captured even if the
        # counselor never generates a report. Best-effort: never break chat.
        protocol = analytics.get("safety_protocol")
        flag_type = protocol.get("flag_type") if protocol else None
        if flag_type and flag_type not in st.session_state.session_risk_flags:
            st.session_state.session_risk_flags.append(flag_type)
        topic = analytics.get("topic")
        if topic and topic not in st.session_state.session_topics:
            st.session_state.session_topics.append(topic)

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

        # Refresh display to show new message
        st.rerun()

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
                st.write(recommendations)
                
            except Exception as e:
                logger.exception("Error generating report")
                st.error("An error occurred while generating the patient report. Please try again later.")

# Main Navigation
if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chat":
    chat_page()

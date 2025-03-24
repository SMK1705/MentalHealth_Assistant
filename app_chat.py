import asyncio
import streamlit as st
import logging
import uuid
from logging_config import setup_logging
from unified_guidance import generate_counselor_guidance
from archiver import archive_conversation
from schemas import Conversation, Message
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import simple_sentiment_analysis
from llm_rag import generate_advice

# Ensure an event loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

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

    if st.button("Get Started"):
        st.session_state.page = "chat"
        st.rerun()

# Chat Page
def chat_page():
    st.title("🧠 Mental Health Counselor Guidance")

    # Display messages freshly each time
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Chat input
    user_message = st.chat_input("Type your message here...")
    if user_message:
        # User message append
        st.session_state.conversation.append({"role": "user", "content": user_message})
        st.session_state.conversation_model.add_message(Message(content=user_message, is_user=True))

        # Conversation context
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
        )

        try:
            # Generate reply
            guidance = generate_counselor_guidance(user_message, conversation_text)
            assistant_reply = guidance.get("generated_advice", "No advice available.")
        except Exception as e:
            logger.exception("Guidance generation error")
            assistant_reply = "I'm sorry, something went wrong."

        # Assistant message append
        st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
        st.session_state.conversation_model.add_message(Message(content=assistant_reply, is_user=False))

        # Archive conversation
        archive_conversation(st.session_state.conversation_model)

        # Refresh to display new message
        st.rerun()

    # Report button
    if st.button("Exit Conversation and Show Report", key="exit_button", help="Generate detailed patient report"):
        with st.spinner("Generating your detailed patient report..."):
            try:
                # Compile full conversation text for analysis
                conversation_text = "\n".join(
                    f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
                )
                
                # Predict primary topic and confidence
                classifier = load_topic_classifier()
                predicted_topic, topic_confidence = predict_topic(conversation_text, classifier)
                
                # Evaluate overall sentiment of the conversation
                sentiment_score = simple_sentiment_analysis(conversation_text)
                sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
                
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
                
                # Generate recommendations using LLM-based advice generation
                recommendations_obj = generate_advice(prompt)
                if isinstance(recommendations_obj, list) and hasattr(recommendations_obj[0], "content"):
                    recommendations = recommendations_obj[0].content
                else:
                    recommendations = str(recommendations_obj)
                
                # Display the report in a structured format
                st.divider()
                st.subheader("📝 Patient Report")
                st.markdown(f"**Predicted Topic:** {predicted_topic}  \n**Confidence:** {topic_confidence:.2f}")
                st.markdown(f"**Overall Sentiment:** {sentiment}  \n**Sentiment Score:** {sentiment_score}")
                st.markdown("### Recommendations")
                st.write(recommendations)
                
            except Exception as e:
                logger.exception("Error generating report")
                st.error("An error occurred while generating the patient report. Please try again later.")

# Page Navigation
if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "chat":
    chat_page()

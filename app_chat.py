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

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize session state for conversation and conversation model
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of messages: each dict with keys "role" and "content"
if "conversation_model" not in st.session_state:
    st.session_state.conversation_model = Conversation(
        session_id=str(uuid.uuid4()),
        patient_id="test_patient",  # Replace with a dynamic patient ID if available
        messages=[]
    )

st.title("Mental Health Counselor Chatbot")
st.markdown("### Chat with your assistant. Your conversation is maintained below.")

# Create an empty placeholder for the chat history
chat_placeholder = st.empty()

def display_chat():
    # Use custom markdown styling for a chat-like display
    st.markdown("### Conversation History")
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f"<p style='color:blue'><strong>User:</strong> {msg['content']}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='color:green'><strong>Assistant:</strong> {msg['content']}</p>",
                unsafe_allow_html=True,
            )
    st.markdown("---")

# Function to generate a patient report based on the conversation text
def generate_patient_report(conversation_text):
    # Use topic classifier to get predicted topic and confidence
    classifier = load_topic_classifier()
    predicted_topic, topic_confidence = predict_topic(conversation_text, classifier)
    
    # Perform sentiment analysis on the conversation text
    sentiment_score = simple_sentiment_analysis(conversation_text)
    sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    
    # Generate recommendations using your LLM-based function with a tailored prompt
    prompt = (
        f"Patient Report Analysis:\n"
        f"The patient's conversation indicates a topic of '{predicted_topic}' with a confidence of {topic_confidence:.2f}. "
        f"The overall sentiment is {sentiment} (score: {sentiment_score}). "
        "Provide detailed recommendations on steps the patient should take to overcome these challenges."
    )
    recommendations_obj = generate_advice(prompt)
    # Extract text from the response if it's a list, otherwise convert to string
    if isinstance(recommendations_obj, list) and hasattr(recommendations_obj[0], "content"):
        recommendations = recommendations_obj[0].content
    else:
        recommendations = str(recommendations_obj)
    
    report = {
        "predicted_topic": predicted_topic,
        "topic_confidence": topic_confidence,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "recommendations": recommendations,
    }
    return report

# Display the current conversation in the chat placeholder
with chat_placeholder.container():
    display_chat()

# Chat input form at the bottom: single text input field
with st.form(key="chat_form", clear_on_submit=True):
    user_message = st.text_input("Type your message here...", key="user_input")
    send_submitted = st.form_submit_button("Send")

if send_submitted and user_message:
    try:
        # Append the user's message to session state and conversation model
        st.session_state.conversation.append({"role": "user", "content": user_message})
        st.session_state.conversation_model.add_message(Message(content=user_message, is_user=True))
        
        # Build a combined conversation text from session state for context
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
        )
        
        # Generate guidance using your existing chain logic with memory
        guidance = generate_counselor_guidance(user_message, conversation_text)
        assistant_reply = guidance.get("generated_advice", "No advice generated.")
        
        # Append the assistant's reply to session state and conversation model
        st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})
        st.session_state.conversation_model.add_message(Message(content=assistant_reply, is_user=False))
        
        # Archive the conversation to the database
        archive_conversation(st.session_state.conversation_model)
        
        # Update the chat placeholder display
        chat_placeholder.empty()
        with chat_placeholder.container():
            display_chat()
            
    except Exception as e:
        logger.exception("Error during chat processing")
        st.error(f"An error occurred: {e}")

# Exit Conversation button to generate and display the patient report
exit_clicked = st.button("Exit Conversation and Show Report")
if exit_clicked:
    try:
        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.conversation
        )
        with st.spinner("Generating patient report, please wait..."):
            report = generate_patient_report(conversation_text)
        st.markdown("## Patient Report")
        st.markdown(f"**Predicted Topic:** {report['predicted_topic']} (Confidence: {report['topic_confidence']:.2f})")
        st.markdown(f"**Overall Sentiment:** {report['sentiment']} (Score: {report['sentiment_score']})")
        st.markdown("### Recommendations")
        st.write(report['recommendations'])
    except Exception as e:
        logger.exception("Error generating report")
        st.error(f"An error occurred while generating the report: {e}")


import logging
from semantic_search import semantic_search
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import simple_sentiment_analysis
from llm_rag import generate_advice

logger = logging.getLogger(__name__)

def generate_counselor_guidance(
    user_input: str,
    patient_profile: dict | None = None,
    conversation_history: str = "",
):
    """Generate guidance for counselors based on the latest patient message.

    Args:
        user_input: Latest message from the patient.
        patient_profile: Attributes describing the patient.
        conversation_history: Prior conversation transcript.

    Returns:
        Dictionary with generated advice and analytics for the counselor.
    """

    logger.debug("Generating counselor guidance. User input: %s", user_input)

    # Topic classification and sentiment analysis for the latest message
    classifier = load_topic_classifier()
    predicted_topic, topic_score = predict_topic(user_input, classifier)
    logger.debug("Predicted topic: %s with score: %s", predicted_topic, topic_score)

    sentiment_score = simple_sentiment_analysis(user_input)
    sentiment = (
        "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    )
    logger.debug("Sentiment score: %s (%s)", sentiment_score, sentiment)

    # Prepare context for advice generation incorporating profile and analytics
    profile_text = "".join(
        f"{k}: {v}\n" for k, v in patient_profile.items()
    ) if patient_profile else ""

    analysis_context = (
        f"Patient Profile:\n{profile_text}\n"
        f"Conversation History:\n{conversation_history}\n"
        f"Latest Message: {user_input}\n"
        f"Predicted Topic: {predicted_topic} (Confidence: {topic_score})\n"
        f"Sentiment: {sentiment} (Score: {sentiment_score})"
    )

    examples = semantic_search(user_input, top_k=3)
    logger.debug("Retrieved %d historical examples.", len(examples))

    advice_obj = generate_advice(analysis_context)
    
    if isinstance(advice_obj, list):
        advice_text = advice_obj[0].content if hasattr(advice_obj[0], "content") else str(advice_obj[0])
    elif hasattr(advice_obj, "content"):
        advice_text = advice_obj.content
    else:
        advice_text = str(advice_obj)
    
    logger.debug("Generated advice: %s", advice_text)
    
    guidance = {
        "generated_advice": advice_text,
        "predicted_topic": predicted_topic,
        "topic_confidence": topic_score,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "historical_examples": examples,
        "patient_profile": patient_profile or {},
    }
    return guidance

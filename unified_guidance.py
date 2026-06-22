import logging
from semantic_search import semantic_search
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import simple_sentiment_analysis
from llm_rag import generate_advice
from safety import SafetyChecker
from urgency_detector import load_urgency_detector, detect_urgency

logger = logging.getLogger(__name__)

# Stateless crisis screener reused across calls.
_safety_checker = SafetyChecker()

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

    # Crisis-safety screen first — a cheap regex that must never be lost to a
    # later (LLM/model) failure, so it is computed before any heavy work.
    safety_protocol = _safety_checker.check_input(user_input)
    if safety_protocol:
        logger.warning(
            "Safety protocol triggered (%s) for latest message.",
            safety_protocol.get("action"),
        )

    # Seed with safe, format-friendly defaults so a downstream failure still
    # returns the safety signal and renders without errors.
    guidance = {
        "generated_advice": "I'm sorry, something went wrong.",
        "predicted_topic": None,
        "topic_confidence": 0.0,
        "sentiment": None,
        "sentiment_score": 0,
        "historical_examples": [],
        "patient_profile": patient_profile or {},
        "safety_protocol": safety_protocol,
        "urgency": {"is_urgent": False, "label": None, "score": None},
    }

    # Best-effort emotional-urgency detection (heavy model; degrade gracefully).
    try:
        is_urgent, urgency_label, urgency_score = detect_urgency(
            user_input, load_urgency_detector()
        )
        guidance["urgency"] = {
            "is_urgent": is_urgent,
            "label": urgency_label,
            "score": urgency_score,
        }
    except Exception:
        logger.exception("Urgency detection failed; defaulting to not urgent.")

    try:
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

        guidance.update({
            "generated_advice": advice_text,
            "predicted_topic": predicted_topic,
            "topic_confidence": topic_score,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "historical_examples": examples,
        })
    except Exception:
        logger.exception("Counselor guidance generation failed; returning safety signal only.")

    return guidance

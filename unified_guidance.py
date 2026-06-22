import logging
from semantic_search import semantic_search
from topic_classifier import predict_topic, load_topic_classifier
from patient_ml import analyze_sentiment
from llm_rag import generate_advice
from safety import SafetyChecker
from urgency_detector import load_urgency_detector, detect_urgency

logger = logging.getLogger(__name__)

# Stateless crisis screener reused across calls.
_safety_checker = SafetyChecker()


def analyze_message(
    user_input: str,
    patient_profile: dict | None = None,
    conversation_history: str = "",
):
    """Run the non-LLM analysis for a message and assemble the LLM context.

    Returns a dict with safety, urgency, topic, sentiment, the retrieved
    ``historical_examples`` and the assembled ``analysis_context`` — but NOT the
    generated advice, so the advice can be streamed separately. Resilient: a
    downstream failure still returns the safety signal.
    """
    logger.debug("Analyzing message: %s", user_input)

    # Crisis-safety screen first — a cheap regex that must never be lost to a
    # later (model) failure, so it is computed before any heavy work.
    safety_protocol = _safety_checker.check_input(user_input)
    if safety_protocol:
        logger.warning(
            "Safety protocol triggered (%s) for latest message.",
            safety_protocol.get("action"),
        )

    # Seed with safe, format-friendly defaults so a downstream failure still
    # returns the safety signal and renders without errors.
    result = {
        "predicted_topic": None,
        "topic_confidence": 0.0,
        "sentiment": None,
        "sentiment_score": 0,
        "historical_examples": [],
        "patient_profile": patient_profile or {},
        "safety_protocol": safety_protocol,
        "urgency": {"is_urgent": False, "label": None, "score": None},
        "analysis_context": "",
        # Names of pipeline stages that degraded, so callers/UI can say so
        # instead of presenting a partial result as a clean one.
        "errors": [],
    }

    # Best-effort emotional-urgency detection (heavy model; degrade gracefully).
    try:
        is_urgent, urgency_label, urgency_score = detect_urgency(
            user_input, load_urgency_detector()
        )
        result["urgency"] = {
            "is_urgent": is_urgent,
            "label": urgency_label,
            "score": urgency_score,
        }
    except Exception:
        logger.exception("Urgency detection failed; defaulting to not urgent.")
        result["errors"].append("urgency")

    # A detected safety crisis is authoritative: always treat it as urgent, even
    # when the emotion model (which is not a crisis detector) did not flag it.
    if safety_protocol and not result["urgency"]["is_urgent"]:
        result["urgency"] = {
            "is_urgent": True,
            "label": result["urgency"]["label"] or "crisis",
            "score": result["urgency"]["score"] if result["urgency"]["score"] is not None else 1.0,
        }

    # Local topic + sentiment models. Isolated from retrieval below so a failure
    # in one does not discard the other's result.
    try:
        classifier = load_topic_classifier()
        predicted_topic, topic_score = predict_topic(user_input, classifier)
        logger.debug("Predicted topic: %s with score: %s", predicted_topic, topic_score)

        sentiment, sentiment_score = analyze_sentiment(user_input)
        logger.debug("Sentiment score: %s (%s)", sentiment_score, sentiment)

        result.update({
            "predicted_topic": predicted_topic,
            "topic_confidence": topic_score,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
        })
    except Exception:
        logger.exception("Topic/sentiment analysis failed.")
        result["errors"].append("analysis")

    # Always assemble the LLM context from whatever signals we have (using the
    # seeded defaults when a stage degraded), so downstream generation never
    # silently runs on an empty context.
    profile_text = "".join(
        f"{k}: {v}\n" for k, v in patient_profile.items()
    ) if patient_profile else ""
    result["analysis_context"] = (
        f"Patient Profile:\n{profile_text}\n"
        f"Conversation History:\n{conversation_history}\n"
        f"Latest Message: {user_input}\n"
        f"Predicted Topic: {result['predicted_topic']} (Confidence: {result['topic_confidence']})\n"
        f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']})"
    )

    # Retrieval (Pinecone + Mongo) isolated in its own block: a RAG outage must
    # not throw away the topic/sentiment/context computed above.
    try:
        examples = semantic_search(user_input, top_k=3)
        logger.debug("Retrieved %d historical examples.", len(examples))
        result["historical_examples"] = examples
    except Exception:
        logger.exception("Semantic retrieval failed; continuing without examples.")
        result["errors"].append("retrieval")

    return result


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

    guidance = analyze_message(user_input, patient_profile, conversation_history)
    guidance["generated_advice"] = "I'm sorry, something went wrong."

    try:
        advice_obj = generate_advice(
            guidance["analysis_context"], examples=guidance["historical_examples"]
        )
        if isinstance(advice_obj, list):
            advice_text = advice_obj[0].content if hasattr(advice_obj[0], "content") else str(advice_obj[0])
        elif hasattr(advice_obj, "content"):
            advice_text = advice_obj.content
        else:
            advice_text = str(advice_obj)
        guidance["generated_advice"] = advice_text
    except Exception:
        logger.exception("Advice generation failed.")

    return guidance

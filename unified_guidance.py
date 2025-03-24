import logging
from semantic_search import semantic_search
from topic_classifier import predict_topic, load_topic_classifier
from llm_rag import generate_advice

logger = logging.getLogger(__name__)

def generate_counselor_guidance(user_input: str, conversation_history: str = ""):
    logger.debug("Generating counselor guidance. User input: %s", user_input)
    combined_input = conversation_history + "\n" + user_input if conversation_history else user_input
    examples = semantic_search(user_input, top_k=3)
    logger.debug("Retrieved %d historical examples.", len(examples))
    
    classifier = load_topic_classifier()
    predicted_topic, topic_score = predict_topic(user_input, classifier)
    logger.debug("Predicted topic: %s with score: %s", predicted_topic, topic_score)
    
    advice_obj = generate_advice(combined_input)
    
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
        "historical_examples": examples
    }
    return guidance

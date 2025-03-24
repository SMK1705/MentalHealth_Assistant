import logging

logger = logging.getLogger(__name__)

def simple_sentiment_analysis(text):
    positive_words = {"happy", "good", "great", "positive", "joy", "love"}
    negative_words = {"sad", "bad", "terrible", "negative", "hate", "depressed"}
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in positive_words)
    neg_count = sum(text_lower.count(word) for word in negative_words)
    return pos_count - neg_count

def train_patient_ml_model(patient_id: str):
    from patient_profile import get_patient_conversation
    conv = get_patient_conversation(patient_id)
    if not conv:
        logger.warning("No conversation found for patient %s", patient_id)
        return None
    patient_texts = [msg.get("content", "") for msg in conv.get("messages", []) if msg.get("is_user")]
    if not patient_texts:
        logger.warning("No patient messages found for sentiment analysis for patient %s", patient_id)
        return None
    full_text = " ".join(patient_texts)
    score = simple_sentiment_analysis(full_text)
    if score > 0:
        sentiment = "positive"
    elif score < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    logger.info("Patient %s sentiment: %s (score: %d)", patient_id, sentiment, score)
    return sentiment

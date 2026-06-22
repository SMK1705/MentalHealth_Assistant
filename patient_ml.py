import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# 3-class sentiment model with an explicit Neutral class — replaces binary
# SST-2, which over-confidently mislabeled neutral/ambiguous text.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def simple_sentiment_analysis(text):
    """Lightweight word-count heuristic, used as a fallback when the
    transformer sentiment model is unavailable."""
    positive_words = {"happy", "good", "great", "positive", "joy", "love"}
    negative_words = {"sad", "bad", "terrible", "negative", "hate", "depressed"}
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in positive_words)
    neg_count = sum(text_lower.count(word) for word in negative_words)
    return pos_count - neg_count


@lru_cache(maxsize=1)
def load_sentiment_model():
    """Return a cached 3-class transformers sentiment pipeline."""
    from transformers import pipeline
    model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    logger.debug("Sentiment model loaded: %s", SENTIMENT_MODEL)
    return model


def _normalize_sentiment_label(raw: str) -> str:
    """Map model label variants (positive/negative/neutral or LABEL_0/1/2)."""
    r = raw.lower()
    if "pos" in r or r == "label_2":
        return "positive"
    if "neg" in r or r == "label_0":
        return "negative"
    return "neutral"


def analyze_sentiment(text: str):
    """Classify the sentiment of ``text``.

    Returns a ``(label, score)`` tuple where label is one of
    ``"Positive"``/``"Negative"``/``"Neutral"`` and score is a signed
    confidence in ``[-1.0, 1.0]`` (0.0 for Neutral). Falls back to the
    word-count heuristic if the transformer model cannot be loaded or run.
    """
    if not text or not text.strip():
        return "Neutral", 0.0
    try:
        classifier = load_sentiment_model()
        # Truncate to roughly the model's max sequence length to avoid errors.
        result = classifier(text[:512])[0]
        label = _normalize_sentiment_label(result["label"])
        confidence = float(result["score"])
        if label == "positive":
            return "Positive", confidence
        if label == "negative":
            return "Negative", -confidence
        return "Neutral", 0.0
    except Exception:
        logger.exception("Sentiment model failed; using word-count fallback.")
        score = simple_sentiment_analysis(text)
        label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        return label, float(score)


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
    sentiment, score = analyze_sentiment(full_text)
    logger.info("Patient %s sentiment: %s (score: %s)", patient_id, sentiment.lower(), score)
    return sentiment.lower()

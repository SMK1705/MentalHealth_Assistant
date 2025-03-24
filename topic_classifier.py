import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

CANDIDATE_LABELS = [
    "addiction",
    "anger-management",
    "anxiety",
    "behavioral-change",
    "children-adolescents",
    "counseling-fundamentals",
    "depression",
    "diagnosis",
    "domestic-violence",
    "eating-disorders",
    "family-conflict",
    "grief-and-loss",
    "human-sexuality",
    "intimacy",
    "legal-regulatory",
    "lgbtq",
    "marriage",
    "military-issues",
    "parenting",
    "professional-ethics",
    "relationship-dissolution",
    "relationships",
    "self-esteem",
    "self-harm",
    "sleep-improvement",
    "social-relationships",
    "spirituality",
    "stress",
    "substance-abuse",
    "trauma",
    "workplace-relationships"
]

def load_topic_classifier():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.debug("Topic classifier loaded.")
    return classifier

def predict_topic(text: str, classifier=None):
    if classifier is None:
        classifier = load_topic_classifier()
    result = classifier(text, candidate_labels=CANDIDATE_LABELS)
    logger.debug("Topic classification result: %s", result)
    predicted_label = result["labels"][0]
    score = result["scores"][0]
    return predicted_label, score

if __name__ == "__main__":
    classifier = load_topic_classifier()
    text = "I have been feeling very anxious and stressed lately, and I worry about my sleep."
    topic, confidence = predict_topic(text, classifier)
    logger.info("Predicted Topic: %s", topic)
    logger.info("Confidence: %s", confidence)

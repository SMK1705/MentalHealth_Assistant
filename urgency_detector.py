import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

def load_urgency_detector():
    detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    logger.debug("Urgency detector loaded.")
    return detector

def detect_urgency(text: str, detector=None, threshold=0.7):
    if detector is None:
        detector = load_urgency_detector()
    predictions = detector(text)
    logger.debug("Urgency detector predictions: %s", predictions)
    urgent_emotions = {"anger", "fear", "sadness"}
    for pred in predictions[0]:
        if pred['label'].lower() in urgent_emotions and pred['score'] > threshold:
            logger.info("Urgency detected: %s with score %s", pred['label'], pred['score'])
            return True, pred['label'], pred['score']
    return False, None, None

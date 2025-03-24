import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

def load_response_classifier():
    # In a real system, fine-tune on your response type labeled data.
    classifier = pipeline("text-classification", model="bert-base-uncased", return_all_scores=True)
    logger.debug("Response classifier loaded.")
    return classifier

def predict_response_type(text: str, classifier=None):
    if classifier is None:
        classifier = load_response_classifier()
    prediction = classifier(text)
    logger.debug("Response classification prediction: %s", prediction)
    if prediction and isinstance(prediction, list):
        top_pred = max(prediction[0], key=lambda x: x['score'])
        return top_pred['label'], top_pred['score']
    return None, None

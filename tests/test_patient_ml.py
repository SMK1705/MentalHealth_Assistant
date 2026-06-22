import patient_ml
from patient_ml import simple_sentiment_analysis, analyze_sentiment, _normalize_sentiment_label


def test_normalize_sentiment_label():
    assert _normalize_sentiment_label("positive") == "positive"
    assert _normalize_sentiment_label("LABEL_2") == "positive"
    assert _normalize_sentiment_label("negative") == "negative"
    assert _normalize_sentiment_label("LABEL_0") == "negative"
    assert _normalize_sentiment_label("neutral") == "neutral"
    assert _normalize_sentiment_label("LABEL_1") == "neutral"


def test_simple_sentiment_counts():
    assert simple_sentiment_analysis("I am happy and full of joy") > 0
    assert simple_sentiment_analysis("I feel sad and depressed") < 0
    assert simple_sentiment_analysis("the meeting is at noon") == 0


def test_analyze_sentiment_empty_is_neutral():
    assert analyze_sentiment("") == ("Neutral", 0.0)
    assert analyze_sentiment("   ") == ("Neutral", 0.0)


def test_analyze_sentiment_falls_back_when_model_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(patient_ml, "load_sentiment_model", boom)

    label, score = analyze_sentiment("I am so happy and full of joy")
    assert label == "Positive"
    assert score > 0

    label, score = analyze_sentiment("I feel sad and depressed")
    assert label == "Negative"
    assert score < 0

import importlib.util

import pytest

from explain import parse_advice


def test_parse_advice_structured():
    text = "Advice: Do X.\nRationale: Because Y.\nSuggested Actions:\n1. A\n2. B"
    p = parse_advice(text)
    assert p["advice"] == "Do X."
    assert p["rationale"] == "Because Y."
    assert "1. A" in p["actions"] and "2. B" in p["actions"]


def test_parse_advice_bold_labels():
    text = "**Advice:** Do X.\n\n**Rationale:** Y\n\n**Suggested Actions:**\n1. A"
    p = parse_advice(text)
    assert p["advice"] == "Do X."
    assert p["rationale"] == "Y"
    assert "1. A" in p["actions"]


def test_parse_advice_unstructured_fallback():
    text = "Just plain advice with no sections."
    p = parse_advice(text)
    assert p["advice"] == text
    assert p["rationale"] == "" and p["actions"] == ""


def test_parse_advice_empty():
    assert parse_advice("") == {"advice": "", "rationale": "", "actions": ""}
    assert parse_advice(None) == {"advice": "", "rationale": "", "actions": ""}


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
def test_topic_classifier_is_lru_cached():
    # Regression: load_topic_classifier must be cached so the 1.6GB model is
    # not reloaded per message. @lru_cache adds a cache_info attribute.
    from topic_classifier import load_topic_classifier
    assert hasattr(load_topic_classifier, "cache_info")

from safety import SafetyChecker


def test_suicide_risk_detected():
    result = SafetyChecker().check_input("I want to kill myself")
    assert result is not None
    assert result["action"] == "CRITICAL"
    assert result["flag_type"] == "suicide_risk"


def test_violence_self_harm_fires():
    # Regression: previously returned None because PROTOCOLS had no
    # violence_risk entry, silently dropping self-harm alerts.
    result = SafetyChecker().check_input("I want to harm myself")
    assert result is not None
    assert result["action"] == "CRITICAL"
    assert result["flag_type"] == "violence_risk"


def test_abuse_disclosure_detected():
    result = SafetyChecker().check_input("there was abuse at home")
    assert result is not None
    assert result["action"] == "URGENT"
    assert result["flag_type"] == "abuse_disclosure"


def test_inflections_detected():
    for text in [
        "I feel suicidal",
        "she was raped",
        "I keep cutting myself",
        "there is no reason to live",
    ]:
        assert SafetyChecker().check_input(text) is not None, text


def test_self_harm_gerund_forms_detected():
    # Regression (P1): inflected self-harm phrasing must flag a CRITICAL crisis.
    # "harming myself" / "hurting myself" previously returned None.
    checker = SafetyChecker()
    for text in ["I've started thinking about harming myself", "I keep hurting myself"]:
        r = checker.check_input(text)
        assert r is not None and r["action"] == "CRITICAL" and r["flag_type"] == "violence_risk", text


def test_benign_text_not_flagged():
    for text in [
        "I had a good day today",
        "I love hiking and grapes",
        "we discussed scheduling next week",
    ]:
        assert SafetyChecker().check_input(text) is None, text

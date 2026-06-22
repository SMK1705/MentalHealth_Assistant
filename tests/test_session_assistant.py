from session_assistant import parse_suggestions
from patient_overview import build_patient_summary, build_history_summary


def test_parse_valid_json():
    d = parse_suggestions(
        '{"emotional_state":"anxious","state_confidence":"Medium",'
        '"next_questions":["q1","q2"],"red_flags":["a - b"],'
        '"missing_info":[],"follow_ups":["f"],"caveat":"c"}'
    )
    assert d["emotional_state"] == "anxious"
    assert d["state_confidence"] == "medium"  # normalized to lowercase
    assert d["next_questions"] == ["q1", "q2"]
    assert d["red_flags"] == ["a - b"]


def test_parse_code_fenced_and_string_to_list():
    d = parse_suggestions('```json\n{"emotional_state":"sad","next_questions":"just one"}\n```')
    assert d["emotional_state"] == "sad"
    assert d["next_questions"] == ["just one"]


def test_parse_prose_wrapped():
    d = parse_suggestions('Sure, here it is: {"emotional_state":"ok","red_flags":["a"]} hope it helps')
    assert d["emotional_state"] == "ok"
    assert d["red_flags"] == ["a"]


def test_parse_malformed_fallback_keeps_raw():
    d = parse_suggestions("not json at all")
    assert d["_raw"] == "not json at all"
    assert d["next_questions"] == [] and d["emotional_state"] == ""


def test_parse_empty():
    d = parse_suggestions("")
    assert d["_raw"] == "" and d["next_questions"] == []


def test_build_patient_summary():
    s = build_patient_summary({"patient_id": "P1", "medical_history": ["GAD"], "therapy_goals": ["sleep"]})
    assert "patient_id: P1" in s and "GAD" in s and "sleep" in s


def test_build_history_summary():
    assert build_history_summary([]) == "(no prior sessions)"
    s = build_history_summary([
        {"created_at": "2026-06-01T00:00:00", "detected_topics": ["anxiety"], "risk_flags": ["suicide_risk"]}
    ])
    assert "2026-06-01" in s and "anxiety" in s and "suicide_risk" in s

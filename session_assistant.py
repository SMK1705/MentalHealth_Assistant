import json
import logging

logger = logging.getLogger(__name__)

_EMPTY = {
    "emotional_state": "",
    "state_confidence": "low",
    "next_questions": [],
    "follow_ups": [],
    "missing_info": [],
    "red_flags": [],
    "caveat": "",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _extract_json(text):
    """Pull a JSON object out of a model reply, tolerating code fences/prose."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_suggestions(text):
    """Parse the model's JSON decision-support into a normalized dict.

    On unparseable output, returns the empty shape plus ``_raw`` so the raw
    text can still be surfaced (nothing is silently dropped).
    """
    data = _extract_json(text)
    if not isinstance(data, dict):
        out = dict(_EMPTY)
        out["_raw"] = (text or "").strip()
        return out
    return {
        "emotional_state": str(data.get("emotional_state") or "").strip(),
        "state_confidence": str(data.get("state_confidence") or "low").strip().lower(),
        "next_questions": _as_list(data.get("next_questions")),
        "follow_ups": _as_list(data.get("follow_ups")),
        "missing_info": _as_list(data.get("missing_info")),
        "red_flags": _as_list(data.get("red_flags")),
        "caveat": str(data.get("caveat") or "").strip(),
    }


def generate_session_suggestions(transcript, patient_summary, history_summary,
                                 signals, examples, doctor_questions):
    """Call the LLM for decision-support and return parsed suggestions.

    The deterministic crisis flag is NOT produced here — it comes from
    analyze_message's safety_protocol and is rendered first by render_suggestions.
    """
    from langchain.schema import HumanMessage
    from model_cache import get_chat_groq
    from prompt_templates import SESSION_ASSISTANT_TEMPLATE

    cases_text = ""
    for ex in (examples or [])[:3]:
        cases_text += (
            f"- Patient: {(ex.get('questionText') or '')[:200]}\n"
            f"  Therapist: {(ex.get('answerText') or '')[:200]}\n"
        )
    prompt = SESSION_ASSISTANT_TEMPLATE.format(
        patient_summary=patient_summary or "(none)",
        history_summary=history_summary or "(no prior sessions)",
        signals=signals or "(none)",
        retrieved_cases=cases_text or "(none)",
        doctor_questions=doctor_questions or "(none yet)",
        transcript=transcript or "",
    )
    try:
        response = get_chat_groq().invoke([HumanMessage(content=prompt)])
        text = response.content if hasattr(response, "content") else str(response)
        return parse_suggestions(text)
    except Exception:
        logger.exception("Session suggestion generation failed.")
        # Flag the failure so the UI can distinguish "the model errored" from
        # "the model ran and had nothing high-value to add".
        out = dict(_EMPTY)
        out["_error"] = "generation_failed"
        return out


def render_suggestions(data, analysis):
    """Render the cockpit decision-support panel + the grounding ('why') toggle.

    The deterministic crisis banner is rendered separately (above the
    transcript) by the app; here we show the LLM aids and degraded-state notes.
    """
    import streamlit as st
    import ui
    from explain import render_why

    data = data or {}
    analysis = analysis or {}

    st.markdown(
        ui.decision_support(
            state=data.get("emotional_state"),
            confidence=data.get("state_confidence"),
            red_flags=data.get("red_flags") or [],
            missing=data.get("missing_info") or [],
            questions=data.get("next_questions") or [],
            follow_ups=data.get("follow_ups") or [],
            caveat=data.get("caveat"),
            error=bool(data.get("_error")),
            degraded=analysis.get("errors") or [],
        ),
        unsafe_allow_html=True,
    )

    if data.get("_raw"):
        with st.expander("Raw assistant output (could not parse as JSON)"):
            st.code(data["_raw"])

    render_why(analysis)

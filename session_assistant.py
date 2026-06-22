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
        return dict(_EMPTY)


_CONF_LABEL = {
    "high": "high confidence",
    "medium": "medium confidence",
    "low": "low confidence — treat as a hint",
}


def render_suggestions(data, analysis):
    """Render the assistant panel: deterministic safety first, then LLM aids."""
    import streamlit as st
    from explain import render_why

    data = data or {}
    analysis = analysis or {}

    # 1. Deterministic safety — never depends on the LLM.
    sp = analysis.get("safety_protocol")
    if sp:
        st.error(
            f"Safety — {sp.get('action')}: {sp.get('flag_type')}. {sp.get('response', '')}"
        )

    red = data.get("red_flags") or []
    if red:
        st.markdown("**Red flags / concerns**")
        for item in red:
            st.markdown(f"- {item}")

    missing = data.get("missing_info") or []
    if missing:
        st.markdown("**Missing info to clarify**")
        for item in missing:
            st.markdown(f"- {item}")

    state = data.get("emotional_state")
    if state:
        conf = _CONF_LABEL.get((data.get("state_confidence") or "low"), _CONF_LABEL["low"])
        st.markdown(f"**Likely state** — {state}")
        st.caption(conf)

    questions = data.get("next_questions") or []
    if questions:
        st.markdown("**Next questions to consider**")
        for q in questions:
            st.markdown(f"- {q}")

    follow_ups = data.get("follow_ups") or []
    if follow_ups:
        st.markdown("**Follow-up points**")
        for item in follow_ups:
            st.markdown(f"- {item}")

    if data.get("caveat"):
        st.caption(f"Note: {data['caveat']}")
    if data.get("_raw"):
        with st.expander("Raw assistant output (could not parse as JSON)"):
            st.code(data["_raw"])

    if not any(data.get(k) for k in ("red_flags", "missing_info", "emotional_state",
                                     "next_questions", "follow_ups")) and not sp:
        st.caption("No high-value suggestions for this turn.")

    st.caption("Decision support — not a diagnosis. Clinical judgment required.")
    render_why(analysis)

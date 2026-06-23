"""Streamlit entry point — the Live Session Cockpit prototype WIRED to the real backend.

The pixel-perfect design (``cockpit_component/``) is mounted as a bidirectional
Streamlit component. Typed PATIENT turns are routed to the real Python pipeline —
deterministic crisis screen (safety.py), emotion/urgency (DistilRoBERTa),
sentiment (RoBERTa), topic (BART zero-shot), semantic retrieval (MiniLM +
Pinecone over the Mongo corpus), and Groq decision-support — and the chips,
decision-support panel, RAG "why" panel, and crisis banner reflect REAL outputs.

The scripted "Advance" button keeps the canned demo arc (PT-0042). Heavy ML
imports are lazy, so the launch screen appears instantly; the first analyzed turn
pays the model-load cost.

The earlier Streamlit-native UI is preserved in ``app_live.py``.
"""
import hmac
import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from config import settings
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Live Session Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Full-bleed: hide Streamlit chrome and stretch the component iframe to the viewport.
st.markdown(
    """
<style>
#MainMenu, header, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] { background: #eef1f6; }
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe { display: block; border: none; width: 100%; height: 100vh !important; }
[data-testid="stElementContainer"]:has(iframe),
[data-testid="element-container"]:has(iframe) { height: 100vh !important; }
</style>
""",
    unsafe_allow_html=True,
)

if not settings.app_password:
    logger.warning("APP_PASSWORD is not set — the app is running without authentication.")


def check_authentication() -> bool:
    """Gate the app behind a shared password when settings.app_password is set."""
    if not settings.app_password:
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("🔒 Sign in")
    password = st.text_input("Password", type="password", key="auth_password")
    if st.button("Sign in"):
        if hmac.compare_digest(password, settings.app_password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_authentication():
    st.stop()

# Bidirectional component wrapping the prototype.
_COMPONENT_DIR = Path(__file__).parent / "cockpit_component"
_lsa_cockpit = components.declare_component("lsa_cockpit", path=str(_COMPONENT_DIR))

# Demo patient profile — matches the cockpit's PT-0042 overview card so the real
# pipeline's grounding is consistent with what the clinician sees.
_DEMO_PROFILE = {
    "patient_id": "PT-0042",
    "medical_history": ["generalized anxiety disorder", "panic attacks"],
    "therapy_goals": ["reduce daily anxiety", "manage panic episodes"],
}


def _fmt_score(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0.00"
    return ("+%.2f" % n) if n >= 0 else ("%.2f" % n)


def _run_real_turn(text: str, transcript: str) -> dict:
    """Run the real analysis + decision-support pipeline for one patient turn and
    shape it into the payload the prototype's React component expects."""
    from unified_guidance import analyze_message
    from session_assistant import generate_session_suggestions
    from patient_overview import build_patient_summary

    analysis = analyze_message(text, _DEMO_PROFILE, transcript)
    urgency = analysis.get("urgency") or {}
    sp = analysis.get("safety_protocol")

    emotion = urgency.get("label") or "neutral"
    e_score = urgency.get("score") or 0.0
    sentiment = (analysis.get("sentiment") or "neutral").lower()
    s_score = analysis.get("sentiment_score") or 0.0
    topic = analysis.get("predicted_topic") or "general"
    t_conf = analysis.get("topic_confidence") or 0.0

    signals = (
        f"topic: {topic} ({t_conf}); sentiment: {sentiment} ({s_score}); "
        f"emotion: {emotion} ({e_score}); "
        f"crisis flag: {(sp.get('flag_type') + '/' + sp.get('action')) if sp else 'none'}"
    )
    suggestions = generate_session_suggestions(
        transcript=transcript,
        patient_summary=build_patient_summary(_DEMO_PROFILE),
        history_summary="(no prior sessions in this demo)",
        signals=signals,
        examples=analysis.get("historical_examples"),
        doctor_questions="(see transcript)",
    )

    why_signals = [
        {"k": "Emotion (DistilRoBERTa)", "v": f"{emotion} · {round(float(e_score) * 100)}%"},
        {"k": "Sentiment (RoBERTa)", "v": f"{sentiment} · {_fmt_score(s_score)}"},
        {"k": "Topic (BART zero-shot)", "v": f"{topic} · {round(float(t_conf) * 100)}%"},
        {"k": "Crisis screen", "v": (f"{sp['flag_type']} · {sp['action']}") if sp else "clear"},
    ]
    cases = [
        {
            "id": str(ex.get("questionID") or ex.get("id") or "—"),
            "sim": "",
            "q": (ex.get("questionText") or "")[:200],
            "a": (ex.get("answerText") or "")[:200],
        }
        for ex in (analysis.get("historical_examples") or [])[:3]
    ]

    return {
        "analysis": {
            "emotion": emotion,
            "emotionScore": float(e_score),
            "sentiment": sentiment,
            "sentimentScore": float(s_score),
            "topic": topic,
            "topicConf": float(t_conf),
        },
        "suggestions": {
            k: suggestions.get(k)
            for k in ("emotional_state", "state_confidence", "red_flags",
                      "missing_info", "next_questions", "follow_ups", "caveat")
        },
        "why": {"signals": why_signals, "cases": cases,
                "context": analysis.get("analysis_context") or ""},
        "crisis": ({"flag_type": sp["flag_type"], "action": sp["action"], "response": sp["response"]}
                   if sp else None),
        "errors": analysis.get("errors") or [],
    }


_ERROR_RESULT = {
    "analysis": {"emotion": "neutral", "emotionScore": 0.0, "sentiment": "neutral",
                 "sentimentScore": 0.0, "topic": "general", "topicConf": 0.0},
    "suggestions": {"emotional_state": "", "state_confidence": "low", "red_flags": [],
                    "missing_info": [], "next_questions": [], "follow_ups": [],
                    "caveat": "The analysis pipeline errored for this turn."},
    "why": {"signals": [], "cases": [], "context": ""},
    "crisis": None,
}

# Render the component, passing the most recent result back to the iframe.
payload = _lsa_cockpit(result=st.session_state.get("lsa_result"), key="lsa_cockpit", default=None)

# A new patient turn arrived from the iframe — run the real pipeline and push it back.
if isinstance(payload, dict) and payload.get("kind") == "patient_turn":
    nonce = payload.get("nonce")
    if nonce is not None and nonce != st.session_state.get("lsa_handled_nonce"):
        st.session_state["lsa_handled_nonce"] = nonce
        try:
            result = _run_real_turn(payload.get("text", ""), payload.get("transcript", ""))
        except Exception:
            logger.exception("Real pipeline failed for a patient turn")
            result = dict(_ERROR_RESULT)
        result["nonce"] = nonce
        st.session_state["lsa_result"] = result
        st.rerun()

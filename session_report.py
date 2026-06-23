"""End-of-session clinician report.

A Groq-generated, structured clinician-facing summary grounded in the live
session transcript, the signals accumulated this session (topics + risk flags),
the patient record, and prior sessions. Decision support — not a diagnosis.
Read-only: it never writes to the database.
"""
import logging

logger = logging.getLogger(__name__)

_MAX_TRANSCRIPT = 8000


def generate_session_report(active: dict, transcript: str, notes: str) -> str:
    """Return a markdown end-of-session summary for the active patient session."""
    from langchain.schema import HumanMessage
    from model_cache import get_chat_groq
    from prompt_templates import REPORT_TEMPLATE
    from patient_profile import get_patient_sessions
    from patient_overview import build_patient_summary, build_history_summary

    active = active or {}
    profile = active.get("profile") or {}
    pid = active.get("patient_id")
    cur_session_id = active.get("session_id")

    try:
        sessions = get_patient_sessions(pid) if pid else []
    except Exception:
        logger.exception("Failed to load sessions for the report")
        sessions = []
    prior = [s for s in sessions if s.get("session_id") != cur_session_id]

    t = transcript or "(no turns logged)"
    if len(t) > _MAX_TRANSCRIPT:
        t = t[:2000] + "\n…[earlier turns omitted]…\n" + t[-5000:]

    prompt = REPORT_TEMPLATE.format(
        patient_summary=build_patient_summary(profile),
        history_summary=build_history_summary(prior, limit=5),
        topics=", ".join(active.get("topics") or []) or "—",
        risk_flags=", ".join(active.get("risk_flags") or []) or "none",
        notes=(notes or "").strip() or "(none)",
        transcript=t,
    )

    try:
        resp = get_chat_groq().invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return (text or "").strip() or "No summary could be generated for this session."
    except Exception:
        logger.exception("Session report generation failed")
        return ("The report could not be generated right now (the language model was "
                "unreachable). Please try again.")

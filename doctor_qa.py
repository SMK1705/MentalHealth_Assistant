"""Doctor-facing query assistant.

Answers a clinician's free-form question during a live session, grounded in the
patient record, prior sessions (incl. their archived transcripts, excluding
today's), the live session so far + accumulated signals, and the RAG knowledge
corpus. One general LLM-with-context engine — no per-question hardcoding.

Read-only: it never writes to the database. Returns a concise markdown answer.
"""
import logging

logger = logging.getLogger(__name__)

_MAX_LIVE_CHARS = 6000
_MAX_PRIOR_MSGS = 30


def _prior_transcript(pid: str, session_id: str) -> str:
    """The archived transcript of the patient's most recent prior session."""
    from patient_profile import get_patient_conversations
    try:
        convs = get_patient_conversations(pid)
    except Exception:
        logger.exception("Failed to load prior conversations")
        return "(transcript unavailable)"
    conv = next((c for c in convs if c.get("session_id") == session_id), None)
    if not conv:
        return "(transcript for the previous session is not archived)"
    lines = []
    for m in (conv.get("messages") or [])[-_MAX_PRIOR_MSGS:]:
        speaker = m.get("speaker") or ("patient" if m.get("is_user") else "doctor")
        lines.append(f"{speaker.capitalize()}: {(m.get('content') or '')[:300]}")
    return "\n".join(lines) or "(empty transcript)"


def _retrieved_cases(question: str) -> str:
    """Top corpus cases for the question (graceful if Pinecone is unavailable)."""
    from semantic_search import semantic_search
    try:
        cases = semantic_search(question, top_k=3)
    except Exception:
        logger.exception("Corpus retrieval failed")
        return "(knowledge base unavailable)"
    if not cases:
        return "(no closely related cases found)"
    return "\n".join(
        f"- [corpus #{ex.get('questionID')}] Q: {(ex.get('questionText') or '')[:200]} "
        f"| A: {(ex.get('answerText') or '')[:300]}"
        for ex in cases
    )


def _clip_live(transcript: str) -> str:
    t = transcript or "(no turns logged yet)"
    if len(t) <= _MAX_LIVE_CHARS:
        return t
    return t[:1500] + "\n…[earlier turns omitted]…\n" + t[-4000:]


def answer_doctor_query(question: str, active: dict, transcript: str) -> str:
    """Answer the clinician's question grounded in the patient's data + corpus.

    ``active`` is ``st.session_state['lsa_active']`` =
    ``{patient_id, session_id, profile(dict), risk_flags, topics}``.
    """
    from langchain.schema import HumanMessage
    from model_cache import get_chat_groq
    from prompt_templates import DOCTOR_QA_TEMPLATE
    from patient_profile import get_patient_sessions
    from patient_overview import build_patient_summary, build_history_summary, _fmt_date, _days_ago

    pid = active.get("patient_id")
    profile = active.get("profile") or {}
    cur_session_id = active.get("session_id")

    # Prior sessions, excluding today's in-progress session.
    try:
        sessions = get_patient_sessions(pid)
    except Exception:
        logger.exception("Failed to load patient sessions")
        sessions = []
    prior = [s for s in sessions if s.get("session_id") != cur_session_id]

    last = prior[0] if prior else None
    if last:
        ago = _days_ago(last.get("created_at"))
        date = _fmt_date(last.get("created_at")) + (f" ({ago})" if ago else "")
        topics = ", ".join(last.get("detected_topics") or []) or "—"
        flags = ", ".join(last.get("risk_flags") or []) or "none"
        notes = (last.get("doctor_notes") or "").strip() or "(none)"
        last_detail = (
            f"date: {date}\ntopics: {topics}\nrisk flags: {flags}\n"
            f"sentiment score: {last.get('sentiment_score')}\nclinician notes: {notes}"
        )
        last_transcript = _prior_transcript(pid, last.get("session_id"))
    else:
        last_detail = "(no previous session on record)"
        last_transcript = "(no previous session on record)"

    session_signals = (
        f"topics so far: {', '.join(active.get('topics') or []) or '—'}\n"
        f"risk flags so far: {', '.join(active.get('risk_flags') or []) or 'none'}"
    )

    prompt = DOCTOR_QA_TEMPLATE.format(
        question=question,
        patient_summary=build_patient_summary(profile),
        history_summary=build_history_summary(prior, limit=5),
        last_session_detail=last_detail,
        last_session_transcript=last_transcript,
        live_transcript=_clip_live(transcript),
        session_signals=session_signals,
        retrieved_cases=_retrieved_cases(question),
    )

    try:
        resp = get_chat_groq().invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return (text or "").strip() or "I couldn't find an answer in the available record."
    except Exception:
        logger.exception("Doctor query answering failed")
        return ("The assistant is temporarily unavailable (the language model could not be "
                "reached). Please try again.")

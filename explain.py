import re

# Matches the section labels the advice prompt asks for, tolerating markdown
# bold (**Advice:**) and leading whitespace.
_LABEL_RE = re.compile(
    r"^[ \t]*\**[ \t]*(Advice|Rationale|Suggested Actions|Actions)[ \t]*:[ \t]*\**[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def parse_advice(text):
    """Split advice text into {advice, rationale, actions}.

    Falls back to putting the whole text in 'advice' when the structure
    (Advice / Rationale / Suggested Actions) is not present.
    """
    text = (text or "").strip()
    if not text:
        return {"advice": "", "rationale": "", "actions": ""}

    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        return {"advice": text, "rationale": "", "actions": ""}

    out = {"advice": "", "rationale": "", "actions": ""}
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        key = "actions" if key in ("suggested actions", "actions") else key
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[key] = text[start:end].strip()

    # Any text before the first labeled section is part of the advice.
    lead = text[: matches[0].start()].strip()
    if lead:
        out["advice"] = (lead + ("\n" + out["advice"] if out["advice"] else "")).strip()
    return out


def render_advice(advice_text):
    """Render advice as structured sections (Advice / Why / Suggested actions)."""
    import streamlit as st
    parts = parse_advice(advice_text)
    if parts["advice"]:
        st.markdown(parts["advice"])
    if parts["rationale"]:
        st.markdown(f"**Why:** {parts['rationale']}")
    if parts["actions"]:
        st.markdown("**Suggested actions**")
        st.markdown(parts["actions"])
    if not any(parts.values()):
        st.markdown(advice_text or "")


def render_why(analysis):
    """Per-message 'Why this guidance' expander: the signals, retrieved cases,
    and the exact context sent to the model."""
    import streamlit as st
    if not analysis:
        return
    with st.expander("Why this guidance"):
        st.markdown("**Signals that informed this**")
        rows = []
        topic, tconf = analysis.get("topic"), analysis.get("topic_confidence")
        if topic:
            note = " · weak hint" if isinstance(tconf, (int, float)) and tconf < 0.4 else ""
            conf = f" · {tconf:.0%} confidence{note}" if isinstance(tconf, (int, float)) else ""
            rows.append(f"- Topic: `{topic}`{conf}")
        sentiment, sscore = analysis.get("sentiment"), analysis.get("sentiment_score")
        if sentiment:
            score = f" ({sscore:+.2f})" if isinstance(sscore, (int, float)) else ""
            rows.append(f"- Sentiment: {sentiment}{score}")
        urgency = analysis.get("urgency") or {}
        if urgency.get("label"):
            score = f" ({urgency['score']:.0%})" if isinstance(urgency.get("score"), (int, float)) else ""
            rows.append(f"- Emotion: {urgency['label']}{score}")
        sp = analysis.get("safety_protocol")
        if sp:
            rows.append(f"- Safety: **{sp.get('action')}** — {sp.get('flag_type')}")
        st.markdown("\n".join(rows) if rows else "_No signals recorded._")

        examples = analysis.get("historical_examples") or []
        st.markdown(f"**Similar past cases retrieved ({len(examples)})**")
        if examples:
            for ex in examples[:3]:
                q = (ex.get("questionText") or "")[:200]
                a = (ex.get("answerText") or "")[:200]
                st.markdown(f"- *Patient:* {q}\n\n  *Therapist:* {a}")
        else:
            st.caption("None retrieved (semantic search returned no matches).")

        context = analysis.get("analysis_context")
        if context:
            st.markdown("**Context sent to the model**")
            st.code(context)

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
    """'Why this guidance' grounding panel: the driving signals, retrieved
    cases, and the exact context sent to the model — in the cockpit style."""
    import streamlit as st
    import ui
    if not analysis:
        return

    signals = []
    topic, tconf = analysis.get("topic"), analysis.get("topic_confidence")
    if topic:
        conf = f" · {tconf:.0%}" if isinstance(tconf, (int, float)) else ""
        signals.append(("Topic (BART zero-shot)", f"{topic}{conf}"))
    sentiment, sscore = analysis.get("sentiment"), analysis.get("sentiment_score")
    if sentiment:
        score = f" · {sscore:+.2f}" if isinstance(sscore, (int, float)) else ""
        signals.append(("Sentiment (RoBERTa)", f"{sentiment}{score}"))
    urgency = analysis.get("urgency") or {}
    if urgency.get("label"):
        score = f" · {urgency['score']:.0%}" if isinstance(urgency.get("score"), (int, float)) else ""
        signals.append(("Emotion (DistilRoBERTa)", f"{urgency['label']}{score}"))
    sp = analysis.get("safety_protocol")
    signals.append(("Crisis screen", f"{sp.get('flag_type')} · {sp.get('action')}" if sp else "clear"))

    cases = []
    for ex in (analysis.get("historical_examples") or [])[:3]:
        cases.append({
            "id": ex.get("questionID") or ex.get("id") or "—",
            "sim": "",
            "q": (ex.get("questionText") or "")[:200],
            "a": (ex.get("answerText") or "")[:200],
        })

    with st.expander("Why this guidance — grounding"):
        st.markdown(
            ui.why_panel(signals, cases, analysis.get("analysis_context") or "(no context assembled)"),
            unsafe_allow_html=True,
        )

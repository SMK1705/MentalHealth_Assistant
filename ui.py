"""Cockpit UI theme + presentational HTML builders.

Pure string builders (no Streamlit import) so the markup can be previewed
standalone. The Streamlit render_* functions call these and pass the result to
st.markdown(..., unsafe_allow_html=True). Every builder returns a single-line
HTML string (no embedded newlines) so Streamlit's markdown pass never treats
indented lines as a code block.
"""
from __future__ import annotations
import html as _html

# ---- design tokens (ported from the Live Session Cockpit mockup) ----
ACCENT = "#3563d4"
ACCENT_SOFT = "rgba(53,99,212,.1)"
BG = "#eef1f6"
INK = "#1a2235"
MUTED = "#8a93a8"
BORDER = "#dde3ee"
CRISIS = "#c2362f"
GREEN = "#2f8f6b"
MONO = "'IBM Plex Mono',ui-monospace,monospace"


def esc(value) -> str:
    """HTML-escape arbitrary (possibly user-entered) content."""
    return _html.escape("" if value is None else str(value))


def _pre(value) -> str:
    """Escape and keep the string single-line (newlines -> entities)."""
    return esc(value).replace("\n", "&#10;")


# ---------------------------------------------------------------- global CSS
def css() -> str:
    """The global <style> block: fonts, chrome hiding, theming, widgets."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

:root { --accent:#3563d4; --ink:#1a2235; --muted:#8a93a8; --border:#dde3ee; }

.stApp, [data-testid="stAppViewContainer"] { background:#eef1f6; }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family:'Public Sans',system-ui,-apple-system,sans-serif; color:#1a2235;
}
.block-container { padding:1.1rem 1.4rem 7rem; max-width:1560px; }
[data-testid="stVerticalBlock"] { gap:.7rem; }
[data-testid="stHorizontalBlock"] { gap:.85rem; }
hr { margin:.5rem 0; border-color:#dde3ee; }

/* section label */
.lsa-h { display:flex; align-items:center; gap:8px; margin-bottom:13px; }
.lsa-h .bar { width:3px; height:13px; border-radius:2px; background:var(--accent); }
.lsa-h .t { font:600 10.5px/1 'IBM Plex Mono',monospace; letter-spacing:.07em; color:#8a93a8; }
.lsa-card { background:#fff; border:1px solid #dde3ee; border-radius:11px; padding:16px; }

/* buttons */
.stButton>button, .stFormSubmitButton>button, [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  border-radius:8px; font-weight:600; font-family:inherit; font-size:13.5px;
  transition:filter .15s ease, background .15s ease; box-shadow:none;
}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"],
.stFormSubmitButton>button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
  background:#3563d4 !important; color:#fff !important; border:none !important;
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover { filter:brightness(1.06); }
.stButton>button[kind="secondary"], [data-testid="stBaseButton-secondary"] {
  background:#fff !important; color:#3a4458 !important; border:1px solid #dde3ee !important;
}
.stButton>button[kind="secondary"]:hover { border-color:#3563d4 !important; color:#28304a !important; }

/* inputs */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {
  border:1px solid #dde3ee !important; border-radius:9px !important; background:#fbfcfe !important;
  color:#1a2235 !important; font-family:inherit; font-size:13.5px;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
  border-color:#3563d4 !important; box-shadow:0 0 0 2px rgba(53,99,212,.15) !important;
}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="base-input"] {
  background:#fbfcfe !important; border-radius:9px !important;
}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  font:600 10.5px/1 'IBM Plex Mono',monospace !important; letter-spacing:.05em; color:#8a93a8 !important;
  text-transform:uppercase;
}

/* radio as a segmented toggle */
[data-testid="stRadio"] [role="radiogroup"] {
  flex-direction:row; gap:4px; background:#eef1f6; padding:3px; border-radius:8px;
  width:max-content;
}
[data-testid="stRadio"] [role="radiogroup"] label {
  margin:0; padding:6px 14px; border-radius:6px; cursor:pointer;
  font:600 12px/1 'Public Sans',sans-serif; color:#7a849c;
}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { background:#3563d4; color:#fff; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none; }

/* chat input pinned at bottom */
[data-testid="stChatInput"] { border-radius:9px; }
[data-testid="stBottomBlockContainer"] { background:#eef1f6; }

/* expander = light card */
[data-testid="stExpander"] { border:1px solid #dde3ee; border-radius:11px; background:#fff; }
[data-testid="stExpander"] summary { font:600 10.5px/1 'IBM Plex Mono',monospace; letter-spacing:.05em; color:#8a93a8; }

/* alerts */
[data-testid="stAlert"] { border-radius:9px; }

/* scrollbars */
.lsa-scroll::-webkit-scrollbar { width:9px; }
.lsa-scroll::-webkit-scrollbar-thumb { background:#d3dae8; border-radius:6px; border:2px solid #fff; }
.lsa-scroll::-webkit-scrollbar-track { background:transparent; }

@keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
@keyframes spin { to { transform:rotate(360deg); } }
@keyframes softpulse { 0%,100% { opacity:1; } 50% { opacity:.45; } }
@keyframes crisisPulse { 0%,100% { box-shadow:0 0 0 0 rgba(194,54,47,.28); } 50% { box-shadow:0 0 0 5px rgba(194,54,47,0); } }
</style>
"""


def _h(label: str) -> str:
    return (f'<div class="lsa-h"><span class="bar"></span>'
            f'<span class="t">{esc(label)}</span></div>')


# --------------------------------------------------------------- launch hero
def launch_hero() -> str:
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;text-align:center;margin:8px 0 22px;">'
        '<div style="width:46px;height:46px;border-radius:13px;background:#3563d4;display:flex;align-items:center;justify-content:center;">'
        '<div style="width:19px;height:19px;border:2.6px solid #fff;border-radius:50%;border-right-color:transparent;"></div></div>'
        '<h1 style="font-weight:700;font-size:27px;letter-spacing:-.025em;margin:16px 0 0;">Live Session Assistant</h1>'
        f'<div style="font:600 11px/1 {MONO};letter-spacing:.16em;color:#8a93a8;margin-top:9px;">CLINICAL DECISION SUPPORT</div>'
        '<p style="font-size:14px;line-height:1.6;color:#5e6a82;max-width:460px;margin:14px 0 0;">A real-time co-pilot for '
        'mental-health sessions. Open a patient to load their history and session context, then get grounded, in-the-moment '
        'decision support.</p>'
        '<div style="display:inline-flex;align-items:center;gap:8px;margin-top:16px;padding:7px 13px;border:1px solid #f0d9d6;background:#fdf4f3;border-radius:20px;">'
        '<span style="width:7px;height:7px;border-radius:50%;background:#c2362f;"></span>'
        '<span style="font-size:12px;font-weight:600;color:#9c453f;">Decision support — not a diagnosis</span></div></div>'
    )


def launch_footer() -> str:
    return (
        '<div style="text-align:center;margin-top:18px;font-size:11.5px;color:#9aa3b6;line-height:1.55;">'
        f'Access is gated by a shared password when <span style="font-family:{MONO};">APP_PASSWORD</span> is configured. '
        'Use de-identified patient data only.</div>'
    )


# --------------------------------------------------------------- header bar
def header_bar(patient_id: str, clock: str, turns: int) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:16px;background:#fff;border:1px solid #dde3ee;'
        'border-radius:11px;padding:11px 18px;margin-bottom:4px;">'
        '<div style="width:30px;height:30px;border-radius:8px;background:#3563d4;display:flex;align-items:center;justify-content:center;flex:none;">'
        '<div style="width:13px;height:13px;border:2.2px solid #fff;border-radius:50%;border-right-color:transparent;"></div></div>'
        '<div style="display:flex;flex-direction:column;line-height:1.1;">'
        '<span style="font-weight:700;font-size:15px;letter-spacing:-.01em;">Live Session Assistant</span>'
        f'<span style="font:500 10px/1 {MONO};color:#8a93a8;letter-spacing:.04em;margin-top:3px;">CLINICAL DECISION SUPPORT</span></div>'
        '<div style="display:flex;align-items:center;gap:8px;padding:5px 11px;background:#f4f6fa;border:1px solid #e3e8f2;border-radius:8px;">'
        f'<span style="font:500 11px/1 {MONO};color:#8a93a8;">PATIENT</span>'
        f'<span style="font:600 13px/1 {MONO};color:#1a2235;">{esc(patient_id)}</span>'
        '<span style="width:6px;height:6px;border-radius:50%;background:#2f8f6b;box-shadow:0 0 0 3px rgba(47,143,107,.16);"></span>'
        '<span style="font-size:11.5px;font-weight:600;color:#2f8f6b;">Session live</span></div>'
        '<div style="margin-left:auto;display:flex;align-items:center;gap:16px;">'
        f'<div style="display:flex;align-items:center;gap:7px;"><span style="font:500 10px/1 {MONO};color:#9aa3b6;">ELAPSED</span>'
        f'<span style="font:600 14px/1 {MONO};color:#1a2235;">{esc(clock)}</span></div>'
        f'<div style="display:flex;align-items:center;gap:7px;"><span style="font:500 10px/1 {MONO};color:#9aa3b6;">TURNS</span>'
        f'<span style="font:600 14px/1 {MONO};color:#1a2235;">{turns}</span></div>'
        '<div style="display:flex;align-items:center;gap:6px;padding:6px 11px;border:1px solid #f0d9d6;background:#fdf4f3;border-radius:8px;">'
        '<span style="width:7px;height:7px;border-radius:50%;background:#c2362f;flex:none;"></span>'
        '<span style="font-size:11.5px;font-weight:600;color:#9c453f;">Not a diagnosis</span></div></div></div>'
    )


# --------------------------------------------------------- patient overview
def patient_overview_card(patient_id, subtitle, sessions, history, goals, last_seen) -> str:
    hist = "".join(
        f'<span style="font:500 11.5px/1 {MONO};background:#f0f3f9;color:#4a5670;padding:5px 8px;border-radius:6px;">{esc(h)}</span>'
        for h in (history or [])
    ) or '<span style="font-size:12px;color:#aab2c4;">No history recorded</span>'
    goal_items = "".join(
        f'<li style="display:flex;gap:8px;font-size:13px;color:#3a4458;"><span style="color:{ACCENT};">›</span>{esc(g)}</li>'
        for g in (goals or [])
    ) or '<li style="font-size:12px;color:#aab2c4;list-style:none;">No goals recorded</li>'
    return (
        '<div class="lsa-card">' + _h("PATIENT OVERVIEW") +
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;">'
        f'<div><div style="font:600 19px/1 {MONO};">{esc(patient_id)}</div>'
        f'<div style="font-size:12.5px;color:#647089;margin-top:4px;">{esc(subtitle)}</div></div>'
        '<div style="text-align:center;background:#f4f6fa;border:1px solid #e6eaf2;border-radius:8px;padding:7px 12px;">'
        f'<div style="font:600 17px/1 {MONO};">{sessions}</div>'
        '<div style="font-size:9.5px;color:#8a93a8;margin-top:3px;letter-spacing:.02em;">SESSIONS</div></div></div>'
        '<div style="margin-top:15px;"><div style="font-size:10.5px;font-weight:600;color:#8a93a8;letter-spacing:.04em;margin-bottom:7px;">CLINICAL HISTORY</div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{hist}</div></div>'
        '<div style="margin-top:14px;"><div style="font-size:10.5px;font-weight:600;color:#8a93a8;letter-spacing:.04em;margin-bottom:7px;">THERAPY GOALS</div>'
        f'<ul style="margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:6px;">{goal_items}</ul></div>'
        '<div style="margin-top:15px;padding-top:13px;border-top:1px dashed #e3e8f2;display:flex;justify-content:space-between;font-size:12px;">'
        f'<span style="color:#8a93a8;">Last seen</span><span style="font-weight:600;color:#3a4458;font-family:{MONO};">{esc(last_seen)}</span></div></div>'
    )


def history_timeline(items) -> str:
    if not items:
        body = '<div style="font-size:12.5px;color:#aab2c4;">No prior sessions on record.</div>'
    else:
        rows = []
        n = len(items)
        for i, t in enumerate(items):
            dot = CRISIS if t.get("flag") else (ACCENT if i == 0 else "#c2cad8")
            line = ("display:none;" if i == n - 1
                    else "flex:1;width:1.5px;background:#e6eaf2;margin:3px 0;")
            sc = t.get("score")
            sc_color = "#647089"
            if isinstance(sc, (int, float)):
                sc_color = "#b23a31" if sc < -0.6 else "#b5862f" if sc < -0.3 else "#647089"
            sc_txt = (f'{sc:+.2f}' if isinstance(sc, (int, float)) else "—")
            topics = "".join(
                f'<span style="font:500 10.5px/1 {MONO};background:#eef1f6;color:#647089;padding:3px 6px;border-radius:5px;">{esc(tp)}</span>'
                for tp in (t.get("topics") or [])
            )
            flag = ""
            if t.get("flag"):
                flag = ('<div style="display:inline-flex;align-items:center;gap:5px;margin-top:7px;padding:3px 7px;background:#fdeceb;border:1px solid #f3c9c5;border-radius:5px;">'
                        '<span style="width:5px;height:5px;border-radius:50%;background:#c2362f;"></span>'
                        f'<span style="font:600 10px/1 {MONO};color:#a8423b;">risk: {esc(t.get("flag"))}</span></div>')
            rows.append(
                '<div style="display:grid;grid-template-columns:14px 1fr;gap:11px;">'
                '<div style="display:flex;flex-direction:column;align-items:center;">'
                f'<span style="width:10px;height:10px;border-radius:50%;flex:none;margin-top:3px;background:{dot};"></span>'
                f'<span style="{line}"></span></div>'
                '<div style="padding-bottom:14px;"><div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">'
                f'<span style="font:600 12px/1 {MONO};color:#3a4458;">{esc(t.get("date"))}</span>'
                f'<span style="font:600 11px/1 {MONO};color:{sc_color};">{sc_txt}</span></div>'
                f'<div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:7px;">{topics}</div>{flag}</div></div>'
            )
        body = "".join(rows)
    return '<div class="lsa-card">' + _h("SESSION HISTORY") + body + '</div>'


# ------------------------------------------------------------- crisis banner
def crisis_banner(action, flag_type, response) -> str:
    return (
        '<div style="background:#fbeceb;border:1.5px solid #e9b8b3;border-radius:11px;padding:14px 16px;animation:crisisPulse 2.2s ease-in-out infinite;margin-bottom:4px;">'
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        '<div style="width:26px;height:26px;border-radius:7px;background:#c2362f;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;flex:none;">!</div>'
        f'<span style="font-weight:700;font-size:14px;color:#9c2f29;">CRISIS PROTOCOL — {esc(action)}</span>'
        f'<span style="font:600 11px/1 {MONO};background:#f3d4d1;color:#9c2f29;padding:4px 8px;border-radius:5px;">{esc(flag_type)}</span>'
        f'<span style="margin-left:auto;font:500 10.5px/1 {MONO};color:#b07a76;">DETERMINISTIC · RULE-BASED · NOT LLM</span></div>'
        f'<p style="margin:10px 0 0;font-size:13px;line-height:1.55;color:#7a3a35;">{esc(response)}</p></div>'
    )


# ------------------------------------------------------------- transcript
def transcript(messages, turns, height=420) -> str:
    bubbles = []
    for m in messages:
        is_p = m.get("speaker") == "patient"
        is_crisis = bool(m.get("crisis"))
        base = "font-size:13.5px;line-height:1.55;padding:11px 13px;border-radius:10px;max-width:88%;border-top-left-radius:3px;"
        if not is_p:
            bub = base + "background:#f4f6fa;border:1px solid #e6eaf2;color:#2a3450;"
            lab = f"font:600 10px/1 {MONO};letter-spacing:.05em;color:#8a93a8;text-transform:uppercase;"
        elif is_crisis:
            bub = base + "background:#fbeceb;border:1px solid #e9b8b3;color:#7a3a35;"
            lab = f"font:600 10px/1 {MONO};letter-spacing:.05em;color:#c2362f;text-transform:uppercase;"
        else:
            bub = base + "background:#eef3fd;border:1px solid #d7e2fa;color:#26365e;"
            lab = f"font:600 10px/1 {MONO};letter-spacing:.05em;color:{ACCENT};text-transform:uppercase;"
        chips = "".join(
            f'<span style="font:500 10.5px/1 {MONO};padding:4px 7px;border-radius:5px;background:{bg};color:{col};letter-spacing:.02em;">{esc(txt)}</span>'
            for (txt, bg, col) in (m.get("chips") or [])
        )
        chips_html = (f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>' if chips else "")
        pending = ('<div style="display:flex;align-items:center;gap:7px;animation:softpulse 1.1s ease-in-out infinite;">'
                   '<span style="width:6px;height:6px;border-radius:50%;background:#aab2c4;"></span>'
                   f'<span style="font:500 11px/1 {MONO};color:#aab2c4;">analyzing turn…</span></div>') if m.get("pending") else ""
        who = "Patient" if is_p else "Doctor"
        bubbles.append(
            '<div style="display:flex;flex-direction:column;gap:7px;animation:fadeUp .3s ease;">'
            f'<div style="display:flex;align-items:center;gap:8px;"><span style="{lab}">{who}</span>'
            f'<span style="font:500 10px/1 {MONO};color:#b4bccc;">{esc(m.get("time",""))}</span></div>'
            f'<div style="{bub}">{esc(m.get("text"))}</div>{chips_html}{pending}</div>'
        )
    inner = "".join(bubbles) or '<div style="font-size:13px;color:#aab2c4;text-align:center;margin-top:30px;">No turns logged yet. Use the composer below to begin.</div>'
    return (
        '<div class="lsa-card" style="padding:0;">'
        '<div style="display:flex;align-items:center;gap:8px;padding:14px 18px;border-bottom:1px solid #eef1f6;">'
        '<span style="width:3px;height:13px;border-radius:2px;background:#3563d4;"></span>'
        f'<span style="font:600 10.5px/1 {MONO};letter-spacing:.07em;color:#8a93a8;">LIVE TRANSCRIPT</span>'
        '<span style="font-size:11.5px;color:#aab2c4;">two-channel · doctor &amp; patient</span>'
        f'<span style="margin-left:auto;font:500 11px/1 {MONO};color:#aab2c4;">{turns} turns</span></div>'
        f'<div class="lsa-scroll" style="max-height:{height}px;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:16px;">{inner}</div></div>'
    )


# ----------------------------------------------------------- analysis pipeline
def pipeline_panel(stages, running, status_label) -> str:
    smap = {
        "pending": ("background:#26314c;color:#5d6b8a;border:1px solid #2c3852;", "color:#6b7793;", ""),
        "done": ("background:#2f8f6b;color:#fff;border:1px solid #2f8f6b;", "color:#cdd6e8;", "✓"),
        "alert": ("background:#c2362f;color:#fff;border:1px solid #c2362f;", "color:#f0c4c0;font-weight:600;", "!"),
        "running": ("background:#6f8fdc;color:#fff;border:1px solid #6f8fdc;", "color:#e6ecf8;font-weight:600;", "•"),
    }
    rows = []
    for st_ in stages:
        dot, lab, icon = smap.get(st_.get("status"), smap["pending"])
        rows.append(
            '<div style="display:flex;align-items:center;gap:10px;padding:5px 0;">'
            f'<span style="width:18px;height:18px;border-radius:50%;flex:none;display:flex;align-items:center;justify-content:center;font:700 10px/1 {MONO};{dot}">{icon}</span>'
            f'<span style="font-size:12.5px;{lab}">{esc(st_.get("label"))}</span>'
            f'<span style="font:500 10px/1 {MONO};color:#5c6885;margin-left:auto;">{esc(st_.get("sub"))}</span></div>'
        )
    badge = ("background:#2a3a63;color:#9fb6f0;" if running else "background:#1f4a3a;color:#7fd6b0;")
    return (
        '<div style="background:#172033;border:1px solid #172033;border-radius:11px;padding:15px 16px;color:#cdd6e8;">'
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:13px;">'
        '<span style="width:3px;height:13px;border-radius:2px;background:#6f8fdc;"></span>'
        f'<span style="font:600 10.5px/1 {MONO};letter-spacing:.07em;color:#8b97b4;">ANALYSIS PIPELINE</span>'
        f'<span style="margin-left:auto;font:600 9.5px/1 {MONO};letter-spacing:.06em;padding:3px 7px;border-radius:5px;{badge}">{esc(status_label)}</span></div>'
        f'<div style="display:flex;flex-direction:column;gap:3px;">{"".join(rows)}</div></div>'
    )


# ----------------------------------------------------------- decision support
_CONF = {
    "high": ("#e8f5ef", "#2f8f6b", "high confidence"),
    "medium": ("#fbf6e9", "#b5862f", "medium confidence"),
    "low": ("#eef1f6", "#8a93a8", "low — treat as a hint"),
}


def _list_block(items, *, bg, border, color, marker, marker_color):
    rows = "".join(
        f'<div style="display:flex;gap:8px;background:{bg};border:1px solid {border};border-radius:8px;padding:9px 11px;font-size:12.5px;line-height:1.45;color:{color};">'
        f'<span style="color:{marker_color};font-weight:700;flex:none;">{marker}</span>{esc(it)}</div>'
        for it in items
    )
    return f'<div style="display:flex;flex-direction:column;gap:7px;">{rows}</div>'


def decision_support(state, confidence, red_flags, missing, questions,
                     follow_ups, caveat, error=False, degraded=None) -> str:
    parts = ['<div class="lsa-card">' + _h("DECISION SUPPORT")]

    if error:
        parts.append('<div style="background:#fdf4f3;border:1px solid #f0d9d6;border-radius:9px;padding:10px 12px;font-size:12.5px;color:#9c453f;margin-bottom:12px;">'
                     'The assistant could not generate suggestions for this turn (model error). The deterministic safety check still applies.</div>')
    if degraded:
        parts.append(f'<div style="font:500 11px/1.4 {MONO};color:#b5862f;margin-bottom:11px;">⚠ Limited analysis — unavailable: {esc(", ".join(degraded))}</div>')

    if state:
        bg, col, lab = _CONF.get((confidence or "low"), _CONF["low"])
        parts.append(
            '<div style="background:#f4f6fa;border:1px solid #e6eaf2;border-radius:9px;padding:12px;margin-bottom:13px;">'
            '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;">'
            '<span style="font-size:10.5px;font-weight:600;color:#8a93a8;letter-spacing:.04em;">LIKELY EMOTIONAL STATE</span>'
            f'<span style="font:600 9.5px/1 {MONO};letter-spacing:.03em;padding:4px 7px;border-radius:5px;background:{bg};color:{col};">{lab}</span></div>'
            f'<div style="font-size:14px;font-weight:600;color:#28304a;line-height:1.4;">{esc(state)}</div></div>'
        )
    if red_flags:
        parts.append('<div style="margin-bottom:14px;"><div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">'
                     '<span style="width:6px;height:6px;border-radius:2px;background:#c2362f;"></span>'
                     '<span style="font-size:11px;font-weight:700;color:#a8423b;letter-spacing:.02em;">RED FLAGS / CONCERNS</span></div>'
                     + _list_block(red_flags, bg="#fdeceb", border="#f3c9c5", color="#8a3833", marker="›", marker_color="#c2362f") + '</div>')
    if missing:
        parts.append('<div style="margin-bottom:14px;"><div style="font-size:11px;font-weight:700;color:#9a7a2a;letter-spacing:.02em;margin-bottom:8px;">MISSING INFO TO CLARIFY</div>'
                     + _list_block(missing, bg="#fbf6e9", border="#efe1c0", color="#7a6420", marker="?", marker_color="#b5862f") + '</div>')
    if questions:
        parts.append('<div style="margin-bottom:14px;"><div style="font-size:11px;font-weight:700;color:#3052b8;letter-spacing:.02em;margin-bottom:8px;">NEXT QUESTIONS TO CONSIDER</div>'
                     + _list_block(questions, bg="#eef3fd", border="#d7e2fa", color="#2f3e63", marker="→", marker_color=ACCENT) + '</div>')
    if follow_ups:
        items = "".join(
            f'<div style="display:flex;gap:8px;font-size:12.5px;line-height:1.45;color:#4a5670;"><span style="color:#aab2c4;">•</span>{esc(f)}</div>'
            for f in follow_ups
        )
        parts.append('<div style="margin-bottom:6px;"><div style="font-size:11px;font-weight:700;color:#647089;letter-spacing:.02em;margin-bottom:8px;">FOLLOW-UP POINTS</div>'
                     f'<div style="display:flex;flex-direction:column;gap:6px;">{items}</div></div>')
    if caveat:
        parts.append(f'<div style="margin-top:10px;font-size:11.5px;font-style:italic;color:#9aa3b6;line-height:1.45;">Note: {esc(caveat)}</div>')

    if not any([state, red_flags, missing, questions, follow_ups]) and not error:
        parts.append('<div style="font-size:12.5px;color:#9aa3b6;">No high-value suggestions for this turn.</div>')

    parts.append('<div style="margin-top:14px;padding-top:12px;border-top:1px dashed #e3e8f2;font-size:11px;color:#9aa3b6;line-height:1.45;">'
                 'Every item is grounded in the patient record, the live transcript, and retrieved cases. The assistant suggests — it never decides. Clinical judgment required.</div>')
    parts.append('</div>')
    return "".join(parts)


def why_panel(signals, cases, context) -> str:
    sig = "".join(
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px;">'
        f'<span style="color:#647089;">{esc(k)}</span>'
        f'<span style="font:600 11.5px/1 {MONO};color:#3a4458;">{esc(v)}</span></div>'
        for (k, v) in (signals or [])
    )
    cases_html = ""
    if cases:
        def _case(c):
            sim = c.get("sim")
            badge = (f'<span style="font:600 10px/1 {MONO};color:#2f8f6b;">{esc(sim)} match</span>'
                     if sim else '<span style="font:600 9.5px/1 ' + MONO + ';color:#aab2c4;">retrieved</span>')
            return (
                '<div style="background:#fff;border:1px solid #e6eaf2;border-radius:7px;padding:9px 10px;">'
                '<div style="display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:5px;">'
                f'<span style="font:600 9.5px/1 {MONO};color:#aab2c4;letter-spacing:.04em;">CORPUS · {esc(c.get("id"))}</span>{badge}</div>'
                f'<div style="font-size:11.5px;color:#3a4458;line-height:1.4;margin-bottom:4px;">“{esc(c.get("q"))}”</div>'
                f'<div style="font-size:11px;color:#8a93a8;line-height:1.4;">{esc(c.get("a"))}</div></div>'
            )
        cards = "".join(_case(c) for c in cases)
        cases_html = ('<div style="font-size:10px;font-weight:700;color:#8a93a8;letter-spacing:.05em;margin:0 0 9px;">RETRIEVED SIMILAR CASES (RAG)</div>'
                      f'<div style="display:flex;flex-direction:column;gap:8px;margin-bottom:13px;">{cards}</div>')
    return (
        '<div style="background:#f7f9fc;border:1px solid #e6eaf2;border-radius:9px;padding:13px;">'
        '<div style="font-size:10px;font-weight:700;color:#8a93a8;letter-spacing:.05em;margin-bottom:9px;">DRIVING SIGNALS</div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:13px;">{sig}</div>{cases_html}'
        '<div style="font-size:10px;font-weight:700;color:#8a93a8;letter-spacing:.05em;margin-bottom:7px;">CONTEXT SENT TO MODEL</div>'
        f'<pre style="margin:0;background:#172033;color:#aeb9d4;border-radius:7px;padding:10px;font:500 10.5px/1.5 {MONO};white-space:pre-wrap;word-break:break-word;">{_pre(context)}</pre></div>'
    )


# ---------------------------------------------------------------- metrics
def metrics_panel(emotion, topic, traj_points, delta_text, delta_color, has_traj) -> str:
    spark = ""
    if has_traj:
        spark = (
            '<div style="background:#f7f9fc;border:1px solid #e6eaf2;border-radius:8px;padding:10px 6px 4px;">'
            '<svg viewBox="0 0 240 60" preserveAspectRatio="none" style="width:100%;height:56px;display:block;">'
            '<line x1="0" y1="30" x2="240" y2="30" stroke="#d9e0ec" stroke-width="1" stroke-dasharray="3 3"></line>'
            f'<polyline points="{traj_points}" fill="none" stroke="{ACCENT}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"></polyline></svg>'
            f'<div style="display:flex;justify-content:space-between;font:500 9px/1 {MONO};color:#aab2c4;padding:0 4px;"><span>turn 1</span><span>now</span></div></div>'
        )
    else:
        spark = '<div style="font-size:11.5px;color:#aab2c4;">Trajectory appears after two analyzed turns.</div>'
    return (
        '<div class="lsa-card">' + _h("SESSION METRICS") +
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:14px;">'
        '<div style="background:#f4f6fa;border:1px solid #e6eaf2;border-radius:8px;padding:10px;">'
        '<div style="font-size:9.5px;color:#8a93a8;letter-spacing:.03em;margin-bottom:5px;">EMOTION</div>'
        f'<div style="font-weight:600;font-size:14px;color:#28304a;">{esc(emotion)}</div></div>'
        '<div style="background:#f4f6fa;border:1px solid #e6eaf2;border-radius:8px;padding:10px;">'
        '<div style="font-size:9.5px;color:#8a93a8;letter-spacing:.03em;margin-bottom:5px;">TOPIC</div>'
        f'<div style="font-weight:600;font-size:14px;color:#28304a;">{esc(topic)}</div></div></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;">'
        '<span style="font-size:11px;color:#8a93a8;">Sentiment trajectory</span>'
        f'<span style="font:600 11px/1 {MONO};color:{delta_color};">{esc(delta_text)}</span></div>{spark}</div>'
    )

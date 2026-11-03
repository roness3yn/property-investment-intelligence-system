"""Investment Intelligence: the investor workspace (Hebrew, right to left).

    streamlit run src/app/streamlit_app.py

Layout (tab "סוכן המחקר"):
* the conversation fills the screen height, with the input pinned under it and a welcome screen
  with suggested questions; while the agents work, a live status shows the pipeline stage and
  each tool call; under each answer, the path it took and the evidence behind every [E#];
* a context panel beside it: the ranked opportunities (with "investigate" buttons that hand an
  opportunity to the agents), a map of Amsterdam, and how reliable the price model is.
Other tabs: the full opportunity table (works without the agents) and the evidence log
(criterion 7). Settings and data status live in the sidebar.
"""
from __future__ import annotations

import html
import math
import os
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st

from agents.config import AgentConfig
from agents.orchestrator import TITLES, Orchestrator

try:                                    # Community Cloud secrets -> environment (root-level keys)
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="Investment Intelligence", page_icon=":material/apartment:", layout="wide",
                   initial_sidebar_state="collapsed" if os.environ.get("ANTHROPIC_API_KEY") else "expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Hebrew:wght@300;400;500;600;700&display=swap');
:root{
  --bg:#F5F6F8; --surface:#FFFFFF; --ink:#101828; --ink-2:#344054; --muted:#667085; --line:#E4E7EC; --line-2:#D0D5DD;
  --accent:#1D3A8A; --accent-2:#2F55C4; --accent-soft:#EEF2FB; --accent-line:#C7D3F2;
  --ok:#067647; --ok-soft:#ECFDF3; --warn:#B54708; --warn-soft:#FFFAEB; --bad:#B42318; --bad-soft:#FEF3F2;
  --pane-h: calc(100vh - 262px);
}
html, body, p, li, td, th, label, input, textarea, button, h1, h2, h3, h4,
.stMarkdown, [data-testid="stCaptionContainer"]{font-family:'IBM Plex Sans Hebrew', system-ui, sans-serif !important;}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"]{
  font-family:'Material Symbols Rounded' !important; direction:ltr;}
.stApp{background:var(--bg); color:var(--ink);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer{display:none !important;}
[data-testid="stMainBlockContainer"], .main .block-container{max-width:1500px; padding:1rem 1.6rem 0 1.6rem;}
.stMarkdown, .stChatMessage, [data-testid="stSidebar"], .stTabs, .stAlert, .stExpander,
[data-testid="stChatInput"] textarea, .stTextInput input, [data-testid="stHorizontalBlock"]{direction:rtl; text-align:right;}
h1,h2,h3{letter-spacing:-.01em; color:var(--ink);}
code{direction:ltr; unicode-bidi:isolate; background:#F2F4F7; color:var(--ink-2); border-radius:4px;}
.stMarkdown ul, .stMarkdown ol{padding-right:1.25rem; padding-left:0;}
table{font-variant-numeric:tabular-nums; border-collapse:collapse; width:100%;}
th{background:#F9FAFB; font-weight:600; color:var(--ink-2);}
td,th{border:1px solid var(--line)!important; padding:.35rem .6rem!important; unicode-bidi:plaintext; font-size:.92rem;}

/* app bar */
.appbar{display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap; direction:rtl;
        padding:.2rem 0 .75rem 0; border-bottom:1px solid var(--line); margin-bottom:.25rem;}
.brand{display:flex; align-items:center; gap:.7rem;}
.mark{width:34px; height:34px; border-radius:8px; background:var(--accent); color:#fff; display:flex; align-items:center;
      justify-content:center; font-weight:700; font-size:.9rem; letter-spacing:.02em; direction:ltr;}
.brand .name{font-size:1.08rem; font-weight:600; color:var(--ink); direction:ltr; text-align:right; line-height:1.2;}
.brand .sub{font-size:.82rem; color:var(--muted);}
.chips{display:flex; gap:.35rem; flex-wrap:wrap; direction:rtl; align-items:center;}
.chip{font-size:.76rem; line-height:1.5; border:1px solid var(--line); background:var(--surface); border-radius:6px; padding:.15rem .55rem;
      color:var(--ink-2); white-space:nowrap;}
.chips .chip{display:inline-flex; align-items:center; height:26px;}
.chip b{font-weight:600; font-variant-numeric:tabular-nums; color:var(--ink);}
.chip.warn{background:var(--warn-soft); border-color:#FEDF89; color:var(--warn);}
.chip.live{background:var(--ok-soft); border-color:#ABEFC6; color:var(--ok);}
.chip.off{color:var(--muted); border-style:dashed;}
.stTabs [data-baseweb="tab-list"]{gap:1.4rem; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-size:.92rem; padding:.55rem 0;}

/* panes */
.st-key-chatlog{flex:0 0 var(--pane-h) !important; height:var(--pane-h) !important; max-height:var(--pane-h);
                min-height:380px; overflow-y:auto !important; padding:0 .25rem 0 .5rem; justify-content:flex-start;}
.st-key-ctx{flex:0 0 calc(var(--pane-h) + 72px) !important; height:calc(var(--pane-h) + 72px) !important;
            max-height:calc(var(--pane-h) + 72px); min-height:452px; overflow-y:auto !important; background:var(--surface);
            border:1px solid var(--line); border-radius:12px; padding:.6rem .9rem .9rem .9rem;}
.pane-hint{font-size:.8rem; color:var(--muted); margin:0 0 .5rem 0; direction:rtl;}

/* welcome */
.welcome{max-width:640px; margin:9vh auto 1.2rem auto; text-align:center; direction:rtl;}
.welcome .mark{margin:0 auto .9rem auto; width:44px; height:44px; font-size:1.05rem; border-radius:10px;}
.welcome h2{font-size:1.65rem; font-weight:600; margin:0 0 .35rem 0;}
.welcome p{color:var(--muted); margin:0; font-size:.95rem; line-height:1.6;}
[class*="st-key-sug_"] button{min-height:64px; text-align:right; justify-content:flex-start; align-items:flex-start;
   background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.6rem .85rem; color:var(--ink-2);}
[class*="st-key-sug_"] button p{font-size:.88rem; line-height:1.45; text-align:right;}
[class*="st-key-sug_"] button:hover{border-color:var(--accent-line); background:var(--accent-soft); color:var(--accent);}

/* messages */
[data-testid="stChatMessage"]{background:transparent; border:0; padding:.35rem 0; gap:.7rem;}
[data-testid="stChatMessage"]:has(.umsg) [data-testid="stChatMessageContent"]{
   background:var(--accent-soft); border:1px solid var(--accent-line); border-radius:12px; padding:.55rem .85rem; max-width:80%;}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{line-height:1.72; font-size:.95rem; color:var(--ink);}
[data-testid="stChatMessage"] h3{font-size:1.02rem; font-weight:600; margin:.9rem 0 .3rem 0;}
[data-testid="stChatInput"]{border-radius:12px; border:1px solid var(--line-2); background:var(--surface);}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent-2); box-shadow:0 0 0 3px rgba(47,85,196,.12);}
.ref{display:inline-block; background:var(--accent-soft); color:var(--accent); border:1px solid var(--accent-line);
     border-radius:4px; padding:0 .3rem; font-size:.72em; font-weight:600; direction:ltr; margin:0 .08rem; vertical-align:1px;}
.rail{display:flex; flex-wrap:wrap; align-items:center; gap:.25rem; margin:.8rem 0 .2rem 0; direction:rtl;}
.step{font-size:.74rem; border-radius:5px; padding:.1rem .45rem; border:1px solid var(--line); background:var(--surface); color:var(--ink-2); white-space:nowrap;}
.step.agent{background:var(--accent); color:#fff; border-color:var(--accent);}
.step.tool b{color:var(--accent); font-weight:600;}
.step.web{border-style:dashed;}
.step.ok{background:var(--ok-soft); border-color:#ABEFC6; color:var(--ok);}
.step.stop{background:var(--bad-soft); border-color:#FECDCA; color:var(--bad);}
.join{color:var(--line-2); font-size:.72rem;}
.facts{color:var(--muted); font-size:.76rem; margin:.15rem 0 .4rem 0; direction:rtl;}
.facts b{color:var(--ink-2); font-weight:600;}
.evi{border:1px solid var(--line); border-radius:8px; padding:.45rem .7rem; margin:.4rem 0; background:var(--surface); direction:rtl;}
.evi .meta{color:var(--muted); font-size:.74rem; direction:ltr; text-align:right; word-break:break-all;}

/* live status */
.live{border:1px solid var(--line); background:var(--surface); border-radius:10px; padding:.7rem .85rem; direction:rtl;}
.live .head{display:flex; align-items:center; gap:.5rem; font-size:.86rem; color:var(--ink-2); margin-bottom:.5rem;}
.spinner{width:12px; height:12px; border-radius:50%; border:2px solid var(--accent-line); border-top-color:var(--accent);}
.stages{display:flex; align-items:center; gap:.2rem; flex-wrap:wrap;}
.stage{font-size:.74rem; padding:.12rem .5rem; border-radius:5px; border:1px solid var(--line); color:var(--muted); background:#fff;}
.stage.done{color:var(--ink-2); border-color:var(--line-2);}
.stage.done::before{content:"✓ "; color:var(--ok);}
.stage.active{background:var(--accent); color:#fff; border-color:var(--accent);}
.stage.alert{background:var(--bad-soft); color:var(--bad); border-color:#FECDCA;}
.arrow{color:var(--line-2); font-size:.72rem;}
.log{margin:.55rem 0 0 0; padding:0; list-style:none; font-size:.8rem; color:var(--ink-2);}
[data-testid="stChatMessage"] .log li, .log li{padding:.12rem 0; font-size:.8rem; line-height:1.5; color:var(--ink-2);}
.log .ms{color:var(--muted); direction:ltr; unicode-bidi:isolate;}
@media (prefers-reduced-motion: no-preference){
  .spinner{animation:spin .8s linear infinite;} @keyframes spin{to{transform:rotate(360deg)}}
}

/* opportunity rows in the context panel */
.orow{direction:rtl; display:grid; grid-template-columns:1.9rem 1fr auto; gap:.15rem .6rem; align-items:center;}
.orow .rk{width:1.9rem; height:1.9rem; border-radius:6px; background:#F2F4F7; color:var(--ink-2); font-weight:600; font-size:.82rem;
          display:flex; align-items:center; justify-content:center; font-variant-numeric:tabular-nums;}
.orow .rk.top{background:var(--accent); color:#fff;}
.orow .nm{font-weight:600; font-size:.9rem; direction:ltr; text-align:right; unicode-bidi:plaintext; color:var(--ink);}
.orow .sg{font-size:.76rem; color:var(--muted); unicode-bidi:plaintext;}
.orow .sc{text-align:left; font-weight:600; font-size:.95rem; color:var(--ink); direction:ltr; font-variant-numeric:tabular-nums;}
.orow .sc small{display:block; font-weight:400; font-size:.7rem; color:var(--muted); direction:rtl; text-align:left;}
.mini{display:flex; gap:.9rem; flex-wrap:wrap; margin:.5rem 0 .55rem 0; font-size:.74rem; color:var(--muted); direction:rtl;}
.mini b{color:var(--ink-2); font-weight:600; direction:ltr; unicode-bidi:isolate; font-variant-numeric:tabular-nums;}
.flag{font-size:.7rem; border-radius:4px; padding:.02rem .35rem; background:var(--warn-soft); color:var(--warn); border:1px solid #FEDF89;}
[class*="st-key-row_"]{border:1px solid var(--line); border-radius:10px; padding:.6rem .75rem; background:#fff; gap:.4rem;}
[class*="st-key-row_"]:hover{border-color:var(--accent-line);}
[class*="st-key-inv_"] button{min-height:28px; padding:.05rem .7rem; font-size:.78rem; border-radius:6px; color:var(--accent);
   border:1px solid var(--accent-line); background:var(--accent-soft);}
[class*="st-key-inv_"] button:hover{background:var(--accent); color:#fff;}

/* model card */
.kv{display:grid; grid-template-columns:1fr auto; gap:.4rem .8rem; font-size:.86rem; direction:rtl;}
.kv span:nth-child(odd){color:var(--muted);} .kv span:nth-child(even){font-weight:600; direction:ltr; text-align:left; font-variant-numeric:tabular-nums;}
.callout{border:1px solid #FEDF89; background:var(--warn-soft); color:var(--warn); border-radius:8px; padding:.5rem .7rem; font-size:.82rem; margin-top:.8rem; direction:rtl;}

/* misc */
.stButton>button{border-radius:8px;}
.stButton>button:focus-visible{outline:2px solid var(--accent-2); outline-offset:2px;}
[data-testid="stSidebar"]{background:var(--surface); border-left:1px solid var(--line);}
[data-testid="stSidebar"] h3{font-size:.95rem; font-weight:600; margin-top:.7rem;}
.status-line{font-size:.84rem; margin:.18rem 0; color:var(--ink-2);}
.dot{display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-left:.45rem;}
.dot.on{background:var(--ok);} .dot.off{background:var(--line-2);}
.notice{border:1px solid var(--line); background:var(--surface); border-radius:10px; padding:.6rem .85rem; color:var(--ink-2);
        font-size:.86rem; direction:rtl; margin:.4rem 0 .8rem 0;}
@media (max-width: 900px){ :root{--pane-h: 70vh;} .st-key-ctx{height:auto;} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = [
    ("המובילות ולמה", "מהן 5 ההזדמנויות המובילות באמסטרדם, ולמה הן מדורגות כך?"),
    ("סתירה במקום הראשון", "למה חדר פרטי ב-Centrum-West נמצא במקום הראשון? מה הראיות בעד ונגד?"),
    ("הערכת מחיר", "כמה תכניס ללילה דירה שלמה עם 2 חדרי שינה ב-De Pijp ל-4 אורחים, וכמה לסמוך על ההערכה?"),
    ("אמינות המודל", "עד כמה מודל המחיר אמין, ובאילו מקרים הוא חלש?"),
]
TOOL_HE = {"rank_opportunities": "דירוג הזדמנויות", "predict_listing_price": "מודל מחיר",
           "neighbourhood_stats": "סטטיסטיקת שכונה", "review_signals": "ביקורות אורחים",
           "data_availability": "זמינות נתונים", "roi_calculator": "מחשבון תשואה",
           "web_search": "חיפוש ברשת", "web_source": "מקור מהרשת", "model_card": "אמינות המודל",
           "segment_profile": "פרופיל הקטע", "listing_profile": "פרופיל נכסים"}
GUARD_HE = {"input": "חסימת קלט", "allow_list": "כלי לא מורשה", "schema": "קלט לא תקין לכלי",
            "budget": "תקציב קריאות", "policy": "מדיניות", "max_rounds": "מגבלת סבבים",
            "agent_unavailable": "סוכן לא זמין"}
BAND_HE = {"0-1": "0–1 חדרי שינה", "2": "2 חדרי שינה", "3+": "3+ חדרי שינה"}
ROOM_HE = {"Entire home/apt": "דירה שלמה", "Private room": "חדר פרטי", "Hotel room": "חדר מלון",
           "Shared room": "חדר משותף", "הכל": "הכל"}
AV_USER, AV_AI = ":material/person:", ":material/insights:"


# ---------------------------------------------------------------------- setup
def default_root() -> str:
    env = os.environ.get("PIIS_DATA_ROOT")
    if env:
        return env
    if list(ROOT.glob("reports/*/opportunities.json")):
        return str(ROOT)
    return str(ROOT / "demo_repo") if (ROOT / "demo_repo").exists() else str(ROOT)


@st.cache_resource(show_spinner="טוען מודלים ונתונים…")
def get_orchestrator(data_root: str, api_key: str, agent_model: str, router_model: str) -> Orchestrator:
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    cfg = AgentConfig(data_root=Path(data_root), agent_model=agent_model, router_model=router_model)
    return Orchestrator(cfg, api_key=api_key or None)


ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("pending", None)
ss.setdefault("sel_hood", None)
ss.setdefault("data_root", default_root())

# ---------------------------------------------------------------------- sidebar: settings and data status
with st.sidebar:
    st.markdown("### הגדרות")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input("מפתח Anthropic API", value="", type="password",
                            placeholder="מוגדר מהסביבה" if env_key else "sk-ant-…",
                            help="בלי מפתח המערכת עובדת במצב ללא סוכן: טבלת ההזדמנויות בלבד.") or env_key
    ss.data_root = st.text_input("תיקיית הנתונים", value=ss.data_root,
                                 help="תיקייה עם models/, reports/ ו-data/processed/ (שורש הריפו, או demo_repo).")
    with st.expander("מודלים"):
        agent_model = st.text_input("מודל הסוכנים", AgentConfig().agent_model)
        router_model = st.text_input("מודל הנתב והמתקן", AgentConfig().router_model)

    orch = get_orchestrator(ss.data_root, api_key, agent_model, router_model)
    status = orch.ctx.status()

    st.markdown("### מצב הנתונים")

    def line(ok: bool, label: str):
        st.markdown(f'<div class="status-line"><span class="dot {"on" if ok else "off"}"></span>{label}</div>',
                    unsafe_allow_html=True)
    tables = status["opportunity_tables"]
    line(bool(tables), "טבלת הזדמנויות: " + (", ".join(f"{c} ({t['label']}, {t['snapshot_date']})"
                                                       for c, t in tables.items()) or "לא נמצאה"))
    line(bool(status["price_models"]), "מודל מחיר: " + (", ".join(status["price_models"]) or "לא נמצא"))
    line(status["listings_rows"] > 0, f"נכסים בנתונים: {status['listings_rows']:,}")
    line(bool(status["reviews_file"]), "טקסט ביקורות: " + ("זמין" if status["reviews_file"] else
                                                          "לא בריפו (משתמשים בציוני ביקורת ובפעילות)"))
    caps = orch.capabilities
    n_tools = sum(1 for why in caps.values() if why is None)
    line(n_tools == len(caps), f"כלים זמינים לסוכנים: {n_tools} מתוך {len(caps)}")
    line(orch.llm is not None, "סוכנים: " + ("פעילים" if orch.llm is not None else "כבויים (אין מפתח API)"))
    if not tables:
        st.info("לא נמצאו קבצי המודל והדירוג בתיקייה הזו. אפשר לבנות נתוני הדגמה סינתטיים.")
        if st.button("בנה נתוני הדגמה (כחצי דקה)"):
            from agents.demo_data import build
            with st.spinner("בונה נתוני הדגמה…"):
                build(ROOT / "demo_repo", verbose=False)
            ss.data_root = str(ROOT / "demo_repo")
            st.cache_resource.clear()
            st.rerun()
    if ss.messages and st.button("שיחה חדשה"):
        ss.messages = []
        st.rerun()


# ---------------------------------------------------------------------- shared data
opp_tool = orch.ctx.opportunity_tool()
city = next(iter(opp_tool.cities)) if opp_tool else None
ct = opp_tool.cities[city] if opp_tool else None
cur = (ct.currency if ct else None) or "EUR"
SYM = {"EUR": "€", "GBP": "£", "USD": "$"}.get(cur, cur + " ")


@st.cache_data(show_spinner=False)
def model_summary(data_root: str) -> dict:
    out = orch.registry["model_card"].handler({}) if orch.capabilities.get("model_card") is None else {}
    return out if out.get("status") == "ok" else {}


card = model_summary(ss.data_root)


def money(v) -> str:
    return f"{SYM}{v:,.0f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "–"


# ---------------------------------------------------------------------- app bar
chips = []
if ct is not None:
    m = ct.meta
    chips += [f'<span class="chip">{html.escape(city)}</span>',
              f'<span class="chip">נתונים מ-<b>{m.get("snapshot_date")}</b></span>',
              f'<span class="chip"><b>{m.get("ranked_segments")}</b> הזדמנויות מדורגות</span>']
    if m.get("label") in ("PROVISIONAL", "SYNTHETIC"):
        tip = "נתוני הדגמה" if m.get("label") == "SYNTHETIC" else "קבוצת המבחן לא נכללה בדירוג; מבחן המודל היה exploratory"
        chips.append(f'<span class="chip warn" title="{tip}">{m.get("label")}</span>')
chips.append(f'<span class="chip {"live" if orch.llm else "off"}">{"סוכנים פעילים" if orch.llm else "סוכנים כבויים"}</span>')
st.markdown(f'<div class="appbar"><div class="brand"><div class="mark">PI</div><div>'
            f'<div class="name">Investment Intelligence</div>'
            f'<div class="sub">חקר הזדמנויות השקעה בהשכרה לטווח קצר</div></div></div>'
            f'<div class="chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- answer rendering
REF_RE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")


def render_answer_text(text: str) -> str:
    safe = text.replace("<", "&lt;")
    return REF_RE.sub(lambda m: "".join(f'<span class="ref">{r.strip()}</span>' for r in m.group(1).split(",")), safe)


def rail_html(r: dict) -> str:
    steps = []
    for e in r["events"]:
        t = e["type"]
        if t == "agent_start" and e["agent"] != "router":
            steps.append(f'<span class="step agent">{html.escape(e.get("title") or e["agent"])}</span>')
        elif t == "route":
            steps.append(f'<span class="step agent">{TITLES["router"]}</span>')
        elif t == "tool_result" and e.get("tool") != "web_source":
            cls = "web" if e.get("server") else ("tool" if e.get("status") in ("ok", None) else "stop")
            ref = f' <b>{e["ref"]}</b>' if e.get("ref") else ""
            steps.append(f'<span class="step {cls}">{TOOL_HE.get(e["tool"], e["tool"])}{ref}</span>')
        elif t == "guardrail":
            steps.append(f'<span class="step stop">{GUARD_HE.get(e.get("guard"), e.get("guard"))}</span>')
        elif t == "verify":
            ok, g = e.get("passed"), e.get("groundedness")
            txt = "אומת" if ok else "תוקן" if not e.get("after_revision") else "אומת חלקית"
            steps.append(f'<span class="step {"ok" if ok else "stop"}">{TITLES["verifier"]}: {txt}'
                         f'{f" {g:.0%}" if g is not None else ""}</span>')
    return '<div class="rail">' + '<span class="join">←</span>'.join(steps) + "</div>" if steps else ""


def facts_html(r: dict) -> str:
    v = r.get("verification") or {}
    parts = []
    if v.get("claims"):
        parts.append(f'<b>{v["claims"]}</b> מספרים נבדקו, <b>{(v["groundedness"] or 0):.0%}</b> מבוססים על כלים')
    calls = sum(1 for e in r["events"] if e["type"] == "tool_result" and e.get("tool") != "web_source")
    if calls:
        parts.append(f'<b>{calls}</b> קריאות לכלים')
    parts.append(f'<b>{r["elapsed_s"]:.1f}</b> שניות')
    u = r.get("usage") or {}
    if u.get("llm_calls"):
        parts.append(f'<b>{u["llm_calls"]}</b> קריאות למודל, {u["input_tokens"] + u["output_tokens"]:,} טוקנים')
    return '<div class="facts">' + " | ".join(parts) + "</div>"


def evidence_block(ev: dict) -> str:
    res = ev.get("result") or {}
    head = res.get("url") if ev["tool"] == "web_search" else (res.get("status_reason") or "")
    return (f'<div class="evi"><span class="ref">{ev["ref"]}</span> <b>{TOOL_HE.get(ev["tool"], ev["tool"])}</b> '
            f'<span style="color:var(--muted)">({html.escape(str(res.get("status")))}'
            f'{", מקור חיצוני" if ev.get("external") else ""})</span>'
            f'<div class="meta">{html.escape(str(ev.get("evidence_id")))}</div>'
            f'{"<div>" + html.escape(str(head)) + "</div>" if head else ""}</div>')


def render_turn(r: dict):
    st.markdown(render_answer_text(r["answer"]), unsafe_allow_html=True)
    st.markdown(rail_html(r) + facts_html(r), unsafe_allow_html=True)
    if r["evidence"]:
        with st.expander(f"ראיות ({len(r['evidence'])})"):
            for ev in r["evidence"]:
                st.markdown(evidence_block(ev), unsafe_allow_html=True)
                with st.popover("תוצאת הכלי"):
                    st.json(ev["result"], expanded=1)
    with st.expander("מסלול התשובה"):
        st.markdown(f"**נתב:** כוונה `{r['intent']}`, עיר `{r['city']}`. {r.get('route_reason') or ''}")
        for e in r["events"]:
            t = e["type"]
            if t == "agent_start":
                st.markdown(f"- **{e.get('title') or e['agent']}**: מודל `{e.get('model')}`, פרומפט `{e.get('prompt_version')}`")
            elif t == "tool_call":
                st.markdown(f"  - קריאה ל-`{e['tool']}` עם `{e.get('args')}`")
            elif t == "tool_result":
                st.markdown(f"  - תוצאה `{e.get('ref') or ''}`, סטטוס `{e.get('status')}`, {e.get('ms', '–')} ms")
            elif t == "tool_cached":
                st.markdown(f"  - קריאה זהה, שימוש חוזר ב-`{e['ref']}`")
            elif t == "delegate":
                st.markdown(f"  - העברה לסוכן ההעשרה: `{e.get('args')}`")
            elif t == "guardrail":
                st.markdown(f"- **{GUARD_HE.get(e.get('guard'), e.get('guard'))}**: {e.get('reason')}")
            elif t == "verify":
                st.markdown(f"- **מאמת ראיות**{' (אחרי תיקון)' if e.get('after_revision') else ''}: "
                            f"עבר={e.get('passed')}, מספרים={e.get('claims')}, לא מבוססים={e.get('ungrounded')}, "
                            f"ללא ציטוט={e.get('uncited')}, הפרות={e.get('violations')}")
            elif t == "error":
                st.markdown(f"- שגיאה: `{e.get('error')}`")


class LiveActivity:
    """What the agents are doing right now: pipeline stage, and the latest steps."""

    def __init__(self, slot):
        self.slot, self.agent, self.log, self.headline = slot, "סוכן", [], "בודק את השאלה"
        self.state = {"קלט": "active", "נתב": "", "סוכן": "", "כלים": "", "אימות": ""}
        self.render()

    def set(self, stage, value="active"):
        order = list(self.state)
        for s in order[:order.index(stage)]:
            if self.state[s] in ("", "active"):
                self.state[s] = "done"
        self.state[stage] = value

    def __call__(self, e):
        t = e["type"]
        if t == "agent_start" and e["agent"] == "router":
            self.set("נתב")
            self.headline = "מנתב את השאלה לסוכן המתאים"
        elif t == "route":
            self.set("סוכן")
            self.log.append(f"הנתב בחר: {html.escape(str(e.get('intent')))}"
                            + (f", {html.escape(str(e['city']))}" if e.get("city") else ""))
        elif t == "agent_start" and e["agent"] == "reviser":
            self.headline = "מתקן את הטיוטה לפי בדיקת הראיות"
            self.log.append("המתקן מתקן את הטיוטה")
        elif t == "agent_start":
            if e["agent"] == "enrichment" and self.agent != "סוכן":
                self.log.append("העברה לסוכן העשרת הנתונים")
            else:
                self.agent = e.get("title") or e["agent"]
            self.headline = f"{self.agent} עובד"
            self.set("סוכן")
        elif t == "tool_call":
            self.set("כלים")
            self.headline = f"{self.agent}: {TOOL_HE.get(e['tool'], e['tool'])}"
            self.log.append(f"{TOOL_HE.get(e['tool'], e['tool'])}…")
        elif t == "tool_result" and e.get("tool") != "web_source":
            ref = f" ← <b>{e['ref']}</b>" if e.get("ref") else ""
            ms = f' <span class="ms">{e["ms"]} ms</span>' if e.get("ms") is not None else ""
            self.log[-1:] = [f"{TOOL_HE.get(e['tool'], e['tool'])}{ref}{ms}"]
        elif t == "guardrail":
            self.log.append(f"guardrail: {GUARD_HE.get(e.get('guard'), e.get('guard'))}")
            if e.get("guard") == "input":
                self.state["קלט"] = "alert"
        elif t == "verify":
            self.set("אימות", "done" if e.get("passed") else "active")
            self.headline = "בודק כל מספר מול הראיות"
            self.log.append(f"נבדקו {e.get('claims', 0)} מספרים: " + ("עבר" if e.get("passed") else "נדרש תיקון"))
        self.render()

    def render(self):
        names = {**{k: k for k in self.state}, "סוכן": self.agent}
        stages = '<span class="arrow">←</span>'.join(
            f'<span class="stage {v}">{html.escape(names[k])}</span>' for k, v in self.state.items())
        items = "".join(f"<li>{x}</li>" for x in self.log[-5:])
        self.slot.markdown(f'<div class="live"><div class="head"><span class="spinner"></span>'
                           f'{html.escape(self.headline)}</div><div class="stages">{stages}</div>'
                           f'<ul class="log">{items}</ul></div>', unsafe_allow_html=True)


def ask(prompt: str, log_box):
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": prompt})
    with log_box:
        with st.chat_message("user", avatar=AV_USER):
            st.markdown(f'<span class="umsg"></span>{html.escape(prompt)}', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=AV_AI):
            slot = st.empty()
            live = LiveActivity(slot)
            result = orch.handle(prompt, history, on_event=live)
            slot.empty()
            r = result.__dict__
            render_turn(r)
    ss.messages.append({"role": "assistant", "content": result.answer, "result": r})


def investigate_prompt(r) -> str:
    return (f"הכן תדריך ראיות על ההזדמנות {r['neighbourhood_cleansed']} / {r['room_type']} / "
            f"{r['bedroom_band']} חדרי שינה: למה היא מדורגת כך, ראיות בעד ונגד, ומה לבדוק הלאה.")


# ---------------------------------------------------------------------- context panel pieces
@st.cache_data(show_spinner=False)
def map_frames(data_root: str, city: str):
    lst = orch.ctx.listings()
    t = orch.ctx.opportunity_tool().cities[city].table
    pts = lst[(lst["location"] == city)][["neighbourhood_cleansed", "latitude", "longitude"]].dropna()
    pts = pts.astype({"latitude": float, "longitude": float})
    cent = pts.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()
    ok = t[t["status"] == "ok"].sort_values("score", ascending=False)
    best = ok.groupby("neighbourhood_cleansed").head(1).set_index("neighbourhood_cleansed")
    hoods = cent.join(best[["score", "room_type", "bedroom_band", "rank"]], how="left")
    hoods["ranked"] = ok.groupby("neighbourhood_cleansed").size().reindex(hoods.index).fillna(0).astype(int)
    return pts, hoods.reset_index().rename(columns={"neighbourhood_cleansed": "hood"})


def draw_map():
    """Dot-density map (Plotly, no tiles or tokens): listings as faint dots, neighbourhoods as
    bubbles sized and coloured by their best opportunity. Click a bubble to filter the list."""
    import plotly.graph_objects as go
    pts, hoods = map_frames(ss.data_root, city)
    lat0 = float(pts["latitude"].median())
    smax = float(hoods["score"].max() or 1)
    h = hoods.copy()
    h["score0"] = h["score"].fillna(0)
    h["size"] = h["score0"].map(lambda v: 10 + 30 * math.sqrt(v / smax))
    h["sel"] = h["hood"] == ss.sel_hood
    h["hover"] = h.apply(lambda r: (f"<b>{r['hood']}</b><br>מקום {int(r['rank'])}: {r['room_type']}, "
                                    f"{r['bedroom_band']} חדרי שינה<br>{money(r['score'])} בשנה")
                         if pd.notna(r["score"]) else f"<b>{r['hood']}</b><br>אין הזדמנות מדורגת", axis=1)
    top = list(h.nlargest(3, "score0")["hood"])
    h["text"] = h["hood"].map(lambda x: x.split(" - ")[0] if x in top else "")
    h["pos"] = h["hood"].map(lambda x: ["top center", "bottom center", "middle left"][top.index(x)]
                             if x in top else "top center")
    sample = pts.sample(min(len(pts), 5000), random_state=1)
    pad = lambda lo, hi: [lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)]
    xr = pad(pts["longitude"].quantile(.03), pts["longitude"].quantile(.97))
    yr = pad(pts["latitude"].quantile(.03), pts["latitude"].quantile(.97))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample["longitude"], y=sample["latitude"], mode="markers", hoverinfo="skip",
                             marker=dict(size=2.5, color="rgba(102,112,133,0.18)")))
    fig.add_trace(go.Scatter(
        x=h["longitude"], y=h["latitude"], mode="markers+text", text=h["text"], textposition=h["pos"],
        textfont=dict(size=11, color="#344054"), customdata=h[["hood"]].to_numpy(),
        hovertext=h["hover"], hoverinfo="text",
        marker=dict(size=h["size"], color=h["score0"], cmin=0, cmax=smax,
                    colorscale=[[0, "#EEF2FB"], [0.4, "#9DB1E6"], [1, "#1D3A8A"]],
                    line=dict(width=h["sel"].map({True: 3, False: 1}),
                              color=h["sel"].map({True: "#B54708", False: "#FFFFFF"})), opacity=0.95)))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", dragmode="pan",
                      font=dict(family="IBM Plex Sans Hebrew, sans-serif"),
                      hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E4E7EC", font=dict(size=12, color="#101828")),
                      xaxis=dict(visible=False, range=xr, constrain="domain"),
                      yaxis=dict(visible=False, range=yr, scaleanchor="x", constrain="domain",
                                 scaleratio=1 / math.cos(math.radians(lat0))))
    cfg = {"displayModeBar": False, "scrollZoom": True}
    try:
        ev = st.plotly_chart(fig, config=cfg, on_select="rerun", selection_mode="points", key="amap")
        points = (ev.selection or {}).get("points", []) if ev else []
        hit = next((p for p in points if p.get("curve_number") == 1), None)
        if hit:
            cd = hit.get("customdata")
            hood = cd[0] if isinstance(cd, (list, tuple)) else cd
            if hood and hood != ss.sel_hood:
                ss.sel_hood = hood
                st.rerun()
    except TypeError:
        st.plotly_chart(fig, config=cfg)


def opportunity_row(r):
    flags = []
    if r["price_certainty"] < 0.6:
        flags.append("ודאות נמוכה")
    if r["n_listings"] < 30:
        flags.append("מעט נכסים")
    if r.get("inside_airbnb_occupancy") and abs(r["inside_airbnb_occupancy"] - r["booked_nights"]) > 50:
        flags.append("פער בלילות")
    with st.container(key=f"row_{r['segment_key']}"):
        st.markdown(
            f'<div class="orow"><div class="rk {"top" if r["rank"] <= 3 else ""}">{int(r["rank"])}</div><div>'
            f'<div class="nm">{html.escape(r["neighbourhood_cleansed"])}</div>'
            f'<div class="sg">{ROOM_HE.get(r["room_type"], r["room_type"])}, '
            f'{BAND_HE.get(r["bedroom_band"], r["bedroom_band"])}, {int(r["n_listings"])} נכסים</div></div>'
            f'<div class="sc">{money(r["score"])}<small>בשנה, שמרני</small></div></div>'
            f'<div class="mini"><span>מחיר <b>{money(r["expected_price"])}</b></span>'
            f'<span>לילות <b>{r["booked_nights"]:.0f}</b></span><span>ודאות <b>{r["price_certainty"]:.2f}</b></span>'
            f'{"".join(f"<span class=flag>{f}</span>" for f in flags)}</div>', unsafe_allow_html=True)
        if orch.llm is not None and st.button("חקור עם הסוכנים", key=f"inv_{r['segment_key']}"):
            ss.pending = investigate_prompt(r)
            st.rerun()


def model_panel():
    if not card:
        st.markdown('<div class="pane-hint">אין model card בתיקיית הנתונים.</div>', unsafe_allow_html=True)
        return
    t, iv = card["test"], card["interval_80"]
    gain = 100 * (1 - t["model_MAE"] / t["baseline_MAE"]) if t.get("baseline_MAE") else 0
    st.markdown(
        f'<div class="pane-hint">תוצאות המבחן של מודל המחיר ({html.escape(str(card.get("selected_model")))}) '
        f'על {card.get("test_listings")} נכסים שהמודל לא ראה. תווית הריצה: {card.get("run_label")}.</div>'
        f'<div class="kv"><span>טעות ממוצעת ללילה (MAE)</span><span>{t["model_MAE"]:.1f} {cur}</span>'
        f'<span>קו בסיס (חציון שכונה × סוג)</span><span>{t["baseline_MAE"]:.1f} {cur}</span>'
        f'<span>שיפור מול קו הבסיס</span><span>{gain:.0f}%</span>'
        f'<span>תחזיות בטווח ±20%</span><span>{100 * (t.get("share_within_20pct") or 0):.0f}%</span>'
        f'<span>כיסוי טווח ה-80%</span><span>{100 * (iv.get("coverage") or 0):.1f}%</span>'
        f'<span>R² על לוג המחיר</span><span>{(t.get("r2_log_price") or 0):.2f}</span></div>',
        unsafe_allow_html=True)
    weak = card.get("weak_segments") or []
    if weak:
        items = ", ".join(f'{w["segment"].replace("price_decile=", "עשירון מחיר ")} (כיסוי {100 * w["coverage_80"]:.0f}%)'
                          for w in weak)
        st.markdown(f'<div class="callout">חולשה ידועה: טווחי המחיר פחות אמינים ב-{items}. '
                    f'הסוכנים מציינים זאת כשזה רלוונטי.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------- layout
tab_chat, tab_table, tab_log = st.tabs(["סוכן המחקר", "טבלת הזדמנויות", "יומן ראיות"])

with tab_chat:
    col_chat, col_ctx = st.columns([1.7, 1], gap="large")

    with col_ctx:
        with st.container(key="ctx"):
            sub_opps, sub_map, sub_model = st.tabs(["הזדמנויות", "מפה", "אמינות המודל"])
            with sub_opps:
                if ct is None:
                    st.markdown('<div class="pane-hint">אין טבלת הזדמנויות בתיקיית הנתונים.</div>',
                                unsafe_allow_html=True)
                else:
                    ranked = ct.table[ct.table["status"] == "ok"].sort_values("rank")
                    room = st.pills("סוג נכס", ["הכל", "Entire home/apt", "Private room"], default="הכל",
                                    key="f_room", format_func=lambda x: ROOM_HE.get(x, x),
                                    label_visibility="collapsed")
                    sel = ranked if not room or room == "הכל" else ranked[ranked["room_type"] == room]
                    if ss.sel_hood:
                        sel = sel[sel["neighbourhood_cleansed"] == ss.sel_hood]
                        if st.button(f"הסר סינון: {ss.sel_hood}", key="clear_hood"):
                            ss.sel_hood = None
                            st.rerun()
                    st.markdown(f'<div class="pane-hint">ציון = לילות × מחיר צפוי × ודאות: הכנסה שמרנית '
                                f'מהשכרה בשנה, לא תשואה ולא המלצת קנייה. {len(sel)} מתאימות.</div>',
                                unsafe_allow_html=True)
                    for r in sel.head(10).to_dict("records"):
                        opportunity_row(r)
            with sub_map:
                if ct is not None:
                    st.markdown('<div class="pane-hint">כל עיגול הוא שכונה, לפי ההזדמנות הטובה בה. '
                                'לחיצה מסננת את רשימת ההזדמנויות.</div>', unsafe_allow_html=True)
                    try:
                        draw_map()
                    except Exception as e:
                        st.caption(f"המפה לא זמינה: {e}")
            with sub_model:
                model_panel()

    with col_chat:
        log_box = st.container(key="chatlog")
        with log_box:
            if not ss.messages and not ss.pending:
                st.markdown('<div class="welcome"><div class="mark">PI</div><h2>במה נתמקד היום?</h2>'
                            '<p>שאלו על הזדמנויות, מחירים או שכונות באמסטרדם. סוכנים מתמחים אוספים ראיות '
                            'מהמודלים ומהנתונים, וכל מספר בתשובה נבדק מול המקור שלו.</p></div>',
                            unsafe_allow_html=True)
                if orch.llm is None:
                    st.markdown('<div class="notice">הסוכנים כבויים כי לא הוגדר מפתח API. פתחו את סרגל ההגדרות '
                                '(החץ בפינה העליונה) והדביקו מפתח. טבלת ההזדמנויות זמינה גם בלעדיו.</div>',
                                unsafe_allow_html=True)
                c1, c2 = st.columns(2, gap="small")
                for i, (title, q) in enumerate(SUGGESTIONS):
                    with (c1 if i % 2 == 0 else c2):
                        if st.button(f"**{title}**  \n{q}", key=f"sug_{i}", width="stretch",
                                     disabled=orch.llm is None):
                            ss.pending = q
                            st.rerun()
            for m in ss.messages:
                with st.chat_message(m["role"], avatar=AV_USER if m["role"] == "user" else AV_AI):
                    if m["role"] == "user":
                        st.markdown(f'<span class="umsg"></span>{html.escape(m["content"])}', unsafe_allow_html=True)
                    else:
                        render_turn(m["result"])
        prompt = st.chat_input("שאלו את סוכן המחקר…", key="chat", disabled=orch.llm is None)
        prompt = prompt or ss.pending
        ss.pending = None
        if prompt:
            ask(prompt, log_box)

with tab_table:
    if ct is None:
        st.info("אין טבלת הזדמנויות בתיקיית הנתונים.")
    else:
        t = ct.table
        c1, c2, c3 = st.columns(3)
        hoods = c1.multiselect("שכונה", sorted(t["neighbourhood_cleansed"].unique()))
        rooms = c2.multiselect("סוג נכס", sorted(t["room_type"].unique()))
        show_all = c3.checkbox("הצג גם ללא מספיק ראיות", value=False)
        f = t if show_all else t[t["status"] == "ok"]
        if hoods:
            f = f[f["neighbourhood_cleansed"].isin(hoods)]
        if rooms:
            f = f[f["room_type"].isin(rooms)]
        view = pd.DataFrame({
            "דירוג": f["rank"], "שכונה": f["neighbourhood_cleansed"], "סוג": f["room_type"],
            "חדרי שינה": f["bedroom_band"], f"ציון ({cur}/שנה)": f["score"].round(0),
            f"מחיר צפוי ({cur}/לילה)": f["expected_price"].round(0), "לילות מוזמנים": f["booked_nights"].round(0),
            "ודאות": f["price_certainty"].round(2), "נכסים": f["n_listings"], "סטטוס": f["status"]})
        st.caption(f"{ct.meta.get('ranked_segments')} הזדמנויות מדורגות, תמונת מצב {ct.meta.get('snapshot_date')}, "
                   f"תווית {ct.meta.get('label')}. ציון = לילות × מחיר צפוי × ודאות; הכנסה שמרנית מהשכרה, לא תשואה.")
        st.dataframe(view, hide_index=True, width="stretch", height=640,
                     column_config={f"ציון ({cur}/שנה)": st.column_config.ProgressColumn(
                         f"ציון ({cur}/שנה)", format="%.0f", min_value=0, max_value=float(t["score"].max() or 1))})

with tab_log:
    turns = [m["result"] for m in ss.messages if m["role"] == "assistant"]
    checked = [r["verification"] for r in turns if r.get("verification") and r["verification"].get("claims")]
    if not turns:
        st.markdown("עוד אין ראיות בשיחה הזו.")
    else:
        claims = sum(v["claims"] for v in checked)
        grounded = sum(round(v["groundedness"] * v["claims"]) for v in checked)
        c1, c2, c3 = st.columns(3)
        c1.metric("מספרים שנבדקו", f"{claims:,}")
        c2.metric("מבוססים על כלים", f"{grounded / claims:.0%}" if claims else "–",
                  help="קריטריון 7: לפחות 95% מהמספרים שהסוכן מצטט תואמים את יומן הראיות")
        c3.metric("קריאות לכלים", sum(len(r["evidence"]) for r in turns))
        rows = [{"שאלה": r["question"][:60], "תור": r["turn_id"], "ראיה": e["ref"], "כלי": e["tool"],
                 "סוכן": e["agent"], "סטטוס": e["status"], "evidence_id": e["evidence_id"]}
                for r in turns for e in r["evidence"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("כל תוצאה נשמרת גם בקובץ evaluation/agent_evidence.jsonl לבדיקת ה-groundedness.")

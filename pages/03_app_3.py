from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from modules.ai_helper import (
    build_structured_context,
    deterministic_operation_answer,
    generate_llm_answer,
    parse_question_constraints,
)
from modules.core import (
    DATE_MAX,
    DATE_MIN,
    airline_area_weights,
    airline_summary,
    build_airline_share_multipliers,
    build_baseline_inputs,
    data_quality_report,
    date_options,
    hhmm_to_minute,
    load_flight_data,
    load_operation_data,
    minute_to_hhmm,
    snapshot,
)
from modules.simulation import (
    ALL_AREAS,
    AREA_TYPES,
    CHECKIN_AREAS,
    IM_AREAS,
    KEEP_RATE,
    MAX_UNITS,
    SELF_AREAS,
    STAFFED_AREAS,
    TARGET_WAIT_MIN,
    UNIT_LABELS,
    clamp_units,
    compute_metrics,
    minimum_operating_units,
    monte_carlo_compare,
    optimize_fixed_allocation,
    simulate_coupled_system,
    staff_from_units,
    total_staff,
    temporal_comparison,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OPERATION_PATH = DATA_DIR / "operation_dashboard_oct2025.csv.gz"
FLIGHT_PATH = DATA_DIR / "flight_counter_oct2025.csv"

st.set_page_config(
    page_title="ICN T2 가상 운영 시나리오 & AI 의사결정",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Visual system · forced dark UI
# -----------------------------------------------------------------------------
DARK_BG = "#07111F"
DARK_SIDEBAR = "#0A1625"
DARK_PANEL = "#0D1B2A"
DARK_PANEL_2 = "#112338"
DARK_LINE = "#263A52"
DARK_GRID = "#1D3045"
TEXT_MAIN = "#E6EDF3"
TEXT_MUTED = "#91A4B7"
ACCENT = "#4DA3FF"
ACCENT_2 = "#38BDF8"
GOOD = "#2DD4BF"
WARN = "#F59E0B"
BAD = "#FB7185"
PLOT_COLORS = [ACCENT, GOOD, WARN, "#A78BFA", "#FB7185", "#22C55E", "#F97316", "#60A5FA"]
DARK_HEAT_SCALE = [
    [0.00, "#08111D"],
    [0.16, "#0B263B"],
    [0.35, "#0E4569"],
    [0.55, "#176A9E"],
    [0.76, "#2D8FD0"],
    [1.00, "#7DD3FC"],
]

# Plotly is forced to the same palette as the Streamlit UI. This remains dark
# even when the browser/OS is using a light appearance.
pio.templates["icn_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_MAIN, family="Arial, 'Noto Sans KR', sans-serif", size=13),
        colorway=PLOT_COLORS,
        hoverlabel=dict(bgcolor="#12263B", bordercolor="#35506C", font_color=TEXT_MAIN),
        legend=dict(bgcolor="rgba(7,17,31,0.76)", bordercolor=DARK_LINE, borderwidth=1),
        title=dict(font=dict(color=TEXT_MAIN, size=17)),
        xaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_LINE, zerolinecolor=DARK_LINE, tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_MAIN)),
        yaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_LINE, zerolinecolor=DARK_LINE, tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_MAIN)),
        colorscale=dict(sequential=DARK_HEAT_SCALE),
    )
)
pio.templates.default = "icn_dark"


def darken_plot(fig: go.Figure) -> go.Figure:
    """Normalize every Plotly figure so it remains legible on the forced dark UI."""
    fig.update_layout(
        template="icn_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK_PANEL,
        font=dict(color=TEXT_MAIN),
        hoverlabel=dict(bgcolor="#12263B", bordercolor="#35506C", font_color=TEXT_MAIN),
        legend=dict(bgcolor="rgba(7,17,31,0.72)", bordercolor=DARK_LINE, borderwidth=1, font=dict(color=TEXT_MAIN)),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=DARK_GRID,
        gridwidth=1,
        linecolor=DARK_LINE,
        zerolinecolor=DARK_LINE,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MAIN),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DARK_GRID,
        gridwidth=1,
        linecolor=DARK_LINE,
        zerolinecolor=DARK_LINE,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(color=TEXT_MAIN),
    )
    return fig


st.markdown(
    r"""
<style>
:root {
  --bg:#07111F;
  --sidebar:#0A1625;
  --panel:#0D1B2A;
  --panel2:#112338;
  --panel3:#152A41;
  --ink:#E6EDF3;
  --muted:#91A4B7;
  --line:#263A52;
  --accent:#4DA3FF;
  --accent2:#38BDF8;
  --good:#2DD4BF;
  --warn:#F59E0B;
  --bad:#FB7185;
}

/* App canvas: force dark regardless of Streamlit/browser appearance. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:var(--bg) !important;
  color:var(--ink) !important;
}
[data-testid="stAppViewBlockContainer"], .block-container {
  padding-top:1.25rem;
  padding-bottom:3rem;
  max-width:1550px;
}
[data-testid="stHeader"] {
  background:rgba(7,17,31,.92) !important;
  border-bottom:1px solid rgba(38,58,82,.75);
  backdrop-filter:blur(10px);
}
[data-testid="stToolbar"], [data-testid="stDecoration"] {background:transparent !important;}
[data-testid="stSidebar"] {
  background:var(--sidebar) !important;
  border-right:1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div:first-child {background:var(--sidebar) !important;}

/* Typography */
h1,h2,h3,h4,h5,h6, p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  color:var(--ink);
}
h1,h2,h3 {letter-spacing:-0.02em;}
[data-testid="stCaptionContainer"], small {color:var(--muted) !important;}
a {color:#70B7FF !important;}
hr {border-color:var(--line) !important;}

/* Hero / cards */
.hero {
  padding:22px 26px;
  border:1px solid #22476A;
  border-radius:18px;
  background:linear-gradient(125deg,#071A2B 0%,#0C2C49 52%,#123E63 100%);
  color:white;
  margin-bottom:14px;
  box-shadow:0 14px 34px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
}
.hero h1 {color:white !important; margin:0 0 5px 0; font-size:1.85rem;}
.hero p {margin:0; color:#C8D8E8 !important; font-size:.98rem;}
.kpi-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:10px 0 18px 0;}
.kpi {
  background:linear-gradient(180deg,#102238 0%,#0D1B2A 100%);
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px 15px;
  box-shadow:0 6px 18px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.025);
}
.kpi .label {font-size:.78rem;color:var(--muted);font-weight:700;margin-bottom:5px;}
.kpi .value {font-size:1.42rem;color:#F4F8FC;font-weight:800;line-height:1.15;}
.kpi .sub {font-size:.75rem;color:var(--muted);margin-top:5px;}
.notice {border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:12px 14px;margin:8px 0 12px 0;}
.section-title {font-size:1.08rem;font-weight:800;margin:14px 0 8px 0;color:#F2F7FB;}

/* Badges */
.badge {display:inline-block;padding:3px 8px;border-radius:999px;font-size:.75rem;font-weight:700;margin-right:5px;border:1px solid transparent;}
.badge-blue {background:#102D49;color:#87C7FF;border-color:#214A6D;}
.badge-green {background:#0E302B;color:#6EE7D2;border-color:#21534B;}
.badge-warn {background:#35250C;color:#F8C86D;border-color:#654814;}
.badge-red {background:#391B25;color:#FDA4AF;border-color:#663040;}

/* Native KPI metrics */
[data-testid="stMetric"] {
  background:linear-gradient(180deg,#102238 0%,#0D1B2A 100%);
  border:1px solid var(--line);
  border-radius:13px;
  padding:12px 14px;
  min-height:98px;
}
[data-testid="stMetricLabel"] {color:var(--muted) !important;}
[data-testid="stMetricValue"] {color:#F4F8FC !important;}
[data-testid="stMetricDelta"] {color:#A9BED1 !important;}

/* Buttons */
.stButton > button, .stDownloadButton > button, button[kind="secondary"] {
  background:#12243A !important;
  color:var(--ink) !important;
  border:1px solid #35506C !important;
  border-radius:9px !important;
  box-shadow:none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background:#17304C !important;
  border-color:#5EA9EE !important;
  color:white !important;
}
.stButton > button:focus {box-shadow:0 0 0 2px rgba(77,163,255,.25) !important;}
button[kind="primary"], .stButton > button[kind="primary"] {
  background:linear-gradient(180deg,#267FE0,#1768BD) !important;
  border-color:#4DA3FF !important;
  color:white !important;
}
button[kind="primary"]:hover {background:linear-gradient(180deg,#3892F1,#1D74D1) !important;}

/* Selects, inputs, textareas, number/date fields */
[data-baseweb="select"] > div,
[data-baseweb="base-input"],
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea,
.stTextArea textarea,
.stTextInput input {
  background:#0F2033 !important;
  color:var(--ink) !important;
  border-color:#2D4661 !important;
}
[data-baseweb="select"] *, [data-baseweb="base-input"] * {color:var(--ink) !important;}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
  background:#0F2033 !important;
  color:var(--ink) !important;
  border-color:var(--line) !important;
}
[role="option"] {background:#0F2033 !important;color:var(--ink) !important;}
[role="option"]:hover, [aria-selected="true"][role="option"] {background:#17304C !important;color:white !important;}
[data-baseweb="calendar"], [data-baseweb="calendar"] > div, [data-baseweb="calendar"] div {background-color:#0F2033 !important;color:var(--ink) !important;}
[data-baseweb="calendar"] button {color:var(--ink) !important;}
[data-baseweb="calendar"] button:hover {background:#17304C !important;}
[data-baseweb="calendar"] [aria-selected="true"] {background:#176FC1 !important;color:white !important;}
input::placeholder, textarea::placeholder {color:#71869A !important;}

/* Sliders and toggles */
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {background:#334A61;}
[data-testid="stSlider"] [role="slider"] {background:#4DA3FF !important;border-color:#8CC8FF !important;}
[data-testid="stSlider"] p, [data-testid="stSlider"] span {color:var(--ink) !important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stTickBarValue"],
[data-testid="stSlider"] [class*="tick-bar"],
[data-testid="stSlider"] [class*="stTickBar"] {
  background:transparent !important;
  color:var(--ink) !important;
  border:none !important;
  box-shadow:none !important;
  padding:0 !important;
  border-radius:0 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"] *,
[data-testid="stSlider"] [data-testid="stTickBarMax"] *,
[data-testid="stSlider"] [data-testid="stTickBarValue"] * {
  background:transparent !important;
  color:var(--ink) !important;
  border:none !important;
  box-shadow:none !important;
}
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label {color:var(--ink) !important;}

/* Tabs */
[data-baseweb="tab-list"] {
  gap:12px;
  border:1px solid var(--line);
  background:linear-gradient(180deg,#091625 0%,#0B1827 100%);
  padding:10px 12px 12px 12px;
  border-radius:16px;
  box-shadow:0 10px 24px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.02);
  margin:6px 0 18px 0;
  flex-wrap:wrap;
}
[data-baseweb="tab"] {
  color:var(--muted) !important;
  background:linear-gradient(180deg,#102238 0%,#0D1B2A 100%) !important;
  border:1px solid #29425A !important;
  border-radius:12px !important;
  padding:10px 18px !important;
  min-height:48px !important;
  box-shadow:0 4px 12px rgba(0,0,0,.12), inset 0 1px 0 rgba(255,255,255,.02);
  transition:all .18s ease;
}
[data-baseweb="tab"]:hover {
  background:linear-gradient(180deg,#15304A 0%,#10253A 100%) !important;
  color:var(--ink) !important;
  border-color:#4C6D8E !important;
  transform:translateY(-1px);
}
[data-baseweb="tab"][aria-selected="true"] {
  color:#DDF1FF !important;
  background:linear-gradient(180deg,#184F84 0%,#12395B 100%) !important;
  border-color:#5FB2FF !important;
  box-shadow:0 10px 18px rgba(6,16,26,.24), 0 0 0 1px rgba(93,178,255,.16) inset;
}
[data-baseweb="tab"] p, [data-baseweb="tab"] span {
  color:inherit !important;
  font-weight:700 !important;
}
[data-baseweb="tab-highlight"] {
  display:none !important;
}
/* clear content separation under tabs */
[data-baseweb="tab-panel"] {
  background:linear-gradient(180deg,#0B1624 0%,#0C1A29 100%);
  border:1px solid rgba(38,58,82,.85);
  border-radius:16px;
  padding:18px 18px 12px 18px;
  box-shadow:0 10px 26px rgba(0,0,0,.14), inset 0 1px 0 rgba(255,255,255,.015);
  margin-top:2px;
}
/* nested tabs use a slightly subtler treatment */
[data-baseweb="tab-panel"] [data-baseweb="tab-list"] {
  gap:8px;
  padding:8px 10px 10px 10px;
  border-radius:14px;
  background:#0C1725;
  margin:0 0 14px 0;
}
[data-baseweb="tab-panel"] [data-baseweb="tab"] {
  min-height:42px !important;
  padding:8px 14px !important;
  border-radius:10px !important;
}

/* Expanders, forms and containers */
[data-testid="stExpander"] {
  background:var(--panel) !important;
  border:1px solid var(--line) !important;
  border-radius:10px !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color:var(--ink) !important;}
[data-testid="stForm"] {background:var(--panel);border:1px solid var(--line);border-radius:12px;}

/* Alerts */
[data-testid="stAlert"] {background:#102238 !important;border:1px solid #2E4965 !important;border-radius:10px !important;color:var(--ink) !important;}
[data-testid="stAlert"] * {color:inherit !important;}
div[data-baseweb="notification"] {background:var(--panel2) !important;color:var(--ink) !important;border-color:var(--line) !important;}

/* Dataframes / editors. Streamlit theme handles the grid canvas; this styles its frame and toolbar. */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  background:var(--panel) !important;
  border:1px solid var(--line) !important;
  border-radius:10px;
  overflow:hidden;
}
[data-testid="stDataFrame"] [data-testid="stElementToolbar"],
[data-testid="stDataEditor"] [data-testid="stElementToolbar"] {background:#0D1B2A !important;}
[data-testid="stDataFrame"] button, [data-testid="stDataEditor"] button {color:var(--ink) !important;}

/* JSON / code */
[data-testid="stJson"], pre, code {
  background:#091827 !important;
  color:#CFE5F7 !important;
  border-color:var(--line) !important;
}

/* Counter line visualization */
.counter-wrap {
  background:linear-gradient(180deg,#102238 0%,#0D1B2A 100%);
  color:var(--ink);
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px;
  margin-top:8px;
}
.counter-grid {display:grid;grid-template-columns:repeat(20,minmax(0,1fr));gap:4px;}
.counter-cell {
  height:29px;border-radius:6px;display:flex;align-items:center;justify-content:center;
  font-size:.68rem;font-weight:700;border:1px solid #2B4058;background:#122238;color:#70869B;
}
.counter-cell.on {background:#176FC1;border-color:#4DA3FF;color:white;box-shadow:inset 0 0 0 1px rgba(255,255,255,.04);}
.counter-cell.full {background:#0E5A99;border-color:#38BDF8;color:white;}
.legend-row {display:flex;gap:12px;align-items:center;font-size:.78rem;color:var(--muted);margin-top:8px;}
.dot {width:10px;height:10px;border-radius:3px;display:inline-block;margin-right:4px;}

/* Plotly containers blend into the page instead of showing a light box. */
[data-testid="stPlotlyChart"] {
  background:var(--panel) !important;
  border:1px solid rgba(38,58,82,.9);
  border-radius:12px;
  padding:4px;
  overflow:hidden;
}


/* Remove bright rectangles around Streamlit slider tick labels */
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background-color: rgb(240"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background: rgb(240"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background-color: rgba(255"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background: rgba(255"] {
  background:transparent !important;
  border:none !important;
  box-shadow:none !important;
}

/* Scrollbars */
* {scrollbar-color:#38536F #0A1625;scrollbar-width:thin;}
::-webkit-scrollbar {width:10px;height:10px;}
::-webkit-scrollbar-track {background:#0A1625;}
::-webkit-scrollbar-thumb {background:#38536F;border-radius:8px;border:2px solid #0A1625;}
::-webkit-scrollbar-thumb:hover {background:#4B6C8A;}

@media (max-width:1100px) {
  .kpi-grid {grid-template-columns:repeat(2,1fr);}
  .counter-grid {grid-template-columns:repeat(10,1fr);}
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="운영 데이터를 불러오는 중입니다...")
def get_operation_data(path_str: str, mtime: float) -> pd.DataFrame:
    return load_operation_data(path_str)


@st.cache_data(show_spinner="항공편 데이터를 불러오는 중입니다...")
def get_flight_data(path_str: str, mtime: float) -> pd.DataFrame:
    return load_flight_data(path_str)


@st.cache_data(show_spinner=False)
def get_baseline_cached(
    path_str: str,
    mtime: float,
    date: str,
    start_minute: int,
    horizon_min: int,
) -> Dict[str, object]:
    op = load_operation_data(path_str)
    return build_baseline_inputs(op, date, start_minute, horizon_min)


def fmt_num(v: float, digits: int = 0) -> str:
    try:
        if digits == 0:
            return f"{float(v):,.0f}"
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)


def delta_text(new: float, old: float, suffix: str = "") -> str:
    d = float(new) - float(old)
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f}{suffix}"


def kpi_html(items: List[Tuple[str, str, str]]) -> None:
    cards = []
    for label, value, sub in items:
        cards.append(
            f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'
        )
    st.markdown('<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def line_visual_html(area: str, active: int, owner_text: str = "") -> str:
    active = clamp_units(area, active)
    cells = []
    for i in range(1, 41):
        cls = "counter-cell on" if i <= active else "counter-cell"
        if active == 40 and i <= active:
            cls = "counter-cell full"
        cells.append(f'<div class="{cls}">{i}</div>')
    owner = f" · {owner_text}" if owner_text else ""
    return f"""
<div class="counter-wrap">
  <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px;">
    <div><b>{area} 라인</b>{owner}</div>
    <div><b>{active}/40</b> 개방</div>
  </div>
  <div class="counter-grid">{''.join(cells)}</div>
  <div class="legend-row"><span><span class="dot" style="background:#176FC1;border:1px solid #4DA3FF"></span>운영</span><span><span class="dot" style="background:#122238;border:1px solid #2B4058"></span>비운영</span></div>
</div>
"""


def area_owner_labels(flight_df: pd.DataFrame) -> Dict[str, str]:
    labels: Dict[str, List[str]] = {a: [] for a in CHECKIN_AREAS}
    for airline, g in flight_df.groupby("항공사"):
        modes = g["체크인카운터_보정"].dropna().astype(str)
        if modes.empty:
            continue
        mapping = modes.mode().iloc[0]
        for area in [x.strip() for x in mapping.split(",")]:
            if area in labels:
                labels[area].append(str(airline))
    return {k: ", ".join(sorted(set(v))) for k, v in labels.items()}


def long_with_time(long_df: pd.DataFrame, minutes: List[int], label: str) -> pd.DataFrame:
    out = long_df.copy()
    minute_map = {i: minutes[i] for i in range(len(minutes))}
    out["분"] = out["minute_index"].map(minute_map)
    out["시각"] = out["분"].apply(minute_to_hhmm)
    out["시나리오"] = label
    return out


def top_bottlenecks(long_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return (
        long_df.groupby("area", as_index=False)
        .agg(
            avg_wait_min=("wait_min", "mean"),
            max_wait_min=("wait_min", "max"),
            max_queue=("queue", "max"),
            avg_utilization=("utilization", "mean"),
        )
        .sort_values(["max_wait_min", "max_queue"], ascending=False)
        .head(n)
    )


def build_minimum_units(snap: pd.DataFrame) -> Dict[str, int]:
    mins: Dict[str, int] = {}
    for _, row in snap.iterrows():
        area = str(row["구역"])
        plan = int(round(float(row["계획오픈수"])))
        demand_present = float(row["실시간인원수"]) > 0 or float(row["계획수요"]) > 0
        mins[area] = minimum_operating_units(area, plan, demand_present)
    return mins


def scenario_signature(
    date: str,
    minute: int,
    horizon: int,
    units: Mapping[str, int],
    airline: str,
    shock: float,
    lag: int,
) -> str:
    payload = {
        "date": date,
        "minute": minute,
        "horizon": horizon,
        "units": {k: int(v) for k, v in sorted(units.items())},
        "airline": airline,
        "shock": float(shock),
        "lag": int(lag),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def safe_secret(name: str) -> str:
    """Read an API key without triggering Streamlit's missing-secrets error banner.

    Priority: environment variable -> existing Streamlit secrets file -> empty string.
    """
    env_value = os.getenv(name, "")
    if env_value:
        return str(env_value)

    candidates = [
        Path.home() / ".streamlit" / "secrets.toml",
        BASE_DIR / ".streamlit" / "secrets.toml",
    ]
    if not any(p.exists() for p in candidates):
        return ""

    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""



def sync_focus_slider(area: str, slider_key: str) -> None:
    if "scenario_units" in st.session_state and slider_key in st.session_state:
        st.session_state["scenario_units"][area] = clamp_units(area, int(st.session_state[slider_key]))


def set_focus_units(area: str, slider_key: str, value: int) -> None:
    """Safely update a focus-line slider from a widget callback."""
    value = clamp_units(area, int(value))
    if "scenario_units" not in st.session_state:
        st.session_state["scenario_units"] = {}
    st.session_state["scenario_units"][area] = value
    st.session_state[slider_key] = value


def queue_focus_slider_update(area: str, slider_key: str, value: int) -> None:
    """Queue a focus slider update so it is applied before the next widget instantiation."""
    value = clamp_units(area, int(value))
    if "scenario_units" not in st.session_state:
        st.session_state["scenario_units"] = {}
    st.session_state["scenario_units"][area] = value
    st.session_state["pending_focus_slider"] = (slider_key, value)


def apply_optimized_plan_callback(opt_units: dict, focus_line: str, state_key: str) -> None:
    """Apply an optimized plan without mutating an already-instantiated slider.

    Streamlit forbids assigning to a widget key after that widget has been created
    in the current run.  A button callback executes before the next rerun, so we
    update the model state here and queue the focus-slider value to be written
    before the slider is instantiated on the following run.
    """
    normalized = {area: clamp_units(area, int(opt_units[area])) for area in ALL_AREAS}
    st.session_state["scenario_units"] = normalized
    slider_key = f"focus_slider_{state_key}_{focus_line}"
    st.session_state["pending_focus_slider"] = (slider_key, normalized[focus_line])
    st.session_state["optimizer_apply_notice"] = state_key


# -----------------------------------------------------------------------------
# Data load & validation
# -----------------------------------------------------------------------------
if not OPERATION_PATH.exists() or not FLIGHT_PATH.exists():
    st.error("필수 데이터 파일을 찾을 수 없습니다. data 폴더에 운영 데이터와 항공편 CSV를 넣어주세요.")
    st.stop()

operation_df = get_operation_data(str(OPERATION_PATH), OPERATION_PATH.stat().st_mtime)
flight_df = get_flight_data(str(FLIGHT_PATH), FLIGHT_PATH.stat().st_mtime)
quality = data_quality_report(operation_df, flight_df)
owners = area_owner_labels(flight_df)

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 관제 기준")
    dates = date_options(operation_df)
    selected_date = st.selectbox("날짜", dates, index=min(13, len(dates) - 1))
    time_options = [minute_to_hhmm(m) for m in range(0, 24 * 60, 15)]
    selected_time = st.selectbox("기준 시각", time_options, index=time_options.index("08:00"))
    start_minute = hhmm_to_minute(selected_time)
    max_horizon = max(30, min(240, 1440 - start_minute))
    horizon_options = [x for x in [60, 90, 120, 180, 240] if x <= max_horizon]
    if not horizon_options:
        horizon_options = [max_horizon]
    horizon_min = st.selectbox("시뮬레이션 범위", horizon_options, index=min(2, len(horizon_options) - 1), format_func=lambda x: f"{x}분")
    travel_lag = st.slider("체크인→출국장 이동 지연", 3, 20, 8, 1, help="체크인 처리량 변화가 IM1/IM2 유입에 반영되기까지의 시간")
    mc_iterations = st.select_slider("불확실성 반복", options=[10, 20, 30, 50], value=30)

    st.divider()
    st.markdown("### 항공사 수요 시나리오")
    airline_list = sorted(flight_df[flight_df["일자_dt"].between(pd.Timestamp(DATE_MIN), pd.Timestamp(DATE_MAX))]["항공사"].dropna().astype(str).unique().tolist())
    selected_airline = st.selectbox("대상 항공사", ["적용 안 함"] + airline_list)
    demand_change_pct = st.slider("해당 항공사 수요 변화", -30, 50, 0, 5, format="%d%%")

    st.divider()
    st.caption("데이터: 2025-09-01 ~ 2025-10-31")
    if quality["operation_duplicates"] == 0:
        st.success("운영 데이터 중복 키 없음", icon="✅")
    if quality["missing_counter_rows"] > 0:
        st.warning(
            f"체크인 구역 결측 {quality['missing_counter_rows']}건은 항공사 대표 구역으로 보정",
            icon="⚠️",
        )


# -----------------------------------------------------------------------------
# Base state and scenario state
# -----------------------------------------------------------------------------
snap = snapshot(operation_df, selected_date, start_minute)
base = get_baseline_cached(
    str(OPERATION_PATH),
    OPERATION_PATH.stat().st_mtime,
    selected_date,
    start_minute,
    int(horizon_min),
)
minutes: List[int] = list(base["minutes"])

baseline_units_snapshot = {
    str(row["구역"]): int(row["권고필요수"])
    for _, row in snap.iterrows()
}

state_key = f"{selected_date}|{start_minute}|{horizon_min}"
if st.session_state.get("scenario_state_key") != state_key:
    st.session_state["scenario_state_key"] = state_key
    st.session_state["scenario_units"] = dict(baseline_units_snapshot)
    st.session_state.pop("mc_result", None)
    st.session_state.pop("optimizer_result", None)

scenario_units: Dict[str, int] = {
    a: clamp_units(a, int(st.session_state.get("scenario_units", {}).get(a, baseline_units_snapshot.get(a, 0))))
    for a in ALL_AREAS
}

multiplier, airline_detail = build_airline_share_multipliers(
    flight_df=flight_df,
    date=selected_date,
    minutes=minutes,
    selected_airline=None if selected_airline == "적용 안 함" else selected_airline,
    demand_change_pct=float(demand_change_pct),
)
scenario_arrivals = {
    a: np.asarray(base["arrivals"][a], dtype=float) * np.asarray(multiplier.get(a, np.ones(len(minutes))), dtype=float)
    for a in ALL_AREAS
}

# Deterministic baseline and current scenario are cheap enough to refresh interactively.
baseline_long, _ = simulate_coupled_system(
    arrivals_by_area=base["arrivals"],
    initial_queue_by_area=base["initial_queue"],
    units_by_area=base["baseline_units"],
    baseline_checkin_processed=base["baseline_checkin_processed"],
    baseline_im_arrivals=base["baseline_im_arrivals"],
    im_split=base["im_split"],
    travel_lag_min=travel_lag,
)
scenario_long, _ = simulate_coupled_system(
    arrivals_by_area=scenario_arrivals,
    initial_queue_by_area=base["initial_queue"],
    units_by_area=scenario_units,
    baseline_checkin_processed=base["baseline_checkin_processed"],
    baseline_im_arrivals=base["baseline_im_arrivals"],
    im_split=base["im_split"],
    travel_lag_min=travel_lag,
)
baseline_metrics = compute_metrics(baseline_long)
scenario_metrics = compute_metrics(scenario_long)

# Header
st.markdown(
    f"""
<div class="hero">
  <h1>ICN T2 가상 운영 시나리오 & AI 의사결정 지원</h1>
  <p>2025년 9월~10월 실제 1분 단위 운영 데이터를 기준으로 체크인 라인·셀프기기·IM 출입문 재배치 효과를 사전 검증합니다.</p>
</div>
""",
    unsafe_allow_html=True,
)

current_people = float(snap["실시간인원수"].sum())
peak_area_row = snap.sort_values("실시간인원수", ascending=False).iloc[0]
base_staff_now = total_staff(baseline_units_snapshot)
scenario_staff_now = total_staff(scenario_units)
active_flights = airline_summary(flight_df, selected_date, start_minute, horizon_min)

kpi_html(
    [
        ("현재 총 구역 인원", f"{fmt_num(current_people)}명", "A~N, IM1·IM2 합계"),
        ("최대 부하 구역", str(peak_area_row["구역"]), f"{fmt_num(peak_area_row['실시간인원수'],1)}명"),
        ("기준 필요 인력", f"{base_staff_now}명", "현재 시각 권고 운영 수 기준"),
        ("시나리오 필요 인력", f"{scenario_staff_now}명", f"기준 대비 {scenario_staff_now-base_staff_now:+d}명"),
        ("분석 범위", f"{horizon_min}분", f"{selected_time}부터 {minute_to_hhmm(min(1439,start_minute+horizon_min-1))}"),
    ]
)

st.markdown(
    '<span class="badge badge-blue">과거 재현 시뮬레이션</span>'
    '<span class="badge badge-green">9~10월 실제 운영 데이터</span>'
    '<span class="badge badge-warn">대기시간은 모델 추정값</span>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_scenario, tab_result, tab_opt, tab_ai, tab_quality = st.tabs(
    ["① 운영 현황", "② 시나리오 설정", "③ 결과 비교", "④ 자동 최적화", "⑤ AI 운영 질의", "⑥ 데이터 검수"]
)


# -----------------------------------------------------------------------------
# 1) Overview
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader(f"{selected_date} {selected_time} 기준 운영 상태")
    st.caption("계획 운영, 센서 인원 기준 필요 수, 기존 시스템 권고 수를 한 화면에서 비교합니다.")

    chart_df = snap[["구역", "계획오픈수", "실시간필요수", "권고필요수"]].melt(
        id_vars="구역",
        var_name="구분",
        value_name="운영 수",
    )
    chart_df["구분"] = chart_df["구분"].replace(
        {"계획오픈수": "항공편 기반 계획", "실시간필요수": "인원 기준 필요", "권고필요수": "기존 시스템 권고"}
    )
    fig = px.bar(chart_df, x="구역", y="운영 수", color="구분", barmode="group", text_auto=".0f")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="", yaxis_title="운영 수", xaxis_title="")
    st.plotly_chart(darken_plot(fig), use_container_width=True)

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="section-title">향후 구역별 인원 부하</div>', unsafe_allow_html=True)
        h = operation_df[
            (operation_df["일자"] == selected_date)
            & (operation_df["분"] >= start_minute)
            & (operation_df["분"] < start_minute + horizon_min)
        ].copy()
        pivot = h.pivot_table(index="구역", columns="분", values="실시간인원수", aggfunc="last").reindex(ALL_AREAS)
        if not pivot.empty:
            tick_step = max(1, len(pivot.columns) // 8)
            xvals = list(pivot.columns)
            fig_h = go.Figure(
                data=go.Heatmap(
                    z=pivot.to_numpy(),
                    colorscale=DARK_HEAT_SCALE,
                    x=[minute_to_hhmm(int(x)) for x in xvals],
                    y=pivot.index.tolist(),
                    colorbar=dict(title="인원"),
                    hovertemplate="구역=%{y}<br>시각=%{x}<br>인원=%{z:.1f}<extra></extra>",
                )
            )
            fig_h.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
            st.plotly_chart(darken_plot(fig_h), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">현재 병목 후보</div>', unsafe_allow_html=True)
        bottleneck_now = snap.copy()
        bottleneck_now["부하/운영"] = bottleneck_now.apply(
            lambda r: float(r["실시간인원수"]) / max(1, int(r["권고필요수"])), axis=1
        )
        show = bottleneck_now.sort_values("부하/운영", ascending=False).head(8)[
            ["구역", "유형", "실시간인원수", "계획오픈수", "실시간필요수", "권고필요수", "부하/운영"]
        ].copy()
        show.columns = ["구역", "유형", "현재 인원", "계획", "인원 기준", "권고", "단위당 부하"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">향후 항공편 수요</div>', unsafe_allow_html=True)
        st.dataframe(active_flights.head(10), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 2) Scenario controls
# -----------------------------------------------------------------------------
with tab_scenario:
    st.subheader("운영안 직접 조정")
    st.caption("유인 체크인 라인은 각 1~40번을 하나의 연속 라인으로 보고, 셀프기기와 IM 출입문은 별도 자원으로 조정합니다.")

    staffed_snap = snap[snap["구역"].isin(STAFFED_AREAS)].copy()
    focus_line = st.selectbox("상세 조정할 체크인 라인", STAFFED_AREAS, index=STAFFED_AREAS.index("H") if "H" in STAFFED_AREAS else 0)
    owner_text = owners.get(focus_line, "")

    slider_key = f"focus_slider_{state_key}_{focus_line}"
    pending_focus = st.session_state.pop("pending_focus_slider", None)
    if pending_focus is not None:
        pending_key, pending_value = pending_focus
        if pending_key == slider_key:
            st.session_state[slider_key] = int(pending_value)
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(scenario_units[focus_line])

    c1, c2, c3, c4, c5 = st.columns([1.25, 1, 1, 1, 1])
    with c1:
        st.slider(
            f"{focus_line} 라인 개방 수",
            min_value=0,
            max_value=40,
            step=1,
            key=slider_key,
            on_change=sync_focus_slider,
            args=(focus_line, slider_key),
        )
    with c2:
        st.button(
            "1~40 전체 개방",
            use_container_width=True,
            on_click=set_focus_units,
            args=(focus_line, slider_key, 40),
        )
    with c3:
        restored = int(baseline_units_snapshot[focus_line])
        st.button(
            "기준 권고로 복원",
            use_container_width=True,
            on_click=set_focus_units,
            args=(focus_line, slider_key, restored),
        )
    with c4:
        reduced = max(0, int(scenario_units[focus_line]) - 5)
        st.button(
            "5개 감축",
            use_container_width=True,
            on_click=set_focus_units,
            args=(focus_line, slider_key, reduced),
        )
    with c5:
        increased = min(40, int(scenario_units[focus_line]) + 5)
        st.button(
            "5개 추가",
            use_container_width=True,
            on_click=set_focus_units,
            args=(focus_line, slider_key, increased),
        )

    st.markdown(line_visual_html(focus_line, scenario_units[focus_line], owner_text), unsafe_allow_html=True)

    st.markdown('<div class="section-title">전체 자원 조정표</div>', unsafe_allow_html=True)
    edit_rows = []
    snap_idx = snap.set_index("구역")
    for area in ALL_AREAS:
        row = snap_idx.loc[area]
        edit_rows.append(
            {
                "구역": area,
                "유형": AREA_TYPES[area],
                "현재 인원": round(float(row["실시간인원수"]), 1),
                "계획 운영": int(row["계획오픈수"]),
                "인원 기준 필요": int(row["실시간필요수"]),
                "기준 권고": int(row["권고필요수"]),
                "시나리오 운영": int(scenario_units[area]),
                "최대": int(MAX_UNITS[area]),
                "기준 직원": staff_from_units(area, int(row["권고필요수"])),
                "시나리오 직원": staff_from_units(area, int(scenario_units[area])),
            }
        )
    editor_df = pd.DataFrame(edit_rows)
    disabled_cols = ["구역", "유형", "현재 인원", "계획 운영", "인원 기준 필요", "기준 권고", "최대", "기준 직원", "시나리오 직원"]
    changed = False
    editor_token = "-".join(str(int(scenario_units[a])) for a in ALL_AREAS)

    edit_tab1, edit_tab2, edit_tab3 = st.tabs(["유인 체크인 라인", "셀프 체크인 기기", "IM1·IM2 출입문"])
    edited_frames = []
    with edit_tab1:
        edited_frames.append(
            st.data_editor(
                editor_df[editor_df["구역"].isin(STAFFED_AREAS)].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                disabled=disabled_cols,
                column_config={"시나리오 운영": st.column_config.NumberColumn("시나리오 운영", min_value=0, max_value=40, step=1, required=True)},
                key=f"resource_editor_staffed_{state_key}_{editor_token}",
            )
        )
    with edit_tab2:
        edited_frames.append(
            st.data_editor(
                editor_df[editor_df["구역"].isin(SELF_AREAS)].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                disabled=disabled_cols,
                column_config={"시나리오 운영": st.column_config.NumberColumn("시나리오 운영", min_value=0, max_value=40, step=1, required=True)},
                key=f"resource_editor_self_{state_key}_{editor_token}",
            )
        )
    with edit_tab3:
        edited_frames.append(
            st.data_editor(
                editor_df[editor_df["구역"].isin(IM_AREAS)].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
                disabled=disabled_cols,
                column_config={"시나리오 운영": st.column_config.NumberColumn("시나리오 운영", min_value=0, max_value=6, step=1, required=True)},
                key=f"resource_editor_im_{state_key}_{editor_token}",
            )
        )

    for edited in edited_frames:
        for _, row in edited.iterrows():
            area = str(row["구역"])
            max_u = MAX_UNITS[area]
            value = max(0, min(max_u, int(row["시나리오 운영"])))
            if value != scenario_units[area]:
                st.session_state["scenario_units"][area] = value
                if area == focus_line:
                    st.session_state["pending_focus_slider"] = (slider_key, value)
                changed = True
    if changed:
        st.rerun()

    # Airline scenario context
    st.markdown('<div class="section-title">항공사 시나리오 영향 구역</div>', unsafe_allow_html=True)
    if selected_airline == "적용 안 함" or demand_change_pct == 0:
        st.info("사이드바에서 항공사와 수요 변화율을 선택하면 해당 항공사의 항공편 좌석 비중에 따라 관련 구역 유입량이 조정됩니다.")
    else:
        airline_rows = flight_df[(flight_df["항공사"] == selected_airline) & flight_df["일자_dt"].between(pd.Timestamp(DATE_MIN), pd.Timestamp(DATE_MAX))]
        mode = airline_rows["체크인카운터_보정"].mode().iloc[0] if not airline_rows.empty else ""
        weights = airline_area_weights(mode)
        tags = " ".join([f'<span class="badge badge-blue">{a} {w*100:.0f}%</span>' for a, w in weights.items()])
        st.markdown(f"**{selected_airline}** 대표 체크인 구역: {mode}<br>{tags}", unsafe_allow_html=True)
        if not airline_detail.empty:
            st.dataframe(
                airline_detail.sort_values("출발기준시각").head(20),
                use_container_width=True,
                hide_index=True,
            )

    # Staffing feasibility
    base_staff = total_staff(baseline_units_snapshot)
    scen_staff = total_staff(scenario_units)
    status_cls = "badge-green" if scen_staff <= base_staff else "badge-warn"
    st.markdown(
        f'<div class="notice"><b>현재 시각 필요 인력</b> &nbsp; 기준 {base_staff}명 → 시나리오 {scen_staff}명 '
        f'<span class="badge {status_cls}">{scen_staff-base_staff:+d}명</span><br>'
        '유인 체크인 라인의 물리적 상한은 각각 40개이며, 자동 최적화에서는 가용 인력 한도를 별도로 적용합니다.</div>',
        unsafe_allow_html=True,
    )

    sim_sig = scenario_signature(selected_date, start_minute, horizon_min, scenario_units, selected_airline, demand_change_pct, travel_lag)
    if st.button("불확실성 포함 시뮬레이션 실행", type="primary", use_container_width=True):
        with st.spinner("기준안과 변경안을 동일한 난수 조건으로 반복 비교하는 중입니다..."):
            mc = monte_carlo_compare(
                arrivals_by_area=base["arrivals"],
                initial_queue_by_area=base["initial_queue"],
                baseline_units=base["baseline_units"],
                scenario_units=scenario_units,
                baseline_checkin_processed=base["baseline_checkin_processed"],
                baseline_im_arrivals=base["baseline_im_arrivals"],
                im_split=base["im_split"],
                scenario_arrival_multiplier=multiplier,
                travel_lag_min=travel_lag,
                iterations=mc_iterations,
                seed=42,
            )
        st.session_state["mc_result"] = mc
        st.session_state["mc_signature"] = sim_sig
        st.success(f"{mc_iterations}회 반복 비교가 완료되었습니다.")


# -----------------------------------------------------------------------------
# 3) Results
# -----------------------------------------------------------------------------
with tab_result:
    st.subheader("기준 운영안 vs 변경 운영안")
    st.caption("절대 대기시간보다 동일 데이터에 대한 기준안 대비 변화량을 중심으로 해석하세요.")

    wait_change = scenario_metrics.avg_wait_min - baseline_metrics.avg_wait_min
    p90_change = scenario_metrics.p90_wait_min - baseline_metrics.p90_wait_min
    queue_change = scenario_metrics.max_queue - baseline_metrics.max_queue
    staff_change = scenario_metrics.peak_staff - baseline_metrics.peak_staff

    kpi_html(
        [
            ("평균 대기시간", f"{scenario_metrics.avg_wait_min:.1f}분", f"기준 {baseline_metrics.avg_wait_min:.1f}분 · {wait_change:+.1f}분"),
            ("P90 대기시간", f"{scenario_metrics.p90_wait_min:.1f}분", f"기준 {baseline_metrics.p90_wait_min:.1f}분 · {p90_change:+.1f}분"),
            ("최대 대기열", f"{scenario_metrics.max_queue:.0f}명", f"기준 {baseline_metrics.max_queue:.0f}명 · {queue_change:+.0f}명"),
            ("혼잡 지속", f"{scenario_metrics.congestion_minutes}분", f"기준 {baseline_metrics.congestion_minutes}분"),
            ("피크 필요 인력", f"{scenario_metrics.peak_staff}명", f"기준 {baseline_metrics.peak_staff}명 · {staff_change:+d}명"),
        ]
    )

    if wait_change < -0.5 and any(a.startswith("IM") for a in top_bottlenecks(scenario_long, 3)["area"].tolist()):
        st.warning("체크인 병목은 개선되지만 상위 병목에 IM 구역이 포함됩니다. 체크인 처리량 증가가 출국장 진입 혼잡으로 전이되는지 확인하세요.")
    elif wait_change > 0.5:
        st.error("현재 변경안은 평균 대기시간을 증가시킵니다. 자원 배치를 다시 조정하거나 자동 최적화를 사용하세요.")
    else:
        st.info("변경안의 전체 평균 효과는 제한적입니다. 특정 구역의 개선과 다른 구역의 악화를 함께 확인하세요.")

    btime = long_with_time(baseline_long, minutes, "기준안")
    stime = long_with_time(scenario_long, minutes, "변경안")
    combined = pd.concat([btime, stime], ignore_index=True)

    # Minute-by-minute diagnostics: a plan is considered strictly optimized only
    # when no observed aggregate wait indicator worsens at any minute.
    temporal_stats, temporal_df = temporal_comparison(baseline_long, scenario_long)
    minute_to_clock = {i: minute_to_hhmm(minutes[i]) for i in range(len(minutes))}
    temporal_df["시각"] = temporal_df["minute_index"].map(minute_to_clock)

    st.markdown('<div class="section-title">시간대별 비악화 검증</div>', unsafe_allow_html=True)
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    with tc1:
        st.metric("전 시간대 비악화", "충족" if temporal_stats["strict_nonworsening"] else "미충족")
    with tc2:
        st.metric("대기부하 개선 시간", f"{temporal_stats['improvement_ratio']*100:.0f}%")
    with tc3:
        st.metric("악화 시간", f"{temporal_stats['worsening_minutes']}분")
    with tc4:
        st.metric("최대 순간 부하 악화", f"+{temporal_stats['max_queue_worsening']:.0f}명")
    with tc5:
        st.metric("누적 대기부하 절감", f"{temporal_stats['cumulative_queue_saved_person_min']:.0f} 인·분")

    if temporal_stats["strict_nonworsening"] and temporal_stats["any_improvement"]:
        st.success("이 변경안은 현재 집계 데이터로 관측 가능한 세 지표에서 전 시간대 비악화 조건을 만족하며, 일부 시간대에서는 실제 개선됩니다.")
    elif temporal_stats["strict_nonworsening"]:
        st.info("전 시간대 비악화 조건은 만족하지만 기준안과 실질적인 차이가 거의 없습니다.")
    else:
        st.warning(
            f"총 평균이 좋아 보여도 {temporal_stats['worsening_minutes']}분 동안 최소 한 지표가 기준보다 악화됩니다. "
            "엄격 최적화에서는 이런 운영안을 최적화안으로 인정하지 않습니다."
        )

    # Total estimated queue/load chart with improvement/worsening intervals.
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(
        x=temporal_df["minute_index"], y=temporal_df["reference_queue"],
        mode="lines", name="기준안", line=dict(color="#94A3B8", width=2.2), customdata=temporal_df["시각"],
        hovertemplate="%{customdata}<br>기준안 %{y:.0f}명<extra></extra>",
    ))
    fig_q.add_trace(go.Scatter(
        x=temporal_df["minute_index"], y=temporal_df["candidate_queue"],
        mode="lines", name="변경안", line=dict(color="#38BDF8", width=2.6), customdata=temporal_df["시각"],
        hovertemplate="%{customdata}<br>변경안 %{y:.0f}명<extra></extra>",
    ))

    states = temporal_df["queue_state"].tolist()
    if states:
        start_i = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start_i]:
                state = states[start_i]
                if state in {"개선", "악화"}:
                    fill = "rgba(45,212,191,0.12)" if state == "개선" else "rgba(251,113,133,0.12)"
                    fig_q.add_vrect(
                        x0=float(temporal_df.iloc[start_i]["minute_index"]) - 0.5,
                        x1=float(temporal_df.iloc[i-1]["minute_index"]) + 0.5,
                        fillcolor=fill, line_width=0, layer="below",
                    )
                start_i = i

    crossings = temporal_stats.get("crossing_minute_indices", [])
    for j, cross_idx in enumerate(crossings[:3]):
        label = minute_to_clock.get(int(cross_idx), "")
        fig_q.add_vline(
            x=int(cross_idx), line_width=1, line_dash="dot", line_color="#CBD5E1",
            annotation_text=(f"교차 {label}" if j == 0 else None),
            annotation_position="top",
        )

    tick_step = max(1, len(temporal_df) // 12)
    tick_rows = temporal_df.iloc[::tick_step]
    fig_q.update_layout(
        height=430, margin=dict(l=10, r=10, t=35, b=10), legend_title_text="",
        xaxis_title="", yaxis_title="전체 추정 대기부하(명)", hovermode="x unified",
        title="전체 추정 대기부하 · 녹색 배경=개선 / 적색 배경=악화",
    )
    fig_q.update_xaxes(tickmode="array", tickvals=tick_rows["minute_index"], ticktext=tick_rows["시각"])
    st.plotly_chart(darken_plot(fig_q), use_container_width=True)

    fig_tw = go.Figure()
    fig_tw.add_trace(go.Scatter(
        x=temporal_df["minute_index"], y=temporal_df["reference_weighted_wait"], mode="lines",
        name="기준안", line=dict(color="#94A3B8", width=2.2), customdata=temporal_df["시각"],
        hovertemplate="%{customdata}<br>기준안 %{y:.1f}분<extra></extra>",
    ))
    fig_tw.add_trace(go.Scatter(
        x=temporal_df["minute_index"], y=temporal_df["candidate_weighted_wait"], mode="lines",
        name="변경안", line=dict(color="#38BDF8", width=2.6), customdata=temporal_df["시각"],
        hovertemplate="%{customdata}<br>변경안 %{y:.1f}분<extra></extra>",
    ))
    fig_tw.update_layout(
        height=330, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="",
        xaxis_title="", yaxis_title="대기열 가중 평균 대기시간(분)", hovermode="x unified",
        title="시간대별 추정 평균 대기시간",
    )
    fig_tw.update_xaxes(tickmode="array", tickvals=tick_rows["minute_index"], ticktext=tick_rows["시각"])
    st.plotly_chart(darken_plot(fig_tw), use_container_width=True)

    if crossings:
        crossing_text = ", ".join(minute_to_clock.get(int(x), str(x)) for x in crossings[:5])
        st.caption(f"총 대기부하 기준 교차 시점: {crossing_text}" + (" …" if len(crossings) > 5 else ""))
    else:
        st.caption("총 대기부하 선의 교차가 없습니다. 변경안이 전 구간에서 한쪽에 위치합니다.")

    c1, c2 = st.columns(2)
    with c1:
        area_cmp = pd.concat(
            [
                btime.groupby("area", as_index=False).agg(평균대기=("wait_min", "mean"), 최대대기열=("queue", "max")).assign(시나리오="기준안"),
                stime.groupby("area", as_index=False).agg(평균대기=("wait_min", "mean"), 최대대기열=("queue", "max")).assign(시나리오="변경안"),
            ],
            ignore_index=True,
        )
        fig_w = px.bar(area_cmp, x="area", y="평균대기", color="시나리오", barmode="group", labels={"area": "구역", "평균대기": "평균 대기시간(분)"})
        fig_w.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="", xaxis_title="")
        st.plotly_chart(darken_plot(fig_w), use_container_width=True)
    with c2:
        scenario_pivot = stime.pivot_table(index="area", columns="시각", values="wait_min", aggfunc="mean").reindex(ALL_AREAS)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=scenario_pivot.to_numpy(),
                colorscale=DARK_HEAT_SCALE,
                x=scenario_pivot.columns.tolist(),
                y=scenario_pivot.index.tolist(),
                colorbar=dict(title="대기분"),
                hovertemplate="구역=%{y}<br>시각=%{x}<br>대기=%{z:.1f}분<extra></extra>",
            )
        )
        fig_heat.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(darken_plot(fig_heat), use_container_width=True)

    st.markdown('<div class="section-title">병목 순위</div>', unsafe_allow_html=True)
    bott = top_bottlenecks(scenario_long, 8).rename(
        columns={
            "area": "구역",
            "avg_wait_min": "평균 대기(분)",
            "max_wait_min": "최대 대기(분)",
            "max_queue": "최대 대기열",
            "avg_utilization": "평균 부하율",
        }
    )
    st.dataframe(bott, use_container_width=True, hide_index=True)

    current_sig = scenario_signature(selected_date, start_minute, horizon_min, scenario_units, selected_airline, demand_change_pct, travel_lag)
    mc = st.session_state.get("mc_result")
    if mc is not None and st.session_state.get("mc_signature") == current_sig:
        st.markdown('<div class="section-title">Monte Carlo 불확실성 비교</div>', unsafe_allow_html=True)
        summary = (
            mc.groupby("scenario")
            .agg(
                평균대기_중앙값=("avg_wait_min", "median"),
                평균대기_P10=("avg_wait_min", lambda s: s.quantile(0.10)),
                평균대기_P90=("avg_wait_min", lambda s: s.quantile(0.90)),
                최대대기열_중앙값=("max_queue", "median"),
                피크인력_중앙값=("peak_staff", "median"),
            )
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        fig_mc = px.box(mc, x="scenario", y="avg_wait_min", points="all", color="scenario", color_discrete_sequence=["#94A3B8", "#38BDF8"], labels={"avg_wait_min": "평균 대기시간(분)", "scenario": ""})
        fig_mc.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(darken_plot(fig_mc), use_container_width=True)
    else:
        st.caption("② 시나리오 설정에서 '불확실성 포함 시뮬레이션 실행'을 누르면 반복 결과의 범위를 확인할 수 있습니다.")


# -----------------------------------------------------------------------------
# 4) Auto optimization
# -----------------------------------------------------------------------------
with tab_opt:
    st.subheader("제한 인력 기반 자동 재배치")
    st.caption("현재 권고 운영안을 시작점으로 두고, 최소 유지 기준과 각 라인의 1~40 상한을 지키면서 추가·감축·재배치를 반복 탐색합니다.")

    oc1, oc2, oc3, oc4 = st.columns([1.2, 1, 1, 1.35])
    with oc1:
        objective = st.selectbox("최적화 목표", ["균형 운영", "대기시간 최소화", "최소 인력 운영"])
    with oc2:
        additional_staff = st.number_input("추가 투입 가능 인력", min_value=0, max_value=50, value=4, step=1)
    with oc3:
        optimization_steps = st.slider("탐색 단계", 5, 40, 20, 1)
    with oc4:
        strict_nonworsening = st.checkbox(
            "전 시간대 비악화 보장", value=True,
            help="매 1분마다 전체 추정 대기부하, 대기열 가중 평균 대기시간, 최악 구역 대기시간이 기준 운영보다 높아지는 후보를 제외합니다.",
        )

    staff_budget = base_staff_now + int(additional_staff)
    st.markdown(
        f'<div class="notice">현재 기준 인력 <b>{base_staff_now}명</b> + 추가 가능 <b>{int(additional_staff)}명</b> = 최적화 인력 한도 <b>{staff_budget}명</b></div>',
        unsafe_allow_html=True,
    )

    if st.button("자동 최적화 실행", type="primary", use_container_width=True):
        mins = build_minimum_units(snap)
        with st.spinner("대기시간·인력·혼잡 지속을 함께 비교하며 자원을 재배치하는 중입니다..."):
            opt_units, opt_metrics, opt_long = optimize_fixed_allocation(
                arrivals_by_area=scenario_arrivals,
                initial_queue_by_area=base["initial_queue"],
                starting_units=baseline_units_snapshot,
                minimum_units=mins,
                staff_budget=staff_budget,
                baseline_checkin_processed=base["baseline_checkin_processed"],
                baseline_im_arrivals=base["baseline_im_arrivals"],
                im_split=base["im_split"],
                travel_lag_min=travel_lag,
                objective=objective,
                max_steps=optimization_steps,
                require_temporal_nonworsening=strict_nonworsening,
            )
            opt_reference_long, _ = simulate_coupled_system(
                arrivals_by_area=scenario_arrivals,
                initial_queue_by_area=base["initial_queue"],
                units_by_area=baseline_units_snapshot,
                baseline_checkin_processed=base["baseline_checkin_processed"],
                baseline_im_arrivals=base["baseline_im_arrivals"],
                im_split=base["im_split"],
                travel_lag_min=travel_lag,
            )
            opt_reference_metrics = compute_metrics(opt_reference_long)
            opt_temporal_stats, _ = temporal_comparison(opt_reference_long, opt_long)
        st.session_state["optimizer_result"] = {
            "units": opt_units,
            "metrics": opt_metrics,
            "long": opt_long,
            "objective": objective,
            "staff_budget": staff_budget,
            "reference_metrics": opt_reference_metrics,
            "temporal_stats": opt_temporal_stats,
            "strict_nonworsening": strict_nonworsening,
            "state_key": state_key,
            "airline": selected_airline,
            "shock": demand_change_pct,
        }

    opt = st.session_state.get("optimizer_result")
    if opt and opt.get("state_key") == state_key and opt.get("airline") == selected_airline and opt.get("shock") == demand_change_pct:
        opt_units = opt["units"]
        opt_metrics = opt["metrics"]
        changes = []
        for area in ALL_AREAS:
            before = baseline_units_snapshot[area]
            after = int(opt_units[area])
            if before != after:
                changes.append(
                    {
                        "구역": area,
                        "유형": AREA_TYPES[area],
                        "기준": before,
                        "최적화": after,
                        "변화": after - before,
                        "기준직원": staff_from_units(area, before),
                        "최적화직원": staff_from_units(area, after),
                    }
                )

        opt_ref_metrics = opt.get("reference_metrics", baseline_metrics)
        opt_temporal = opt.get("temporal_stats", {})
        kpi_html(
            [
                ("최적화 평균 대기", f"{opt_metrics.avg_wait_min:.1f}분", f"동일 수요 기준 {opt_ref_metrics.avg_wait_min:.1f}분"),
                ("최적화 P90", f"{opt_metrics.p90_wait_min:.1f}분", f"동일 수요 기준 {opt_ref_metrics.p90_wait_min:.1f}분"),
                ("최대 대기열", f"{opt_metrics.max_queue:.0f}명", f"동일 수요 기준 {opt_ref_metrics.max_queue:.0f}명"),
                ("피크 인력", f"{opt_metrics.peak_staff}명", f"한도 {opt['staff_budget']}명"),
                ("전 시간대 비악화", "충족" if opt_temporal.get("strict_nonworsening", False) else "미충족", f"악화 {int(opt_temporal.get('worsening_minutes', 0))}분"),
            ]
        )
        if opt.get("strict_nonworsening", True):
            if opt_temporal.get("strict_nonworsening", False) and opt_temporal.get("any_improvement", False):
                st.success("엄격 최적화 조건 충족: 분석 범위의 어느 1분에서도 기준 운영보다 악화되지 않으면서 일부 시간대가 개선됩니다.")
            elif opt_temporal.get("strict_nonworsening", False):
                st.info("전 시간대 비악화 조건은 충족하지만, 현재 인력·운영 제약에서는 기준안보다 실질적으로 더 나은 재배치안을 찾지 못했습니다.")
            else:
                st.error("엄격 최적화 조건을 만족하는 운영안을 찾지 못했습니다. 현재 결과는 적용하지 않는 것이 좋습니다.")
        else:
            st.warning("전 시간대 비악화 보장을 끈 탐색 결과입니다. 평균은 개선되어도 일부 시간대가 악화될 수 있습니다.")
        if changes:
            st.dataframe(pd.DataFrame(changes).sort_values("변화", key=lambda s: s.abs(), ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("주어진 인력 한도와 목적함수에서는 현재 기준 운영안을 유지하는 결과가 나왔습니다.")

        st.button(
            "최적화 운영안을 시나리오에 적용",
            use_container_width=True,
            on_click=apply_optimized_plan_callback,
            args=(dict(opt_units), focus_line, state_key),
        )
        if st.session_state.get("optimizer_apply_notice") == state_key:
            st.success("최적화 운영안을 시나리오 설정에 적용했습니다.")
            st.session_state.pop("optimizer_apply_notice", None)
    else:
        st.info("자동 최적화를 실행하면 현재 항공사 수요 시나리오까지 반영한 자원 재배치안을 생성합니다.")


# -----------------------------------------------------------------------------
# 5) AI assistant
# -----------------------------------------------------------------------------
with tab_ai:
    st.subheader("AI 운영 질의")
    st.caption("AI는 대기시간을 직접 계산하지 않습니다. 질문에서 조건을 추출한 뒤 시뮬레이션/최적화 결과를 받아 설명합니다.")

    default_q = "현재 혼잡도가 높은데 추가 인력 4명으로 앞으로 2시간 동안 어떤 조치를 하는 것이 가장 효과적이야?"
    question = st.text_area("자연어 질문", value=default_q, height=110)

    parsed = parse_question_constraints(question, airline_list)
    if parsed:
        chips = []
        for k, v in parsed.items():
            chips.append(f'<span class="badge badge-blue">{k}: {v}</span>')
        st.markdown("감지된 조건: " + " ".join(chips), unsafe_allow_html=True)

    a1, a2, a3 = st.columns([1, 1, 1.2])
    with a1:
        provider = st.selectbox("답변 엔진", ["내장 분석", "OpenAI", "Gemini"])
    with a2:
        if provider == "OpenAI":
            model = st.text_input("모델", value="gpt-5")
        elif provider == "Gemini":
            model = st.text_input("모델", value="gemini-3.5-flash")
        else:
            model = ""
    with a3:
        secret_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GEMINI_API_KEY"
        stored_key = safe_secret(secret_name) if provider != "내장 분석" else ""
        api_key = stored_key
        if provider != "내장 분석" and not stored_key:
            api_key = st.text_input("API Key", type="password", help="배포 시 Streamlit Secrets 사용을 권장합니다.")
        elif provider != "내장 분석":
            st.success(f"{secret_name} Secrets 사용 중")

    if st.button("질문 분석 및 운영안 생성", type="primary", use_container_width=True):
        q_horizon = int(parsed.get("horizon_min", horizon_min))
        q_horizon = min(q_horizon, max(30, 1440 - start_minute))
        q_airline = str(parsed.get("airline", selected_airline))
        if q_airline == "적용 안 함":
            q_airline = ""
        q_shock = float(parsed.get("demand_change_pct", demand_change_pct))
        q_additional_staff = int(parsed.get("additional_staff", 4))
        q_objective = str(parsed.get("objective", "균형 운영"))

        q_base = get_baseline_cached(
            str(OPERATION_PATH),
            OPERATION_PATH.stat().st_mtime,
            selected_date,
            start_minute,
            q_horizon,
        )
        q_minutes = list(q_base["minutes"])
        q_mult, _ = build_airline_share_multipliers(
            flight_df,
            selected_date,
            q_minutes,
            q_airline or None,
            q_shock,
        )
        q_arrivals = {
            a: np.asarray(q_base["arrivals"][a], dtype=float) * np.asarray(q_mult.get(a, np.ones(len(q_minutes))), dtype=float)
            for a in ALL_AREAS
        }
        q_snap = snapshot(operation_df, selected_date, start_minute)
        q_start_units = {str(r["구역"]): int(r["권고필요수"]) for _, r in q_snap.iterrows()}
        q_mins = build_minimum_units(q_snap)
        q_budget = total_staff(q_start_units) + q_additional_staff

        with st.spinner("질문 조건을 반영해 실행 가능한 운영안을 탐색하는 중입니다..."):
            q_opt_units, q_opt_metrics, q_opt_long = optimize_fixed_allocation(
                arrivals_by_area=q_arrivals,
                initial_queue_by_area=q_base["initial_queue"],
                starting_units=q_start_units,
                minimum_units=q_mins,
                staff_budget=q_budget,
                baseline_checkin_processed=q_base["baseline_checkin_processed"],
                baseline_im_arrivals=q_base["baseline_im_arrivals"],
                im_split=q_base["im_split"],
                travel_lag_min=travel_lag,
                objective=q_objective,
                max_steps=24,
                require_temporal_nonworsening=True,
            )
            q_base_long, _ = simulate_coupled_system(
                arrivals_by_area=q_arrivals,
                initial_queue_by_area=q_base["initial_queue"],
                units_by_area=q_start_units,
                baseline_checkin_processed=q_base["baseline_checkin_processed"],
                baseline_im_arrivals=q_base["baseline_im_arrivals"],
                im_split=q_base["im_split"],
                travel_lag_min=travel_lag,
            )
            q_base_metrics = compute_metrics(q_base_long)
            q_bott = top_bottlenecks(q_opt_long, 5).to_dict("records")
            context = build_structured_context(
                question=question,
                date=selected_date,
                time_text=selected_time,
                horizon_min=q_horizon,
                baseline_metrics=q_base_metrics.as_dict(),
                scenario_metrics=q_opt_metrics.as_dict(),
                units_before=q_start_units,
                units_after=q_opt_units,
                bottlenecks=q_bott,
                airline=q_airline or None,
                demand_change_pct=q_shock,
                assumptions=[
                    "2025년 9월~10월 1분 단위 운영·인원 데이터 재생",
                    "유인 체크인 각 라인은 1~40 연속 슬롯으로 가정",
                    "대기시간은 운영 데이터의 자원 산정 규칙에 맞춘 처리율로 추정",
                    "항공사 수요 변화는 항공편 좌석 가중치 기준으로 관련 구역 유입량에 비례 반영",
                    "AI 자동 운영안은 전 시간대 비악화 조건을 기본 적용",
                ],
            )
            try:
                if provider == "내장 분석" or not api_key:
                    answer = deterministic_operation_answer(context)
                    if provider != "내장 분석" and not api_key:
                        st.info("API Key가 입력되지 않아 내장 분석 엔진으로 결과를 생성했습니다.")
                else:
                    answer = generate_llm_answer(provider, model, api_key, context)
            except Exception as exc:
                answer = deterministic_operation_answer(context)
                st.warning(f"외부 AI 호출에 실패해 내장 분석으로 대체했습니다: {exc}")

        st.session_state["ai_answer"] = answer
        st.session_state["ai_context"] = context

    if st.session_state.get("ai_answer"):
        st.markdown(st.session_state["ai_answer"])
        with st.expander("AI에 전달된 구조화된 시뮬레이션 결과 보기"):
            st.json(st.session_state.get("ai_context", {}))


# -----------------------------------------------------------------------------
# 6) Data QA
# -----------------------------------------------------------------------------
with tab_quality:
    st.subheader("데이터 검수 및 모델 가정")
    st.caption("시뮬레이션에서 발견 가능한 오해를 줄이기 위해 원본 데이터 구조와 보정 내용을 명시합니다.")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("운영 데이터 행", f"{quality['operation_rows']:,}")
    q2.metric("운영 일수", f"{quality['operation_dates']}일")
    q3.metric("9~10월 출발편", f"{quality['flight_rows']:,}편")
    q4.metric("항공사", f"{quality['airlines']}개")

    st.markdown("### 자동 검수 결과")
    checks = pd.DataFrame(
        [
            {"항목": "운영 키 중복(일자·분·구역)", "결과": str(quality["operation_duplicates"]), "판정": "정상" if quality["operation_duplicates"] == 0 else "확인 필요"},
            {"항목": "항공편 체크인 구역 결측", "결과": str(quality["missing_counter_rows"]), "판정": "대표 구역으로 보정" if quality["missing_counter_rows"] else "정상"},
            {"항목": "운영 구역", "결과": ", ".join(quality["areas"]), "판정": "I 구역 데이터 없음" if "I" not in quality["areas"] else "정상"},
        ]
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)

    st.markdown("### 핵심 모델링 가정")
    st.markdown(
        """
- **유인 체크인 라인 A/C/D/E/H/J/K/M/N:** 각 라인을 물리적 1~40번 슬롯으로 모델링합니다. 원본 데이터의 일반 체크인 계획/필요 수 최대값도 40입니다.
- **셀프 체크인 B/F/G/L:** 유인 카운터와 분리된 기기 자원으로 취급합니다.
- **IM1/IM2:** 기존 운영 앱과 동일하게 수요가 있으면 최소 3개, 최대 6개 출입문을 기준으로 합니다.
- **기준 운영안:** 기존 2번 앱의 계획 대비 실시간 필요 수 차이와 최소 유지 비율 로직을 재현합니다.
- **대기시간:** 실제 승객별 체크인 시작·종료 로그가 없으므로 직접 관측값이 아닙니다. 1분 단위 구역 인원과 기준 운영 수에서 유입량을 역산하고, 기존 자원 산정 규칙과 목표 대기시간에 맞춘 처리율로 계산한 추정값입니다.
- **항공사 수요 변화:** 좌석 수 자체를 그대로 승객 수로 더하지 않습니다. 항공편 좌석 가중치로 해당 항공사가 각 구역 수요에서 차지하는 비율을 구한 뒤 기존 9~10월 유입량을 비례 조정합니다.
- **체크인→IM 병목 전이:** 체크인 처리량이 기준보다 증가/감소하면 일정 이동 지연 후 IM1/IM2 유입량에 반영합니다.
"""
    )

    st.markdown("### 데이터에서 확인된 자원 산정 규칙")
    rules = pd.DataFrame(
        [
            {"구분": "일반 체크인", "원본 실시간 필요 수": "ceil(실시간인원수 / 5)", "시뮬레이션 목표 대기": "10분"},
            {"구분": "셀프 체크인", "원본 실시간 필요 수": "ceil(실시간인원수 / 6)", "시뮬레이션 목표 대기": "5분"},
            {"구분": "IM1/IM2", "원본 앱 재계산": "ceil(인원 / 30), 최소 3·최대 6", "시뮬레이션 목표 대기": "3분"},
            {"구분": "프리미엄 A", "계획 운영 수": "계획수요 / 8 기반", "시뮬레이션 목표 대기": "10분"},
        ]
    )
    st.dataframe(rules, use_container_width=True, hide_index=True)

    with st.expander("현재 선택 시각 원본 스냅샷 보기"):
        st.dataframe(snap, use_container_width=True, hide_index=True)

st.caption(
    "프로토타입 범위: 2025년 9월~10월 과거 데이터 재현. 실제 실시간 운영 적용 전에는 S-WARD 신규 데이터 수집, 실제 서비스 완료 로그, 카운터 물리 배치표와 인력 근무 제약을 추가 검증해야 합니다."
)

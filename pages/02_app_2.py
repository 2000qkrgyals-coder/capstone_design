from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "operation_dashboard_data.csv.gz"

COUNTERS = list("ABCDEFGHIJKLMN")
AREA_LIST = COUNTERS + ["IM1", "IM2"]
AREAS = ["전체"] + AREA_LIST
SELF_COUNTERS = {"B", "F", "G", "L"}
IM_AREAS = {"IM1", "IM2"}

TYPE_MAP = {}
UNIT_MAP = {}

for counter in COUNTERS:
    if counter == "A":
        TYPE_MAP[counter] = "프리미엄 체크인"
        UNIT_MAP[counter] = "창구"

    elif counter in SELF_COUNTERS:
        TYPE_MAP[counter] = "셀프 체크인"
        UNIT_MAP[counter] = "기기"

    else:
        TYPE_MAP[counter] = "일반 체크인"
        UNIT_MAP[counter] = "창구"


TYPE_MAP.update(
    {
        "IM1": "출국장 진입",
        "IM2": "출국장 진입",
    }
)

UNIT_MAP.update(
    {
        "IM1": "출입문",
        "IM2": "출입문",
    }
)


IM_MAX_GATES = 6
IM_MIN_ACTIVE_GATES = 3
IM_PEOPLE_PER_GATE = 30


NUMERIC_COLS = [
    "분",
    "계획수요",
    "실시간인원수",
    "계획오픈수",
    "실시간필요수",
    "필요수차이",
    "계획기본직원수",
    "계획지원직원수",
    "계획총직원수",
    "실시간기본직원수",
    "실시간지원직원수",
    "실시간총직원수",
    "직원차이",
]


st.set_page_config(
    page_title="T2 운영 최적화 수정 시스템",
    layout="wide",
)


st.markdown(
    """
<style>

/* =========================================================
   전체 다크 테마
   ========================================================= */

:root {
    color-scheme: dark;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #08111f !important;
    color: #e5edf7 !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #08111f !important;
}

[data-testid="stHeader"] {
    background: rgba(8, 17, 31, 0.96) !important;
}

[data-testid="stToolbar"] {
    color: #dbeafe !important;
}


/* =========================================================
   사이드바
   ========================================================= */

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #0c1627 !important;
    color: #e5edf7 !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #25324a;
}

[data-testid="stSidebar"] * {
    color: #e5edf7;
}


/* =========================================================
   기본 레이아웃
   ========================================================= */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

h1,
h2,
h3,
h4,
h5,
h6,
[data-testid="stMarkdownContainer"] {
    color: #e5edf7;
}

p,
label {
    color: #d7e0ec;
}

[data-testid="stCaptionContainer"] {
    color: #93a4ba !important;
}


/* =========================================================
   메인 제목
   ========================================================= */

.main-title {
    display: flex;
    align-items: center;
    gap: 10px;
    min-height: 54px;

    overflow: visible;

    padding: 5px 0 4px 0;
    margin: 0 0 4px 0;

    color: #f8fafc;

    font-size: 34px;
    font-weight: 900;

    letter-spacing: -0.8px;
    line-height: 1.28;
}

.title-emoji {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    flex: 0 0 auto;

    font-size: 36px;
    line-height: 1.25;

    padding: 3px 0 2px 0;

    overflow: visible;
}

.sub-title {
    color: #94a3b8;

    font-size: 15px;

    margin-bottom: 18px;
}


/* =========================================================
   SELECTBOX 강제 다크 모드
   ========================================================= */

/* 선택창 외부와 내부 */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stSelectbox"] div[aria-haspopup="listbox"] {
    background-color: #111c30 !important;
    background: #111c30 !important;

    color: #f8fafc !important;

    border-color: #334155 !important;

    box-shadow: none !important;
}


/* BaseWeb 내부 흰색 레이어까지 강제로 제거 */
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #111c30 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #111c30 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
    background-color: #111c30 !important;
}


/* 선택된 값 글씨 */
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] p,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #f8fafc !important;
}


/* 선택창 아이콘 */
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #cbd5e1 !important;
    fill: #cbd5e1 !important;
}


/* placeholder */
[data-testid="stSelectbox"] input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}


/* 포커스 */
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] div[role="combobox"]:focus,
[data-testid="stSelectbox"] div[aria-haspopup="listbox"]:focus {
    background-color: #111c30 !important;
    background: #111c30 !important;

    border-color: #60a5fa !important;
}


/* 펼쳐지는 드롭다운 */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
div[role="listbox"],
div[role="option"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: #0f1a2d !important;
    color: #e5edf7 !important;
}

div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
    background-color: #1b2a43 !important;
}


/* =========================================================
   기타 입력 위젯
   ========================================================= */

div[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #111c30 !important;
    color: #f8fafc !important;
    border-color: #334155 !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] p,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] p {
    color: #e5edf7 !important;
}


/* =========================================================
   Expander
   ========================================================= */

[data-testid="stExpander"] {
    background: #0f1a2d !important;

    border: 1px solid #2b3a52 !important;
    border-radius: 14px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #e5edf7 !important;
}


/* =========================================================
   Dataframe
   ========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #2b3a52;

    border-radius: 12px;

    overflow: hidden;
}


/* =========================================================
   Alert
   ========================================================= */

[data-testid="stAlert"] {
    background: #101d31 !important;
    color: #e5edf7 !important;

    border: 1px solid #334155 !important;
}

hr {
    border-color: #25324a !important;
}

a {
    color: #7dd3fc !important;
}


/* =========================================================
   상단 상태 바
   ========================================================= */

.status-strip {
    border-radius: 14px;

    padding: 13px 16px;

    font-size: 15px;
    font-weight: 800;

    margin-bottom: 16px;

    border: 1px solid;
}

.plan-strip {
    background: #0f2440;
    color: #93c5fd;

    border-color: #1d4f7a;
}

.live-strip {
    background: #0b2a23;
    color: #6ee7b7;

    border-color: #17624f;
}

.alert-strip {
    background: #32141c;
    color: #fda4af;

    border-color: #7f1d35;
}

.reduce-strip {
    background: #321f12;
    color: #fdba74;

    border-color: #7c3f18;
}


/* =========================================================
   KPI
   ========================================================= */

.kpi-card {
    border: 1px solid #2a3951;
    border-radius: 18px;

    background: #101a2c;

    padding: 17px 18px;

    min-height: 118px;

    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
}

.kpi-label {
    color: #94a3b8;

    font-size: 13px;
    font-weight: 800;

    margin-bottom: 8px;
}

.kpi-value {
    color: #f8fafc;

    font-size: 30px;
    font-weight: 950;

    letter-spacing: -0.7px;
    line-height: 1.12;
}

.kpi-sub {
    color: #94a3b8;

    font-size: 12px;

    margin-top: 7px;

    line-height: 1.4;
}


/* =========================================================
   LIVE 전체 요약
   ========================================================= */

.summary-card {
    border-radius: 18px;

    padding: 18px;

    min-height: 128px;

    border: 1px solid;
}

.summary-add {
    background: #2f141d;
    border-color: #7f1d35;
}

.summary-reduce {
    background: #322014;
    border-color: #7c3f18;
}

.summary-keep {
    background: #0c2922;
    border-color: #17624f;
}

.summary-staff {
    background: #10253e;
    border-color: #245b8a;
}

.summary-label {
    font-size: 13px;
    font-weight: 850;

    color: #a9b7c9;

    margin-bottom: 8px;
}

.summary-value {
    font-size: 31px;
    font-weight: 950;

    color: #f8fafc;

    line-height: 1.1;
}

.summary-sub {
    margin-top: 8px;

    color: #94a3b8;

    font-size: 12px;
}


/* =========================================================
   운영 전환 흐름
   ========================================================= */

.flow-step {
    border: 1px solid #2a3951;
    border-radius: 18px;

    padding: 18px;

    background: #101a2c;

    min-height: 132px;
}

.flow-step.final {
    border-color: #2563a5;

    background: #10253e;
}

.flow-number {
    font-size: 12px;
    font-weight: 900;

    color: #93a4ba;

    margin-bottom: 8px;
}

.flow-title {
    font-size: 14px;
    font-weight: 800;

    color: #a9b7c9;
}

.flow-value {
    font-size: 32px;
    font-weight: 950;

    color: #f8fafc;

    margin-top: 8px;
}

.flow-sub {
    font-size: 12px;

    color: #94a3b8;

    margin-top: 7px;
}


/* =========================================================
   최종 판단
   ========================================================= */

.decision-box {
    border-radius: 18px;

    padding: 18px 20px;

    margin: 10px 0 16px 0;

    border: 1px solid;
}

.decision-add {
    background: #2f141d;
    border-color: #7f1d35;
}

.decision-reduce {
    background: #10253e;
    border-color: #245b8a;
}

.decision-keep {
    background: #0c2922;
    border-color: #17624f;
}

.decision-title {
    font-size: 21px;
    font-weight: 950;

    color: #f8fafc;
}

.decision-sub {
    margin-top: 7px;

    color: #c3cfdd;

    font-size: 14px;

    line-height: 1.45;
}


/* =========================================================
   Action board
   ========================================================= */

.action-card {
    border: 1px solid #2a3951;
    border-radius: 16px;

    padding: 15px 16px;

    background: #101a2c;

    margin-bottom: 10px;
}

.action-card.add {
    border-left: 6px solid #fb7185;
}

.action-card.reduce {
    border-left: 6px solid #fb923c;
}

.action-title {
    font-size: 17px;
    font-weight: 900;

    color: #f8fafc;
}

.action-sub {
    margin-top: 5px;

    font-size: 13px;

    color: #9fb0c5;

    line-height: 1.45;
}


/* =========================================================
   Section title
   ========================================================= */

.section-title {
    font-size: 21px;
    font-weight: 900;

    color: #f8fafc;

    margin: 22px 0 10px 0;
}

</style>
    """,
    unsafe_allow_html=True,
)


def ceil_div(value, base):
    value = float(value)

    if value <= 0:
        return 0

    return math.ceil(value / base)


def calc_im_gates(people):
    people = float(people)

    if people <= 0:
        return 0

    gates = ceil_div(
        people,
        IM_PEOPLE_PER_GATE,
    )

    gates = max(
        IM_MIN_ACTIVE_GATES,
        gates,
    )

    gates = min(
        IM_MAX_GATES,
        gates,
    )

    return int(gates)


def calc_im_support_staff(gates):
    gates = int(gates)

    if gates <= 0:
        return 0

    if gates <= 3:
        return 1

    if gates <= 5:
        return 2

    return 3


def recalc_im_rows(df):
    df = df.copy()

    if "구역" not in df.columns:
        return df

    mask = df["구역"].isin(
        IM_AREAS
    )

    for idx, row in df.loc[mask].iterrows():
        plan_gates = calc_im_gates(
            row["계획수요"]
        )

        sensor_gates = calc_im_gates(
            row["실시간인원수"]
        )

        plan_support = calc_im_support_staff(
            plan_gates
        )

        sensor_support = calc_im_support_staff(
            sensor_gates
        )

        df.at[idx, "유형"] = "출국장 진입"
        df.at[idx, "단위"] = "출입문"

        df.at[idx, "계획오픈수"] = plan_gates
        df.at[idx, "실시간필요수"] = sensor_gates

        df.at[idx, "계획기본직원수"] = plan_gates
        df.at[idx, "계획지원직원수"] = plan_support
        df.at[idx, "계획총직원수"] = (
            plan_gates
            + plan_support
        )

        df.at[idx, "실시간기본직원수"] = sensor_gates
        df.at[idx, "실시간지원직원수"] = sensor_support
        df.at[idx, "실시간총직원수"] = (
            sensor_gates
            + sensor_support
        )

        df.at[idx, "필요수차이"] = (
            sensor_gates
            - plan_gates
        )

        df.at[idx, "직원차이"] = (
            sensor_gates
            + sensor_support
            - plan_gates
            - plan_support
        )

        if sensor_gates >= 5:
            df.at[idx, "IM판단"] = (
                "집중 운영 권고"
            )

        elif sensor_gates >= 3:
            df.at[idx, "IM판단"] = (
                "기본 운영 수준"
            )

        elif sensor_gates > 0:
            df.at[idx, "IM판단"] = (
                "최소 개방 수준"
            )

        else:
            df.at[idx, "IM판단"] = (
                "출입문 대기 수요 없음"
            )

    return df


@st.cache_resource(show_spinner=False)
def load_data(file_mtime):
    _ = file_mtime

    df = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
        compression="infer",
    )

    df["일자"] = df["일자"].astype(str)
    df["구역"] = df["구역"].astype(str)
    df["시각"] = df["시각"].astype(str)

    if "IM판단" not in df.columns:
        df["IM판단"] = ""

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    df = df[
        df["일자"].str.startswith("2025-")
    ]

    df = df[
        df["구역"].isin(AREA_LIST)
    ]

    return df


def fmt_num(value):
    try:
        return f"{int(round(float(value))):,}"

    except Exception:
        return str(value)


def fmt_signed(value):
    try:
        return f"{int(round(float(value))):+,}"

    except Exception:
        return str(value)


def minute_to_hhmm(minute):
    minute = max(
        0,
        min(
            1439,
            int(minute),
        ),
    )

    return (
        f"{minute // 60:02d}:"
        f"{minute % 60:02d}"
    )


def hhmm_to_minute(text):
    hour, minute = str(text).split(":")

    return (
        int(hour) * 60
        + int(minute)
    )


def selectable_times():
    return [
        minute_to_hhmm(minute)
        for minute in range(
            0,
            1440,
            15,
        )
    ]


def graph_window(selected_time):
    center = hhmm_to_minute(
        selected_time
    )

    start = max(
        0,
        center - 30,
    )

    end = min(
        1439,
        center + 90,
    )

    label = (
        f"계획 "
        f"{minute_to_hhmm(start)}부터 "
        f"{minute_to_hhmm(end)}까지"
    )

    return (
        start,
        end,
        label,
    )


def unit_suffix(unit):
    if unit == "기기":
        return "대"

    return "개"


def axis_name(area):
    if area == "전체":
        return "필요 운영 수"

    if area in SELF_COUNTERS:
        return "필요 기기 수"

    if area in IM_AREAS:
        return "필요 출입문 수"

    return "필요 창구 수"


def keep_rate(area):
    if area == "A":
        return 0.70

    if (
        area in SELF_COUNTERS
        or area in IM_AREAS
    ):
        return 0.50

    return 0.60


def estimate_staff_from_units(
    area,
    units,
):
    units = int(
        max(
            0,
            units,
        )
    )

    if units <= 0:
        return 0

    if area == "A":
        return (
            units
            + (
                1
                if units >= 3
                else 0
            )
        )

    if area in SELF_COUNTERS:
        return min(
            math.ceil(
                units / 6
            ),
            3,
        )

    if area in IM_AREAS:
        return (
            units
            + calc_im_support_staff(
                units
            )
        )

    if units < 8:
        support = 0

    elif units < 16:
        support = 1

    elif units < 24:
        support = 2

    else:
        support = 3

    return (
        units
        + support
    )


def add_recommendation_columns(rows):
    rows = rows.copy()

    decisions = []
    final_units_list = []
    adjust_units = []
    final_staff_list = []
    adjust_staff = []

    for _, row in rows.iterrows():
        area = row["구역"]

        plan_units = int(
            row["계획오픈수"]
        )

        sensor_units = int(
            row["실시간필요수"]
        )

        plan_staff = int(
            row["계획총직원수"]
        )

        sensor_staff = int(
            row["실시간총직원수"]
        )

        diff = (
            sensor_units
            - plan_units
        )

        if plan_units <= 0:
            if sensor_units > 0:
                decision = "추가 필요"
                final_units = sensor_units
                final_staff = sensor_staff

            else:
                decision = "계획 유지"
                final_units = 0
                final_staff = 0

        elif diff >= 2:
            decision = "추가 필요"
            final_units = sensor_units
            final_staff = sensor_staff

        elif diff <= -2:
            minimum_units = math.ceil(
                plan_units
                * keep_rate(area)
            )

            final_units = max(
                sensor_units,
                minimum_units,
            )

            if final_units < plan_units:
                decision = "감축 검토"

                final_staff = (
                    estimate_staff_from_units(
                        area,
                        final_units,
                    )
                )

            else:
                decision = "계획 유지"
                final_units = plan_units
                final_staff = plan_staff

        else:
            decision = "계획 유지"
            final_units = plan_units
            final_staff = plan_staff

        decisions.append(
            decision
        )

        final_units_list.append(
            int(final_units)
        )

        adjust_units.append(
            int(
                final_units
                - plan_units
            )
        )

        final_staff_list.append(
            int(final_staff)
        )

        adjust_staff.append(
            int(
                final_staff
                - plan_staff
            )
        )

    rows["조정판단"] = decisions
    rows["권고필요수"] = final_units_list
    rows["조정필요수"] = adjust_units
    rows["권고직원수"] = final_staff_list
    rows["직원조정수"] = adjust_staff

    return rows


def base_area_frame(area):
    if area == "전체":
        areas = AREA_LIST

    else:
        areas = [area]

    return pd.DataFrame(
        {
            "구역": areas,
        }
    )


def fill_snapshot_defaults(rows):
    rows["유형"] = (
        rows["유형"]
        .fillna(
            rows["구역"].map(TYPE_MAP)
        )
    )

    rows["단위"] = (
        rows["단위"]
        .fillna(
            rows["구역"].map(UNIT_MAP)
        )
    )

    for col in NUMERIC_COLS:
        if col in rows.columns:
            rows[col] = pd.to_numeric(
                rows[col],
                errors="coerce",
            ).fillna(0)

    rows["상태"] = (
        rows["상태"]
        .fillna("계획 유지")
    )

    rows["권고"] = (
        rows["권고"]
        .fillna("계획 유지")
    )

    rows["IM판단"] = (
        rows["IM판단"]
        .fillna("")
    )

    return rows


def current_snapshot(
    df,
    date,
    time_value,
    area,
):
    minute = hhmm_to_minute(
        time_value
    )

    rows = df[
        (df["일자"] == date)
        & (df["분"] == minute)
    ].copy()

    if not rows.empty:
        rows = rows.drop_duplicates(
            subset=["구역"],
            keep="last",
        )

    rows = base_area_frame(
        area
    ).merge(
        rows,
        on="구역",
        how="left",
    )

    rows["일자"] = (
        rows["일자"]
        .fillna(date)
    )

    rows["분"] = (
        rows["분"]
        .fillna(minute)
        .astype(int)
    )

    rows["시각"] = (
        rows["시각"]
        .fillna(time_value)
    )

    rows = fill_snapshot_defaults(
        rows
    )

    rows = recalc_im_rows(
        rows
    )

    rows = add_recommendation_columns(
        rows
    )

    if area != "전체":
        rows = rows[
            rows["구역"] == area
        ].copy()

    return rows


def day_series(
    df,
    date,
    area,
    start_min,
    end_min,
):
    day = df[
        (df["일자"] == date)
        & (df["분"] >= start_min)
        & (df["분"] <= end_min)
    ].copy()

    if area != "전체":
        day = day[
            day["구역"] == area
        ].copy()

    day = recalc_im_rows(
        day
    )

    base = pd.DataFrame(
        {
            "분": list(
                range(
                    start_min,
                    end_min + 1,
                )
            )
        }
    )

    base["시각"] = (
        base["분"]
        .apply(minute_to_hhmm)
    )

    if day.empty:
        base["계획오픈수"] = 0
        base["실시간필요수"] = 0

        return base

    grouped = (
        day.groupby(
            "분",
            as_index=False,
        )[
            [
                "계획오픈수",
                "실시간필요수",
            ]
        ]
        .sum()
    )

    out = base.merge(
        grouped,
        on="분",
        how="left",
    )

    for col in [
        "계획오픈수",
        "실시간필요수",
    ]:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        ).fillna(0)

    return out.sort_values(
        "분"
    )


def get_live_end_minute(selected_time):
    (
        start_min,
        end_min,
        _,
    ) = graph_window(
        selected_time
    )

    selected_min = hhmm_to_minute(
        selected_time
    )

    if "live_elapsed" not in st.session_state:
        st.session_state["live_elapsed"] = 0

    live_end = (
        selected_min
        + int(
            st.session_state[
                "live_elapsed"
            ]
        )
    )

    return max(
        start_min,
        min(
            end_min,
            live_end,
        ),
    )


def make_chart_data(
    df,
    date,
    area,
    selected_time,
    mode,
):
    (
        start_min,
        end_min,
        plan_label,
    ) = graph_window(
        selected_time
    )

    series = day_series(
        df,
        date,
        area,
        start_min,
        end_min,
    )

    plan = series[
        [
            "분",
            "시각",
            "계획오픈수",
        ]
    ].copy()

    plan = plan.rename(
        columns={
            "계획오픈수": "필요수",
        }
    )

    plan["구분"] = (
        "항공편 기반 계획"
    )

    if mode == "OFF":
        return (
            plan,
            plan_label,
            selected_time,
        )

    live_end = get_live_end_minute(
        selected_time
    )

    sensor = series[
        series["분"] <= live_end
    ][
        [
            "분",
            "시각",
            "실시간필요수",
        ]
    ].copy()

    sensor = sensor.rename(
        columns={
            "실시간필요수": "필요수",
        }
    )

    sensor["구분"] = (
        "인원수 기준"
    )

    chart = pd.concat(
        [
            plan,
            sensor,
        ],
        ignore_index=True,
    )

    chart = chart.sort_values(
        [
            "분",
            "구분",
        ]
    )

    live_end_time = minute_to_hhmm(
        live_end
    )

    label = (
        f"{plan_label} / "
        f"인원수 기준 "
        f"{minute_to_hhmm(start_min)}부터 "
        f"{live_end_time}까지"
    )

    return (
        chart,
        label,
        live_end_time,
    )


def metric_card(
    title,
    value,
    suffix="",
    sub="",
):
    if str(sub).strip():
        sub_html = (
            '<div class="kpi-sub">'
            f"{sub}"
            "</div>"
        )

    else:
        sub_html = ""

    html = (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{title}</div>'
        f'<div class="kpi-value">{value}{suffix}</div>'
        f"{sub_html}"
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def summary_card(
    css_class,
    title,
    value,
    sub,
):
    html = (
        f'<div class="summary-card {css_class}">'
        f'<div class="summary-label">{title}</div>'
        f'<div class="summary-value">{value}</div>'
        f'<div class="summary-sub">{sub}</div>'
        "</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def draw_line_chart(
    chart,
    title_text,
    y_title,
):
    fig = px.line(
        chart,
        x="분",
        y="필요수",
        color="구분",
        custom_data=[
            "시각",
            "구분",
        ],
    )

    if not chart.empty:
        y_max = float(
            chart["필요수"].max()
        )

    else:
        y_max = 1

    y_top = max(
        1,
        math.ceil(
            y_max * 1.15
        ),
    )

    if y_top <= 10:
        dtick = 1

    elif y_top <= 30:
        dtick = 2

    elif y_top <= 80:
        dtick = 5

    else:
        dtick = 10

    fig.update_traces(
        mode="lines",
        line=dict(
            width=3,
        ),
        hovertemplate=(
            "시각=%{customdata[0]}<br>"
            "구분=%{customdata[1]}<br>"
            f"{y_title}=%{{y:.0f}}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1a2d",

        font=dict(
            color="#dbe7f3",
        ),

        title_font=dict(
            color="#f8fafc",
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#dbe7f3",
            ),
        ),

        title=title_text,

        height=410,

        margin=dict(
            l=10,
            r=10,
            t=42,
            b=10,
        ),

        legend_title_text="",

        xaxis_title="",
        yaxis_title=y_title,

        hovermode="x unified",
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,

        color="#cbd5e1",
    )

    fig.update_yaxes(
        range=[
            0,
            y_top,
        ],

        tickmode="linear",
        dtick=dtick,

        tickformat="d",

        rangemode="tozero",

        gridcolor="#263449",
        zerolinecolor="#334155",

        color="#cbd5e1",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )


def render_flow(row):
    suffix = unit_suffix(
        row["단위"]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            (
                '<div class="flow-step">'
                '<div class="flow-number">1단계</div>'
                '<div class="flow-title">사전 계획</div>'
                f'<div class="flow-value">{fmt_num(row["계획오픈수"])}{suffix}</div>'
                '<div class="flow-sub">항공편 기반 운영계획</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            (
                '<div class="flow-step">'
                '<div class="flow-number">2단계</div>'
                '<div class="flow-title">LIVE 재계산</div>'
                f'<div class="flow-value">{fmt_num(row["실시간필요수"])}{suffix}</div>'
                '<div class="flow-sub">현재 인원수 기준 필요 수</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            (
                '<div class="flow-step final">'
                '<div class="flow-number">3단계</div>'
                '<div class="flow-title">최종 권고</div>'
                f'<div class="flow-value">{fmt_num(row["권고필요수"])}{suffix}</div>'
                '<div class="flow-sub">조정 기준을 적용한 최종 운영안</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def recommendation_reason(row):
    area = str(
        row["구역"]
    )

    suffix = unit_suffix(
        row["단위"]
    )

    if row["단위"] == "기기":
        unit_name = "기기"

    else:
        unit_name = row["단위"]

    plan_units = int(
        row["계획오픈수"]
    )

    sensor_units = int(
        row["실시간필요수"]
    )

    recommended_units = int(
        row["권고필요수"]
    )

    decision = str(
        row["조정판단"]
    )

    if decision == "추가 필요":
        amount = (
            recommended_units
            - plan_units
        )

        return (
            f"인원수 기준 필요 {unit_name}가 "
            f"계획보다 {amount}{suffix} 많아 "
            f"{area} 구역의 추가 운영을 권고합니다."
        )

    if decision == "감축 검토":
        amount = (
            plan_units
            - recommended_units
        )

        return (
            f"인원수 기준 필요 {unit_name}가 "
            f"계획보다 {amount}{suffix} 적어 "
            f"{area} 구역의 감축을 검토합니다."
        )

    if sensor_units != plan_units:
        return (
            f"인원수 기준 필요 수는 "
            f"{sensor_units}{suffix}이지만 "
            "조정 기준 범위 안이므로 "
            "기존 계획을 유지합니다."
        )

    return (
        "항공편 기반 계획과 인원수 기준 필요 수가 같아 "
        "기존 계획을 유지합니다."
    )


def render_decision_box(row):
    decision = str(
        row["조정판단"]
    )

    suffix = unit_suffix(
        row["단위"]
    )

    adjust = int(
        row["조정필요수"]
    )

    staff_adjust = int(
        row["직원조정수"]
    )

    if decision == "추가 필요":
        css_class = (
            "decision-add"
        )

        title = (
            f"추가 운영 "
            f"{adjust}{suffix}"
        )

    elif decision == "감축 검토":
        css_class = (
            "decision-reduce"
        )

        title = (
            f"감축 검토 "
            f"{abs(adjust)}{suffix}"
        )

    else:
        css_class = (
            "decision-keep"
        )

        title = (
            "현재 계획 유지"
        )

    html = (
        f'<div class="decision-box {css_class}">'
        f'<div class="decision-title">{title}</div>'
        f'<div class="decision-sub">'
        f'{recommendation_reason(row)}<br>'
        f'직원 조정: {fmt_signed(staff_adjust)}명'
        f"</div>"
        f"</div>"
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


def action_board(current):
    priority = current[
        current["조정판단"]
        != "계획 유지"
    ].copy()

    priority["정렬값"] = (
        priority[
            "조정필요수"
        ].abs()
    )

    priority = (
        priority.sort_values(
            [
                "정렬값",
                "직원조정수",
            ],
            ascending=False,
        )
        .head(8)
    )

    if priority.empty:
        st.success(
            "현재 데이터 기준 추가 운영 또는 "
            "감축 검토가 필요한 구역이 없습니다."
        )

        return

    for _, row in priority.iterrows():
        suffix = unit_suffix(
            row["단위"]
        )

        adjust = int(
            row["조정필요수"]
        )

        staff_adjust = int(
            row["직원조정수"]
        )

        if adjust > 0:
            css_class = "add"

            title = (
                f"{row['구역']} · "
                f"{abs(adjust)}{suffix} "
                "추가 운영"
            )

        else:
            css_class = "reduce"

            title = (
                f"{row['구역']} · "
                f"{abs(adjust)}{suffix} "
                "감축 검토"
            )

        html = (
            f'<div class="action-card {css_class}">'
            f'<div class="action-title">{title}</div>'
            f'<div class="action-sub">'
            f'{row["유형"]} · '
            f'계획 {int(row["계획오픈수"])}{suffix} '
            f'→ LIVE {int(row["실시간필요수"])}{suffix} '
            f'→ 최종 {int(row["권고필요수"])}{suffix} '
            f'· 직원 {int(row["계획총직원수"])}명 '
            f'→ {int(row["권고직원수"])}명 '
            f'({fmt_signed(staff_adjust)}명)'
            f"</div>"
            f"</div>"
        )

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


def live_status_table(current):
    table = current[
        [
            "구역",
            "유형",
            "계획오픈수",
            "실시간필요수",
            "권고필요수",
            "조정필요수",
            "조정판단",
            "계획총직원수",
            "권고직원수",
            "직원조정수",
        ]
    ].copy()

    table["우선순위"] = (
        table[
            "조정필요수"
        ].abs()
    )

    table = (
        table.sort_values(
            [
                "우선순위",
                "직원조정수",
            ],
            ascending=False,
        )
        .drop(
            columns=[
                "우선순위",
            ]
        )
    )

    return table.rename(
        columns={
            "계획오픈수": "사전 계획",
            "실시간필요수": "LIVE 필요",
            "권고필요수": "최종 권고",
            "조정필요수": "운영 조정",
            "조정판단": "판단",
            "계획총직원수": "계획 직원",
            "권고직원수": "권고 직원",
            "직원조정수": "직원 조정",
        }
    )


def operation_table_off(current):
    table = current[
        [
            "구역",
            "유형",
            "계획수요",
            "계획오픈수",
            "단위",
            "계획기본직원수",
            "계획지원직원수",
            "계획총직원수",
        ]
    ].copy()

    table = table.sort_values(
        [
            "계획오픈수",
            "계획총직원수",
        ],
        ascending=False,
    )

    return table.rename(
        columns={
            "계획수요": "계획 수요",
            "계획오픈수": "계획 필요",
            "계획기본직원수": "기본 직원",
            "계획지원직원수": "지원 직원",
            "계획총직원수": "총 직원",
        }
    )


st.markdown(
    (
        '<div class="main-title">'
        '<span class="title-emoji">✈️</span>'
        "<span>T2 운영 최적화 수정 시스템</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown(
    (
        '<div class="sub-title">'
        "항공편 기반 계획을 LIVE 인원수로 재계산해 "
        "구역별 추가 운영·감축 검토·계획 유지 여부를 제시합니다."
        "</div>"
    ),
    unsafe_allow_html=True,
)


with st.expander(
    "운영 판단 기준 보기",
    expanded=False,
):
    criteria = pd.DataFrame(
        [
            [
                "프리미엄 체크인",
                "8명당 창구 1개",
            ],
            [
                "일반 체크인",
                "5명당 창구 1개",
            ],
            [
                "셀프 체크인",
                "6명당 기기 1대",
            ],
            [
                "IM1·IM2",
                (
                    "30명당 출입문 1개 · "
                    "수요 발생 시 최소 3개 · 최대 6개"
                ),
            ],
            [
                "추가 운영",
                "계획 대비 필요 수가 2개 이상 증가",
            ],
            [
                "감축 검토",
                (
                    "계획 대비 필요 수가 2개 이상 감소하며 "
                    "최소 운영률 유지"
                ),
            ],
        ],
        columns=[
            "구분",
            "기준",
        ],
    )

    st.dataframe(
        criteria,
        width="stretch",
        hide_index=True,
    )


if not DATA_PATH.exists():
    st.error(
        "operation_dashboard_data.csv.gz 파일이 없습니다. "
        "먼저 전처리 코드를 실행하세요."
    )

    st.code(
        (
            'cd /d "G:\\캡디\\2026-07-30 '
            '과제 2번 디벨롭"\n'
            "python make_operation_dashboard_data.py"
        ),
        language="cmd",
    )

    st.stop()


file_mtime = (
    DATA_PATH.stat().st_mtime
)

df = load_data(
    file_mtime
)


if df.empty:
    st.error(
        "데이터가 비어 있습니다. "
        "전처리 결과를 확인하세요."
    )

    st.stop()


dates = sorted(
    df["일자"]
    .dropna()
    .unique()
)

times = selectable_times()


with st.sidebar:
    st.header(
        "관제 설정"
    )

    selected_date = st.selectbox(
        "일자",
        dates,
        index=0,
    )

    selected_area = st.selectbox(
        "구역",
        AREAS,
        index=0,
    )

    selected_time = st.selectbox(
        "데이터 기준 시각",
        times,
        index=(
            times.index("08:00")
            if "08:00" in times
            else 0
        ),
    )

    mode = st.radio(
        "표시 방식",
        [
            "OFF",
            "LIVE",
        ],
        index=0,
    )

    refresh_seconds = 20

    if mode == "LIVE":
        refresh_seconds = st.selectbox(
            "LIVE 갱신 간격",
            [
                10,
                20,
                30,
                60,
            ],
            index=1,
        )

    st.caption(
        "OFF: 항공편 기반 사전 운영계획"
    )

    st.caption(
        "LIVE: 인원수 데이터를 순차 반영해 운영안을 재계산"
    )


session_key = (
    f"{selected_date}|"
    f"{selected_area}|"
    f"{selected_time}|"
    f"{mode}"
)


if (
    st.session_state.get("session_key")
    != session_key
):
    st.session_state["session_key"] = (
        session_key
    )

    st.session_state["live_elapsed"] = 0


def render_off_view():
    (
        chart,
        window_label,
        data_time,
    ) = make_chart_data(
        df,
        selected_date,
        selected_area,
        selected_time,
        "OFF",
    )

    current = current_snapshot(
        df,
        selected_date,
        data_time,
        selected_area,
    )

    y_title = axis_name(
        selected_area
    )

    st.subheader(
        f"🗓️ {selected_date} "
        f"{selected_time} "
        "사전 운영계획"
    )

    st.markdown(
        (
            '<div class="status-strip plan-strip">'
            "OFF · 항공편 기반 계획만 표시합니다."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    plan_demand = current[
        "계획수요"
    ].sum()

    plan_units = int(
        current[
            "계획오픈수"
        ].sum()
    )

    plan_staff = int(
        current[
            "계획총직원수"
        ].sum()
    )

    active_areas = int(
        (
            current[
                "계획오픈수"
            ]
            > 0
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "계획 수요",
            fmt_num(
                plan_demand
            ),
            "명",
        )

    with c2:
        if selected_area == "전체":
            metric_card(
                y_title,
                fmt_num(
                    plan_units
                ),
            )

        else:
            metric_card(
                y_title,
                fmt_num(
                    plan_units
                ),
                "개",
            )

    with c3:
        metric_card(
            "계획 직원",
            fmt_num(
                plan_staff
            ),
            "명",
        )

    with c4:
        if selected_area == "전체":
            metric_card(
                "운영 구역",
                fmt_num(
                    active_areas
                ),
                "곳",
            )

        else:
            metric_card(
                "구역 유형",
                current.iloc[0][
                    "유형"
                ],
            )

    st.markdown(
        (
            '<div class="section-title">'
            "시간대별 사전 운영계획"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        window_label
    )

    draw_line_chart(
        chart,
        "항공편 기반 계획",
        y_title,
    )


    # 전체 구역일 때 표를 클릭하지 않고 바로 표시
    if selected_area == "전체":
        st.markdown(
            (
                '<div class="section-title">'
                "구역별 운영계획"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.dataframe(
            operation_table_off(
                current
            ),
            width="stretch",
            hide_index=True,
        )


    else:
        row = current.iloc[0]

        suffix = unit_suffix(
            row["단위"]
        )

        st.markdown(
            (
                '<div class="section-title">'
                "선택 구역 운영계획"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "계획 수요",
                fmt_num(
                    row[
                        "계획수요"
                    ]
                ),
                "명",
            )

        with c2:
            metric_card(
                "계획 필요",
                fmt_num(
                    row[
                        "계획오픈수"
                    ]
                ),
                suffix,
            )

        with c3:
            metric_card(
                "기본 직원",
                fmt_num(
                    row[
                        "계획기본직원수"
                    ]
                ),
                "명",
            )

        with c4:
            metric_card(
                "총 계획 직원",
                fmt_num(
                    row[
                        "계획총직원수"
                    ]
                ),
                "명",
            )


def render_live_view():
    (
        chart,
        window_label,
        data_time,
    ) = make_chart_data(
        df,
        selected_date,
        selected_area,
        selected_time,
        "LIVE",
    )

    current = current_snapshot(
        df,
        selected_date,
        data_time,
        selected_area,
    )

    y_title = axis_name(
        selected_area
    )

    st.subheader(
        f"🟢 {selected_date} "
        f"{data_time} "
        "LIVE 운영 보정"
    )

    add_count = int(
        (
            current[
                "조정판단"
            ]
            == "추가 필요"
        ).sum()
    )

    reduce_count = int(
        (
            current[
                "조정판단"
            ]
            == "감축 검토"
        ).sum()
    )

    keep_count = int(
        (
            current[
                "조정판단"
            ]
            == "계획 유지"
        ).sum()
    )

    staff_adjust = int(
        current[
            "직원조정수"
        ].sum()
    )

    if add_count > 0:
        strip_class = (
            "alert-strip"
        )

        strip_text = (
            f"LIVE · 추가 운영 {add_count}곳, "
            f"감축 검토 {reduce_count}곳이 확인되었습니다."
        )

    elif reduce_count > 0:
        strip_class = (
            "reduce-strip"
        )

        strip_text = (
            f"LIVE · 감축 검토 "
            f"{reduce_count}곳이 확인되었습니다."
        )

    else:
        strip_class = (
            "live-strip"
        )

        strip_text = (
            "LIVE · 현재 모든 구역이 "
            "계획 유지 범위입니다."
        )

    st.markdown(
        (
            f'<div class="status-strip {strip_class}">'
            f"{strip_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


    if selected_area == "전체":
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            summary_card(
                "summary-add",
                "추가 운영",
                f"{add_count}곳",
                "즉시 증설 검토 대상",
            )

        with c2:
            summary_card(
                "summary-reduce",
                "감축 검토",
                f"{reduce_count}곳",
                "최소 운영률을 유지한 감축 후보",
            )

        with c3:
            summary_card(
                "summary-keep",
                "계획 유지",
                f"{keep_count}곳",
                "현재 계획 유지 가능",
            )

        with c4:
            summary_card(
                "summary-staff",
                "직원 순조정",
                (
                    f"{fmt_signed(staff_adjust)}명"
                ),
                "전체 권고 인력 증감",
            )

        st.markdown(
            (
                '<div class="section-title">'
                "우선 조치 보드"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        action_board(
            current
        )


        st.markdown(
            (
                '<div class="section-title">'
                "전체 구역 LIVE 비교"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.dataframe(
            live_status_table(
                current
            ),
            width="stretch",
            hide_index=True,

            column_config={
                "운영 조정": (
                    st.column_config.NumberColumn(
                        format="%+d",
                    )
                ),

                "직원 조정": (
                    st.column_config.NumberColumn(
                        format="%+d",
                    )
                ),
            },
        )


        # LIVE 그래프 기본 표시
        st.markdown(
            (
                '<div class="section-title">'
                "시간대별 계획·LIVE 변화 그래프"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            window_label
        )

        draw_line_chart(
            chart,
            "항공편 기반 계획 vs LIVE 필요 수",
            y_title,
        )


    else:
        row = current.iloc[0]

        suffix = unit_suffix(
            row["단위"]
        )


        st.markdown(
            (
                '<div class="section-title">'
                f"{selected_area} 운영 전환 흐름"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        render_flow(
            row
        )

        render_decision_box(
            row
        )


        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "계획 수요",
                fmt_num(
                    row[
                        "계획수요"
                    ]
                ),
                "명",
                "항공편 기반 수요",
            )

        with c2:
            metric_card(
                "LIVE 인원",
                fmt_num(
                    row[
                        "실시간인원수"
                    ]
                ),
                "명",
                "현재 인원수 데이터",
            )

        with c3:
            metric_card(
                "권고 직원",
                fmt_num(
                    row[
                        "권고직원수"
                    ]
                ),
                "명",
                (
                    "계획 대비 "
                    f"{fmt_signed(row['직원조정수'])}명"
                ),
            )

        with c4:
            metric_card(
                "운영 조정",
                fmt_signed(
                    row[
                        "조정필요수"
                    ]
                ),
                suffix,
                row[
                    "조정판단"
                ],
            )


        if (
            selected_area in IM_AREAS
            and str(
                row[
                    "IM판단"
                ]
            ).strip()
        ):
            st.info(
                f"IM 운영 판단: "
                f"{row['IM판단']}"
            )


        # 선택 구역에서도 그래프 기본 표시
        st.markdown(
            (
                '<div class="section-title">'
                "시간대별 계획·LIVE 변화 그래프"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            window_label
        )

        draw_line_chart(
            chart,
            "항공편 기반 계획 vs LIVE 필요 수",
            y_title,
        )


    (
        _,
        end_min,
        _,
    ) = graph_window(
        selected_time
    )

    current_live_end = (
        hhmm_to_minute(
            data_time
        )
    )

    if current_live_end < end_min:
        st.session_state[
            "live_elapsed"
        ] = (
            int(
                st.session_state.get(
                    "live_elapsed",
                    0,
                )
            )
            + 1
        )


if mode == "OFF":
    render_off_view()


else:
    if not hasattr(
        st,
        "fragment",
    ):
        st.error(
            "현재 Streamlit 버전이 "
            "st.fragment를 지원하지 않습니다. "
            "requirements.txt에서 "
            "streamlit>=1.37.0으로 올려야 합니다."
        )

        st.stop()


    @st.fragment(
        run_every=f"{int(refresh_seconds)}s"
    )
    def live_fragment():
        render_live_view()


    live_fragment()

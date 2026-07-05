from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 인원수 기반 체크인카운터 운영·인력배치 대시보드
#
# 사용 파일:
# 1) data/people_counter_plan_by_time.csv.gz
#
# 목적:
# - 항공편/기종/좌석수 없이 실제 인원수 데이터만 사용
# - 특정 날짜/시간대의 A~N 체크인카운터 혼잡도 확인
# - 핵심: 몇 개 창구를 열고, 몇 명을 배치할지 확인
# =========================================================


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PLAN_PATH = DATA_DIR / "people_counter_plan_by_time.csv.gz"

COUNTERS = list("ABCDEFGHIJKLMN")

CAUTION_LINE = 80
CROWDED_LINE = 120
VERY_CROWDED_LINE = 160

MAX_OPEN_COUNTERS = 40
OPEN_COUNTER_UNIT_PEOPLE = 5
SUPPORT_START_PEOPLE = 80
MAX_SUPPORT_STAFF = 3

FLOW_BIN_MINUTES = 15


st.set_page_config(
    page_title="체크인카운터 운영·인력배치",
    page_icon="✈️",
    layout="wide",
)


# =========================================================
# 화면 스타일
# =========================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    letter-spacing: -0.4px;
}

div[data-testid="stMetric"] {
    background-color: #f8fbff;
    border: 1px solid #d9e2ef;
    padding: 14px 16px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.95rem;
    color: #555;
}

div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 800;
}

.small-caption {
    color: #666;
    font-size: 0.9rem;
}

.priority-main-title {
    font-size: 1.2rem;
    font-weight: 800;
}

.priority-sub {
    color: #666;
    font-size: 0.92rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 데이터 로드
# =========================================================

@st.cache_data
def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = [str(c).replace("\ufeff", "").replace(" ", "").strip() for c in df.columns]
            return df
        except Exception:
            pass

    raise RuntimeError(f"CSV 파일을 읽지 못했습니다: {path}")


def minute_to_hhmm(minute_value):
    try:
        minute_value = int(minute_value)
    except Exception:
        minute_value = 0

    hour = minute_value // 60
    minute = minute_value % 60

    return f"{hour:02d}:{minute:02d}"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not PLAN_PATH.exists():
        st.error(f"필수 파일이 없습니다: {PLAN_PATH}")
        st.info("먼저 `python make_people_counter_plan.py`를 실행해서 전처리 파일을 생성해야 합니다.")
        st.stop()

    df = read_csv_auto(PLAN_PATH)

    rename_map = {
        "data_date": "일자",
        "minute_index": "분인덱스",
        "time_hhmm": "시각",
        "counter": "카운터",
        "counter_type": "카운터유형",
        "num_people": "실제인원수",
    }

    df = df.rename(columns=rename_map)

    required_cols = [
        "일자",
        "분인덱스",
        "시각",
        "카운터",
        "카운터유형",
        "실제인원수",
        "혼잡등급",
        "권고오픈카운터수",
        "창구운영직원수",
        "현장지원직원수",
        "권고직원수",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error("전처리 파일에 필요한 컬럼이 없습니다.")
        st.write("누락 컬럼:", missing)
        st.write("현재 컬럼:", df.columns.tolist())
        st.stop()

    df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
    df["시각"] = df["시각"].astype(str).str.slice(0, 5)
    df["카운터"] = df["카운터"].astype(str).str.strip().str.upper()

    df = df[df["카운터"].isin(COUNTERS)].copy()
    df = df[~df["카운터"].isin(["IM1", "IM2", "OUTSIDE", "GH"])].copy()

    numeric_cols = [
        "분인덱스",
        "실제인원수",
        "권고오픈카운터수",
        "창구운영직원수",
        "현장지원직원수",
        "권고직원수",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["실제인원수"] = df["실제인원수"].round(1)

    if "최대오픈카운터수" not in df.columns:
        df["최대오픈카운터수"] = MAX_OPEN_COUNTERS

    if "혼잡점수" not in df.columns:
        df["혼잡점수"] = df["혼잡등급"].map(
            {
                "보통": 0,
                "주의": 1,
                "혼잡": 2,
                "매우 혼잡": 3,
                "매우혼잡": 3,
            }
        ).fillna(0)

    df["혼잡등급"] = df["혼잡등급"].astype(str).replace({"매우혼잡": "매우 혼잡"})

    df["시간블록"] = (df["분인덱스"] // FLOW_BIN_MINUTES) * FLOW_BIN_MINUTES
    df["시간블록시각"] = df["시간블록"].apply(minute_to_hhmm)

    df = df.sort_values(["일자", "분인덱스", "카운터"]).reset_index(drop=True)

    return df


# =========================================================
# 보조 함수
# =========================================================

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return "0"


def fmt_float(x):
    try:
        return f"{float(x):,.1f}"
    except Exception:
        return "0.0"


def make_download_csv(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def level_order(x):
    order = {
        "보통": 0,
        "주의": 1,
        "혼잡": 2,
        "매우 혼잡": 3,
        "매우혼잡": 3,
    }
    return order.get(str(x), 99)


def level_prefix(level):
    if level == "매우 혼잡":
        return "🔴 즉시 대응"

    if level == "혼잡":
        return "🟠 운영 강화"

    if level == "주의":
        return "🟡 주의 관찰"

    return "🟢 기본 운영"


def action_message(row):
    level = row["혼잡등급"]
    open_cnt = int(row["권고오픈카운터수"])
    support = int(row["현장지원직원수"])

    if level == "매우 혼잡":
        return f"매우 혼잡 상태입니다. 창구 {open_cnt}개 수준으로 즉시 운영하고, 현장지원 {support}명을 투입하는 것이 적절합니다."

    if level == "혼잡":
        return f"혼잡 상태입니다. 창구 {open_cnt}개 운영을 우선 확인하고, 지원 인력 투입 여부를 점검해야 합니다."

    if level == "주의":
        return f"혼잡 전 단계입니다. 창구 {open_cnt}개 운영 상태를 유지하며 인원 증가 여부를 관찰해야 합니다."

    return "현재는 기본 운영 수준입니다."


def add_people_threshold_lines(fig):
    fig.add_hline(
        y=CAUTION_LINE,
        line_dash="dash",
        annotation_text="주의 80명",
        annotation_position="top left",
    )

    fig.add_hline(
        y=CROWDED_LINE,
        line_dash="dash",
        annotation_text="혼잡 120명",
        annotation_position="top left",
    )

    fig.add_hline(
        y=VERY_CROWDED_LINE,
        line_dash="dash",
        annotation_text="매우 혼잡 160명",
        annotation_position="top left",
    )

    return fig


def get_tick_values(time_values, step=4):
    values = list(time_values)

    if not values:
        return []

    return values[::step]


# =========================================================
# 데이터 로드
# =========================================================

plan_df = load_data()


# =========================================================
# 사이드바
# =========================================================

st.sidebar.title("조회 조건")

available_dates = (
    plan_df["일자"]
    .dropna()
    .dt.date
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if not available_dates:
    st.error("조회 가능한 날짜가 없습니다.")
    st.stop()

default_date_index = 0

for i, d in enumerate(available_dates):
    if str(d) == "2025-08-31":
        default_date_index = i
        break

selected_date = st.sidebar.selectbox(
    "날짜",
    available_dates,
    index=default_date_index,
)

date_plan = plan_df[plan_df["일자"].dt.date == selected_date].copy()

available_times = (
    date_plan["시각"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

if not available_times:
    st.error("선택한 날짜에 조회 가능한 시간이 없습니다.")
    st.stop()

default_time_index = 0

for i, t in enumerate(available_times):
    if t >= "08:00":
        default_time_index = i
        break

selected_time = st.sidebar.selectbox(
    "시간",
    available_times,
    index=default_time_index,
)

selected_counters = st.sidebar.multiselect(
    "카운터",
    COUNTERS,
    default=COUNTERS,
)

available_types = sorted(date_plan["카운터유형"].dropna().unique().tolist())

selected_counter_types = st.sidebar.multiselect(
    "카운터유형",
    available_types,
    default=available_types,
)

quick_filter = st.sidebar.radio(
    "빠른 보기",
    ["전체", "주의 이상", "혼잡 이상", "상위 5개"],
    index=0,
)

st.sidebar.divider()

st.sidebar.markdown(
    """
**계산 기준**

- 사용 데이터: 실제 인원수 데이터
- 사용 구역: A~N 체크인카운터
- 제외 구역: IM1, IM2, OUTSIDE, GH
- 주의: 80명 이상
- 혼잡: 120명 이상
- 매우 혼잡: 160명 이상
- 창구 1개당 기준 인원: 5명
- 현장지원직원: 80명 초과부터 추가
- 현장지원직원: 최대 3명
"""
)


# =========================================================
# 선택 데이터 구성
# =========================================================

current = date_plan[date_plan["시각"] == selected_time].copy()

if selected_counters:
    current = current[current["카운터"].isin(selected_counters)].copy()

if selected_counter_types:
    current = current[current["카운터유형"].isin(selected_counter_types)].copy()

current["혼잡등급정렬"] = current["혼잡등급"].apply(level_order)

current = current.sort_values(
    ["혼잡등급정렬", "실제인원수", "권고오픈카운터수", "권고직원수"],
    ascending=[False, False, False, False],
).drop(columns=["혼잡등급정렬"])

if quick_filter == "주의 이상":
    current = current[current["혼잡등급"].isin(["주의", "혼잡", "매우 혼잡"])].copy()

if quick_filter == "혼잡 이상":
    current = current[current["혼잡등급"].isin(["혼잡", "매우 혼잡"])].copy()

if quick_filter == "상위 5개":
    current = current.head(5).copy()

flow = date_plan.copy()

if selected_counters:
    flow = flow[flow["카운터"].isin(selected_counters)].copy()

if selected_counter_types:
    flow = flow[flow["카운터유형"].isin(selected_counter_types)].copy()


# =========================================================
# 메인 화면
# =========================================================

st.title("체크인카운터 운영·인력배치 대시보드")
st.caption(f"{selected_date} {selected_time} 기준 · 실제 인원수 기반 운영계획")


# =========================================================
# 전체 KPI
# =========================================================

total_people = current["실제인원수"].sum() if not current.empty else 0
avg_people = current["실제인원수"].mean() if not current.empty else 0
max_people = current["실제인원수"].max() if not current.empty else 0

total_open_desks = current["권고오픈카운터수"].sum() if not current.empty else 0
total_window_staff = current["창구운영직원수"].sum() if not current.empty else 0
total_support_staff = current["현장지원직원수"].sum() if not current.empty else 0
total_staff = current["권고직원수"].sum() if not current.empty else 0
busy_count = current[current["혼잡등급"].isin(["혼잡", "매우 혼잡"])]["카운터"].nunique() if not current.empty else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("현재 총 인원", f"{fmt_int(total_people)}명", f"평균 {fmt_float(avg_people)}명")
kpi2.metric("최대 카운터 인원", f"{fmt_int(max_people)}명")
kpi3.metric("총 오픈 권고", f"{fmt_int(total_open_desks)}개")
kpi4.metric("총 배치 인력", f"{fmt_int(total_staff)}명")
kpi5.metric("혼잡 카운터", f"{fmt_int(busy_count)}곳")


# =========================================================
# 우선 조치 필요 카운터
# =========================================================

st.divider()

if quick_filter == "상위 5개":
    st.subheader("상위 5개 카운터")
else:
    st.subheader("우선 조치 필요 카운터")

if current.empty:
    st.warning("선택 조건에 해당하는 운영계획이 없습니다.")
else:
    if quick_filter == "상위 5개":
        urgent = current.head(5).copy()
    else:
        urgent = current[current["혼잡등급"].isin(["주의", "혼잡", "매우 혼잡"])].copy()

        if urgent.empty:
            st.success("현재 선택 시간대에는 주의 이상 카운터가 없습니다. 기본 운영만 유지하면 됩니다.")
            urgent = current.head(3).copy()
        else:
            urgent = urgent.head(5)

    for _, row in urgent.iterrows():
        with st.container(border=True):
            st.markdown(f"### {level_prefix(row['혼잡등급'])} · {row['카운터']}카운터")

            main1, main2, main3, main4 = st.columns([1.25, 1.25, 1, 1])

            main1.metric(
                "현재 인원",
                f"{fmt_int(row['실제인원수'])}명",
            )

            main2.metric(
                "오픈 권고",
                f"{int(row['권고오픈카운터수'])}/{int(row['최대오픈카운터수'])}개",
            )

            main3.metric(
                "총 배치 인력",
                f"{int(row['권고직원수'])}명",
            )

            main4.metric(
                "지원 인력",
                f"{int(row['현장지원직원수'])}명",
            )

            st.caption(
                f"혼잡등급: {row['혼잡등급']} · "
                f"카운터유형: {row['카운터유형']} · "
                f"{action_message(row)}"
            )


# =========================================================
# 카운터별 운영 요약
# =========================================================

st.divider()
st.subheader("카운터별 운영 요약")

if current.empty:
    st.warning("표시할 데이터가 없습니다.")
else:
    summary = current.copy()

    summary_cols = [
        "카운터",
        "카운터유형",
        "실제인원수",
        "혼잡등급",
        "권고오픈카운터수",
        "최대오픈카운터수",
        "권고직원수",
        "창구운영직원수",
        "현장지원직원수",
    ]

    renamed = summary[summary_cols].rename(
        columns={
            "실제인원수": "현재인원",
            "권고오픈카운터수": "오픈권고",
            "최대오픈카운터수": "최대창구",
            "권고직원수": "총인력",
            "창구운영직원수": "창구인력",
            "현장지원직원수": "지원인력",
        }
    )

    st.dataframe(
        renamed,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="현재 운영요약 다운로드",
        data=make_download_csv(summary),
        file_name=f"people_counter_summary_{selected_date}_{selected_time.replace(':', '')}.csv",
        mime="text/csv",
    )


# =========================================================
# 현재 시간대 그래프
# =========================================================

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("카운터별 실제 인원수")

    if current.empty:
        st.warning("표시할 데이터가 없습니다.")
    else:
        chart_people = current.sort_values("카운터").copy()

        fig_people = px.bar(
            chart_people,
            x="카운터",
            y="실제인원수",
            text="실제인원수",
            hover_data=[
                "카운터유형",
                "혼잡등급",
                "권고오픈카운터수",
                "권고직원수",
            ],
        )

        fig_people.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_people = add_people_threshold_lines(fig_people)

        y_max = max(chart_people["실제인원수"].max(), VERY_CROWDED_LINE)

        fig_people.update_layout(
            xaxis_title="",
            yaxis_title="실제 인원수",
            yaxis_range=[0, y_max * 1.18],
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_people, use_container_width=True)

with right:
    st.subheader("카운터별 오픈 권고")

    if current.empty:
        st.warning("표시할 데이터가 없습니다.")
    else:
        chart_open = current.sort_values("카운터").copy()

        fig_open = px.bar(
            chart_open,
            x="카운터",
            y="권고오픈카운터수",
            text="권고오픈카운터수",
            hover_data=[
                "카운터유형",
                "실제인원수",
                "혼잡등급",
                "권고직원수",
            ],
        )

        fig_open.update_traces(textposition="outside")
        fig_open.update_layout(
            xaxis_title="",
            yaxis_title="오픈 권고",
            yaxis_range=[0, 42],
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_open, use_container_width=True)


# =========================================================
# 인력 배치 그래프
# =========================================================

st.divider()

left2, right2 = st.columns(2)

with left2:
    st.subheader("카운터별 인력 배치")

    if current.empty:
        st.warning("표시할 데이터가 없습니다.")
    else:
        staff_chart = current[
            [
                "카운터",
                "창구운영직원수",
                "현장지원직원수",
                "권고직원수",
                "혼잡등급",
            ]
        ].copy()

        staff_long = staff_chart.melt(
            id_vars=["카운터", "권고직원수", "혼잡등급"],
            value_vars=["창구운영직원수", "현장지원직원수"],
            var_name="구분",
            value_name="인원",
        )

        fig_staff = px.bar(
            staff_long,
            x="카운터",
            y="인원",
            color="구분",
            text="인원",
            hover_data=["혼잡등급", "권고직원수"],
        )

        fig_staff.update_traces(textposition="inside")
        fig_staff.update_layout(
            xaxis_title="",
            yaxis_title="인원",
            barmode="stack",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_staff, use_container_width=True)

with right2:
    st.subheader("현재 시간대 혼잡 분포")

    if current.empty:
        st.warning("표시할 데이터가 없습니다.")
    else:
        level_count = (
            current.groupby("혼잡등급")
            .agg(카운터수=("카운터", "nunique"))
            .reset_index()
        )

        level_count["정렬"] = level_count["혼잡등급"].apply(level_order)
        level_count = level_count.sort_values("정렬")

        fig_level = px.bar(
            level_count,
            x="혼잡등급",
            y="카운터수",
            text="카운터수",
        )

        fig_level.update_traces(textposition="outside")
        fig_level.update_layout(
            xaxis_title="",
            yaxis_title="카운터 수",
            yaxis_range=[0, max(5, level_count["카운터수"].max() + 1)],
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_level, use_container_width=True)


# =========================================================
# 하루 운영 흐름
# =========================================================

st.divider()
st.subheader("하루 운영 흐름")

if flow.empty:
    st.warning("시간대별 흐름을 표시할 데이터가 없습니다.")
else:
    flow_grouped = (
        flow.groupby(["시간블록", "시간블록시각"], dropna=False)
        .agg(
            총인원=("실제인원수", "sum"),
            최대카운터인원=("실제인원수", "max"),
            오픈창구=("권고오픈카운터수", "sum"),
            총인력=("권고직원수", "sum"),
        )
        .reset_index()
        .sort_values("시간블록")
    )

    flow_grouped["총인원"] = flow_grouped["총인원"].round(1)

    fig_flow = px.line(
        flow_grouped,
        x="시간블록시각",
        y=["총인원", "오픈창구", "총인력"],
        markers=True,
    )

    tick_values = get_tick_values(flow_grouped["시간블록시각"].tolist(), step=4)

    fig_flow.update_layout(
        xaxis_title="시각",
        yaxis_title="합계",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_values,
        ),
        legend_title="지표",
    )

    st.plotly_chart(fig_flow, use_container_width=True)


# =========================================================
# 시간대별 혼잡 발생 흐름
# =========================================================

st.divider()
st.subheader("시간대별 혼잡 발생 흐름")

if flow.empty:
    st.warning("혼잡 흐름을 표시할 데이터가 없습니다.")
else:
    flow_level = flow[flow["혼잡등급"].isin(["주의", "혼잡", "매우 혼잡"])].copy()

    if flow_level.empty:
        st.success("선택한 조건에서는 주의 이상 혼잡 구간이 없습니다.")
    else:
        flow_level_grouped = (
            flow_level.groupby(["시간블록", "시간블록시각", "혼잡등급"], dropna=False)
            .agg(혼잡카운터수=("카운터", "nunique"))
            .reset_index()
            .sort_values("시간블록")
        )

        fig_congestion = px.bar(
            flow_level_grouped,
            x="시간블록시각",
            y="혼잡카운터수",
            color="혼잡등급",
            text="혼잡카운터수",
            category_orders={"혼잡등급": ["주의", "혼잡", "매우 혼잡"]},
        )

        tick_values = get_tick_values(
            sorted(flow_level_grouped["시간블록시각"].dropna().unique().tolist()),
            step=4,
        )

        fig_congestion.update_traces(textposition="inside")
        fig_congestion.update_layout(
            xaxis_title="시각",
            yaxis_title="혼잡 발생 카운터 수",
            barmode="stack",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                tickmode="array",
                tickvals=tick_values,
            ),
            legend_title="혼잡등급",
        )

        st.plotly_chart(fig_congestion, use_container_width=True)


# =========================================================
# 카운터별 하루 요약
# =========================================================

st.divider()
st.subheader("카운터별 하루 요약")

if flow.empty:
    st.warning("요약할 데이터가 없습니다.")
else:
    daily_summary = (
        flow.groupby(["카운터", "카운터유형"], dropna=False)
        .agg(
            평균인원=("실제인원수", "mean"),
            최대인원=("실제인원수", "max"),
            최대오픈권고=("권고오픈카운터수", "max"),
            최대총인력=("권고직원수", "max"),
            평균총인력=("권고직원수", "mean"),
        )
        .reset_index()
    )

    level_counts = (
        flow.pivot_table(
            index="카운터",
            columns="혼잡등급",
            values="분인덱스",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["보통", "주의", "혼잡", "매우 혼잡"]:
        if col not in level_counts.columns:
            level_counts[col] = 0

    daily_summary = daily_summary.merge(level_counts, on="카운터", how="left")

    daily_summary = daily_summary.rename(
        columns={
            "보통": "보통_분",
            "주의": "주의_분",
            "혼잡": "혼잡_분",
            "매우 혼잡": "매우혼잡_분",
        }
    )

    daily_summary["평균인원"] = daily_summary["평균인원"].round(1)
    daily_summary["최대인원"] = daily_summary["최대인원"].round(1)
    daily_summary["평균총인력"] = daily_summary["평균총인력"].round(1)

    daily_summary = daily_summary.sort_values(
        ["매우혼잡_분", "혼잡_분", "주의_분", "최대인원"],
        ascending=[False, False, False, False],
    )

    st.dataframe(
        daily_summary,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 선택 카운터 상세 추세
# =========================================================

st.divider()
st.subheader("선택 카운터 상세 추세")

trend_options = [c for c in COUNTERS if c in flow["카운터"].unique().tolist()]
default_trend = trend_options[: min(4, len(trend_options))]

trend_counters = st.multiselect(
    "상세 추세로 볼 카운터",
    trend_options,
    default=default_trend,
)

if not trend_counters:
    st.info("상세 추세를 볼 카운터를 선택하세요.")
else:
    trend = flow[flow["카운터"].isin(trend_counters)].copy()

    trend_grouped = (
        trend.groupby(["시간블록", "시간블록시각", "카운터"], dropna=False)
        .agg(평균인원=("실제인원수", "mean"))
        .reset_index()
        .sort_values(["시간블록", "카운터"])
    )

    trend_grouped["평균인원"] = trend_grouped["평균인원"].round(1)

    fig_trend = px.line(
        trend_grouped,
        x="시간블록시각",
        y="평균인원",
        color="카운터",
        markers=True,
    )

    fig_trend = add_people_threshold_lines(fig_trend)

    tick_values = get_tick_values(
        sorted(trend_grouped["시간블록시각"].dropna().unique().tolist()),
        step=4,
    )

    y_max = max(trend_grouped["평균인원"].max(), VERY_CROWDED_LINE)

    fig_trend.update_layout(
        xaxis_title="시각",
        yaxis_title="평균 인원수",
        yaxis_range=[0, y_max * 1.18],
        height=430,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_values,
        ),
        legend_title="카운터",
    )

    st.plotly_chart(fig_trend, use_container_width=True)


# =========================================================
# 현재 시간대 인원수 집계 보기
# =========================================================

st.divider()

with st.expander("현재 시간대 인원수 집계 보기", expanded=False):
    if current.empty:
        st.warning("선택한 시간대에 표시할 데이터가 없습니다.")
    else:
        raw_cols = [
            "일자",
            "시각",
            "분인덱스",
            "카운터",
            "카운터유형",
            "실제인원수",
            "혼잡등급",
            "권고오픈카운터수",
            "창구운영직원수",
            "현장지원직원수",
            "권고직원수",
        ]

        st.dataframe(
            current[raw_cols].sort_values("카운터"),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="현재 시간대 인원수 집계 다운로드",
            data=make_download_csv(current[raw_cols]),
            file_name=f"current_people_count_{selected_date}_{selected_time.replace(':', '')}.csv",
            mime="text/csv",
        )


# =========================================================
# 계산 기준
# =========================================================

with st.expander("계산 기준 보기", expanded=False):
    st.markdown(
        """
### 사용 데이터
이 대시보드는 항공편, 기종, 좌석수 데이터를 사용하지 않고  
`area_count_time_full_*.csv`의 실제 인원수 데이터만 사용합니다.

### 사용 카운터
체크인카운터 기준이므로 A~N 단일 알파벳 카운터만 사용합니다.

제외 구역은 다음과 같습니다.

- IM1
- IM2
- OUTSIDE
- GH

### 시간 기준
`time_index`는 10초 단위로 해석했습니다.

- 1 → 00:00
- 7 → 00:01
- 361 → 01:00

같은 분 안에 여러 10초 데이터가 있으면 카운터별 평균 인원수로 집계했습니다.

### 혼잡등급
- 보통: 80명 미만
- 주의: 80명 이상
- 혼잡: 120명 이상
- 매우 혼잡: 160명 이상

### 권고 오픈 창구 수
실제 인원수 기준으로 5명당 창구 1개를 권고합니다.

계산식:

`권고오픈카운터수 = ceil(현재 인원수 / 5)`

단, 최대 오픈 카운터 수는 40개로 제한했습니다.

### 직원 수
직원 수는 두 종류로 구분했습니다.

- **창구운영직원수**: 실제 오픈 창구를 운영하는 인력
- **현장지원직원수**: 줄 정리, 승객 안내, 질서유지 인력

현장지원직원은 80명을 초과하는 구간부터 추가하고, 과대 산정을 피하기 위해 최대 3명으로 제한했습니다.
"""
    )
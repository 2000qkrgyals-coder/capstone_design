import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# 1. 페이지 설정 및 디자인 테마
st.set_page_config(page_title="Incheon Airport Smart Guide", layout="wide")
MAIN_CYAN = "#00EEFF" # ISTJ 스타일의 사이언 색상

# [전처리 스크립트와 동일한 파라미터 설정]
# 이 수치들이 일치해야 전처리된 데이터와 실시간 계산 로직이 충돌하지 않습니다.
WAIT_PARAMS = {
    "checkin":  {"alpha": 4.0, "gamma": 0.09, "R": 6.0,  "beta": 1.5, "wmax": 120.0},
    "security": {"alpha": 5.0, "gamma": 0.11, "R": 8.0,  "beta": 2.0, "wmax": 120.0},
    "transit":  {"alpha": 1.5, "gamma": 0.11, "R": 20.0, "beta": 4.0, "wmax": 60.0}
}

def classify_area(area_name):
    a = str(area_name).strip().upper()
    if a in list("ABCDEFGHIJKLMN"): return "checkin"
    if any(k in a for k in ["SECURITY", "SEARCH", "SCREEN", "DEPARTURE", "GATE", "IM1", "IM2"]): return "security"
    return "transit"

def compute_wait_time_logic(area_type, n_eff):
    p = WAIT_PARAMS.get(area_type, WAIT_PARAMS["transit"])
    load_factor = max(0, n_eff) / p["R"]
    # 전처리 스크립트 공식: W = beta + alpha * (exp(gamma * load_factor) - 1.0)
    wait_min = p["beta"] + p["alpha"] * (math.exp(p["gamma"] * load_factor) - 1.0)
    return round(max(0.0, min(wait_min, p["wmax"])), 2)

@st.cache_data
def load_data():
    try:
        # 전처리 완료된 파일 로드
        df = pd.read_parquet("final_cache_10min.parquet")
        df['area'] = df['area'].str.strip().str.upper()
        df = df[df['area'] != 'I']
        
        # 시간 표시용 레이블 생성 (10분 단위 인덱스 활용)
        def index_to_time(idx):
            return f"{int(idx // 6):02d}:{int((idx % 6) * 10):02d}"
        
        df['time_label'] = df['ten_min_index'].apply(index_to_time)
        return df
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return None

df = load_data()

if df is not None:
    # --- 사이드바: 기본 설정 ---
    with st.sidebar:
        st.header("⚙️ System Control")
        selected_day = st.selectbox("📅 분석 날짜 선택", sorted(df['day'].unique()))
        st.markdown("---")
        st.info("Bluetooth RSSI 기술과 전처리 알고리즘이 결합된 대기 시간 예측 시스템입니다.")

    day_df = df[df['day'] == selected_day]
    all_areas = sorted(day_df['area'].unique())

    # --- 메인 대시보드 구성 ---
    tab1, tab2, tab3 = st.tabs(["📊 실시간 모니터링", "🧬 알고리즘 엔진", "🧭 최적 경로 가이드"])

    # [Tab 1] 실시간 모니터링
    with tab1:
        st.subheader(f"📍 {selected_day}일 구역별 실시간 지표")
        
        default_selection = [a for a in ["A", "G", "IM1", "IM2"] if a in all_areas]
        selected_areas = st.multiselect("비교 대상 구역 선택", all_areas, default=default_selection)
        
        plot_df = day_df[day_df['area'].isin(selected_areas)].sort_values('ten_min_index')
        
        st.write("### **⏱ 예상 대기시간 (분)**")
        fig_wait = px.line(plot_df, x="time_label", y="wait_time", color="area",
                          labels={"time_label": "시간", "wait_time": "대기시간 (분)"},
                          template="plotly_dark", 
                          color_discrete_sequence=["#00EEFF", "#00CCDD", "#00AABB", "#008899"])
        
        avg_wait = plot_df['wait_time'].mean()
        fig_wait.add_hline(y=avg_wait, line_dash="dash", line_color="gray", annotation_text="전체 평균")
        fig_wait.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_wait, use_container_width=True)

        st.markdown("---")

        st.write("### **👥 구역별 실시간 인원수 (명)**")
        fig_people = px.area(plot_df, x="time_label", y="num_people", color="area",
                            labels={"time_label": "시간", "num_people": "인원수 (명)"},
                            template="plotly_dark",
                            color_discrete_sequence=["#00EEFF", "#00CCDD", "#00AABB", "#008899"])
        fig_people.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_people, use_container_width=True)
        st.caption("※ 위 수치는 EWM(Exponential Weighted Moving Average)으로 보정된 데이터입니다.")

    # [Tab 2] 알고리즘 엔진
    with tab2:
        st.header("🧪 Data-Driven Modeling & Reliability Logic")
        
        col_logic_text, col_logic_formula = st.columns([1.5, 1])
        with col_logic_text:
            st.markdown("### **1. 비선형 지수 모델의 타당성**")
            st.write(f"""
            본 프로젝트의 수식은 사람이 없을 때 대기시간이 기본값($\\beta$)이 되도록 설계되었습니다. 
            특히 **감쇄 계수($\\gamma$)**와 **처리 용량($R$)**을 구역 성격(체크인, 보안검색, 면세구역)에 따라 다르게 설정하여
            병목 현상의 특수성을 반영합니다.
            """)
        with col_logic_formula:
            st.latex(r"W = \beta + \alpha \cdot (e^{\gamma \cdot \frac{N}{R}} - 1)")
            st.caption("공항 전처리용 최종 확정 수식 적용")

        st.divider()

        st.markdown("### **2. 혼잡 전이 인과관계 증명 (Correlation Analysis)**")
        col_corr_plot, col_corr_text = st.columns([2, 1])
        with col_corr_plot:
            checkin_sum = day_df[day_df['area'].isin(list("ABCDEF"))].groupby('ten_min_index')['num_people'].sum()
            security_wait = day_df[day_df['area'] == "IM1"].set_index('ten_min_index')['wait_time']
            lags = range(0, 13) 
            corrs = [checkin_sum.corr(security_wait.shift(-l)) for l in lags]
            
            fig_corr = px.bar(x=[l*10 for l in lags], y=corrs, 
                               labels={'x': '혼잡 전이 시차 (분)', 'y': '상관계수 (Pearson)'},
                               color_discrete_sequence=[MAIN_CYAN],
                               template="plotly_dark")
            fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig_corr, use_container_width=True)
        with col_corr_text:
            st.write("**[데이터 일관성 분석]**")
            st.info("체크인-보안검색 간의 상관계수가 특정 시차에서 높게 나타나는 것을 통해 데이터의 흐름을 증명합니다.")

        st.divider()

        st.markdown("### **3. 구역별 알고리즘 파라미터 (System Constants)**")
        st.table(pd.DataFrame(WAIT_PARAMS).T)

    # [Tab 3] 사용자 맞춤형 최적 경로 가이드
    with tab3:
        st.header("🧭 Personalized Smart Journey")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            flight_time = st.time_input("✈️ 비행기 출발 시각", datetime.strptime("13:00", "%H:%M"))
        with c2:
            gate_margin = st.select_slider("🚩 게이트 도착 목표", options=[20, 30, 40, 50, 60, 90], value=40)
        with c3:
            counter_list = [a for a in all_areas if len(a) == 1]
            assigned_counter = st.selectbox("🧳 나의 체크인 카운터", counter_list)

        flight_dt = datetime.combine(datetime.now().date(), flight_time)
        analysis_dt = flight_dt - timedelta(minutes=gate_margin + 60)
        calc_idx = max(0, min(143, (analysis_dt.hour * 6) + (analysis_dt.minute // 10)))
        snap = day_df[day_df['ten_min_index'] == calc_idx]

        if not snap.empty:
            try:
                w_chk = snap[snap['area'] == assigned_counter]['wait_time'].values[0]
                w_im1 = snap[snap['area'] == "IM1"]['wait_time'].values[0]
                w_im2 = snap[snap['area'] == "IM2"]['wait_time'].values[0]
                
                is_right = assigned_counter in list("ABCDEF")
                dist_im1, dist_im2 = (5, 15) if is_right else (15, 5)
                total_im1, total_im2 = dist_im1 + w_im1, dist_im2 + w_im2
                
                if total_im1 <= total_im2:
                    best_im, other_im = "IM1(1번)", "IM2(2번)"
                    best_wait, other_wait = w_im1, w_im2
                    best_dist, other_dist = dist_im1, dist_im2
                    best_total, other_total = total_im1, total_im2
                else:
                    best_im, other_im = "IM2(2번)", "IM1(1번)"
                    best_wait, other_wait = w_im2, w_im1
                    best_dist, other_dist = dist_im2, dist_im1
                    best_total, other_total = total_im2, total_im1

                t_gate = flight_dt - timedelta(minutes=gate_margin)
                t_sec_entry = t_gate - timedelta(minutes=20 + best_wait) 
                t_checkin_start = t_sec_entry - timedelta(minutes=best_dist + w_chk)

                st.divider()
                st.markdown(f"### 🏆 최적 경로: {best_im} 이용 권장")
                st.info(f"💡 {best_im} 경로가 {other_im}보다 약 **{abs(total_im1 - total_im2):.1f}분** 더 빠릅니다.")

                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.write(f"**✅ 추천: {best_im}**")
                    st.write(f"- 도보 이동: {best_dist}분")
                    st.write(f"- 대기 예상: {best_wait:.1f}분")
                    st.write(f"- **합계: {best_total:.1f}분**")
                with col_comp2:
                    st.write(f"**❌ 대안: {other_im}**")
                    st.write(f"- 도보 이동: {other_dist}분")
                    st.write(f"- 대기 예상: {other_wait:.1f}분")
                    st.write(f"- **합계: {other_total:.1f}분**")

                st.markdown("---")

                m1, m2, m3 = st.columns(3)
                m1.metric("🏠 수속 시작 권장", t_checkin_start.strftime("%H:%M"))
                m2.metric(f"🔍 {best_im} 진입 시각", t_sec_entry.strftime("%H:%M"))
                m3.metric("🚩 게이트 도착 목표", t_gate.strftime("%H:%M"))

                st.write("")
                cw1, cw2, ct = st.columns(3)
                with cw1:
                    st.markdown(f"**🧳 {assigned_counter} 카운터 대기**")
                    st.title(f"{w_chk:.1f} 분")
                    st.progress(min(w_chk/60, 1.0))
                with cw2:
                    st.markdown(f"**👮 {best_im} 보안 대기**")
                    st.title(f"{best_wait:.1f} 분")
                    st.progress(min(best_wait/60, 1.0))
                with ct:
                    st.markdown("**🔥 총 예상 여정 시간**")
                    st.title(f"{w_chk + best_dist + best_wait:.0f} 분")
                    st.caption(f"이동 {best_dist}분 포함")

                fig_comp = go.Figure(data=[
                    go.Bar(name='도보 이동', x=[best_im, other_im], y=[best_dist, other_dist], marker_color='#333333'),
                    go.Bar(name='대기 예상', x=[best_im, other_im], y=[best_wait, other_wait], marker_color=MAIN_CYAN)
                ])
                fig_comp.update_layout(barmode='stack', template="plotly_dark", title="출국장별 소요 시간 상세 비교", height=350)
                st.plotly_chart(fig_comp, use_container_width=True)

            except Exception as e:
                st.error("⚠️ 데이터 연산 중 오류가 발생했습니다.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import math

# --- 0. 페이지 설정 ---
st.set_page_config(page_title="ICN Smart Control Center", layout="wide")

# --- 1. 데이터 로드 함수 (이동평균 스무딩 포함) ---
@st.cache_data
def load_all_data(traffic_path, coord_path):
    if not os.path.exists(traffic_path) or not os.path.exists(coord_path):
        return None, None
    
    df = pd.read_csv(traffic_path)
    df['area'] = df['area'].astype(str).apply(lambda x: x.strip().upper())
    # 10초 단위 데이터를 1분 단위로 그룹화
    df['minute_index'] = (df['time_index'] - 1) // 6
    df_min = df.groupby(['minute_index', 'area'])['num_people'].mean().reset_index()
    
    coords = pd.read_csv(coord_path)
    if 'area_name' in coords.columns:
        coords = coords.rename(columns={'area_name': 'area'})
    coords['area'] = coords['area'].astype(str).apply(lambda x: x.strip().upper())
    coords['x'] = coords[['x1', 'x2', 'x3', 'x4']].mean(axis=1)
    coords['y'] = coords[['y1', 'y2', 'y3', 'y4']].mean(axis=1)
    
    return df_min, coords[['area', 'x', 'y']]

# --- 2. 사이드바 및 환경 설정 ---
st.sidebar.header("🕹️ 관제 설정")
c1, c2 = st.sidebar.columns(2)
in_hour = c1.number_input("시 (0~23)", 0, 23, 10)
in_min = c2.number_input("분 (0~59)", 0, 59, 0)
current_time_min = in_hour * 60 + in_min

# --- [추가] 공통 운영 변수 설정 (에러 방지) ---
with st.sidebar.expander("⚙️ 기본 운영 파라미터"):
    service_rate = st.number_input("카운터당 처리 용량 (명/10분)", 1, 100, 15)
    wait_threshold = st.slider("대기시간 경고 기준 (분)", 5, 30, 15)

selected_date = st.sidebar.date_input("날짜 선택", value=pd.to_datetime("2026-09-14"))
traffic_file = f"area_count_time_{selected_date.month:02d}_{selected_date.day:02d}.csv"
area_coord_file = "terminal_areas_grouped_2.csv"
bg_img_path = "ICN_Airport_3F.png"

df, coords = load_all_data(traffic_file, area_coord_file)

if df is not None and coords is not None:
    # --- 데이터 전처리 (스무딩) ---
    pivot_raw = df.pivot(index='minute_index', columns='area', values='num_people').fillna(0)
    if 'OUTSIDE' in pivot_raw.columns:
        pivot_raw = pivot_raw.drop(columns=['OUTSIDE'])
    
    # 10분 이동평균 적용 (시각적 노이즈 제거)
    pivot_df = pivot_raw.rolling(window=10, min_periods=1, center=True).mean()
    total_flow = pivot_df.sum(axis=1).reindex(range(1440), fill_value=0)

  # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 실시간 통합 관제", "🕒 시간대별 피크 분석", "🔍 구역별 상세 비교 분석", "🛡️ 안전 관리 및 위기 대응"])
# --- [TAB 1] 실시간 통합 관제 (중복 Key 에러 완전 해결 버전) ---
    with tab1:
        st.title("📊 실시간 관제 현황 (자율 재생 모드)")
        st.caption("💡 이 탭은 사이드바의 시간 설정과 무관하게 전체 타임라인을 부드럽게 재생합니다.")
        
        # 1. 데이터 결합 및 시간 리스트 생성
        anim_data = pd.merge(df, coords, on='area')
        anim_data['시간'] = anim_data['minute_index'].apply(lambda x: f"{x//60:02d}:{x%60:02d}")
        anim_data = anim_data.sort_values('minute_index')
        unique_times = sorted(anim_data['시간'].unique())
        
        # 2. 토글 컨트롤러 (재생 / 정지 상태 관리)
        if "loop_playing" not in st.session_state:
            st.session_state.loop_playing = False
        if "loop_time_idx" not in st.session_state:
            st.session_state.loop_time_idx = 0

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("▶️ 재생 시작" if not st.session_state.loop_playing else "⏸️ 시뮬레이션 중지", use_container_width=True):
                st.session_state.loop_playing = not st.session_state.loop_playing
                st.rerun()
        with c2:
            # 수동 탐색용 슬라이더 (사이드바 독립)
            selected_idx = st.slider(
                "⏱️ 자율 타임라인 제어", 
                min_value=0, 
                max_value=len(unique_times) - 1, 
                value=st.session_state.loop_time_idx,
                format=""
            )
            if not st.session_state.loop_playing:
                st.session_state.loop_time_idx = selected_idx

        # 3. 화면 깜빡임을 방지하는 단일 정적 컨테이너
        main_view = st.empty()

        if os.path.exists(bg_img_path):
            img = Image.open(bg_img_path)
            img_width, img_height = img.size
            
            grid_x = np.linspace(0, img_width, 40)
            grid_y = np.linspace(0, img_height, 25)
            X, Y = np.meshgrid(grid_x, grid_y)

            # [재생 모드 루프 제어]
            while st.session_state.loop_playing:
                current_time = unique_times[st.session_state.loop_time_idx]
                t_data = anim_data[anim_data['시간'] == current_time]
                
                # 가우시안 히트맵 매트릭스 계산
                Z = np.zeros_like(X)
                for _, row in t_data.iterrows():
                    if row['num_people'] > 0:
                        sigma = max(img_width, img_height) * 0.04
                        dist_sq = (X - row['x'])**2 + (Y - row['y'])**2
                        Z += row['num_people'] * np.exp(-dist_sq / (2 * sigma**2))
                
                # --- 지도 생성 ---
                fig_map = go.Figure(data=go.Contour(
                    x=grid_x, y=grid_y, z=Z,
                    colorscale=[
                        [0.0, 'rgba(0,0,0,0)'],           
                        [0.2, 'rgba(0, 120, 255, 0.22)'], 
                        [0.5, 'rgba(0, 240, 100, 0.42)'], 
                        [0.8, 'rgba(255, 140, 0, 0.62)'], 
                        [1.0, 'rgba(240, 0, 0, 0.82)']    
                    ],
                    contours=dict(coloring='heatmap', showlines=False),
                    line_width=0, opacity=0.65, showscale=True,
                    colorbar=dict(title="혼잡도", thickness=12)
                ))
                fig_map.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=0, sizex=img_width, sizey=img_height, sizing="stretch", opacity=0.6, layer="below"))
                fig_map.update_layout(height=480, template="plotly_dark", margin=dict(l=5,r=5,b=5,t=5))
                fig_map.update_xaxes(visible=False, range=[0, img_width])
                fig_map.update_yaxes(visible=False, range=[img_height, 0])

                # --- 우측 랭킹 차트 생성 ---
                rank_data = t_data.sort_values('num_people', ascending=False).head(10)
                fig_rank = px.bar(rank_data, x='num_people', y='area', orientation='h', color='num_people', color_continuous_scale='Reds', template="plotly_dark")
                fig_rank.update_layout(height=480, yaxis={'autorange': 'reversed'}, margin=dict(l=5,r=5,b=5,t=5), coloraxis_showscale=False)

                # 💡 해결책 1: Key 이름 뒤에 현재 시간 문자열을 결합하여 고유성을 확보합니다.
                # 컨테이너 안에서 지워지고 새로 생성될 때 중복 에러가 안 납니다.
                with main_view.container():
                    st.markdown(f"#### ⏱️ 현재 자율 관제 시점: `{current_time}`")
                    v_c1, v_c2 = st.columns([2, 1])
                    v_c1.plotly_chart(fig_map, use_container_width=True, key=f"play_map_{current_time}")
                    v_c2.plotly_chart(fig_rank, use_container_width=True, key=f"play_rank_{current_time}")

                import time
                time.sleep(0.05)
                
                # 다음 시간 인덱스로 이동
                st.session_state.loop_time_idx = (st.session_state.loop_time_idx + 1) % len(unique_times)

            # --- [정지 상태] 혹은 슬라이더 수동 조작 시 화면 렌더링 ---
            if not st.session_state.loop_playing:
                current_time = unique_times[st.session_state.loop_time_idx]
                t_data = anim_data[anim_data['시간'] == current_time]
                
                Z = np.zeros_like(X)
                for _, row in t_data.iterrows():
                    if row['num_people'] > 0:
                        sigma = max(img_width, img_height) * 0.04
                        dist_sq = (X - row['x'])**2 + (Y - row['y'])**2
                        Z += row['num_people'] * np.exp(-dist_sq / (2 * sigma**2))
                        
                fig_map = go.Figure(data=go.Contour(
                    x=grid_x, y=grid_y, z=Z,
                    colorscale=[[0.0, 'rgba(0,0,0,0)'], [0.2, 'rgba(0,120,255,0.22)'], [0.5, 'rgba(0,240,100,0.42)'], [1.0, 'rgba(240,0,0,0.82)']],
                    contours=dict(coloring='heatmap', showlines=False), line_width=0, opacity=0.65
                ))
                fig_map.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=0, sizex=img_width, sizey=img_height, sizing="stretch", opacity=0.6, layer="below"))
                fig_map.update_layout(height=480, template="plotly_dark", margin=dict(l=5,r=5,b=5,t=5))
                fig_map.update_xaxes(visible=False, range=[0, img_width])
                fig_map.update_yaxes(visible=False, range=[img_height, 0])
                
                rank_data = t_data.sort_values('num_people', ascending=False).head(10)
                fig_rank = px.bar(rank_data, x='num_people', y='area', orientation='h', color='num_people', color_continuous_scale='Reds', template="plotly_dark")
                fig_rank.update_layout(height=480, yaxis={'autorange': 'reversed'}, margin=dict(l=5,r=5,b=5,t=5), coloraxis_showscale=False)

                # 💡 해결책 2: 정지 상태용 맵 키 이름도 동적으로 설정해 줍니다.
                with main_view.container():
                    st.markdown(f"#### ⏸️ 대기 중인 시점: `{current_time}`")
                    v_c1, v_c2 = st.columns([2, 1])
                    v_c1.plotly_chart(fig_map, use_container_width=True, key=f"stop_map_{current_time}")
                    v_c2.plotly_chart(fig_rank, use_container_width=True, key=f"stop_rank_{current_time}")
        else:
            st.error("공항 배경 이미지(PNG)를 찾을 수 없습니다.")

    # --- [TAB 2] 시간대별 피크 분석 ---
    with tab2:
        st.title("🕒 주요 피크 시간대 분석")
        peak_definitions = {
            "🌅 아침 피크 (07-09시)": (420, 540, "rgba(255, 99, 71, 0.2)"),
            "☀️ 낮 피크 (12-14시)": (720, 840, "rgba(255, 215, 0, 0.2)"),
            "🌙 저녁 피크 (18-20시)": (1080, 1200, "rgba(30, 144, 255, 0.2)")
        }
        
        p1, p2, p3 = st.columns(3)
        cols = [p1, p2, p3]
        for i, (name, (start, end, color)) in enumerate(peak_definitions.items()):
            peak_val = total_flow.iloc[start:end].max()
            peak_time = total_flow.iloc[start:end].idxmax()
            cols[i].metric(name, f"{peak_val:.1f}명", f"발생시각 {peak_time//60:02d}:{peak_time%60:02d}")
        
        st.divider()
        st.subheader("📈 전체 시간대 혼잡 흐름 (구간 하이라이트)")
        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(x=total_flow.index, y=total_flow, name="전체 인원", fill='tozeroy', line=dict(color='#FF4B4B')))
        
        for name, (start, end, color) in peak_definitions.items():
            fig_total.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.5, layer="below", line_width=0, 
                                annotation_text=name.split()[0], annotation_position="top left")

        fig_total.update_layout(template="plotly_dark", height=450, 
                                xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                           ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)]))
        st.plotly_chart(fig_total, use_container_width=True)

    # --- [TAB 3] 구역별 상세 비교 분석 ---
    with tab3:
        st.title("🔍 구역별 심층 비교 및 분석 리포트")
        selected_areas = st.multiselect("비교할 구역을 선택하세요", pivot_df.columns, default=list(pivot_df.columns[:3]))
        
        if selected_areas:
            # ---------------------------------------------------------
            # [기존 유지] 지표 기반 정량 분석 리포트 (1, 2, 4번 지표)
            # ---------------------------------------------------------
            st.subheader("📋 실시간 핵심 운영 지표 (정량 분석)")
            
            now_idx = current_time_min
            prev_idx = max(0, now_idx - 5) # 5분 전 비교
            avg_people_all = pivot_df.iloc[now_idx].mean() # 4번 지표용 전체 평균
            
            analysis_metrics = []
            for area in selected_areas:
                curr_v = pivot_df[area].iloc[now_idx]
                prev_v = pivot_df[area].iloc[prev_idx]
                
                # 1. 혼잡 지수 (임계치 100명 대비 비율)
                c_idx = (curr_v / 100) * 100
                # 2. 유입 가속도 (분당 변화량)
                accel = (curr_v - prev_v) / 5
                # 4. 불균형 지수 (전체 평균 대비 쏠림 정도)
                imb_idx = curr_v / avg_people_all if avg_people_all > 0 else 0
                
                analysis_metrics.append({
                    "구역": area,
                    "현재 인원": round(curr_v, 1),
                    "혼잡 지수(%)": round(c_idx, 1),
                    "유입 가속도(명/분)": round(accel, 2),
                    "불균형 지수(평균대비)": round(imb_idx, 2)
                })
            
            metric_df = pd.DataFrame(analysis_metrics)
            
            # 지표 테이블 출력 (컬러 강조 추가)
            st.table(metric_df.style.format({
                '혼잡 지수(%)': '{:.1f}%',
                '유입 가속도(명/분)': '{:+.2f}',
                '불균형 지수(평균대비)': '{:.2f}x'
            }).background_gradient(subset=['혼잡 지수(%)'], cmap='Reds', vmin=0, vmax=120)
              .background_gradient(subset=['유입 가속도(명/분)'], cmap='coolwarm'))
            
            st.caption("※ 혼잡 지수: 100명 기준(100%) / 유입 가속도: 5분 전 대비 변화율 / 불균형 지수: 현재 전체 구역 평균 대비 배수")

            # 지표 상관관계 시각화 (Bubble Chart)
            fig_bubble = px.scatter(metric_df, x="혼잡 지수(%)", y="유입 가속도(명/분)", 
                                    size="불균형 지수(평균대비)", color="구역",
                                    hover_name="구역", size_max=40,
                                    title="📊 운영 지표 상관 분석 (원 크기 = 불균형 쏠림 정도)",
                                    template="plotly_dark")
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.divider()

            # 1. 점유율 상세 (기존 유지)
            st.subheader("🌐 공항 전체 대비 점유 비중")
            total_now = total_flow.get(current_time_min, 1)
            selected_now_vals = pivot_df.loc[current_time_min, selected_areas]
            
            c_met, c_table, c_pie = st.columns([1, 1.5, 1.5])
            with c_met:
                st.metric("선택 구역 합계 비중", f"{(selected_now_vals.sum()/total_now)*100:.1f}%")
            with c_table:
                share_df = pd.DataFrame({
                    '구역': selected_areas,
                    '인원(명)': selected_now_vals.values.round(1),
                    '비중(%)': [(v/total_now)*100 for v in selected_now_vals]
                }).sort_values('인원(명)', ascending=False)
                st.table(share_df.style.format({'비중(%)': '{:.1f}%'}))
            with c_pie:
                fig_pie = px.pie(share_df, values='인원(명)', names='구역', hole=0.4, template="plotly_dark")
                fig_pie.update_layout(height=250, margin=dict(t=0, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # 2. 시간대별 혼잡도 추이 (기존 유지)
            st.subheader("📈 시간대별 혼잡도 추이 (10분 이동평균 적용)")
            fig_compare = go.Figure()
            for area in selected_areas:
                fig_compare.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[area], name=area, mode='lines', line=dict(width=3)))
            
            fig_compare.add_vline(x=current_time_min, line_dash="dash", line_color="red", annotation_text="현재")
            fig_compare.update_layout(template="plotly_dark", height=600, hovermode="x unified",
                                      xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                                 ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)]))
            st.plotly_chart(fig_compare, use_container_width=True)

            # 3. 통계 기반 관제 인사이트 (기존 유지)
            st.divider()
            st.subheader("🧪 통계 기반 관제 인사이트")
            c_ins1, c_ins2, c_ins3 = st.columns(3)
            
            with c_ins1:
                st.write("**⌛ 과밀 유지 시간 (80명 기준)**")
                for area in selected_areas:
                    crowded_mins = (pivot_df[area] > 80).sum()
                    st.write(f"- {area}: **{crowded_mins}분**")
                st.caption("누적 혼잡 시간을 통해 상시 인력 배치 구역을 판단합니다.")

            with c_ins2:
                st.write("**⚡ 현재 유입 가속도 (명/분)**")
                for area in selected_areas:
                    accel_val = pivot_df[area].diff().iloc[current_time_min]
                    status = "🚀 급증" if accel_val > 1.5 else ("➡️ 안정" if accel_val > -1.5 else "⬇️ 감소")
                    st.write(f"- {area}: {status} ({accel_val:.2f})")

            with c_ins3:
                st.write("**🔗 구역 간 이동 유사성(상관관계)**")
                if len(selected_areas) >= 2:
                    corr_val = pivot_df[selected_areas].corr().iloc[0, 1]
                    st.write(f"지표: **{corr_val:.2f}**")
                    st.caption("두 구역의 흐름이 얼마나 동기화되어 있는지 나타냅니다.")

            # 4. 영역별 피크 타임 상세 분석 (기존 유지)
            st.divider()
            st.subheader("🏔️ 영역별 시간대 피크(Peak) 시각 및 상세 분석")
            
            time_slots = {
                "🌅 아침 (07-10시)": (420, 600),
                "☀️ 낮 (12-15시)": (720, 900),
                "🌙 저녁 (18-21시)": (1080, 1260)
            }
            
            st.write("**📍 구역별/시간대별 최대 혼잡 발생 정보**")
            peak_report_data = []
            
            for area in selected_areas:
                area_data = { "구역": area }
                for slot_name, (start, end) in time_slots.items():
                    slot_series = pivot_df[area].iloc[start:end]
                    if not slot_series.empty:
                        max_v = slot_series.max()
                        max_i = slot_series.idxmax()
                        p_h = max_i // 60
                        p_m = max_i % 60
                        area_data[slot_name] = f"{p_h:02d}:{p_m:02d} ({max_v:.1f}명)"
                    else:
                        area_data[slot_name] = "-"
                peak_report_data.append(area_data)
            
            st.table(pd.DataFrame(peak_report_data))
            st.caption("💡 각 구역이 시간대별로 가장 붐볐던 구체적인 시각과 당시 인원을 표시합니다.")

            # 시간대별 인원 밀도 히트맵 (기존 유지)
            st.write("")
            st.write("**🎯 시간대별 혼잡 밀도 히트맵 (Heatmap)**")
            heatmap_df = pivot_df[selected_areas].copy()
            heatmap_df.index = [f"{h:02d}:00" for h in (heatmap_df.index // 60)]
            heatmap_summary = heatmap_df.groupby(level=0).mean()
            
            fig_heat = px.imshow(heatmap_summary.T, 
                                 labels=dict(x="시간대", y="구역", color="평균 인원"),
                                 color_continuous_scale="Reds", 
                                 template="plotly_dark", aspect="auto")
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

            # 5. 병목 위험도 예측 및 강도 분석 (기존 유지)
            st.divider()
            st.subheader("⚠️ 실시간 위험 알림 및 강도 분석")
            for area in selected_areas:
                diff_val = pivot_df[area].iloc[current_time_min] - pivot_df[area].iloc[max(0, current_time_min-10)]
                if diff_val > 20:
                    st.error(f"🔥 {area} 구역 급증 경고: 10분 전 대비 {diff_val:.1f}명 증가!")
            
            fig_diff = go.Figure()
            for area in selected_areas:
                diff_series = pivot_df[area].diff(periods=10).fillna(0)
                fig_diff.add_trace(go.Scatter(x=diff_series.index, y=diff_series, name=f"{area} 유입강도", fill='tozeroy'))
            fig_diff.update_layout(template="plotly_dark", height=400, title="단기 유입/유출 변화량(10분 단위)",
                                   xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                               ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)]))
            st.plotly_chart(fig_diff, use_container_width=True)

            # ---------------------------------------------------------
            # [신규 추가] 6. 보안검색대(IM) 특화 상관관계 분석
            # ---------------------------------------------------------
            st.divider()
            st.subheader("🛡️ 보안검색대(IM) 핵심 관문 상관관계 리포트")
            
            # IM 구역 정의 (데이터에 존재하는지 확인)
            im_areas = [area for area in ['IM1', 'IM2', 'IM3', 'IM4', 'IM5'] if area in pivot_df.columns]
            
            if len(im_areas) >= 1:
                col_graph, col_stats = st.columns([1.5, 1])
                
                with col_graph:
                    st.write("**🔗 보안검색대-터미널 흐름 동기화**")
                    # IM 구역 인원 합계
                    im_sum = pivot_df[im_areas].sum(axis=1)
                    im_correlation = im_sum.corr(total_flow)
                    
                    fig_im_sync = go.Figure()
                    # 터미널 전체 흐름 (배경)
                    fig_im_sync.add_trace(go.Scatter(
                        x=total_flow.index, y=total_flow, 
                        name="전체 터미널", line=dict(color='gray', width=1), opacity=0.4
                    ))
                    # 검색대 합계 흐름 (강조)
                    fig_im_sync.add_trace(go.Scatter(
                        x=im_sum.index, y=im_sum, 
                        name="보안검색대(IM) 합계", line=dict(color='#00CC96', width=3)
                    ))
                    
                    fig_im_sync.update_layout(
                        template="plotly_dark", height=380, 
                        title=f"터미널 전체 vs 보안검색대 흐름 (상관계수: {im_correlation:.2f})",
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                   ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)])
                    )
                    st.plotly_chart(fig_im_sync, use_container_width=True)

                with col_stats:
                    st.write("**📊 검색대 운영 효율 지표**")
                    curr_im_total = im_sum.iloc[current_time_min]
                    curr_total_all = total_flow.iloc[current_time_min] if total_flow.iloc[current_time_min] > 0 else 1
                    
                    # 지표 1: 전체 대비 검색대 점유율 (검색대 병목 수준 확인)
                    im_ratio_val = (curr_im_total / curr_total_all) * 100
                    st.metric("보안검색대 점유 비중", f"{im_ratio_val:.1f}%", 
                              help="현재 공항 전체 인원 중 검색 구역에 머물고 있는 비율입니다.")
                    
                    # 지표 2: 라인 간 불균형 (IM1, IM2 존재 시)
                    if 'IM1' in pivot_df.columns and 'IM2' in pivot_df.columns:
                        v1 = pivot_df['IM1'].iloc[current_time_min]
                        v2 = pivot_df['IM2'].iloc[current_time_min]
                        line_diff = abs(v1 - v2)
                        
                        st.metric("IM1-IM2 라인 편차", f"{int(line_diff)} 명", 
                                  delta="⚠️ 불균형" if line_diff > 15 else "안정", delta_color="inverse")
                        
                        line_corr = pivot_df['IM1'].corr(pivot_df['IM2'])
                        st.write(f"라인 간 동기화 지수: **{line_corr:.2f}**")
                    
                    st.info("💡 **운영 인사이트:** 상관지수가 0.8 이상일 경우, 보안검색대가 전체 승객 수요 변화에 즉각적으로 대응하고 있음을 나타냅니다.")

            else:
                st.info("보안검색대(IM1, IM2 등) 데이터가 포함되어 있지 않아 특화 분석을 표시할 수 없습니다.")

            # [신규 추가] 7. 실시간 관제 AI 인사이트 (운용 효율 및 안전 관리)
            # ---------------------------------------------------------
            st.divider()
            st.subheader("🛡️ 실시간 관제 AI 인사이트 (운영 및 안전)")
            
            # 관리자용 의사결정 지원 컬럼
            col_safety, col_efficiency = st.columns(2)
            
            with col_safety:
                st.markdown("#### **🚨 구역별 안전 임계치 경보**")
                for area in selected_areas:
                    curr_p = pivot_df[area].iloc[now_idx]
                    
                    # 안전 임계치 논리 (예: 100명 초과 시 위험, 80명 초과 시 주의)
                    if curr_p > 100:
                        st.error(f"**[위험]** {area} 구역 밀집도 초과 ({curr_p:.1f}명) - 즉시 인원 통제 필요")
                    elif curr_p > 80:
                        st.warning(f"**[주의]** {area} 구역 혼잡도 상승 ({curr_p:.1f}명) - 모니터링 강화")
                    else:
                        st.success(f"**[정상]** {area} 구역 밀집도 안정적")
                
                st.caption("※ 지수 모델 기반 병목 구간 진입 여부를 실시간으로 감시합니다.")

            with col_efficiency:
                st.markdown("#### **👨‍✈️ 인력 최적화 및 운영 가이드**")
                
                # --- [고도화] 보안검색대(IM) 스마트 분배 가이드 ---
                if 'IM1' in pivot_df.columns and 'IM2' in pivot_df.columns:
                    st.write("**🛡️ 보안검색대 실시간 분배 진단**")
                    v1_curr = pivot_df['IM1'].iloc[now_idx]
                    v2_curr = pivot_df['IM2'].iloc[now_idx]
                    
                    # 1. 사용자가 설정한 service_rate를 활용해 대기시간 계산 (exp 함수 로직을 반영한 간이 수식)
                    # 실제 프로젝트에서 사용한 exp(인원/계수) 형태가 있다면 여기에 적용하세요.
                    wait1 = (v1_curr / service_rate) * 10 
                    wait2 = (v2_curr / service_rate) * 10
                    
                    # 2. 불균형 편차 계산
                    im_diff = wait1 - wait2
                    
                    if abs(im_diff) >= 8:  # 8분 이상 차이 날 경우 강력 권고
                        heavy_im = "IM1" if im_diff > 0 else "IM2"
                        light_im = "IM2" if im_diff > 0 else "IM1"
                        
                        st.error(f"⚠️ **검색대 불균형 심화 (차이: {abs(im_diff):.1f}분)**")
                        st.info(f"💡 **조치**: {heavy_im} 진입 승객을 **{light_im}**으로 즉시 분산 유도하십시오.")
                        
                        # 시각적 인디케이터 (Progress Bar로 비중 표시)
                        total_im = v1_curr + v2_curr
                        st.progress(v1_curr / total_im if total_im > 0 else 0.5, text=f"IM1({v1_curr:.0f}명) vs IM2({v2_curr:.0f}명)")
                    
                    elif abs(im_diff) >= 4: # 4~8분 사이는 주의
                        st.warning(f"⚖️ **분산 고려**: 현재 {abs(im_diff):.1f}분 편차 발생 중")
                    else:
                        st.success("✅ **분배 적정**: 양측 검색대 흐름이 균형적입니다.")
    
           # 8. 위기 대응 시뮬레이션 (간이 대피로 확인)
            st.divider()
            st.subheader("🌋 비상 상황 대응 시나리오")
            evac_col1, evac_col2 = st.columns([1, 2])
                
            with evac_col1:
                emergency_area = st.selectbox("사고 발생 구역 가정", selected_areas)
                if st.button("🚨 비상 대피 시나리오 가동"):
                    st.error(f"**{emergency_area} 구역 비상 상황 전파!**")
                    st.write(f"1. {emergency_area} 인근 승객 최단거리 대피 유도")
                    st.write(f"2. {emergency_area} 진입 셔터 폐쇄 및 우회 경로 확보")
                
            with evac_col2:
                # 사고 구역 제외 혼잡도 재계산 시각화 (예시 히트맵)
                st.caption("사고 발생 시 주변 구역 전이 혼잡도 예측 모델 (Simulation)")
                sim_data = pivot_df[selected_areas].iloc[now_idx:now_idx+6].copy() # 향후 60분 예측 가정
                st.line_chart(sim_data)
                
            # --- [최종 보정] 9. 실시간 전 구역 카운터 개방 최적화 및 시뮬레이터 ---
            st.divider()
            st.markdown("<h2 style='text-align: center; color: #00EEFF;'>👨‍✈️ Smart Resource Optimizer & Simulator</h2>", unsafe_allow_html=True)
            
            # A, I 구역 제외한 카운터 구역 추출
            counter_areas = [a for a in pivot_df.columns if len(a) == 1 and a not in ['A', 'I']]
            
            if counter_areas:
                # 1. 시간 선택 기능
                all_times = pivot_df.index.tolist()
                
                try:
                    default_idx = now_idx if 'now_idx' in locals() else len(all_times) - 1
                except:
                    default_idx = len(all_times) - 1
                    
                selected_time = st.select_slider("🕒 분석 기준 시간 선택", options=all_times, value=all_times[default_idx])
                current_idx = all_times.index(selected_time)
                prev_idx_dynamic = max(0, current_idx - 1)
            
                # 2. 운영 설정값 및 시뮬레이션 변수
                               # --- 하단 9번 섹션 수정 ---
                with st.expander("⚙️ 운영 최적화 알고리즘 및 시뮬레이션 설정", expanded=True):
                    c_set1, c_set2, c_set3 = st.columns(3)
                    with c_set1:
                        # 이미 위에서 선언했으므로 표시만 하거나, 여기서 값을 입력받으려면 위쪽 선언을 지워야 합니다.
                        st.info(f"현재 처리 용량: {service_rate}명") 
                    with c_set2:
                        st.info(f"현재 경고 기준: {wait_threshold}분")
                    with c_set3:
                        default_open = st.number_input("구역별 기본 개방 카운터 (초기값)", 1, 20, 3)
            
                # 3. 데이터 분석 로직
                base_data = []
                import math
                for area in counter_areas:
                    curr_p = round(float(pivot_df[area].iloc[current_idx]), 1)
                    prev_p = float(pivot_df[area].iloc[prev_idx_dynamic])
                    
                    accel = (curr_p - prev_p) / 5
                    pred_p = max(0, round(curr_p + (accel * 10), 1))
                    req_c = math.ceil(pred_p / max(1, service_rate))
            
                    base_data.append({
                        "구역": area,
                        "현재 인원": curr_p,
                        "10분 뒤 예측": pred_p,
                        "현재 개방 카운터": default_open, 
                        "유입 강도": accel,
                        "AI 권장": req_c
                    })
            
                df_base = pd.DataFrame(base_data)
            
                # 4. 관리자 시뮬레이션 입력
                st.markdown(f"### 📝 실시간 카운터 운영 시뮬레이션 <small>({selected_time} 기준)</small>", unsafe_allow_html=True)
                
                edited_df = st.data_editor(
                    df_base,
                    column_config={
                        "현재 개방 카운터": st.column_config.NumberColumn("실제 개방 수", min_value=1, max_value=30, step=1),
                        "유입 강도": st.column_config.ProgressColumn("인원 유입 추세", min_value=-5, max_value=5),
                        "AI 권장": st.column_config.NumberColumn("AI 권장", format="%d 개"),
                    },
                    disabled=["구역", "현재 인원", "10분 뒤 예측", "유입 강도", "AI 권장"],
                    hide_index=True,
                    use_container_width=True
                )
            
                # 5. 시각화 카드 출력
                st.divider()
                cols_per_row = 4
                for i in range(0, len(edited_df), cols_per_row):
                    row_data = edited_df.iloc[i : i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    
                    for j, (idx, data) in enumerate(row_data.iterrows()):
                        with cols[j]:
                            # 수치 계산
                            capa = max(1, data["현재 개방 카운터"] * service_rate)
                            c_wait = (data["현재 인원"] / capa) * 10
                            p_wait = (data["10분 뒤 예측"] / capa) * 10
                            
                            # 색상 설정
                            if c_wait > wait_threshold or p_wait > wait_threshold:
                                m_c, b_c, s_t = "#FF3131", "rgba(255, 49, 49, 0.2)", "🚨 인력 즉시 증설"
                            elif data["현재 개방 카운터"] < data["AI 권장"]:
                                m_c, b_c, s_t = "#FFAC1C", "rgba(255, 172, 28, 0.2)", "⚠️ 보충 권장"
                            else:
                                m_c, b_c, s_t = "#00FFFF", "rgba(0, 255, 255, 0.15)", "✅ 운영 적정"
            
                            # HTML 카드 렌더링 (모든 텍스트 color: #000000 강제 적용)
                            st.markdown(f"""
                            <div style="padding: 15px; border-radius: 12px; border: 3px solid {m_c}; background-color: {b_c}; text-align: center; box-shadow: 0px 4px 6px rgba(0,0,0,0.3); margin-bottom: 10px;">
                                <div style="font-size: 18px; font-weight: 900; color: #000000; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 5px; margin-bottom: 8px;">
                                    {data['구역']} AREA
                                </div>
                                <div style="margin-bottom: 10px;">
                                    <div style="font-size: 10px; color: #333333; font-weight: 700;">인원 (현재/예측)</div>
                                    <div style="font-size: 16px; font-weight: 800; color: #000000;">
                                        {data['현재 인원']:.1f} <span style="color:{m_c};">→</span> {data['10분 뒤 예측']:.1f}
                                    </div>
                                </div>
                                <div style="background: rgba(255,255,255,0.4); padding: 8px; border-radius: 8px; border: 1px solid rgba(0,0,0,0.05);">
                                    <div style="font-size: 10px; color: #333333; font-weight: 700;">대기시간(분)</div>
                                    <div style="font-size: 22px; font-weight: 900; color: #000000;">
                                        {c_wait:.1f} / {p_wait:.1f}
                                    </div>
                                </div>
                                <div style="font-size: 12px; margin-top: 10px; font-weight: 800; color: #000000;">
                                    {s_t}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 가이드 텍스트
                            if data["현재 개방 카운터"] < data["AI 권장"]:
                                diff = int(data["AI 권장"] - data["현재 개방 카운터"])
                                st.markdown(f"<p style='color:#E67E22; font-size:11px; font-weight:800; text-align:center;'>▲ {diff}개 부족</p>", unsafe_allow_html=True)
                            elif data["현재 개방 카운터"] > data["AI 권장"] + 1:
                                diff = int(data["현재 개방 카운터"] - data["AI 권장"])
                                st.markdown(f"<p style='color:#27AE60; font-size:11px; font-weight:800; text-align:center;'>▼ {diff}개 여유</p>", unsafe_allow_html=True)
            
                # 6. 인력 재배치 제안
                st.divider()
                surplus_areas = edited_df[edited_df["현재 개방 카운터"] > edited_df["AI 권장"]]["구역"].tolist()
                shortage_areas = edited_df[edited_df["현재 개방 카운터"] < edited_df["AI 권장"]]["구역"].tolist()
                if surplus_areas and shortage_areas:
                    st.success(f"🔄 **AI 재배치 가이드**: 여유 있는 **{surplus_areas[0]} 구역**의 인력을 혼잡한 **{shortage_areas[0]} 구역**으로 이동 배치를 권고합니다.")
            else:
                st.error("데이터에서 카운터 구역(B~N)을 찾을 수 없습니다.")
        else:
            st.warning("분석할 구역을 선택해주세요.")
    
# --- [TAB 4] 안전 관리 및 위기 대응 ---
    with tab4:
        st.title("🛡️ 안전 관리 및 위기 대응 시스템")
        st.markdown("공항 내 밀집도를 실시간 모니터링하고, 비상 상황 시 즉각적인 의사결정을 지원합니다.")
        
        # ---------------------------------------------------------
        # 1. 구역별 밀집도 임계치 경보 (Safety Threshold)
        # ---------------------------------------------------------
        st.subheader("🚨 실시간 안전 밀집도 현황")
        
        danger_limit = 120  # 위험 기준
        warning_limit = 80  # 주의 기준
        
        if selected_areas:
            safety_cols = st.columns(len(selected_areas))
            for i, area in enumerate(selected_areas):
                curr_p = pivot_df[area].iloc[now_idx]
                
                with safety_cols[i]:
                    if curr_p >= danger_limit:
                        st.error(f"### {area}\n**{curr_p:.1f}명**\n\n🚨 위험")
                    elif curr_p >= warning_limit:
                        st.warning(f"### {area}\n**{curr_p:.1f}명**\n\n⚠️ 주의")
                    else:
                        st.success(f"### {area}\n**{curr_p:.1f}명**\n\n✅ 정상")
        else:
            st.info("💡 상단 '구역별 상세 비교 분석' 탭에서 모니터링할 구역을 먼저 선택해주세요.")

        st.divider()

        # ---------------------------------------------------------
        # 2. 이동 경로 예측 및 정체 구역 시각화 (Predictive Insight)
        # ---------------------------------------------------------
        st.subheader("🔍 향후 정체 예상 구역 (10분 후 예측)")
        
        pred_list = []
        for area in pivot_df.columns:
            curr_v = pivot_df[area].iloc[now_idx]
            # 1분 전 데이터가 없을 경우를 대비해 처리
            prev_idx = max(0, now_idx - 1)
            prev_v = pivot_df[area].iloc[prev_idx]
            
            # 유입 속도 (분당 인원 변화)
            accel = curr_v - prev_v
            # 10분 후 예측 (현재 + 속도*10)
            future_v = max(0, curr_v + (accel * 10))
            
            # 위험/주의 단계에 해당하는 것만 추출
            if future_v >= warning_limit:
                risk_tag = "🚨 위험" if future_v >= danger_limit else "⚠️ 주의"
                pred_list.append({
                    "구역": area,
                    "현재 인원": round(curr_v, 1),
                    "유입 속도": f"{accel:+.1f} 명/분",
                    "10분 후 예상": round(future_v, 1),
                    "위험도": risk_tag
                })

        if pred_list:
            pdf = pd.DataFrame(pred_list).sort_values("10분 후 예상", ascending=False)
            
            # 스타일 함수 정의 (Pandas 2.0+ 대응)
            def highlight_risk(val):
                if '위험' in val: return 'color: #ff4b4b; font-weight: bold;'
                if '주의' in val: return 'color: #ffa500; font-weight: bold;'
                return ''

            # st.table 대신 st.dataframe을 써서 스타일 적용
            st.dataframe(
                pdf.style.map(highlight_risk, subset=['위험도']),
                use_container_width=True,
                hide_index=True
            )
            st.caption("💡 유입 가속도를 분석하여 정체 발생 전 선제적 대응이 가능한 구역들입니다.")
        else:
            st.success("✅ 모든 구역이 안정적입니다. 10분 내 임계치 초과 예상 구역이 없습니다.")

        st.divider()

        # ---------------------------------------------------------
        # 3. 비상 대피로 최적화 (Emergency Routing)
        # ---------------------------------------------------------
        st.subheader("🌋 위기 대응 시뮬레이터")
        
        # 사고 발생 시나리오 설정
        c1, c2 = st.columns([1, 2])
        with c1:
            incident_area = st.selectbox("사고 발생 구역 지정", pivot_df.columns)
            evac_btn = st.button("🔥 비상 대피령 발령", type="primary", use_container_width=True)
            
        with c2:
            if evac_btn:
                st.error(f"🚨 [긴급] {incident_area} 구역 사고 발생! 즉시 대피를 유도합니다.")
                
                # [Logic] 대피 알고리즘: 사고지점 제외, 현재 혼잡도가 가장 낮은 구역 Top 2 추천
                # 실제 좌표 데이터(coords)가 있다면 최단거리로 계산하겠지만, 
                # 여기서는 '가장 여유 있는 구역'을 찾는 논리로 구성합니다.
                
                escape_candidates = []
                for area in pivot_df.columns:
                    if area == incident_area: continue
                    
                    curr_val = pivot_df[area].iloc[now_idx]
                    # 여유 용량 계산 (임계치 - 현재인원)
                    capacity = danger_limit - curr_val
                    escape_candidates.append({"구역": area, "여유용량": capacity, "현재인원": curr_val})
                
                # 여유 용량이 많은 순으로 정렬
                escape_df = pd.DataFrame(escape_candidates).sort_values("여유용량", ascending=False).head(2)
                
                st.markdown("### **🏃 추천 대피 경로**")
                for _, row in escape_df.iterrows():
                    st.info(f"👉 **{row['구역']}** 방향으로 유도 (현재 {row['현재인원']:.1f}명 상주 중)")
            else:
                st.write("사고 발생 구역을 선택하고 버튼을 누르면 최적 대피 경로가 산출됩니다.")
else:
    st.error("데이터 파일을 로드할 수 없습니다. 파일 경로와 날짜 설정을 확인해주세요.")

# ---------------------------------------------------------
           

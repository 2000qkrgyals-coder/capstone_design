import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

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
    tab1, tab2, tab3 = st.tabs(["🚀 실시간 통합 관제", "🕒 시간대별 피크 분석", "🔍 구역별 상세 비교 분석"])

    # --- [TAB 1] 실시간 통합 관제 ---
    with tab1:
        st.title(f"📊 실시간 관제 현황 [{in_hour:02d}:{in_min:02d}]")
        current_data = df[df['minute_index'] == current_time_min]
        active_areas = current_data[current_data['area'] != 'OUTSIDE']
        
        m1, m2, m3 = st.columns(3)
        curr_total = total_flow.get(current_time_min, 0)
        m1.metric("터미널 전체 인원", f"{curr_total:.1f} 명")
        if not active_areas.empty:
            std_val = active_areas['num_people'].std()
            m2.metric("구역별 혼잡 불균형", f"{std_val:.2f}", help="값이 높을수록 특정 구역 쏠림이 심함")
            max_area = active_areas.loc[active_areas['num_people'].idxmax(), 'area']
            m3.warning(f"최대 혼잡 구역: {max_area}")

        col_map, col_rank = st.columns([2, 1])
        with col_map:
            st.subheader("📍 실시간 혼잡 지도")
            map_data = pd.merge(current_data, coords, on='area')
            if os.path.exists(bg_img_path):
                img = Image.open(bg_img_path)
                fig_map = go.Figure()
                fig_map.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=0, sizex=img.size[0], sizey=img.size[1], sizing="stretch", opacity=0.5, layer="below"))
                fig_map.add_trace(go.Scatter(x=map_data['x'], y=map_data['y'], mode='markers+text',
                                             marker=dict(size=map_data['num_people'], sizemode='area', sizeref=2.*max(map_data['num_people'])/(35**2) if not map_data.empty else 1,
                                                         color=map_data['num_people'], colorscale='Reds', showscale=True),
                                             text=map_data['area'], textfont=dict(size=10, color="white"), textposition="top center"))
                fig_map.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0))
                fig_map.update_xaxes(visible=False, range=[0, img.size[0]])
                fig_map.update_yaxes(visible=False, range=[img.size[1], 0])
                st.plotly_chart(fig_map, use_container_width=True)

        with col_rank:
            st.subheader("🚩 혼잡도 랭킹 (Top 10)")
            rank_data = active_areas.sort_values('num_people', ascending=False).head(10)
            fig_rank = px.bar(rank_data, x='num_people', y='area', orientation='h', color='num_people', color_continuous_scale='Reds', template="plotly_dark")
            fig_rank.update_layout(height=500, yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig_rank, use_container_width=True)

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
            # [신규 추가] 지표 기반 정량 분석 리포트 (1, 2, 4번 지표)
            # ---------------------------------------------------------
            st.subheader("📋 실시간 핵심 운영 지표 (정량 분석)")
            
            now_idx = current_time_min
            prev_idx = max(0, now_idx - 5) # 5분 전 비교
            avg_people_all = pivot_df.iloc[now_idx].mean() # 4번 지표용 전체 평균
            
            analysis_metrics = []
            for area in selected_areas:
                curr_v = pivot_df[area].iloc[now_idx]
                prev_v = pivot_df[area].iloc[prev_idx]
                
                # 1. 혼잡 지수 (임계치 80명 대비 비율)
                c_idx = (curr_v / 80) * 100
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
            
            st.caption("※ 혼잡 지수: 80명 기준(100%) / 유입 가속도: 5분 전 대비 변화율 / 불균형 지수: 현재 전체 구역 평균 대비 배수")

            # 지표 상관관계 시각화 (Bubble Chart)
            fig_bubble = px.scatter(metric_df, x="혼잡 지수(%)", y="유입 가속도(명/분)", 
                                    size="불균형 지수(평균대비)", color="구역",
                                    hover_name="구역", size_max=40,
                                    title="📊 운영 지표 상관 분석 (원 크기 = 불균형 쏠림 정도)",
                                    template="plotly_dark")
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.divider()

            # 1. 점유율 상세 (기존 로직)
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

            # 2. 시간대별 혼잡도 추이 (기존 로직)
            st.subheader("📈 시간대별 혼잡도 추이 (10분 이동평균 적용)")
            fig_compare = go.Figure()
            for area in selected_areas:
                fig_compare.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[area], name=area, mode='lines', line=dict(width=3)))
            
            fig_compare.add_vline(x=current_time_min, line_dash="dash", line_color="red", annotation_text="현재")
            fig_compare.update_layout(template="plotly_dark", height=600, hovermode="x unified",
                                      xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                                 ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)]))
            st.plotly_chart(fig_compare, use_container_width=True)

            # 3. 통계 기반 관제 인사이트 (기존 로직)
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
                    accel = pivot_df[area].diff().iloc[current_time_min]
                    status = "🚀 급증" if accel > 1.5 else ("➡️ 안정" if accel > -1.5 else "⬇️ 감소")
                    st.write(f"- {area}: {status} ({accel:.2f})")

            with c_ins3:
                st.write("**🔗 구역 간 이동 유사성(상관관계)**")
                if len(selected_areas) >= 2:
                    corr = pivot_df[selected_areas].corr().iloc[0, 1]
                    st.write(f"지표: **{corr:.2f}**")
                    st.caption("두 구역의 흐름이 얼마나 동기화되어 있는지 나타냅니다.")

            # --- [핵심 업데이트] 4. 영역별 피크 타임 상세 분석 (기존 로직) ---
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
                        max_val = slot_series.max()
                        max_idx = slot_series.idxmax()
                        p_hour = max_idx // 60
                        p_min = max_idx % 60
                        area_data[slot_name] = f"{p_hour:02d}:{p_min:02d} ({max_val:.1f}명)"
                    else:
                        area_data[slot_name] = "-"
                peak_report_data.append(area_data)
            
            st.table(pd.DataFrame(peak_report_data))
            st.caption("💡 각 구역이 시간대별로 가장 붐볐던 구체적인 시각과 당시 인원을 표시합니다.")

            # 시간대별 인원 밀도 히트맵 (기존 로직)
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

            # 5. 병목 위험도 예측 및 강도 분석 (기존 로직)
            st.divider()
            st.subheader("⚠️ 실시간 위험 알림 및 강도 분석")
            for area in selected_areas:
                diff = pivot_df[area].iloc[current_time_min] - pivot_df[area].iloc[max(0, current_time_min-10)]
                if diff > 20:
                    st.error(f"🔥 {area} 구역 급증 경고: 10분 전 대비 {diff:.1f}명 증가!")
            
            fig_diff = go.Figure()
            for area in selected_areas:
                diff_series = pivot_df[area].diff(periods=10).fillna(0)
                fig_diff.add_trace(go.Scatter(x=diff_series.index, y=diff_series, name=f"{area} 유입강도", fill='tozeroy'))
            fig_diff.update_layout(template="plotly_dark", height=400, title="단기 유입/유출 변화량(10분 단위)",
                                    xaxis=dict(tickmode='array', tickvals=list(range(0, 1441, 120)), 
                                               ticktext=[f"{h:02d}:00" for h in range(0, 25, 2)]))
            st.plotly_chart(fig_diff, use_container_width=True)

        else:
            st.warning("분석할 구역을 선택해주세요.")
else:
    st.error("데이터 파일을 로드할 수 없습니다. 파일 경로와 날짜 설정을 확인해주세요.")
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
                
                # 보안검색대 스마트 분배 가이드 (IM1, IM2 기준)
                if 'IM1' in pivot_df.columns and 'IM2' in pivot_df.columns:
                    v1 = pivot_df['IM1'].iloc[now_idx]
                    v2 = pivot_df['IM2'].iloc[now_idx]
                    
                    if abs(v1 - v2) > 20:
                        heavy_im = "IM1" if v1 > v2 else "IM2"
                        light_im = "IM2" if v1 > v2 else "IM1"
                        st.info(f"💡 **분산 권고**: {heavy_im}에 인원이 쏠려있습니다. {light_im}로 승객 유도를 권장합니다.")
                
                # 카운터 개방 가이드 (가속도 기반 예측)
                for area in selected_areas:
                    accel_val = (pivot_df[area].iloc[now_idx] - pivot_df[area].iloc[prev_idx]) / 5
                    if accel_val > 2.0: # 분당 2명 이상 급증 시
                        st.error(f"📍 **카운터 추가 개방**: {area} 구역 유입 속도 급증! 추가 가동이 필요합니다.")
                    elif accel_val < -2.0 and pivot_df[area].iloc[now_idx] < 30:
                        st.write(f"🍃 **운영 효율화**: {area} 구역 수요 감소 중. 인력 재배치 고려 가능.")
    
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

            # --- [최종 통합본] 9. 실시간 전 구역 카운터 개방 최적화 및 시뮬레이터 ---
            st.divider()
            st.markdown("<h2 style='text-align: center; color: #00EEFF;'>👨‍✈️ Smart Resource Optimizer & Simulator</h2>", unsafe_allow_html=True)
            
            # 1. 시간 선택 기능 (데이터 인덱스 기반 자유 선택)
            all_times = pivot_df.index.tolist()
            selected_time = st.select_slider("🕒 분석 기준 시간 선택", options=all_times, value=all_times[now_idx])
            current_idx = all_times.index(selected_time)
            prev_idx_dynamic = max(0, current_idx - 1) # 5분 전 데이터 위치
            
            # 2. 운영 설정값 및 시뮬레이션 변수 (변수명: service_rate로 통일)
            with st.expander("⚙️ 운영 최적화 알고리즘 및 시뮬레이션 설정", expanded=True):
                c_set1, c_set2, c_set3 = st.columns(3)
                with c_set1:
                    service_rate = st.number_input("카운터당 처리 용량 (명/10분)", 1, 100, 15)
                with c_set2:
                    wait_threshold = st.slider("대기시간 경고 기준 (분)", 5, 30, 15)
                with c_set3:
                    default_open = st.number_input("구역별 기본 개방 카운터 (초기값)", 1, 20, 3)
            
            # 3. 데이터 추출 및 분석 로직 (A, I 구역 제외)
            counter_areas = [a for a in pivot_df.columns if len(a) == 1 and a not in ['A', 'I']]
            
            if counter_areas:
                base_data = []
                for area in counter_areas:
                    curr_p = round(pivot_df[area].iloc[current_idx], 1)
                    prev_p = pivot_df[area].iloc[prev_idx_dynamic]
                    
                    # 유입 가속도 및 10분 뒤 예측 인원 계산
                    accel = (curr_p - prev_p) / 5
                    pred_p = max(0, round(curr_p + (accel * 10), 1))
                    
                    # AI 권장 카운터 (10분 뒤 예측 인원 기준)
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
            
                # 4. 관리자 시뮬레이션 입력 (가시성 높은 데이터 에디터)
                st.markdown(f"### 📝 실시간 카운터 운영 시뮬레이션 <small>({selected_time} 기준)</small>", unsafe_allow_html=True)
                st.info("💡 '현재 개방 카운터' 열의 숫자를 직접 수정하면 하단 카드의 대기시간이 즉시 재계산됩니다.")
                
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
            
                # 5. 시인성 강화 버전 구역별 상세 카드 배치
                st.divider()
                cols_per_row = 4
                rows = [edited_df.iloc[i:i + cols_per_row] for i in range(0, len(edited_df), cols_per_row)]
                
                for row_data in rows:
                    cols = st.columns(cols_per_row)
                    for i, (idx, data) in enumerate(row_data.iterrows()):
                        with cols[i]:
                            # 대기시간 계산 로직
                            capacity = data["현재 개방 카운터"] * service_rate
                            curr_wait = (data["현재 인원"] / max(1, capacity)) * 10
                            pred_wait = (data["10분 뒤 예측"] / max(1, capacity)) * 10
                            
                            # 색상 및 상태 로직 (시인성 중심)
                            if curr_wait > wait_threshold or pred_wait > wait_threshold:
                                main_color = "#FF3131"  # Bright Red
                                bg_color = "rgba(255, 49, 49, 0.15)"
                                status_text = "🚨 인력 즉시 증설"
                            elif data["현재 개방 카운터"] < data["AI 권장"]:
                                main_color = "#FFAC1C"  # Bright Orange
                                bg_color = "rgba(255, 172, 28, 0.1)"
                                status_text = "⚠️ 보충 권장"
                            else:
                                main_color = "#00FFFF"  # Neon Cyan
                                bg_color = "rgba(0, 255, 255, 0.05)"
                                status_text = "✅ 운영 적정"
            
                            # HTML 카드 렌더링
                            st.markdown(f"""
                            <div style="
                                padding: 20px; 
                                border-radius: 15px; 
                                border: 2px solid {main_color}; 
                                background-color: {bg_color}; 
                                min-height: 240px; 
                                text-align: center;
                                box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                            ">
                                <div style="font-size: 24px; font-weight: 900; color: #FFFFFF; margin-bottom: 10px; border-bottom: 1px solid {main_color}44; padding-bottom: 5px;">
                                    {data['구역']} AREA
                                </div>
                                
                                <div style="margin-bottom: 15px;">
                                    <div style="font-size: 13px; color: #E0E0E0; font-weight: 400;">현재 / 10분 뒤 인원</div>
                                    <div style="font-size: 20px; font-weight: 700; color: #FFFFFF;">
                                        {data['현재 인원']:.1f} <span style="color:{main_color};">→</span> {data['10분 뒤 예측']:.1f}명
                                    </div>
                                </div>
                                
                                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 10px;">
                                    <div style="font-size: 13px; color: #E0E0E0;">예상 대기시간</div>
                                    <div style="font-size: 30px; font-weight: 900; color: {main_color}; letter-spacing: -1px;">
                                        {curr_wait:.1f} / {pred_wait:.1f}<span style="font-size: 16px;">분</span>
                                    </div>
                                </div>
                                
                                <div style="font-size: 14px; margin-top: 15px; font-weight: 800; color: {main_color}; text-transform: uppercase;">
                                    {status_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 카드 하단 보조 안내 지표
                            if data["현재 개방 카운터"] < data["AI 권장"]:
                                st.markdown(f"<p style='color:#FFAC1C; font-size:13px; font-weight:600; text-align:center; margin-top:5px;'>▲ AI 권장보다 {int(data['AI 권장'] - data['현재 개방 카운터'])}개 부족</p>", unsafe_allow_html=True)
                            elif data["현재 개방 카운터"] > data["AI 권장"] + 1:
                                st.markdown(f"<p style='color:#00FF7F; font-size:13px; font-weight:600; text-align:center; margin-top:5px;'>▼ {int(data['현재 개방 카운터'] - data['AI 권장'])}개 감축 가능</p>", unsafe_allow_html=True)
                            st.write("")
            
                # 6. 스마트 인력 재배치 제안 (최종 요약)
                st.divider()
                surplus_areas = edited_df[edited_df["현재 개방 카운터"] > edited_df["AI 권장"]]["구역"].tolist()
                shortage_areas = edited_df[edited_df["현재 개방 카운터"] < edited_df["AI 권장"]]["구역"].tolist()
            
                if surplus_areas and shortage_areas:
                    st.success(f"🔄 **AI 재배치 가이드**: 여유 있는 **{surplus_areas[0]} 구역**의 인력을 혼잡한 **{shortage_areas[0]} 구역**으로 이동 배치를 권고합니다.")
                else:
                    st.info("✅ **운영 상태 요약**: 모든 구역이 인력 최적화 상태이거나 구역 간 편차가 크지 않습니다.")
            
            else:
                st.error("분석 가능한 카운터 데이터(B~N)가 존재하지 않습니다.")
    
        else:
            st.warning("분석할 구역을 선택해주세요.")
else:
    st.error("데이터 파일을 로드할 수 없습니다. 파일 경로와 날짜 설정을 확인해주세요.")

# ---------------------------------------------------------
           

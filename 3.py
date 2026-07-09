import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal

# --- [설정] 기본 경로 설정 ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"        
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"          

st.set_page_config(page_title="인천공항 T2 3층 데이터 분석 센터", layout="wide")

# --- 인력 배치 로직 함수 ---
def calculate_staffing(people_count):
    # 인원당 5명 기준 창구 오픈 권고 (최대 40개)
    open_counters = min(40, -(-people_count // 5))  # ceil 연산
    
    # 현장 지원 인력 (80명 초과 시 투입, 최대 3명)
    support_staff = 0
    if people_count > 80:
        support_staff = min(3, (people_count - 80) // 40 + 1)
        
    total_staff = open_counters + support_staff
    return open_counters, support_staff, total_staff

# --- 공통 함수 ---
def index_to_time_str(t_index):
    total_seconds = int(t_index) * 10
    hours, minutes = total_seconds // 3600, (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}:{total_seconds % 60:02d}"

@st.cache_data
def load_data_by_date(selected_date_str):
    area_df = pd.read_csv(AREA_FILE_PATH)
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None: bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)
    try:
        counts_df = pd.read_csv(f"area_count_time_full_{selected_date_str}.csv")
    except FileNotFoundError:
        return area_df, {}, [], bg_img, False
    
    time_grouped_data = {}
    for t_index, group in counts_df.groupby('time_index'):
        filtered = group[group['area'] != 'Outside']
        time_grouped_data[t_index] = {'counts': dict(zip(filtered['area'], filtered['num_people']))}
    return area_df, time_grouped_data, sorted(list(time_grouped_data.keys())), bg_img, True

def get_daily_peaks(df_trend):
    peaks = {}
    # 레이블을 더 전문적인 용어로 변경
    ranges = [
        ("1차 피크", "05:00", "09:00"),
        ("2차 피크", "09:00", "17:00"),
        ("3차 피크", "17:00", "21:00")
    ]
    
    for label, start, end in ranges:
        subset = df_trend.between_time(start, end)
        if not subset.empty:
            max_val = subset['이동평균'].max()
            max_time = subset['이동평균'].idxmax()
            peaks[label] = (max_time, max_val)
    return peaks

def generate_density_heatmap(area_df, current_counts, img_shape):
    height, width, _ = img_shape
    heatmap_grid = np.zeros((height, width), dtype=np.float32)
    np.random.seed(42)
    
    for _, row in area_df.iterrows():
        people_cnt = current_counts.get(row['area_name'], 0)
        if people_cnt > 0:
            cX = int((row['x1'] + row['x2'] + row['x3'] + row['x4']) / 4)
            cY = int((row['y1'] + row['y2'] + row['y3'] + row['y4']) / 4)
            num_particles = int(people_cnt * 4)
            rand_x = np.random.normal(cX, 100, num_particles).astype(np.int32)
            rand_y = np.random.normal(cY, 50, num_particles).astype(np.int32)
            valid = (rand_x >= 0) & (rand_x < width) & (rand_y >= 0) & (rand_y < height)
            for x, y in zip(rand_x[valid], rand_y[valid]): heatmap_grid[y, x] += 1.0

    if heatmap_grid.max() > 0:
        heatmap_smooth = cv2.GaussianBlur(heatmap_grid, (175, 175), 0)
        heatmap_norm = (heatmap_smooth / heatmap_smooth.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        _, alpha = cv2.threshold(heatmap_norm, 20, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(heatmap_color, heatmap_color, mask=alpha)
    return np.zeros((height, width, 3), dtype=np.uint8)

@st.fragment
def render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD):
    time_options = [int(t) for t in past_unique_times]
    idx_to_label = {t: index_to_time_str(t) for t in time_options}

    st.sidebar.subheader("⚙️ 분석 설정")
    # 1분(6개) ~ 10분(60개) 단위로 조절 가능하도록 설정
    window_size = st.sidebar.select_slider(
        "이동평균 윈도우 크기 (분)",
        options=[1, 3, 5, 10],
        value=5,
        help="데이터의 노이즈를 제거하고 추세를 파악하기 위한 평균 구간을 설정합니다."
    )
    
    # 1. 시간 선택
    selected_t_index = st.select_slider("🕒 조회 시간 선택", options=time_options, format_func=lambda x: idx_to_label[x])
    current_counts = past_time_data[selected_t_index]['counts']
    excluded = ["GH", "IM1", "IM2"]
    filtered_counts = {k: v for k, v in current_counts.items() if k not in excluded}
    
    # [KPI 계산]
    total_people = sum(filtered_counts.values())
    urgent_areas = {k: v for k, v in filtered_counts.items() if v >= 80}
    max_area = max(filtered_counts, key=filtered_counts.get) if filtered_counts else "없음"
    norm_ratio = (1 - (len(urgent_areas) / len(filtered_counts))) * 100 if filtered_counts else 100

    # 2. [개선] 상단 KPI 카드 영역
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 체류 여객", f"{total_people:,} 명")
    col2.metric("혼잡 구역", f"{len(urgent_areas)} 곳", delta="주의" if urgent_areas else None, delta_color="inverse")
    col3.metric("최대 밀집 구역", max_area)
    col4.metric("운영 정상도", f"{norm_ratio:.1f}%")

    st.divider()

    # [분석 요약 전, 데이터 계산 로직 추가]
    # 모든 시간대별 인원을 합산하여 가장 인원이 많은 시간대를 찾습니다.
    time_totals = {
        t: sum(past_time_data[t]['counts'].values()) 
        for t in past_time_data
    }
    
    # 인원이 가장 많았던 시간대 index 찾기
    peak_t_index = max(time_totals, key=time_totals.get)
    peak_time = index_to_time_str(peak_t_index) # 시간 문자열 변환
    
    # 가장 혼잡했던 구역 계산
    max_area = max(filtered_counts, key=filtered_counts.get) if filtered_counts else "없음"
    
    # 📝 일일 운영 분석 요약 패널
    st.subheader("📝 일일 운영 분석 요약")
    st.info(f"""
        **{target_date_str} 운영 분석 결과:**
        - **피크 시간대:** 데이터상 가장 인원이 몰렸던 시간은 **{peak_time}**입니다.
        - **최대 혼잡 구역:** 금일 가장 혼잡도가 높았던 구역은 **{max_area}**입니다.
    """)

    # 6. [신규] 지능형 운영 제언 패널 (Actionable Insight)
    st.divider()
    st.subheader("💡 지능형 운영 제언")
    
    if urgent_areas:
        # 가장 혼잡한 구역 추출
        top_urgent = max(urgent_areas, key=urgent_areas.get)
        msg = f"현재 **{top_urgent}** 구역의 밀집도가 임계치를 초과했습니다. " \
              f"최대 {urgent_areas[top_urgent]}명의 여객이 체류 중입니다. " \
              f"즉시 추가 창구 운영 및 현장 안내 요원 배치를 권고합니다."
        st.warning(msg)
    else:
        st.success("현재 모든 구역이 원활하게 운영 중입니다. 추가 조치가 필요하지 않습니다.")

    # 3. [개선] 레이아웃 분할 (좌: 히트맵, 우: Top 5 혼잡 구역 차트)
    st.divider()
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.subheader("📊 구역별 혼잡도 히트맵")
        heatmap = generate_density_heatmap(area_df, filtered_counts, bg_img.shape)
        blended = cv2.addWeighted(bg_img, 0.6, heatmap, 0.4, 0)
        st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with c2:
        st.subheader("🚨 혼잡 Top 5 구역")
        # 혼잡 순위 정렬
        sorted_areas = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        df_top5 = pd.DataFrame(sorted_areas, columns=["구역", "인원"])
        st.bar_chart(df_top5.set_index("구역"), color="#FF4B4B") # 경고색인 빨간색 활용

   # 5. [전체 인원 흐름 차트]
    st.divider()
    st.subheader("📈 전체 여객 인원 흐름 분석")
    
    # [수정] 이동평균 윈도우 슬라이더
    window_size = st.select_slider(
        "분석 구간 선택 (이동평균 분)", 
        options=[1, 3, 5, 10], 
        value=5,
        help="데이터 노이즈를 제거하기 위한 평균 구간 설정입니다."
    )
    
    # --- 데이터 생성 로직 ---
    time_trend_data = []
    for t in sorted(past_time_data.keys()):
        counts = past_time_data[t]['counts']
        filtered = {k: v for k, v in counts.items() if k not in ["GH", "IM1", "IM2"]}
        time_trend_data.append({"시간": idx_to_label[t], "인원": sum(filtered.values())})
    
    # 1. 원본 시간 데이터를 가져와 변환
    df_trend = pd.DataFrame(time_trend_data)
    
    # 2. errors='coerce'를 사용하여 변환 불가한 값을 NaT로 처리
    df_trend['시간'] = pd.to_datetime(df_trend['시간'], format='%H:%M:%S', errors='coerce')
    
    # 3. 변환에 실패한 행(NaT) 제거
    df_trend = df_trend.dropna(subset=['시간'])
    
    # 4. 인덱스 설정 및 정렬
    df_trend = df_trend.set_index("시간").sort_index()
    
    # 5. 이제 이동평균 계산 진행
    df_trend['이동평균'] = df_trend['인원'].rolling(window=window_size * 6, min_periods=1).mean()
        
    # 4. 차트 출력
    import altair as alt

    # 1. 인덱스를 컬럼으로 꺼내기 (Altair는 컬럼 데이터를 선호함)
    df_plot = df_trend.reset_index()

    # 2. Altair 차트 생성
    chart = alt.Chart(df_plot).mark_area(
        color="#3498db", 
        opacity=0.6
    ).encode(
        # 'hour'를 직접 문자열로 전달하거나, 정수를 사용하여 눈금 개수를 대략적으로 조절합니다.
        x=alt.X('시간:T', axis=alt.Axis(format='%H:%M', tickCount='hour')), 
        y=alt.Y('이동평균:Q', title="체류 인원")
    ).properties(
        height=300
    )

    # 피크 데이터 계산 (수정된 함수 사용)
    peak_data = get_daily_peaks(df_trend)
    
    # 상단 요약 카드 출력
    cols = st.columns(3)
    for i, (label, (t, val)) in enumerate(peak_data.items()):
        cols[i].metric(f"{label} 피크", t.strftime('%H:%M'), f"{int(val)}명")

    # 차트 주석(Annotation) 데이터 생성
    peak_annotations = []
    for label, (t, val) in peak_data.items():
        peak_annotations.append({"시간": t, "인원": val, "라벨": label})
    
    df_peaks = pd.DataFrame(peak_annotations)

    # 차트 레이어 추가 (세로선 및 텍스트)
    rules = alt.Chart(df_peaks).mark_rule(color='#e74c3c', strokeDash=[3,3]).encode(x='시간:T')
    text = alt.Chart(df_peaks).mark_text(align='left', dx=5, dy=-10, color='#e74c3c', fontWeight='bold').encode(
        x='시간:T', y='인원:Q', text='라벨:N'
    )

    # 최종 차트 결합
    final_chart = (chart + rules + text)
    st.altair_chart(final_chart, use_container_width=True)

    # 5. [전체 인원 흐름 차트] 아래에 추가
    st.divider()
    st.subheader("🔍 특정 구역 상세 인원 추이")
    
    all_areas = sorted(list(filtered_counts.keys()))
    selected_areas = st.multiselect("분석할 구역을 선택하세요", options=all_areas, default=[all_areas[0]] if all_areas else [])

    if selected_areas:
        area_trend_data = []
        for t in sorted(past_time_data.keys()):
            counts = past_time_data[t]['counts']
            for area in selected_areas:
                area_trend_data.append({
                    "시간": idx_to_label[t],
                    "인원": counts.get(area, 0),
                    "구역": area
                })
        
        df_area = pd.DataFrame(area_trend_data)
        
        if df_area.empty:
            st.warning("선택한 구역에 대한 데이터가 없습니다.")
        else:
            df_area['시간'] = pd.to_datetime(df_area['시간'], format='%H:%M:%S', errors='coerce')
            df_area = df_area.dropna(subset=['시간'])
            
            # 구역별 이동평균 계산
            df_area['이동평균'] = df_area.groupby('구역')['인원'].transform(lambda x: x.rolling(window=window_size * 6, min_periods=1).mean())
            
            # 4. Altair 차트 생성 (시각적 최적화)
            chart_area = alt.Chart(df_area).mark_line(
                strokeWidth=1.5,                
                point=False
            ).encode(
                x=alt.X('시간:T', axis=alt.Axis(format='%H:%M', title='시간')),
                y=alt.Y('이동평균:Q', title="체류 인원"),
                color=alt.Color('구역:N', legend=alt.Legend(title="선택 구역")),
                tooltip=['시간', '구역', alt.Tooltip('이동평균', format='.1f')]
            ).properties(
                height=300,
                title="선택 구역별 인원 추이 상세"
            ).interactive() # 마우스 휠로 확대/축소 가능하게 설정

            st.altair_chart(chart_area, use_container_width=True)
    else:
        st.info("비교할 구역을 선택해 주세요.")

    # 5. 상세 운영 권고
    st.divider()
    detailed_data = []
    for area in sorted(filtered_counts.keys()):
        count = filtered_counts.get(area, 0)
        level = "🔴 매우 혼잡" if count >= 160 else "🟠 혼잡" if count >= 120 else "🟡 주의" if count >= 80 else "🟢 보통"
        open_cnt = 0 if count <= 0 else min(40, -(-int(count) // 5))
        support = 0 if count <= 80 else min(3, (count - 80) // 40 + 1)
        detailed_data.append({"구역": area, "혼잡등급": level, "현재 인원": int(count), "권고 오픈 창구": open_cnt, "현장 지원": support})
    
    # Streamlit 데이터프레임으로 시각적 고급화
    def color_congestion(row):
        color = ''
        if "매우 혼잡" in row['혼잡등급']: color = 'background-color: #ffcccc' # 연한 빨강
        elif "혼잡" in row['혼잡등급']: color = 'background-color: #ffe6cc'     # 연한 주황
        elif "주의" in row['혼잡등급']: color = 'background-color: #ffffcc'     # 연한 노랑
        return [color] * len(row)
    
    # 표 시각화 부분
    st.subheader("📍 구역별 운영 권고 상세")
    df_display = pd.DataFrame(detailed_data)
    
    # 스타일 적용
    st.dataframe(
        df_display.style.apply(color_congestion, axis=1),
        use_container_width=True,
        column_config={
            "권고 오픈 창구": st.column_config.ProgressColumn("권고 오픈 창구", format="%d 개", min_value=0, max_value=40),
            "현장 지원": st.column_config.ProgressColumn("현장 지원", format="%d 명", min_value=0, max_value=3)
        },
        hide_index=True
    )

# --- 메인 실행부 ---
st.sidebar.title("🏢 대시보드 메뉴")
menu = st.sidebar.radio("모드 선택", ["📊 과거 데이터 분석", "📡 실시간 모니터링"])

if menu == "📊 과거 데이터 분석":
    st.title("✈️ 인천국제공항 T2 3층 데이터 분석 시스템")
    selected_date = st.date_input("📅 조회할 날짜 선택", value=datetime.date(2025, 10, 4))
    target_date_str = selected_date.strftime("%Y-%m-%d")
    area_df, past_time_data, past_unique_times, bg_img, exists = load_data_by_date(target_date_str)
    
    if exists:
        # 슬라이더를 사이드바에서 본문으로 옮겼으므로 THRESHOLD 인자만 남김
        render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, 75)
    else:
        st.error("해당 날짜의 데이터 파일이 없습니다.")

elif menu == "📡 실시간 모니터링":
    st.title("📡 실시간 모니터링 센터")
    st.info("실시간 데이터 파이프라인 연동 대기 중...")

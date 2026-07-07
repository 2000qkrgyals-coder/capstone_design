import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

    # 3. [개선] 레이아웃 분할 (좌: 히트맵, 우: Top 5 혼잡 구역 차트)
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

    st.divider()

    # 5. [수정] 사이드바 값을 반영한 이동평균 적용
    st.divider()
    st.subheader(f"📈 전체 여객 인원 흐름 ({window_size}분 이동평균)")
    
    time_trend_data = []
    for t in sorted(past_time_data.keys()):
        counts = past_time_data[t]['counts']
        filtered = {k: v for k, v in counts.items() if k not in ["GH", "IM1", "IM2"]}
        time_trend_data.append({"시간": idx_to_label[t], "인원": sum(filtered.values())})
    
    df_trend = pd.DataFrame(time_trend_data).set_index("시간")
    
    # 윈도우 사이즈 반영 (10초 단위 데이터이므로 *6을 하여 분 단위로 환산)
    df_trend['이동평균'] = df_trend['인원'].rolling(window=window_size * 6).mean()

    st.area_chart(df_trend[['이동평균']], color="#3498db")
    
    # 5. 상세 운영 권고 (고급 표 적용)
    st.subheader("📍 구역별 운영 권고 상세")
    
    detailed_data = []
    for area in sorted(filtered_counts.keys()):
        count = filtered_counts.get(area, 0)
        level = "🔴 매우 혼잡" if count >= 160 else "🟠 혼잡" if count >= 120 else "🟡 주의" if count >= 80 else "🟢 보통"
        open_cnt = 0 if count <= 0 else min(40, -(-int(count) // 5))
        support = 0 if count <= 80 else min(3, (count - 80) // 40 + 1)
        detailed_data.append({"구역": area, "혼잡등급": level, "현재 인원": int(count), "권고 오픈 창구": open_cnt, "현장 지원": support})
    
    # Streamlit 데이터프레임으로 시각적 고급화
    st.dataframe(
        pd.DataFrame(detailed_data),
        use_container_width=True,
        column_config={
            "권고 오픈 창구": st.column_config.ProgressColumn("권고 오픈 창구", format="%d 개", min_value=0, max_value=40),
            "현장 지원": st.column_config.ProgressColumn("현장 지원", format="%d 명", min_value=0, max_value=3)
        },
        hide_index=True
    )

# --- 메인 실행부 ---
st.title("✈️ 인천국제공항 T2 3층 데이터 분석 시스템")
tab1 = st.tabs(["📊 과거 데이터 이력 분석"])

with tab1[0]:
    selected_date = st.date_input("📅 조회할 날짜 선택", value=datetime.date(2025, 10, 4))
    target_date_str = selected_date.strftime("%Y-%m-%d")
    area_df, past_time_data, past_unique_times, bg_img, exists = load_data_by_date(target_date_str)
    
    if exists:
        render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, 75)
    else:
        st.error("해당 날짜의 데이터 파일이 없습니다.")


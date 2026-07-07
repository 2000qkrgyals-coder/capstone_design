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
def calculate_staffing(people_count, current_open_counters=None):
    """
    people_count: 현재 인원
    current_open_counters: 현재 열려있는 창구 수 (이전 상태 유지용)
    """
    # 1. 인원당 필요 창구 계산 (5명 기준)
    needed = -(-people_count // 5)
    needed = min(40, max(0, needed))
    
    # 2. 히스테리시스(완충) 적용
    # 현재 창구 수가 있다면, 급격한 변동을 막기 위한 여유범위 설정
    if current_open_counters is not None:
        # 1~2개 정도의 창구는 일시적 인원 변화로 보고 현재 상태를 유지함
        if abs(needed - current_open_counters) <= 2:
            return current_open_counters, ... # 기존 값 반환

    open_counters = needed
    
    # ... (이하 현장 지원 인력 계산 로직)
    
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
    
    # 1. 시간 선택 및 데이터 필터링
    selected_t_index = st.select_slider("🕒 조회 시간 선택", options=time_options, format_func=lambda x: idx_to_label[x])
    current_counts = past_time_data[selected_t_index]['counts']
    excluded = ["GH", "IM1", "IM2"]
    filtered_counts = {k: v for k, v in current_counts.items() if k not in excluded}
    total_people = sum(filtered_counts.values())
    
    # 2. 상단 요약 카드 (Metric)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("현재 총 체류 여객", f"{total_people:,} 명")
        
    # 3. 중요 경고 (혼잡 구역이 있으면 경고 표시)
    urgent_areas = {k: v for k, v in filtered_counts.items() if v >= 80}
    if urgent_areas:
        st.error(f"🚨 주의 필요: {len(urgent_areas)}개 구역에서 혼잡 발생")
    else:
        st.success("✅ 전 구역 정상 운영 중")

    st.divider()
    
    # 4. 히트맵
    st.subheader("📊 구역별 혼잡도 히트맵")
    heatmap = generate_density_heatmap(area_df, filtered_counts, bg_img.shape)
    blended = cv2.addWeighted(bg_img, 0.6, heatmap, 0.4, 0)
    st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.divider()
    
    # 5. 상세 운영 권고 (고급 표 적용)
    st.subheader("📍 구역별 운영 권고 상세")
    
    detailed_data = []
    for area in sorted(filtered_counts.keys()):
        count = filtered_counts.get(area, 0)
        # 알고리즘 계산
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

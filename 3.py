import datetime
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --- [설정] 기본 경로 설정 ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"        
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"          

# --- 1. 페이지 설정 및 사이드바 ---
st.set_page_config(
    page_title="인천공항 T2 3층 데이터 분석 센터",
    layout="wide"
)

with st.sidebar:
    st.header("🎛️ 분석 설정")
    THRESHOLD = st.slider(
        "🚨 정체 경보 임계치 설정 (명)",
        min_value=30,
        max_value=150,
        value=75,
        step=5,
        help="특정 구역의 인원이 이 수치를 넘으면 혼잡(🔴) 경보가 발생합니다."
    )
    st.markdown("---")
    st.info(f"💡 설정 기준\n- 주의: {int(THRESHOLD * 0.6)}명 이상\n- 혼잡: {THRESHOLD}명 이상")

# --- 공통 함수 ---
def index_to_time_str(t_index):
    total_seconds = int(t_index) * 10
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}:{total_seconds % 60:02d}"

@st.cache_data
def load_data_by_date(selected_date_str):
    csv_file_path = f"area_count_time_full_{selected_date_str}.csv"
    area_df = pd.read_csv(AREA_FILE_PATH)
    bg_img = cv2.imread(BACKGROUND_IMAGE_PATH)
    if bg_img is None:
        bg_img = np.full((600, 1900, 3), 240, dtype=np.uint8)
        
    try:
        counts_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return area_df, {}, [], bg_img, False
        
    time_grouped_data = {}
    for t_index, group in counts_df.groupby('time_index'):
        filtered_group = group[group['area'] != 'Outside']
        counts_dict = dict(zip(filtered_group['area'], filtered_group['num_people']))
        time_grouped_data[t_index] = {'counts': counts_dict}
            
    unique_times = sorted(list(time_grouped_data.keys()))
    return area_df, time_grouped_data, unique_times, bg_img, True

def generate_density_heatmap(area_df, current_counts, img_shape):
    height, width, _ = img_shape
    heatmap_grid = np.zeros((height, width), dtype=np.float32)
    np.random.seed(42) # 과거 데이터 재현성을 위해 고정
    
    for _, row in area_df.iterrows():
        area_name = row['area_name']
        people_cnt = current_counts.get(area_name, 0)
        
        if people_cnt > 0:
            pts = np.array([[int(row['x1']), int(row['y1'])], [int(row['x2']), int(row['y2'])], 
                            [int(row['x3']), int(row['y3'])], [int(row['x4']), int(row['y4'])]], dtype=np.int32)
            M = cv2.moments(pts)
            cX, cY = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else (int((row['x1']+row['x3'])/2), int((row['y1']+row['y3'])/2))
            
            num_particles = int(people_cnt * 4)
            rand_x = np.random.normal(cX, 120, num_particles).astype(np.int32)
            rand_y = np.random.normal(cY, 60, num_particles).astype(np.int32)
            valid = (rand_x >= 0) & (rand_x < width) & (rand_y >= 0) & (rand_y < height)
            for x, y in zip(rand_x[valid], rand_y[valid]): heatmap_grid[y, x] += 1.0

    heatmap_smooth = cv2.GaussianBlur(heatmap_grid, (175, 175), 0)
    max_people = max(current_counts.values()) if current_counts else 1
    norm = (heatmap_smooth / heatmap_smooth.max()) * (255 if max_people >= THRESHOLD else (max_people / THRESHOLD * 220))
    
    heatmap_color = cv2.applyColorMap(np.clip(norm, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    _, mask = cv2.threshold(np.clip(norm, 0, 255).astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)

@st.fragment
def render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD):
    # (기존 과거 대시보드 로직 유지)
    time_options = [int(t) for t in past_unique_times]
    idx_to_label = {t: index_to_time_str(t) for t in time_options}
    
    selected_t_index = st.select_slider("🕒 조회 시간 선택", options=time_options, format_func=lambda x: idx_to_label[x])
    
    current_counts = past_time_data[selected_t_index]['counts']
    st.subheader(f"📍 {target_date_str} [{idx_to_label[selected_t_index]}] 분석")
    
    heatmap = generate_density_heatmap(area_df, current_counts, bg_img.shape)
    st.image(cv2.cvtColor(cv2.addWeighted(bg_img, 0.6, heatmap, 0.4, 0), cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # 통계 표시 등...
    st.success("데이터 시각화 완료")

# --- 세션 상태 ---
if "current_index_ptr" not in st.session_state: st.session_state.current_index_ptr = 0

# --- 메인 실행부 ---
st.title("✈️ 인천국제공항 T2 3층 데이터 분석 시스템")
tab1 = st.tabs(["📊 과거 데이터 이력 분석"])

with tab1[0]:
    selected_date = st.date_input("📅 조회할 날짜 선택", value=datetime.date(2025, 10, 4))
    target_date_str = selected_date.strftime("%Y-%m-%d")
    area_df, past_time_data, past_unique_times, bg_img, exists = load_data_by_date(target_date_str)
    
    if exists:
        render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD)
    else:
        st.error("해당 날짜의 데이터가 없습니다.")

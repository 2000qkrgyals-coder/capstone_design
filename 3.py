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
st.set_page_config(page_title="인천공항 T2 3층 데이터 분석 센터", layout="wide")

with st.sidebar:
    st.header("🎛️ 분석 설정")
    THRESHOLD = st.slider("🚨 정체 경보 임계치 설정 (명)", 30, 150, 75, 5)

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
        cnt = current_counts.get(row['area_name'], 0)
        if cnt > 0:
            # 히트맵 입자 생성 로직 (생략된 부분)
            pass
    return np.zeros((height, width, 3), dtype=np.uint8) # 실제 구현 시 위 함수 활용

@st.fragment
def render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD):
    time_options = [int(t) for t in past_unique_times]
    idx_to_label = {t: index_to_time_str(t) for t in time_options}
    
    # 시간 선택
    selected_t_index = st.select_slider("🕒 조회 시간 선택", options=time_options, format_func=lambda x: idx_to_label[x])
    
    # 현재 시간의 데이터
    current_counts = past_time_data[selected_t_index]['counts']
    total_people = sum(current_counts.values())
    
    # 📍 체류 여객 총합 표시 (추가된 부분)
    st.metric(label=f"👥 {idx_to_label[selected_t_index]} 기준 총 체류 여객", value=f"{total_people:,} 명")
    
    st.subheader(f"📍 {target_date_str} 혼잡도 도면")
    heatmap = generate_density_heatmap(area_df, current_counts, bg_img.shape)
    st.image(cv2.cvtColor(cv2.addWeighted(bg_img, 0.6, heatmap, 0.4, 0), cv2.COLOR_BGR2RGB), use_container_width=True)

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

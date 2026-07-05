import datetime
import time
import random
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# --- [설정] 기본 경로 설정 ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"        
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"          

# --- 1. 페이지 설정 및 사이드바 제어 ---
st.set_page_config(
    page_title="인천공항 T2 3층 실시간 혼잡도 제어 센터",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.header("🎛️ 관제 설정 제어")
    THRESHOLD = st.slider(
        "🚨 정체 경보 임계치 설정 (명)",
        min_value=30,
        max_value=150,
        value=75,
        step=5,
        help="특정 구역의 인원이 이 수치를 넘으면 혼잡(🔴) 경보가 발생합니다."
    )
    st.markdown("---")
    st.info(f"💡 현재 설정\n- 주의 기준: {int(THRESHOLD * 0.6)}명 이상\n- 혼잡 기준: {THRESHOLD}명 이상")

def index_to_time_str(t_index):
    total_seconds = int(t_index) * 10
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_virtual_realtime_data(area_df):
    now = datetime.datetime.now()
    hour = now.hour
    if 7 <= hour <= 19:
        base_min, base_max = 30, 95
    else:
        base_min, base_max = 5, 40
        
    virtual_counts = {}
    for _, row in area_df.iterrows():
        area_name = row['area_name']
        virtual_counts[area_name] = random.randint(base_min, base_max)
    return virtual_counts

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
    if not counts_df.empty:
        for t_index, group in counts_df.groupby('time_index'):
            filtered_group = group[group['area'] != 'Outside']
            counts_dict = dict(zip(filtered_group['area'], filtered_group['num_people']))
            time_grouped_data[t_index] = {
                'counts': counts_dict
            }
            
    unique_times = sorted(list(time_grouped_data.keys()))
    return area_df, time_grouped_data, unique_times, bg_img, True

def generate_density_heatmap(area_df, current_counts, img_shape, is_live_mode):
    height, width, _ = img_shape
    heatmap_grid = np.zeros((height, width), dtype=np.float32)
    
    if not is_live_mode:
        np.random.seed(42)
    
    for _, row in area_df.iterrows():
        area_name = row['area_name']
        people_cnt = current_counts.get(area_name, 0)
        
        if people_cnt > 0:
            pts = np.array([
                [int(row['x1']), int(row['y1'])], [int(row['x2']), int(row['y2'])],
                [int(row['x3']), int(row['y3'])], [int(row['x4']), int(row['y4'])]
            ], dtype=np.int32)
            
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cX = int((row['x1'] + row['x2'] + row['x3'] + row['x4']) / 4)
                cY = int((row['y1'] + row['y2'] + row['y3'] + row['y4']) / 4)
            
            std_x, std_y = 120, 60
            num_particles = int(people_cnt * 4)
            
            rand_x = np.random.normal(cX, std_x, num_particles).astype(np.int32)
            rand_y = np.random.normal(cY, std_y, num_particles).astype(np.int32)
            
            valid_indices = (rand_x >= 0) & (rand_x < width) & (rand_y >= 0) & (rand_y < height)
            for x, y in zip(rand_x[valid_indices], rand_y[valid_indices]):
                heatmap_grid[y, x] += 1.0

    blur_size = 175 
    if heatmap_grid.max() > 0:
        heatmap_smooth = cv2.GaussianBlur(heatmap_grid, (blur_size, blur_size), 0)
        max_people_now = max(current_counts.values()) if current_counts else 1
        
        if max_people_now >= THRESHOLD:
            heatmap_norm = (heatmap_smooth / heatmap_smooth.max()) * 255
        else:
            color_sensitivity = max_people_now / float(THRESHOLD)
            heatmap_norm = (heatmap_smooth / heatmap_smooth.max()) * color_sensitivity * 220
            
        heatmap_norm = np.clip(heatmap_norm, 0, 255).astype(np.uint8)
        _, alpha_mask = cv2.threshold(heatmap_norm, 20, 255, cv2.THRESH_BINARY)
        
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.bitwise_and(heatmap_color, heatmap_color, mask=alpha_mask)
        return heatmap_color
    else:
        return np.zeros((height, width, 3), dtype=np.uint8)

@st.fragment
def render_live_dashboard(area_df, bg_img, THRESHOLD):
    current_counts = get_virtual_realtime_data(area_df)
    time_now_str = datetime.datetime.now().strftime("%H:%M:%S")
    total_airport_people = sum(current_counts.values())
    
    prev_total = st.session_state.live_trend_data["전체 여객 수"].iloc[-1] if not st.session_state.live_trend_data.empty else total_airport_people
    delta_people = int(total_airport_people - prev_total)
    
    new_row = pd.DataFrame([{"시간": time_now_str, "전체 여객 수": total_airport_people}])
    st.session_state.live_trend_data = pd.concat([st.session_state.live_trend_data, new_row], ignore_index=True).tail(20)
    
    st.success(f"🔴 LIVE 스트리밍 작동 중 ({time_now_str} | 10초 주기 자동 갱신)")
    st.subheader("📍 인천공항 T2 실시간 여객 밀집도 도면")
    
    heatmap_overlay = generate_density_heatmap(area_df, current_counts, bg_img.shape, is_live_mode=True)
    blended_image = cv2.addWeighted(bg_img, 0.6, heatmap_overlay, 0.4, 0)
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
    st.image(blended_image_rgb, use_container_width=True)
    
    st.markdown("---")
    col_bot1, col_bot2, col_bot3 = st.columns([0.25, 0.45, 0.3])
    
    with col_bot1:
        live_delta_str = f"+{delta_people:,} 명" if delta_people > 0 else (f"{delta_people:,} 명" if delta_people < 0 else "변동 없음")
        st.metric(label="👥 공항 체류 여객 총합", value=f"{total_airport_people:,} 명", delta=live_delta_str)
        st.subheader("📈 실시간 혼잡 추이")
        st.line_chart(st.session_state.live_trend_data.set_index("시간"), height=200)
        
    with col_bot2:
        st.subheader("📊 구역별 상세 모니터링")
        sorted_areas = sorted(current_counts.items(), key=lambda x: x[1], reverse=True)
        for name, cnt in sorted_areas[:4]:
            pct = min(cnt / THRESHOLD, 1.0) 
            status_label = "🔴 혼잡" if cnt >= THRESHOLD else ("🟡 주의" if cnt >= int(THRESHOLD * 0.6) else "🟢 원활")
            st.markdown(f"**카운터 {name}** : `{cnt}명` ({status_label})")
            st.progress(pct)
            
    with col_bot3:
        st.subheader("🚨 실시간 정체 알림")
        alerts_triggered = [ (name, cnt) for name, cnt in current_counts.items() if cnt >= THRESHOLD ]
        if alerts_triggered:
            st.error(f"⚠️ 임계치({THRESHOLD}명) 초과 구역 **{len(alerts_triggered)}곳** 감지")
            for area_name, cnt in alerts_triggered:
                st.markdown(f"❌ **{area_name}** ({cnt}명)")
        else:
            st.success("🟢 모든 구역 임계치 이하 안전 운영 중")
            
    time.sleep(10.0)
    st.rerun()

@st.fragment
def render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD):
    time_options = [int(t) for t in past_unique_times]
    time_labels = [index_to_time_str(t) for t in time_options]
    idx_to_label_map = dict(zip(time_options, time_labels))
    
    peak_time_index = time_options[0]
    max_total_people = 0
    for t_idx in time_options:
        t_sum = sum(past_time_data[t_idx]['counts'].values())
        if t_sum > max_total_people:
            max_total_people = t_sum
            peak_time_index = t_idx
            
    peak_time_str = idx_to_label_map.get(peak_time_index, "알 수 없음")
    
    col_slide, col_b1, col_b2, col_peak = st.columns([0.5, 0.1, 0.1, 0.3])
    selected_t_index = time_options[st.session_state.current_index_ptr]
    
    with col_slide:
        if st.session_state.is_simulating:
            st.info(f"▶ ... 과거 시뮬레이션 재생 중 ... (현재 시각: {idx_to_label_map.get(selected_t_index)})")
        else:
            selected_t_index = st.select_slider(
                "🕒 조회 시간 선택",
                options=time_options,
                value=selected_t_index,
                format_func=lambda x: idx_to_label_map.get(x, str(x)),
                key=f"past_slider_{uuid.uuid4()}" # 매번 새로운 고유 ID 생성
            )
            st.session_state.current_index_ptr = time_options.index(selected_t_index)
    
    with col_b1:
        st.write("")
        if st.button("▶️ 재생", use_container_width=True):
            st.session_state.is_simulating = True
            st.rerun()
    with col_b2:
        st.write("")
        if st.button("⏸️ 정지", use_container_width=True):
            st.session_state.is_simulating = False
            st.rerun()
            
    with col_peak:
        st.write("")
        if st.button(f"🔥 피크 타임 이동 ({peak_time_str})", use_container_width=True, type="primary"):
            st.session_state.is_simulating = False
            st.session_state.current_index_ptr = time_options.index(peak_time_index)
            st.rerun()
    
    current_counts_inner = past_time_data[selected_t_index]['counts'] if selected_t_index in past_time_data else {}
    display_time_str_inner = index_to_time_str(selected_t_index)
    total_airport_people = sum(current_counts_inner.values()) if current_counts_inner else 0
    
    if st.session_state.current_index_ptr > 0:
        prev_t_idx = time_options[st.session_state.current_index_ptr - 1]
        prev_total = sum(past_time_data[prev_t_idx]['counts'].values())
        delta_people = int(total_airport_people - prev_total)
    else:
        delta_people = 0
    
    st.subheader(f"📍 과거 데이터 분석 도면 ({target_date_str} [{display_time_str_inner}])")
    heatmap_overlay = generate_density_heatmap(area_df, current_counts_inner, bg_img.shape, is_live_mode=False)
    blended_image = cv2.addWeighted(bg_img, 0.6, heatmap_overlay, 0.4, 0)
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
    st.image(blended_image_rgb, use_container_width=True)
    
    st.markdown("---")
    col_bot1, col_bot2, col_bot3 = st.columns([0.25, 0.45, 0.3])
    
    with col_bot1:
        delta_str = f"+{delta_people:,} 명" if delta_people > 0 else (f"{delta_people:,} 명" if delta_people < 0 else "변동 없음")
        st.metric(label="👥 당시 체류 여객 총합", value=f"{total_airport_people:,} 명", delta=delta_str)
        st.metric(label="👑 당일 최고 피크 인원", value=f"{max_total_people:,} 명", delta=f"발생시각: {peak_time_str}", delta_color="off")
        
    with col_bot2:
        st.subheader("📊 구역별 상세 모니터링")
        sorted_areas = sorted(current_counts_inner.items(), key=lambda x: x[1], reverse=True) if current_counts_inner else []
        if sorted_areas and max(current_counts_inner.values()) > 0:
            for name, cnt in sorted_areas[:4]:
                pct = min(cnt / THRESHOLD, 1.0)
                status_label = "🔴 혼잡" if cnt >= THRESHOLD else ("🟡 주의" if cnt >= int(THRESHOLD * 0.6) else "🟢 원활")
                st.markdown(f"**카운터 {name}** : `{cnt}명` ({status_label})")
                st.progress(pct)
        else:
            st.info("🟢 대기 구역 한산함")
            
    with col_bot3:
        st.subheader(f"🚨 정체 발생 정보")
        alerts_triggered = [ (name, cnt) for name, cnt in current_counts_inner.items() if cnt >= THRESHOLD ] if current_counts_inner else []
        if alerts_triggered:
            st.error(f"⚠️ 임계치({THRESHOLD}명) 초과 구역 **{len(alerts_triggered)}곳** 기록됨")
            for area_name, cnt in alerts_triggered:
                st.markdown(f"❌ **{area_name}** ({cnt}명)")
        else:
            st.success("🟢 데이터 기록상 정체 구역 없음")

    if st.session_state.is_simulating:
        time.sleep(0.4)
        st.session_state.current_index_ptr = (st.session_state.current_index_ptr + 1) % len(time_options)
        st.rerun()

# --- 세션 상태 초기화 ---
if "current_index_ptr" not in st.session_state:
    st.session_state.current_index_ptr = 0
if "is_simulating" not in st.session_state:
    st.session_state.is_simulating = False
if "live_trend_data" not in st.session_state:
    st.session_state.live_trend_data = pd.DataFrame(columns=["시간", "전체 여객 수"])

# --- 2. 최상단 고정 영역 메인 실행부 ---
st.title("✈️ 인천국제공항 T2 3층 혼잡도 관제 시스템")
st.markdown("---")

area_df, _, _, bg_img, _ = load_data_by_date("2025-10-04")

is_live = st.toggle("🚨 LIVE 실시간 관제 스트리밍 모드", value=False)
st.markdown("---")

# [수정] st.empty()를 지우고, 그냥 컨테이너를 하나 만듭니다.
main_container = st.container()

# [수정] 이 상자(main_container) 안에 내용물을 넣습니다.
with main_container:
    if is_live:
        st.session_state.is_simulating = False
        render_live_dashboard(area_df, bg_img, THRESHOLD)
    else:
        selected_date = st.date_input(
            "📅 조회할 날짜를 선택하세요",
            value=datetime.date(2025, 10, 4),  
            min_value=datetime.date(2025, 9, 1),
            max_value=datetime.date(2025, 10, 31)
        )
        target_date_str = selected_date.strftime("%Y-%m-%d")
        past_area_df, past_time_data, past_unique_times, past_bg_img, past_file_exists = load_data_by_date(target_date_str)
        
        if not past_file_exists:
            st.error(f"❌ 해당 날짜({target_date_str})의 데이터 파일이 존재하지 않습니다.")
        else:
            render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD)
            render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img, target_date_str, THRESHOLD)

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import datetime

# --- [1] Page & Style Configuration ---
st.set_page_config(page_title="ICN T2 Operations Center", layout="wide")

# Dark Mode & Command Center Style
st.markdown("""
    <style>
    /* 전체 배경을 강제로 어둡게 고정 */
    .stApp { background-color: #050505 !important; }
    
    /* 모든 텍스트를 흰색으로 */
    h1, h2, h3, p, div, label, span { color: #ffffff !important; font-family: 'Segoe UI', sans-serif !important; }
    
    /* 표(DataFrame) 배경을 검정색으로 강제 */
    [data-testid="stDataFrame"] { background-color: #000000 !important; border: 1px solid #333 !important; }
    thead tr th { background-color: #1a1a1a !important; color: #00ffcc !important; }
    tbody tr td { background-color: #050505 !important; color: #ffffff !important; }
    
    /* Metric 카드 및 기타 */
    div[data-testid="stMetricValue"] { color: #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- [2] Core Logic Functions ---
AREA_FILE_PATH = "terminal_areas_grouped_2.csv"
BACKGROUND_IMAGE_PATH = "ICN_Airport_3F.png"

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
            cX, cY = int((row['x1']+row['x2']+row['x3']+row['x4'])/4), int((row['y1']+row['y2']+row['y3']+row['y4'])/4)
            num_particles = int(people_cnt * 4)
            rand_x = np.random.normal(cX, 100, num_particles).astype(int)
            rand_y = np.random.normal(cY, 50, num_particles).astype(int)
            valid = (rand_x >= 0) & (rand_x < width) & (rand_y >= 0) & (rand_y < height)
            for x, y in zip(rand_x[valid], rand_y[valid]): heatmap_grid[y, x] += 1.0

    if heatmap_grid.max() > 0:
        heatmap_norm = (cv2.GaussianBlur(heatmap_grid, (175, 175), 0) / heatmap_grid.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        return cv2.bitwise_and(heatmap_color, heatmap_color, mask=cv2.threshold(heatmap_norm, 20, 255, cv2.THRESH_BINARY)[1])
    return np.zeros((height, width, 3), dtype=np.uint8)

@st.fragment
def render_past_dashboard(area_df, past_time_data, past_unique_times, bg_img):
    idx_to_label = {t: f"{(int(t)*10)//3600:02d}:{((int(t)*10)%3600)//60:02d}" for t in past_unique_times}
    t_idx = st.select_slider("🕒 SELECT TIMELINE", options=past_unique_times, format_func=lambda x: idx_to_label[x])
    
    counts = {k: v for k, v in past_time_data[t_idx]['counts'].items() if k not in ["GH", "IM1", "IM2"]}
    
    col1, _ = st.columns([1, 4])
    with col1: st.metric("TOTAL PAX", f"{sum(counts.values()):,}")
    
    if any(v >= 80 for v in counts.values()):
        st.error(f"🚨 ALERT: {sum(1 for v in counts.values() if v >= 80)} zones require intervention")
    else:
        st.success("✅ System Status: Nominal")

    st.subheader("📊 DENSITY HEATMAP")
    heatmap = generate_density_heatmap(area_df, counts, bg_img.shape)
    st.image(cv2.cvtColor(cv2.addWeighted(bg_img, 0.6, heatmap, 0.4, 0), cv2.COLOR_BGR2RGB), use_container_width=True)
    
    st.subheader("📍 REAL-TIME OPERATION RECOMMENDATIONS")
    detailed_data = []
    for area in sorted(counts.keys()):
        c = counts.get(area, 0)
        level = "🔴 CRITICAL" if c >= 160 else "🟠 CONGESTED" if c >= 120 else "🟡 CAUTION" if c >= 80 else "🟢 NORMAL"
        detailed_data.append({
            "Zone": area, "Status": level, "Pax": int(c), 
            "Counters": min(40, -(-int(c)//5)), "Staff": min(3, max(0, (int(c)-80)//40 + 1) if c>80 else 0)
        })
    
    if detailed_data:
        df = pd.DataFrame(detailed_data)
        st.dataframe(df.style.map(
            lambda x: "color: #ff4d4d; font-weight: bold;" if "CRIT" in x else 
                      "color: #ff9900; font-weight: bold;" if "CONG" in x else 
                      "color: #ffff00; font-weight: bold;" if "CAUT" in x else "color: #00ffcc;", 
            subset=["Status"]
        ), use_container_width=True, column_config={
            "Counters": st.column_config.ProgressColumn("COUNTERS", min_value=0, max_value=40),
            "Staff": st.column_config.ProgressColumn("STAFF", min_value=0, max_value=3)
        }, hide_index=True)

# --- [3] Main Execution ---
st.title("✈️ ICN T2 OPERATIONS COMMAND CENTER")

with st.sidebar:
    st.header("⚙️ SYSTEM CONFIG")
    selected_date = st.date_input("SELECT TARGET DATE", value=datetime.date(2025, 10, 4))

area_df, past_data, unique_times, bg, exists = load_data_by_date(selected_date.strftime("%Y-%m-%d"))
if exists:
    render_past_dashboard(area_df, past_data, unique_times, bg)
else:
    st.error("❌ ERROR: Data file for selected date not found.")

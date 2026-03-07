import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.title("Airport Area Congestion & Waiting Time Simulation")

# ============================
# 1️⃣ CSV 파일 읽기
# ============================
df = pd.read_csv("area_count_time_full_2.csv")
df['area'] = df['area'].str.strip()

# ============================
# 2️⃣ 10초 → 1분 단위 집계
# ============================
df['minute_index'] = ((df['time_index'] - 1) // 6).astype(int)
area_counts = df.groupby(['minute_index','area'])['num_people'].sum().reset_index()
area_counts_pivot = area_counts.pivot(index='minute_index', columns='area', values='num_people').fillna(0)

# ============================
# 3️⃣ 구역 유형별 매우혼잡 기준 (1시간 기준)
# ============================
def max_people_1h(area):
    if area in ['A','C','D','E','H','I','J','K','M','N']:  # 일반 체크인
        return min(10, area_counts_pivot[area].max()) * 5 * 60
    elif area in ['B','F','G','L']:  # 셀프 체크인
        return min(10, area_counts_pivot[area].max()) * 2 * 60
    elif area in ['IM1','IM2']:  # 출국장
        return max(area_counts_pivot[area].max(), 1)
    else:  # GH
        return area_counts_pivot[area].max()

# ============================
# 4️⃣ 혼잡도 계산 함수
# ============================
def compute_congestion(area, arrivals, k=5):
    N_max = max_people_1h(area)
    congestion = []
    for n in arrivals:
        raw = min(n / N_max, 1)  # 0~1
        c = 1 / (1 + np.exp(-k*(raw - 0.5)))  # sigmoid
        congestion.append(c)
    return np.array(congestion)

# ============================
# 5️⃣ 구역별 혼잡도 계산
# ============================
congestion_df = pd.DataFrame(index=area_counts_pivot.index)
for area in area_counts_pivot.columns:
    congestion_df[area] = compute_congestion(area, area_counts_pivot[area].values)

# ============================
# 6️⃣ 시간 계산 (HH:MM)
# ============================
start_time = pd.to_datetime('2026-03-07 00:00:00')
time_index = start_time + pd.to_timedelta(congestion_df.index, unit='m')

# ============================
# 7️⃣ Streamlit 구역 선택
# ============================
area_list = list(congestion_df.columns)
selected_area = st.selectbox("Select Area", area_list)

# ============================
# 8️⃣ 혼잡도 그래프
# ============================
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)

# 혼잡도
ax1.plot(time_index, congestion_df[selected_area], label=f"{selected_area} Congestion", color='blue', linewidth=1.2)
ax1.axhline(0.25, color='green', linestyle='--', label='Smooth')
ax1.axhline(0.5, color='yellow', linestyle='--', label='Normal')
ax1.axhline(0.75, color='orange', linestyle='--', label='Crowded')
ax1.axhline(1.0, color='red', linestyle='--', label='Very Crowded')
ax1.set_ylabel("Congestion (0~1)")
ax1.set_title(f"{selected_area} 1-Minute Congestion")
ax1.legend()
ax1.grid(True)

# ============================
# 9️⃣ 대기시간 그래프 (선형 변환)
# ============================
T_max = 60  # 최대 대기시간 60분
waiting_time = congestion_df[selected_area] * T_max
ax2.plot(time_index, waiting_time, label=f"{selected_area} Waiting Time", color='purple', linewidth=1.2)
ax2.set_ylabel("Estimated Waiting Time (min)")
ax2.set_xlabel("Time (HH:MM)")
ax2.set_title(f"{selected_area} Estimated Waiting Time")
ax2.grid(True)

# x축 1시간 간격
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide")
st.title("Airport Area Congestion & Waiting Time Simulation")

# ============================
# 1️⃣ CSV 파일 읽기
# ============================
df = pd.read_csv("area_count_time_full_3.csv")
df['area'] = df['area'].str.strip()

# ============================
# 2️⃣ 10초 → 1분 단위 평균
# ============================
df['minute_index'] = ((df['time_index'] - 1) // 6).astype(int)

area_counts = (
    df.groupby(['minute_index','area'])['num_people']
    .mean()
    .reset_index(name='num_people')
)

area_counts_pivot = area_counts.pivot(
    index='minute_index',
    columns='area',
    values='num_people'
).fillna(0)

# ============================
# 3️⃣ 구역 처리 능력 설정
# ============================
def get_service_params(area):
    if area in ['A','C','D','E','H','J','K','M','N']:
        servers = 10  # 일반 체크인 카운터
        Ts = 4        # 1인당 처리시간 (분)
    elif area in ['B','F','G','L']:
        servers = 10  # 셀프 체크인 기기
        Ts = 5        # 1인당 처리시간 (분)
    elif area in ['IM1','IM2']:
        servers = 10  # 출국장 동시 처리 가능 인원
        Ts = 4        # 1인당 처리시간 (분)
    else:  # Great Hall
        servers = 1000
        Ts = 0.1
    return servers, Ts

# ============================
# 4️⃣ 대기시간 계산
# ============================
waiting_df = pd.DataFrame(index=area_counts_pivot.index)

for area in area_counts_pivot.columns:
    N = area_counts_pivot[area].values
    servers, Ts = get_service_params(area)
    W = np.where(N <= servers, 0, (N - servers) * Ts / servers)
    W = np.maximum(W, 0)
    waiting_df[area] = W

# ============================
# 5️⃣ 혼잡도 계산 (0~1)
# ============================
W_scale = 37.3  # 60분 → congestion 0.8
congestion_df = 1 - np.exp(-waiting_df / W_scale)

# ============================
# 6️⃣ 시간 생성
# ============================
start_time = pd.to_datetime('2025-09-01 00:00:00')
time_index = start_time + pd.to_timedelta(congestion_df.index * 60, unit='s')

# ============================
# 7️⃣ Streamlit 구역 선택
# ============================
area_list = [a for a in congestion_df.columns if a != "Outside"]
selected_area = st.selectbox("Select Area", area_list)

# ============================
# 8️⃣ PPT용 그래프 생성
# ============================
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,8), sharex=True)

# ---- 혼잡도 그래프 ----
ax1.plot(time_index, congestion_df[selected_area],
         label=f"{selected_area} Congestion",
         color='blue', linewidth=2, alpha=0.9)

# 혼잡도 기준선
congestion_levels = {
    0.25: ('green', 'Smooth'),
    0.5: ('yellow', 'Normal'),
    0.75: ('orange', 'Crowded'),
    1.0: ('red', 'Very Crowded')
}

for level, (color, label) in congestion_levels.items():
    ax1.axhline(level, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
    ax1.text(time_index[0], level - 0.05, label, color=color, fontsize=10, fontweight='bold')

ax1.set_ylabel("Congestion (0~1)")
ax1.set_title(f"{selected_area} Congestion")
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, which='major', linestyle='-', linewidth=0.7)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.5)
ax1.minorticks_on()

# ---- 현실적 대기시간 그래프 ----
ax2.plot(time_index, waiting_df[selected_area],
         label=f"{selected_area} Waiting Time (min)",
         color='purple', linewidth=2, alpha=0.9)

ax2.set_ylabel("Estimated Waiting Time (min)")
ax2.set_xlabel("Time (HH:MM)")
ax2.set_title(f"{selected_area} Waiting Time")

# y축 자동 조정
max_wait = waiting_df[selected_area].max()
ax2.set_ylim(0, max_wait * 1.15)

ax2.grid(True, which='major', linestyle='-', linewidth=0.7)
ax2.grid(True, which='minor', linestyle=':', linewidth=0.5)
ax2.minorticks_on()

# x축 1시간 간격
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

st.pyplot(fig)
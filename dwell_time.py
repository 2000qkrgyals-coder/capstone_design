import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide")
st.title("Airport Area Waiting Time Simulation")

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
        Ts = 3        # 1인당 처리시간 (분)
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
# 5️⃣ 시간 생성
# ============================
start_time = pd.to_datetime('2025-09-01 00:00:00')
time_index = start_time + pd.to_timedelta(waiting_df.index * 60, unit='s')

# ============================
# 6️⃣ Streamlit 구역 선택
# ============================
area_list = [a for a in waiting_df.columns if a != "Outside"]
selected_area = st.selectbox("Select Area", area_list)

# ============================
# 7️⃣ 대기시간 그래프
# ============================
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(time_index, waiting_df[selected_area],
        label=f"{selected_area} Waiting Time (min)",
        color='purple', linewidth=2, alpha=0.9)

# ---- 대기시간 기준선 ----
waiting_levels = {
    10: ('green', 'Smooth'),
    20: ('yellow', 'Normal'),
    30: ('orange', 'Crowded'),
    40: ('red', 'Very Crowded')
}

for level, (color, label) in waiting_levels.items():
    ax.axhline(level, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(time_index[0], level + 0.5, label,
            color=color, fontsize=10, fontweight='bold')

ax.set_ylabel("Estimated Waiting Time (min)")
ax.set_xlabel("Time (HH:MM)")
ax.set_title(f"{selected_area} Waiting Time")

# y축 자동 조정
max_wait = waiting_df[selected_area].max()
ax.set_ylim(0, max(max_wait * 1.15, 45))

ax.grid(True, which='major', linestyle='-', linewidth=0.7)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
ax.minorticks_on()

# x축 1시간 간격
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xticks(rotation=45)

ax.legend()

st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide")
st.title("Airport Area Waiting Time & Prediction")

# ============================
# 1️⃣ 데이터 로드
# ============================
df = pd.read_csv("area_count_time_full_3.csv")
df['area'] = df['area'].str.strip()

# ============================
# 2️⃣ 10초 → 1분 변환
# ============================
df['minute'] = (df['time_index'] - 1) // 6

df_1min = (
    df.groupby(['minute','area'])['num_people']
    .mean()
    .reset_index()
)

# ============================
# 3️⃣ 이동평균 예측
# ============================
def moving_average_predict(area_df, window=3):

    area_df = area_df.sort_values("minute").reset_index(drop=True)
    nums = area_df["num_people"].tolist()

    pred = []

    for i in range(len(nums)):

        if i < window:
            avg = sum(nums[:i+1]) / (i+1)
        else:
            avg = sum(nums[i-window+1:i+1]) / window

        pred.append(avg)

    area_df["pred_people"] = pred
    return area_df


areas = sorted(df_1min["area"].unique())

pred_list = []

for area in areas:

    area_df = df_1min[df_1min["area"] == area].copy()

    pred_df = moving_average_predict(area_df)

    pred_list.append(pred_df)

pred_df = pd.concat(pred_list)

# ============================
# 4️⃣ Pivot
# ============================
area_counts = df_1min.pivot(
    index="minute",
    columns="area",
    values="num_people"
).fillna(0)

pred_counts = pred_df.pivot(
    index="minute",
    columns="area",
    values="pred_people"
).fillna(0)

# ============================
# 5️⃣ 서비스 능력
# ============================
def get_service_params(area):

    if area in ['A','C','D','E','H','J','K','M','N']:
        servers = 10
        Ts = 4

    elif area in ['B','F','G','L']:
        servers = 10
        Ts = 5

    elif area in ['IM1','IM2']:
        servers = 10
        Ts = 3

    else:
        servers = 1000
        Ts = 0.1

    return servers, Ts


# ============================
# 6️⃣ 대기시간 계산
# ============================
def calc_waiting(count_df):

    waiting = pd.DataFrame(index=count_df.index)

    for area in count_df.columns:

        N = count_df[area].values

        servers, Ts = get_service_params(area)

        W = np.where(N <= servers, 0, (N - servers) * Ts / servers)

        waiting[area] = np.maximum(W,0)

    return waiting


waiting_df = calc_waiting(area_counts)
future_waiting_df = calc_waiting(pred_counts)

# ============================
# 7️⃣ 시간 생성
# ============================
start_time = pd.to_datetime("2025-09-01 00:00:00")

time_index = start_time + pd.to_timedelta(waiting_df.index * 60, unit="s")

# ============================
# 8️⃣ UI
# ============================
area_list = [a for a in waiting_df.columns if a != "Outside"]

selected_area = st.selectbox("Select Area", area_list)

selected_index = st.slider(
    "Select Time",
    0,
    len(time_index)-1,
    0
)

selected_time = time_index[selected_index]

# ============================
# 9️⃣ 현재 & 미래 계산
# ============================
current_wait = waiting_df[selected_area].iloc[selected_index]

future_index = min(selected_index + 10, len(future_waiting_df)-1)

future_wait = future_waiting_df[selected_area].iloc[future_index]

# ============================
# 🔟 상태 판단
# ============================
def get_status(wait):

    if wait < 10:
        return "Smooth"

    elif wait < 20:
        return "Normal"

    elif wait < 30:
        return "Crowded"

    else:
        return "Very Crowded"


current_status = get_status(current_wait)
future_status = get_status(future_wait)

# ============================
# 11️⃣ 그래프
# ============================
fig, ax = plt.subplots(figsize=(14,6))

ax.plot(
    time_index,
    waiting_df[selected_area],
    color="purple",
    linewidth=2,
    label="Current Waiting Time"
)

ax.plot(
    time_index,
    future_waiting_df[selected_area],
    color="blue",
    linestyle="--",
    linewidth=2,
    label="Predicted Waiting Time"
)

# 기준선
levels = [10,20,30,40]
colors = ["green","yellow","orange","red"]

for l,c in zip(levels,colors):

    ax.axhline(l,color=c,linestyle="--",alpha=0.7)

# 선택 시점
ax.axvline(selected_time,color="black",linestyle=":",linewidth=2)

ax.set_ylabel("Waiting Time (min)")
ax.set_xlabel("Time")

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.xticks(rotation=45)

ax.grid(True)
ax.legend()

st.pyplot(fig)

# ============================
# 12️⃣ 상태 표시
# ============================
st.subheader("Waiting Time Status")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label=f"Current ({selected_time.strftime('%H:%M')})",
        value=f"{current_wait:.1f} min",
        delta=current_status
    )

with col2:

    future_time = selected_time + pd.Timedelta(minutes=10)

    st.metric(
        label=f"Predicted ({future_time.strftime('%H:%M')})",
        value=f"{future_wait:.1f} min",
        delta=future_status
    )
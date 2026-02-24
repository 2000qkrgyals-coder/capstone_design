import pandas as pd
import numpy as np
import random
from dataclasses import dataclass

# ======================
# 설정
# ======================
DATA_FILE = "icn_2025_08_31.csv"
SWARD_FILE = "sward_locations.csv"
OUTPUT_FILE = "trajectory_frames.npz"

IMAGE_WIDTH = 2062
IMAGE_HEIGHT = 662

MIN_RSSI = -100
MAX_RSSI = -60

def rssi_to_distance(rssi, rssi0=-50, n=2):
    return 10 ** ((rssi0 - rssi) / (10 * n))

# ======================
# 데이터 로드
# ======================
print("Loading CSV...")
usecols = ["time_index", "sward_name", "mac_address", "rssi"]
dtypes = {
    "time_index": "int32",
    "sward_name": "category",
    "mac_address": "category",
    "rssi": "int16",
}
df = pd.read_csv(DATA_FILE, usecols=usecols, dtype=dtypes)

# 10초 → 1분
df["minute_index"] = ((df["time_index"] - 1) // 6) + 1

sward = pd.read_csv(SWARD_FILE)
sward_map = {
    row["sward_id"]: (int(row["pos_x"]), int(row["pos_y"]))
    for _, row in sward.iterrows()
}

print("Processing frames...")

frames = []
max_minute = df["minute_index"].max()

for minute in range(1, max_minute + 1):
    df_t = df[df["minute_index"] == minute]

    mac_records = {}

    for mac, sward_name, rssi in df_t[["mac_address","sward_name","rssi"]].values:
        if sward_name not in sward_map:
            continue
        ward_x, ward_y = sward_map[sward_name]
        mac_records.setdefault(mac, []).append((ward_x, ward_y, rssi))

    x_coords = []
    y_coords = []

    for mac, records in mac_records.items():
        records.sort(key=lambda x: x[2], reverse=True)
        ward_x, ward_y, rssi = records[0]

        random.seed(str(mac) + str(minute))

        radius = rssi_to_distance(rssi)
        angle = random.uniform(0, 2*np.pi)

        x = int(ward_x + radius*np.cos(angle))
        y = int(ward_y + radius*np.sin(angle))

        x = min(max(0, x), IMAGE_WIDTH-1)
        y = min(max(0, y), IMAGE_HEIGHT-1)

        x_coords.append(x)
        y_coords.append(y)

    frames.append(np.vstack([x_coords, y_coords]).T)

print("Saving npz...")
np.savez_compressed(
    OUTPUT_FILE,
    frames=np.array(frames, dtype=object)
)
print("Done.")
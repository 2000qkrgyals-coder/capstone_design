from pathlib import Path
import io
import math

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_autorefresh import st_autorefresh

st.write("화면 테스트")

# =============================
# Files
# =============================
BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "trajectory_day_rssi_1min.npz"
FLOORPLAN_PATH = BASE_DIR / "ICN_Airport_3F.png"

# =============================
# Visual params
# =============================
MAX_DISPLAY_WIDTH = 1200
JPEG_QUALITY = 80
TICK_MS = 550

# =============================
# Utils
# =============================
def minute_to_hhmm(m: int) -> str:
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def idx_to_hhmm(idx: int, time_bin_min: int) -> str:
    total_minutes = int(idx) * int(time_bin_min)
    h = (total_minutes // 60) % 24
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"

def fmt_time(i: int, time_bin_min: int) -> str:
    s = i * time_bin_min
    e = (i + 1) * time_bin_min
    return f"{s//60:02d}:{s%60:02d} ~ {e//60:02d}:{e%60:02d}"

@st.cache_resource
def load_npz(path: Path):
    z = np.load(str(path), allow_pickle=True)
    frames = z["frames"]  # object array (T,)
    meta = {k: z[k] for k in z.files if k != "frames"}
    return frames, meta

@st.cache_resource
def load_floorplan_scaled(path: Path, max_width: int):
    img = Image.open(path).convert("RGBA")
    W, H = img.size
    if W <= max_width:
        return img, 1.0
    scale = max_width / float(W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))
    return img.resize((new_w, new_h), resample=Image.BILINEAR), scale

def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def draw_badge(d, xy, text, font, pad=(24, 16), radius=16):  # 박스 크기 확대
    x, y = xy
    pad_x, pad_y = pad
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    d.rounded_rectangle(box, radius=radius, fill=(0, 0, 0, 150))
    d.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255, 255), font=font)

def draw_time_overlays(img, cur_text, start_text, end_text):
    out = img.copy()
    d = ImageDraw.Draw(out, "RGBA")

    font_big = _load_font(60)
    font_mid = _load_font(60)

    draw_badge(d, (16, 16), cur_text, font_big)

    W, _ = out.size
    start_label = f"START {start_text}"
    end_label = f"END {end_text}"

    bbox1 = d.textbbox((0, 0), start_label, font=font_mid)
    bbox2 = d.textbbox((0, 0), end_label, font=font_mid)

    w1 = (bbox1[2] - bbox1[0]) + 28
    w2 = (bbox2[2] - bbox2[0]) + 28

    x_start = max(16, W - w1 - 16)
    x_end = max(16, W - w2 - 16)

    draw_badge(d, (x_start, 16), start_label, font_mid)
    draw_badge(d, (x_end, 60), end_label, font_mid)

    return out

def to_jpeg_bytes(img_rgb, quality=80):
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

# =============================
# Streamlit
# =============================
st.set_page_config(layout="wide")
st.title("ICN Trajectory")

if not NPZ_PATH.exists():
    st.error("trajectory npz 파일이 없습니다.")
    st.stop()

frames, meta = load_npz(NPZ_PATH)
floor_small, scale = load_floorplan_scaled(FLOORPLAN_PATH, MAX_DISPLAY_WIDTH)

T = len(frames)
TIME_BIN_MIN = 1

# -------------------------
# State
# -------------------------
if "playing" not in st.session_state:
    st.session_state.playing = False

if "pos_val" not in st.session_state:
    st.session_state.pos_val = 540.0

if "pos_pick" not in st.session_state:
    st.session_state.pos_pick = int(round(st.session_state.pos_val))

if "skip_once" not in st.session_state:
    st.session_state.skip_once = False

# -------------------------
# UI
# -------------------------
st.markdown("## ⏰ Time Range")

start = st.slider("Start Time", 0, T - 1, 540, format="%d")
end = st.slider("End Time", 0, T - 1, 600, format="%d")

if start > end:
    start, end = end, start

# 슬라이더 위에 실제 시간 표시
st.markdown(f"**Start: {idx_to_hhmm(start, TIME_BIN_MIN)}, End: {idx_to_hhmm(end, TIME_BIN_MIN)}**")

speed = st.slider("Speed", 0.5, 6.0, 2.0, 0.25)

b1, b2, b3 = st.columns([1.2, 1.2, 7.6])
with b1:
    play_clicked = st.button("▶ Play", use_container_width=True)
with b2:
    pause_clicked = st.button("⏸ Pause", use_container_width=True)
with b3:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.playing = False
    st.session_state.pos_val = float(start)
    st.session_state.skip_once = True
elif pause_clicked:
    st.session_state.playing = False
    st.session_state.skip_once = True
elif play_clicked:
    st.session_state.playing = True
    st.session_state.skip_once = True

picked = st.slider(
    "Minute (현재 시각)",
    int(start),
    int(end),
    int(round(st.session_state.pos_val)),
    disabled=st.session_state.playing,
)

if not st.session_state.playing:
    st.session_state.pos_val = float(picked)

# -------------------------
# Playback loop
if st.session_state.playing:
    st_autorefresh(interval=TICK_MS, key="loop")

if st.session_state.skip_once:
    st.session_state.skip_once = False
else:
    if st.session_state.playing:
        st.session_state.pos_val += float(speed)
        if st.session_state.pos_val >= float(end):
            st.session_state.pos_val = float(end)
            st.session_state.playing = False

# -------------------------
# Render
pos = float(st.session_state.pos_val)
i0 = int(math.floor(pos))
i0 = max(int(start), min(i0, int(end)))

coords = frames[i0]

cur_text = idx_to_hhmm(i0, TIME_BIN_MIN)  # HH:MM 표시

composed = floor_small.copy()
draw = ImageDraw.Draw(composed, "RGBA")

# 점 크기 설정
POINT_RADIUS = 2
POINT_COLOR = (0, 120, 255, 200)

if coords is not None and len(coords) > 0:
    for x, y in coords:
        xs = int(x * scale)
        ys = int(y * scale)
        draw.ellipse(
            (xs - POINT_RADIUS, ys - POINT_RADIUS, xs + POINT_RADIUS, ys + POINT_RADIUS),
            fill=POINT_COLOR
        )

start_hhmm = idx_to_hhmm(int(start), TIME_BIN_MIN)
end_hhmm = idx_to_hhmm(int(end), TIME_BIN_MIN)

composed = draw_time_overlays(composed, cur_text, start_hhmm, end_hhmm)

img_bytes = to_jpeg_bytes(composed.convert("RGB"), quality=JPEG_QUALITY)

left, right = st.columns([8, 1])
with left:
    st.image(img_bytes, use_container_width=True)
with right:
    st.empty()

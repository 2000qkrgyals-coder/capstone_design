import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(page_title="Incheon Airport Real-time Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('rf_wait_time_model.pkl')
    df = pd.read_csv("total_learning_data.csv")
    df_ml = pd.get_dummies(df, columns=['area', 'area_type'])
    feature_cols = ['num_people', 'hour', 'is_weekend'] + [col for col in df_ml.columns if 'area_' in col]
    return model, df, feature_cols

try:
    rf_model, raw_df, feature_cols = load_assets()
except Exception as e:
    st.error(f"파일 로드 실패: {e}")
    st.stop()

# 2. 사이드바: 조회 조건 설정
st.sidebar.header("🔍 상세 조회 설정")
all_areas = sorted(raw_df['area'].unique())
selected_area = st.sidebar.selectbox("구역 선택", all_areas)
selected_date = st.sidebar.date_input("날짜 선택", value=datetime(2025, 9, 1))

# [추가] 시간 선택 슬라이더
selected_hour = st.sidebar.slider("상세 조회 시간 (시)", 0, 23, 14)

# 요일 계산
is_weekend_val = 1 if selected_date.weekday() >= 5 else 0
weekday_str = ["월", "화", "수", "목", "금", "토", "일"][selected_date.weekday()]

# 3. 데이터 계산 로직
hours = list(range(24))
predictions = []
avg_people_list = []

for h in hours:
    # 과거 평균 인원 추출
    past_match = raw_df[
        (raw_df['area'] == selected_area) & 
        (raw_df['hour'] == h) & 
        (raw_df['weekday'].apply(lambda x: 1 if x >= 5 else 0) == is_weekend_val)
    ]
    avg_ppl = past_match['num_people'].mean() if len(past_match) > 0 else raw_df[raw_df['hour'] == h]['num_people'].mean()
    avg_people_list.append(avg_ppl)
    
    # 모델 예측
    input_row = pd.DataFrame(columns=feature_cols).fillna(0)
    input_row.loc[0] = 0
    input_row['num_people'] = avg_ppl
    input_row['hour'] = h
    input_row['is_weekend'] = is_weekend_val
    if f'area_{selected_area}' in input_row.columns:
        input_row[f'area_{selected_area}'] = 1
    
    predictions.append(rf_model.predict(input_row)[0])

# 4. 메인 화면 구성
st.title(f"📍 {selected_area} 실시간 혼잡도 예측")
st.markdown(f"### {selected_date.strftime('%Y-%m-%d')} ({weekday_str}) {selected_hour}시 집중 분석")

# 상단 지표 (선택한 시간에 맞춰 변화)
target_wait = predictions[selected_hour]
target_people = avg_people_list[selected_hour]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(f"{selected_hour}시 예상 대기시간", f"{target_wait:.1f} 분")
with c2:
    st.metric(f"{selected_hour}시 예상 인원", f"{target_people:.1f} 명")
with c3:
    congestion = "매우 혼잡" if target_wait > 30 else "보통" if target_wait > 10 else "원활"
    st.metric("예상 혼잡도", congestion)

# 5. 그래프 시각화
fig = go.Figure()

# 전체 흐름 선
fig.add_trace(go.Scatter(
    x=hours, y=predictions, mode='lines', name='대기시간 추이',
    line=dict(color='#00CCFF', width=2), opacity=0.4
))

# [추가] 선택한 시간을 강조하는 포인트
fig.add_trace(go.Scatter(
    x=[selected_hour], y=[target_wait],
    mode='markers+text', name='선택 시점',
    marker=dict(color='#FF4B4B', size=15, symbol='star'),
    text=[f"  {selected_hour}시 ({target_wait:.1f}분)"],
    textposition="top center"
))

fig.update_layout(
    template="plotly_dark",
    xaxis=dict(title="시간 (Hour)", tickmode='linear', range=[-0.5, 23.5]),
    yaxis=dict(title="대기시간 (분)"),
    showlegend=False,
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# 6. 추가 분석 정보
st.divider()
with st.expander("💡 분석 리포트"):
    peak_h = np.argmax(predictions)
    st.write(f"• 해당 날짜의 **피크 타임**은 **{peak_h}시**로 예상되며, 약 **{predictions[peak_h]:.1f}분**의 대기가 발생할 수 있습니다.")
    st.write(f"• 선택하신 **{selected_hour}시**는 하루 중 상대적으로 **{congestion}**한 편에 속합니다.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# ==========================================
# 1. 시스템 설정 및 데이터/모델 조립 (캐싱)
# ==========================================
st.set_page_config(page_title="공항 혼잡도 AI 통합 분석 시스템", layout="wide")

DATA_DIR = Path("./processed_data")
MODEL_DIR = Path("./split_models")

@st.cache_resource
def init_system():
    # [1] 날짜별 데이터 통합 로드 (num_people 포함)
    all_files = sorted(list(DATA_DIR.glob("*_processed.csv")))
    if not all_files:
        st.error(f"'{DATA_DIR}' 폴더에 데이터가 없습니다.")
        st.stop()
        
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df['actual_time'] = pd.to_datetime(df['actual_time'])
    
    # 노이즈 제거 (이동 평균)
    df = df.sort_values(['area', 'actual_time'])
    for col in ['wait_time_min', 'num_people']:
        df[col] = df.groupby('area')[col].transform(
            lambda x: x.rolling(window=120, min_periods=1, center=True).mean()
        )

    # [2] 분할 모델 조립
    rf_model = joblib.load(MODEL_DIR / "rf_skeleton.pkl")
    model_features = joblib.load(MODEL_DIR / "features.pkl")
    
    all_trees = []
    for i in range(1, 6):
        all_trees.extend(joblib.load(MODEL_DIR / f"rf_trees_part_{i}.pkl"))
    rf_model.estimators_ = all_trees
    
    return df, rf_model, model_features

df, rf_model, model_features = init_system()

# ==========================================
# 2. 사이드바 및 필터링
# ==========================================
st.sidebar.header("🕹️ 조회 조건")
all_areas = sorted(df['area'].unique())
selected_area = st.sidebar.selectbox("구역 선택", all_areas)
selected_date = st.sidebar.selectbox("날짜 선택", sorted(df['actual_time'].dt.date.unique()))

plot_df = df[(df['area'] == selected_area) & (df['actual_time'].dt.date == selected_date)].copy()

# ==========================================
# 3. 메인 대시보드 (그래프 영역)
# ==========================================
st.title("✈️ 공항 대기시간 & 인원수 AI 분석 리포트")

if not plot_df.empty:
    # 예측 수행
    X_input = pd.DataFrame(0, index=plot_df.index, columns=model_features)
    X_input['hour'] = plot_df['actual_time'].dt.hour
    X_input['minute'] = plot_df['actual_time'].dt.minute
    X_input['is_weekend'] = (plot_df['actual_time'].dt.dayofweek >= 5).astype(int)
    
    area_col = f"area_{selected_area}"
    if area_col in X_input.columns: X_input[area_col] = 1
    
    plot_df['pred_wait'] = rf_model.predict(X_input)
    # 인원수 추정 (대기시간 트렌드 기반)
    ratio = plot_df['num_people'].mean() / plot_df['wait_time_min'].mean() if plot_df['wait_time_min'].mean() != 0 else 1
    plot_df['pred_num_people'] = plot_df['pred_wait'] * ratio

    # 요약 지표
    m1, m2, m3 = st.columns(3)
    m1.metric(f"[{selected_area}] MAE", f"{mean_absolute_error(plot_df['wait_time_min'], plot_df['pred_wait']):.2f} 분")
    m2.metric(f"[{selected_area}] MAPE", f"{mean_absolute_percentage_error(plot_df['wait_time_min'], plot_df['pred_wait'])*100:.2f} %")
    m3.metric("평균 인원", f"{plot_df['num_people'].mean():.1f} 명")

    # 그래프 1: 대기시간
    st.subheader("⏱️ 대기시간(Wait Time) 실제 vs 예측")
    fig_wait = go.Figure()
    fig_wait.add_trace(go.Scatter(x=plot_df['actual_time'], y=plot_df['wait_time_min'], name="실제", line=dict(color='gray')))
    fig_wait.add_trace(go.Scatter(x=plot_df['actual_time'], y=plot_df['pred_wait'], name="예측", line=dict(color='#FF4B4B', width=3)))
    st.plotly_chart(fig_wait, use_container_width=True)

    # 그래프 2: 인원수
    st.subheader("👥 실시간 인원수(num_people) 실제 vs 추정")
    fig_people = go.Figure()
    fig_people.add_trace(go.Scatter(x=plot_df['actual_time'], y=plot_df['num_people'], name="실제 인원", line=dict(color='#AED6F1')))
    fig_people.add_trace(go.Scatter(x=plot_df['actual_time'], y=plot_df['pred_num_people'], name="추정 인원", line=dict(color='#2E86C1', width=3, dash='dot')))
    st.plotly_chart(fig_people, use_container_width=True)

# ==========================================
# 4. 모델 성능 정밀 평가 (오차 분석 탭 복구)
# ==========================================
st.divider()
st.header("📊 모델 성능 정밀 평가")
tab1, tab2, tab3 = st.tabs(["📉 전체 지표", "📍 구역별 비교", "⏳ 시간대별 상세"])

with tab1:
    with st.spinner("전체 데이터 검증 중..."):
        X_all = pd.DataFrame(0, index=df.index, columns=model_features)
        X_all['hour'] = df['actual_time'].dt.hour
        X_all['minute'] = df['actual_time'].dt.minute
        X_all['is_weekend'] = (df['actual_time'].dt.dayofweek >= 5).astype(int)
        
        df_dummies = pd.get_dummies(df['area'], prefix='area')
        for col in df_dummies.columns:
            if col in X_all.columns: X_all[col] = df_dummies[col].values

        y_pred_all = rf_model.predict(X_all)
        mae = mean_absolute_error(df['wait_time_min'], y_pred_all)
        mape = mean_absolute_percentage_error(df['wait_time_min'], y_pred_all) * 100
        r2 = r2_score(df['wait_time_min'], y_pred_all)

    c1, c2, c3 = st.columns(3)
    c1.metric("통합 MAE", f"{mae:.2f} 분")
    c2.metric("통합 MAPE", f"{mape:.2f} %")
    c3.metric("설명력 (R2)", f"{r2:.4f}")

    st.table(pd.DataFrame({
        "모델명": ["RandomForest (조립형)"],
        "MAE": [f"{mae:.4f}"], "MAPE": [f"{mape:.2f}%"], "R2 Score": [f"{r2:.4f}"]
    }))

with tab2:
    eval_df = df[['actual_time', 'area', 'wait_time_min']].copy()
    eval_df['predicted'] = y_pred_all
    eval_df['abs_error'] = (eval_df['wait_time_min'] - eval_df['predicted']).abs()
    eval_df['hour'] = eval_df['actual_time'].dt.hour
    
    fig_facet = px.line(eval_df.groupby(['area', 'hour'])['abs_error'].mean().reset_index(), 
                        x="hour", y="abs_error", facet_col="area", facet_col_wrap=3)
    st.plotly_chart(fig_facet, use_container_width=True)

with tab3:
    area_spec = eval_df[eval_df['area'] == selected_area].groupby('hour')['abs_error'].mean().reset_index()
    st.plotly_chart(px.bar(area_spec, x='hour', y='abs_error', color='abs_error', color_continuous_scale='Reds'), use_container_width=True)

# ==========================================
# 5. 사용자 방문 시뮬레이션
# ==========================================
st.divider()
st.subheader("🔮 방문 예정 시간 조회")
s1, s2, s3 = st.columns(3)
with s1: in_h = st.slider("시간", 0, 23, 14)
with s2: in_m = st.slider("분", 0, 59, 30)
with s3: is_wk = st.selectbox("날짜 구분", ["평일", "주말"])

u_input = pd.DataFrame(0, index=[0], columns=model_features)
u_input['hour'] = in_h
u_input['minute'] = in_m
u_input['is_weekend'] = 1 if is_wk == "주말" else 0
if f"area_{selected_area}" in u_input.columns: u_input[f"area_{selected_area}"] = 1

st.success(f"📍 [{selected_area}] {in_h:02d}:{in_m:02d} 예상 대기시간: **{rf_model.predict(u_input)[0]:.2f}분**")
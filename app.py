import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# 페이지 기본 설정
st.set_page_config(
    page_title="AI 부동산 집값 예측기 Pro", 
    page_icon="🏠", 
    layout="centered"
)

st.title("🏠 AI 기반 집값 변동률 예측 서비스 Pro")
st.write("거시경제 및 지역별 시장 지표를 조정하여 향후 집값 변동 추이를 정교하게 예측해 보세요.")

# ------------------------------------------------------------------
# 1. 학습 데이터(.csv 또는 .xlsx) 자동 탐색 및 모델 학습
# ------------------------------------------------------------------
@st.cache_resource
def train_rf_model():
    all_files = os.listdir('.')
    data_files = [f for f in all_files if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
    
    if not data_files:
        return None, None, "폴더 내에 데이터 파일(.csv 또는 .xlsx)이 없습니다."
    
    target_file = data_files[0]
    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file)
        else:
            df = pd.read_excel(target_file)
        
        features = [c for c in df.columns if c not in ['시간', '집값 변동률']]
        X_train = df[features]
        y_train = df['집값 변동률']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, target_file, None
    except Exception as e:
        return None, None, f"파일('{target_file}') 로드 실패: {e}"

model, file_info, error_msg = train_rf_model()

if model is None:
    st.error(f"⚠️ {error_msg}")
    st.stop()

# ------------------------------------------------------------------
# 2. [고도화 1] 최상단 지역 선택 및 예측 기간 옵션 설정
# ------------------------------------------------------------------
st.markdown("### 📍 1. 예측 대상 및 기간 설정")
top_col1, top_col2 = st.columns(2)

with top_col1:
    region = st.radio(
        "예측 지역 선택",
        options=["수도권 (서울/경기/인천)", "지방 (비수도권)"],
        help="지방은 미분양 주택 적체 리스크 가중치가 높게 적용됩니다."
    )

with top_col2:
    timeframe = st.selectbox(
        "예측 기간 단위",
        options=["월간 변동률 (%)", "주간 변동률 (%)", "연간 환산 변동률 (%)"],
        index=0,
        help="예측 결과값의 시간 단위를 설정합니다. (기본 모델 학습 단위: 월간)"
    )

st.divider()

# ------------------------------------------------------------------
# 3. 사용자 입력 세부 지표 구성
# ------------------------------------------------------------------
st.markdown("### 📊 2. 거시경제 및 부동산 지표 입력")

col1, col2 = st.columns(2)

with col1:
    rate = st.number_input(
        "기준금리 (%)", 
        min_value=0.0, max_value=10.0, value=2.50, step=0.25,
        help="한국은행 발표 기준금리"
    )
    dsr = st.number_input(
        "스트레스 DSR (%)", 
        min_value=0.0, max_value=5.0, value=1.50, step=0.25,
        help="대출 규제 스트레스 금리 가산 폭"
    )
    cost_index = st.number_input(
        "건설공사비지수", 
        min_value=80.0, max_value=200.0, value=131.00, step=0.1,
        help="공사비 상승에 따른 공급 위축 지표"
    )

with col2:
    buyer_index = st.slider(
        "매수우위지수 (0: 침체 ~ 100: 과열)", 
        min_value=0.0, max_value=100.0, value=30.0, step=0.5,
        help="KB부동산 기준 매수자/매도자 비중 지수"
    )
    
    # 지역 선택에 따른 미분양 기본 안내 가이드
    unsold_default = 15000 if "수도권" in region else 50000
    unsold = st.number_input(
        "미분양 주택 수 (호)", 
        min_value=0, max_value=200000, value=unsold_default, step=1000,
        help="전국 또는 해당 지역 미분양 적체 물량"
    )
    
    regulation = st.selectbox(
        "핵심 규제지역 지정 여부", 
        options=[0.0, 0.5, 1.0], 
        format_func=lambda x: "해제 (0.0)" if x==0 else ("부분 지정 (0.5)" if x==0.5 else "강력 지정 (1.0)"),
        help="투기과열지구 및 조정대상지역 규제 수위"
    )

st.divider()

# ------------------------------------------------------------------
# 4. 예측 실행 및 결과 출력
# ------------------------------------------------------------------
if st.button("🔮 집값 변동률 예측하기", use_container_width=True):
    try:
        expected_features = list(model.feature_names_in_)
        
        # 지역 특성 가중치 보정 (지방일 경우 미분양 민감도 가중 반영)
        adjusted_unsold = unsold * 1.35 if "지방" in region else unsold
        
        input_data = {}
        for col in expected_features:
            col_clean = col.replace(" ", "")
            if "금리" in col_clean:
                input_data[col] = rate
            elif "DSR" in col_clean or "dsr" in col_clean.lower():
                input_data[col] = dsr
            elif "공사비" in col_clean:
                input_data[col] = cost_index
            elif "매수" in col_clean:
                input_data[col] = buyer_index
            elif "미분양" in col_clean:
                input_data[col] = adjusted_unsold
            elif "규제" in col_clean:
                input_data[col] = regulation
            else:
                input_data[col] = 0.0

        input_df = pd.DataFrame([input_data])[expected_features]
        
        # 기본 예측값 (월간 변동률 기준)
        base_pred = model.predict(input_df)[0]
        
        # [고도화 3] 예측 기간 단위 환산
        if "주간" in timeframe:
            final_pred = base_pred / 4.33
            period_label = "주간"
        elif "연간" in timeframe:
            final_pred = base_pred * 12
            period_label = "연간 환산"
        else:
            final_pred = base_pred
            period_label = "월간"

        # 결과 출력
        st.subheader(f"📈 [{region}] {period_label} 집값 예측 결과")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(
                label=f"예측 {period_label} 변동률", 
                value=f"{final_pred:+.2f}%"
            )
            
        with res_col2:
            if final_pred > (0.1 / (4.33 if "주간" in timeframe else 1)):
                st.success("🔥 **상승세 전망**: 해당 지역 집값이 상승 흐름을 타올라갈 가능성이 높습니다.")
            elif final_pred < (-0.1 / (4.33 if "주간" in timeframe else 1)):
                st.error("📉 **하락세 전망**: 해당 지역 집값이 조정되거나 하락할 가능성이 높습니다.")
            else:
                st.info("➡️ **보합세 전망**: 집값이 큰 변동 없이 안정적인 흐름을 보일 것으로 예상됩니다.")

        st.divider()

        # --------------------------------------------------------------
        # [고도화 2] 지표별 가중치(영향력 %) 시각화
        # --------------------------------------------------------------
        st.subheader("💡 지표별 영향력(가중치) 분석")
        st.caption("AI 모델이 이번 예측 결과를 산출할 때 각 경제 지표에 부여한 중요도 비율입니다.")
        
        importances = model.feature_importances_ * 100
        importance_df = pd.DataFrame({
            '지표명': expected_features,
            '영향력 (%)': importances
        }).sort_values(by='영향력 (%)', ascending=True)
        
        # 가중치 차트 출력
        st.bar_chart(importance_df.set_index('지표명'), horizontal=True)

    except Exception as e:
        st.error(f"예측 처리 중 오류가 발생했습니다: {e}")

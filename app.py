import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="AI 부동산 집값 예측기", page_icon="🏠", layout="centered")

st.title("🏠 AI 기반 집값 변동률 예측 서비스")
st.write("거시경제 지표를 조절하여 향후 집값 변동 추이를 예측해 보세요.")

# 2. 모델 파일(.pkcls / .pkl) 자동 탐색 및 로드
@st.cache_resource
def load_model():
    files = [f for f in os.listdir('.') if f.endswith('.pkcls') or f.endswith('.pkl')]
    if not files:
        return None, "모델 파일(.pkcls 또는 .pkl)을 찾을 수 없습니다."
    
    target_file = files[0]
    
    # pickle로 로드 시도
    try:
        with open(target_file, 'rb') as f:
            m = pickle.load(f)
        return m, target_file
    except Exception:
        pass
        
    # joblib으로 로드 시도
    try:
        m = joblib.load(target_file)
        return m, target_file
    except Exception as e:
        return None, str(e)

model, model_name = load_model()

if model is None:
    st.error(f"⚠️ 모델 불러오기 실패: {model_name}")
    st.stop()

st.success(f"✅ 학습된 Random Forest 모델 연결 성공! (`{model_name}`)")

st.divider()

# 3. 사용자 입력 화면 구성
st.subheader("📊 거시경제 및 부동산 지표 입력")

col1, col2 = st.columns(2)

with col1:
    rate = st.number_input("기준금리 (%)", min_value=0.0, max_value=10.0, value=2.50, step=0.25)
    dsr = st.number_input("스트레스 DSR (%)", min_value=0.0, max_value=5.0, value=1.50, step=0.25)
    cost_index = st.number_input("건설공사비지수", min_value=80.0, max_value=200.0, value=131.00, step=0.1)

with col2:
    buyer_index = st.slider("매수우위지수 (0: 침체 ~ 100: 과열)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
    unsold = st.number_input("미분양 주택 수 (호)", min_value=0, max_value=200000, value=65000, step=1000)
    regulation = st.selectbox("핵심 규제지역 지정 여부", options=[0.0, 0.5, 1.0], 
                             format_func=lambda x: "해제(0.0)" if x==0 else ("부분 지정(0.5)" if x==0.5 else "강력 지정(1.0)"))

st.divider()

# 4. 예측 실행 및 결과 출력
if st.button("🔮 집값 변동률 예측하기", use_container_width=True):
    input_df = pd.DataFrame([{
        '건설공사비지수': cost_index,
        '기준금리': rate,
        '스트레스 DSR': dsr,
        '핵심규제지역': regulation,
        '매수우위지수': buyer_index,
        '미분양 현황': unsold
    }])
    
    try:
        if hasattr(model, 'predict'):
            pred_val = model.predict(input_df)
            if isinstance(pred_val, (np.ndarray, list, pd.Series)):
                pred_val = pred_val[0]
        else:
            pred_val = model(input_df.values)[0]
            
        pred_float = float(pred_val)
        
        st.subheader("📈 예측 결과")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="예측 집값 변동률", value=f"{pred_float:+.2f}%")
            
        with col_res2:
            if pred_float > 0.1:
                st.success("🔥 **상승세 전망**: 집값이 상승 흐름을 탈 가능성이 높습니다.")
            elif pred_float < -0.1:
                st.error("📉 **하락세 전망**: 집값이 조정되거나 하락할 가능성이 높습니다.")
            else:
                st.info("➡️ **보합세 전망**: 집값이 큰 변동 없이 안정적인 흐름을 보일 것으로 예상됩니다.")
                
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")

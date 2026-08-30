import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

# 페이지 기본 설정
st.set_page_config(page_title="AI 부동산 집값 예측기", page_icon="🏠", layout="centered")

st.title("🏠 AI 기반 집값 변동률 예측 서비스")
st.write("거시경제 지표를 입력하여 향후 집값 변동 추이를 예측해 보세요.")

# 1. 학습 데이터(.csv 또는 .xlsx) 자동 탐색 및 모델 학습
@st.cache_resource
def train_rf_model():
    all_files = os.listdir('.')
    data_files = [f for f in all_files if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
    
    if not data_files:
        return None, "폴더 내에 데이터 파일(.csv 또는 .xlsx)이 없습니다."
    
    target_file = data_files[0]
    try:
        if target_file.lower().endswith('.csv'):
            df = pd.read_csv(target_file)
        else:
            df = pd.read_excel(target_file)
        
        features = [c for c in df.columns if c not in ['시간', '집값 변동률']]
        X_train = df[features]
        y_train = df['집값 변동률']
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model, target_file
    except Exception as e:
        return None, f"파일('{target_file}') 로드 실패: {e}"

model, file_info = train_rf_model()

if model is None:
    st.error(f"⚠️ {file_info}")
    st.stop()

st.success(f"✅ 학습 데이터 연결 성공! (`{file_info}`)")
st.divider()

# 2. 사용자 입력 화면 구성
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

# 3. 예측 실행
if st.button("🔮 집값 변동률 예측하기", use_container_width=True):
    try:
        # 모델 학습 시 사용된 실제 컬럼 이름 및 순서 가져오기
        expected_features = list(model.feature_names_in_)
        
        # 컬럼명 자동 매핑 (띄어쓰기 및 명칭 차이 해결)
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
                input_data[col] = unsold
            elif "규제" in col_clean:
                input_data[col] = regulation
            else:
                input_data[col] = 0.0

        # 정확한 컬럼 순서대로 데이터프레임 재구성
        input_df = pd.DataFrame([input_data])[expected_features]
        
        pred_val = model.predict(input_df)[0]
        
        st.subheader("📈 예측 결과")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="예측 집값 변동률", value=f"{pred_val:+.2f}%")
            
        with col_res2:
            if pred_val > 0.1:
                st.success("🔥 **상승세 전망**: 집값이 상승 흐름을 탈 가능성이 높습니다.")
            elif pred_val < -0.1:
                st.error("📉 **하락세 전망**: 집값이 조정되거나 하락할 가능성이 높습니다.")
            else:
                st.info("➡️ **보합세 전망**: 집값이 큰 변동 없이 안정적인 흐름을 보일 것으로 예상됩니다.")
    except Exception as e:
        st.error(f"예측 처리 중 오류가 발생했습니다: {e}")

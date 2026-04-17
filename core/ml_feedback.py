# core/ml_feedback.py
import pandas as pd
import numpy as np
import os
try:
    import xgboost as xgb
except ImportError:
    xgb = None

MODEL_PATH = "data/xgb_shrink_model.json"

def train_or_update_model(real_data_csv, cae_features_df):
    """
    실측 치수(real_data_csv)와 CAE 기반 예측 치수(cae_features_df)를 비교해
    오차(Error)를 학습하는 XGBoost 모델 훈련
    """
    if xgb is None:
        return False, "xgboost 라이브러리가 필요해 (pip install xgboost)"
        
    try:
        real_df = pd.read_csv(real_data_csv)
        # 도면 치수, CAE 예측 치수, 공정 조건, 두께 등을 피처로 사용
        merged = pd.merge(real_df, cae_features_df, on="Feature")
        
        X = merged[['도면 공칭 (mm)', '금형 치수 (mm)', 'local_pressure', 'local_temp', 'local_thickness']]
        # Y는 실제 수축률이나 치수 편차
        y = merged['실측 치수 (mm)'] - merged['예측 치수 (mm)']
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
        model.fit(X, y)
        
        # 모델 저장
        os.makedirs("data", exist_ok=True)
        model.save_model(MODEL_PATH)
        
        return True, "모델 학습 완료! 다음 예측부터 현장 오차가 반영될 거야."
    except Exception as e:
        return False, f"학습 중 에러 발생: {e}"

def apply_ml_correction(features_df):
    """예측된 데이터에 ML 모델 오차 보정치 적용"""
    if xgb is None or not os.path.exists(MODEL_PATH):
        return features_df # 모델 없으면 원본 반환
        
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        
        X = features_df[['도면 공칭 (mm)', '금형 치수 (mm)', 'local_pressure', 'local_temp', 'local_thickness']]
        correction_values = model.predict(X)
        
        features_df['ML 보정 치수 (mm)'] = features_df['예측 치수 (mm)'] + correction_values
        return features_df
    except:
        return features_df
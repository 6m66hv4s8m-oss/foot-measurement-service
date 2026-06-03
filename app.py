import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_models():
    return YOLO("models/card_best.pt"), YOLO("models/foot_best.pt")

card_model, foot_model = load_models()

st.title("발 측정 서비스")
uploaded_file = st.file_uploader("영상을 업로드하세요", type=["mp4"])

if uploaded_file is not None:
    # 영상 저장 및 분석 로직 (기존 코드 삽입)
    st.write("분석 중...")
    # ... 여기서 영상 분석하고 결과 출력 ...
    st.success("분석 완료!")
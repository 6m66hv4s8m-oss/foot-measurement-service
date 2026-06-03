import os
import sys
import tempfile
import cv2
import numpy as np
import gdown
import zipfile
import streamlit as st
from ultralytics import YOLO

# 1. 환경 설정 및 라이브러리 경로 강제 지정
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_SKIP_FFMPEG"] = "1"
venv_path = "/home/adminuser/venv/lib/python3.14/site-packages"
if venv_path not in sys.path: sys.path.insert(0, venv_path)

# 2. 모델 설정 및 다운로드 함수
@st.cache_resource
def load_models():
    # 깃허브에는 card_best.pt만 업로드하고, foot_best.pt는 드라이브에서 받음
    card_path = "card_best.pt"
    foot_path = "foot_best.pt"
    FOOT_MODEL_ID = "1eSkFc-EKVuMAKfWQelhjfcK07l8q_vwg"
    
    if not os.path.exists(foot_path):
        temp_zip = "foot_model.zip"
        url = f'https://drive.google.com/uc?id={FOOT_MODEL_ID}'
        gdown.download(url, temp_zip, quiet=False)
        if zipfile.is_zipfile(temp_zip):
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if file.endswith('.pt'):
                        zip_ref.extract(file, ".")
                        if file != foot_path: os.rename(file, foot_path)
        else:
            os.rename(temp_zip, foot_path)
        if os.path.exists(temp_zip): os.remove(temp_zip)
            
    return YOLO(card_path), YOLO(foot_path)

# 3. 측정 로직 함수들 (제공해주신 코드 통합)
# (이곳에 기존에 사용하던 order_points, make_card_homography, transform_point, 
# get_best_card_points, get_best_foot_index, card_shape_is_valid, 
# robust_median_mad, apply_length_correction 함수를 모두 붙여넣으세요)

# 4. 웹 UI 및 동영상 분석 로직
st.title("👟 발 사이즈 측정 서비스")
uploaded_file = st.file_uploader("동영상을 업로드하세요 (mp4)", type=["mp4"])

if uploaded_file is not None:
    if st.button("측정 시작"):
        with st.spinner("분석 중입니다..."):
            card_model, foot_model = load_models()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            # [영상 분석 루프 구현] 
            # 여기에 아까 제공해 드린 'run_measurement' 로직을 배치하세요.
            
            st.success("측정 완료!")
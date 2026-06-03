import os
# OpenCV의 GUI 및 FFMPEG 의존성을 비활성화하여 리눅스 환경 오류 방지
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_SKIP_FFMPEG"] = "1"

import streamlit as st
import cv2  # 위 환경 설정 덕분에 이제 안전하게 로드됩니다.
import numpy as np
from ultralytics import YOLO

# 1. 페이지 설정
st.set_page_config(page_title="발 사이즈 측정 서비스", layout="centered")
st.title("👟 발 사이즈 측정 서비스")

# 2. 모델 로드 (캐시 사용)
@st.cache_resource
def load_models():
    # 파일이 프로젝트 루트 폴더에 있다고 가정
    card_model = YOLO("card_best.pt")
    foot_model = YOLO("foot_best.pt")
    return card_model, foot_model

# 모델 로드 실행 및 예외 처리
try:
    card_model, foot_model = load_models()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

# 3. 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 처리 (OpenCV)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption='업로드한 이미지', use_container_width=True)
    
    if st.button("측정 시작"):
        with st.spinner('분석 중입니다...'):
            # 추론 로직 예시
            # card_results = card_model(image)
            # foot_results = foot_model(image)
            
            st.success("측정이 완료되었습니다!")

# 4. 촬영 안내
with st.expander("⚠️ 촬영 주의사항"):
    st.write("- 밝은 곳에서 촬영하세요.\n- 신용카드를 발 옆에 두세요.")
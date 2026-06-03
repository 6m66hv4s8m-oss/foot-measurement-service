import os
import sys

# 1. 시스템 환경 변수 강제 설정 (핵심!)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_SKIP_FFMPEG"] = "1"

# 2. 파이썬 라이브러리 경로 강제 지정
venv_site_packages = "/home/adminuser/venv/lib/python3.14/site-packages"
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# 3. 안전한 import 처리
try:
    import cv2
except ImportError:
    # 만약 직접 import가 안 된다면 시스템 라이브러리 경로를 확인하는 로직
    os.system('pip install --force-reinstall opencv-python-headless')
    import cv2

import streamlit as st
import numpy as np
from ultralytics import YOLO

st.title("발 사이즈 측정 서비스")
st.write("시스템 환경이 최적화되었습니다.")
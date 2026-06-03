import os
import sys

# 시스템 경로 조작 (가장 먼저 실행되어야 함)
venv_path = "/home/adminuser/venv/lib/python3.14/site-packages"
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# OpenCV 환경 변수 설정
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_SKIP_FFMPEG"] = "1"

# 여기서 import 시도
try:
    import cv2
except ImportError:
    # 만약 위에서 실패하면, 강제로 설치 경로를 다시 한번 확인
    sys.path.append("/usr/local/lib/python3.14/site-packages")
    import cv2

import streamlit as st
import numpy as np
from ultralytics import YOLO

# (이하 기존 코드 작성...)
import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# 모델 로드 (캐시 설정으로 속도 최적화)
@st.cache_resource
def load_models():
    return YOLO("models/card_best.pt"), YOLO("models/foot_best.pt")

card_model, foot_model = load_models()

st.title("👟 스마트 발 측정 서비스")
st.write("측정할 영상을 업로드하면 발 길이와 발볼을 분석합니다.")

# 1. 파일 업로드 기능
uploaded_file = st.file_uploader("영상 파일을 선택하세요 (mp4)", type=["mp4"])

if uploaded_file is not None:
    # 업로드된 파일을 임시로 저장
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)
    
    if st.button("분석 시작하기"):
        with st.spinner('AI가 발을 측정 중입니다... 잠시만 기다려주세요.'):
            # [기존 분석 로직]
            cap = cv2.VideoCapture("temp_video.mp4")
            all_lengths, all_widths = [], []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                card_res = card_model(frame, conf=0.3, verbose=False)[0]
                foot_res = foot_model(frame, conf=0.3, verbose=False)[0]

                if card_res.obb is not None and len(card_res.obb.xyxyxyxy) > 0 and foot_res.keypoints is not None:
                    # (중략: 기존 분석 로직 동일)
                    src_pts = card_res.obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.float32)
                    card_w = np.linalg.norm(src_pts[1] - src_pts[0])
                    pixel_to_mm = 85.6 / (card_w + 1e-6)
                    H = cv2.getPerspectiveTransform(src_pts, np.float32([[0,0], [300,0], [300,200], [0,200]]))
                    
                    if len(foot_res.keypoints.xy) > 0 and len(foot_res.keypoints.xy[0]) > 3:
                        kpts = foot_res.keypoints.xy[0].cpu().numpy()
                        pt0 = cv2.perspectiveTransform(np.array([[[kpts[0][0], kpts[0][1]]]], dtype=np.float32), H)[0][0]
                        pt1 = cv2.perspectiveTransform(np.array([[[kpts[1][0], kpts[1][1]]]], dtype=np.float32), H)[0][0]
                        dist_l = np.linalg.norm(pt0 - pt1) * pixel_to_mm
                        
                        pt2 = cv2.perspectiveTransform(np.array([[[kpts[2][0], kpts[2][1]]]], dtype=np.float32), H)[0][0]
                        pt3 = cv2.perspectiveTransform(np.array([[[kpts[3][0], kpts[3][1]]]], dtype=np.float32), H)[0][0]
                        dist_w = np.linalg.norm(pt2 - pt3) * pixel_to_mm
                        
                        if dist_l < 400: all_lengths.append(dist_l)
                        if dist_w < 200: all_widths.append(dist_w)
            
            cap.release()
            
            # 결과 처리
            if len(all_lengths) > 0:
                f_len = min(np.mean(all_lengths) * 1.15, 400.0)
                f_wid = min(np.mean(all_widths) * 0.85, 200.0) if len(all_widths) > 0 else 0
                st.success("분석 완료!")
                st.metric("발 길이", f"{f_len:.1f} mm")
                st.metric("발볼", f"{f_wid:.1f} mm")
            else:
                st.error("데이터를 찾지 못했습니다. 영상을 다시 찍어주세요.")
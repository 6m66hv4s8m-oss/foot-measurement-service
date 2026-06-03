import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import gdown
import zipfile
from ultralytics import YOLO

# 1. 모델 로드 (구글 드라이브 우회 코드 포함)
@st.cache_resource
def load_models():
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

# 2. 사용자 측정 함수들 (보내주신 로직 그대로)
# (여기에 order_points, make_card_homography, transform_point, 
# get_best_card_points, get_best_foot_index, card_shape_is_valid, 
# robust_median_mad, apply_length_correction 함수를 모두 붙여넣으세요)

# 3. 웹 UI 및 메인 로직
st.title("👟 발 사이즈 자동 측정기")
uploaded_file = st.file_uploader("동영상을 업로드하세요", type=["mp4"])

if uploaded_file is not None:
    if st.button("측정 시작"):
        with st.spinner("영상을 분석 중입니다..."):
            try:
                card_model, foot_model = load_models()
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                cap = cv2.VideoCapture(tfile.name)
                
                all_lengths, all_widths = [], []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # 추론 및 로직 실행
                    card_res = card_model(frame, conf=0.3, verbose=False)[0]
                    foot_res = foot_model(frame, conf=0.3, verbose=False)[0]
                    
                    card_pts = get_best_card_points(card_res)
                    if card_pts is None or not card_shape_is_valid(card_pts): continue
                    
                    foot_idx = get_best_foot_index(foot_res)
                    if foot_idx is None: continue
                    
                    try:
                        H = make_card_homography(card_pts)
                        kpts = foot_res.keypoints.xy[foot_idx].detach().cpu().numpy()
                        
                        # 길이 측정
                        pt0, pt1 = transform_point(kpts[0], H), transform_point(kpts[1], H)
                        dist_l = float(np.linalg.norm(pt0 - pt1))
                        if 150 <= dist_l <= 340: all_lengths.append(dist_l)
                        
                        # 볼 측정
                        pt2, pt3 = transform_point(kpts[2], H), transform_point(kpts[3], H)
                        dist_w = float(np.linalg.norm(pt2 - pt3))
                        if 50 <= dist_w <= 160: all_widths.append(dist_w)
                    except: continue
                
                cap.release()
                os.unlink(tfile.name)
                
                # 결과 출력
                raw_len = robust_median_mad(all_lengths, min_value=150, max_value=340)
                raw_wid = robust_median_mad(all_widths, min_value=50, max_value=160)
                final_len, _ = apply_length_correction(raw_len)
                
                st.success("### 측정 완료")
                st.write(f"👟 **발 길이:** {final_len:.1f} mm")
                st.write(f"📏 **발볼 넓이:** {raw_wid:.1f} mm")
            except Exception as e:
                st.error(f"오류 발생: {e}")
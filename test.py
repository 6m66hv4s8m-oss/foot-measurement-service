import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO

card_model = YOLO("models/card_best.pt")
foot_model = YOLO("models/foot_best.pt")

input_folder = "start_videos"
video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

for video_path in video_files:
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    all_lengths = []
    all_widths = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        card_res = card_model(frame, conf=0.3, verbose=False)[0]
        foot_res = foot_model(frame, conf=0.3, verbose=False)[0]

        if card_res.obb is not None and len(card_res.obb.xyxyxyxy) > 0 and foot_res.keypoints is not None:
            src_pts = card_res.obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.float32)
            card_w = np.linalg.norm(src_pts[1] - src_pts[0])
            pixel_to_mm = 85.6 / (card_w + 1e-6)
            H = cv2.getPerspectiveTransform(src_pts, np.float32([[0,0], [300,0], [300,200], [0,200]]))
            
            if foot_res.keypoints.xy is not None and len(foot_res.keypoints.xy) > 0:
                kpts = foot_res.keypoints.xy[0].cpu().numpy()
                if len(kpts) > 3:
                    # 길이 계산
                    pt0 = cv2.perspectiveTransform(np.array([[[kpts[0][0], kpts[0][1]]]], dtype=np.float32), H)[0][0]
                    pt1 = cv2.perspectiveTransform(np.array([[[kpts[1][0], kpts[1][1]]]], dtype=np.float32), H)[0][0]
                    dist_l = np.linalg.norm(pt0 - pt1) * pixel_to_mm
                    
                    # 발볼 계산
                    pt2 = cv2.perspectiveTransform(np.array([[[kpts[2][0], kpts[2][1]]]], dtype=np.float32), H)[0][0]
                    pt3 = cv2.perspectiveTransform(np.array([[[kpts[3][0], kpts[3][1]]]], dtype=np.float32), H)[0][0]
                    dist_w = np.linalg.norm(pt2 - pt3) * pixel_to_mm
                    
                    # [수정된 필터링] 길이 400mm 미만, 넓이 200mm 미만만 수집
                    if dist_l < 400: all_lengths.append(dist_l)
                    if dist_w < 200: all_widths.append(dist_w)

    cap.release()
    
    # [결과 출력]
    if len(all_lengths) > 0:
        final_len = np.mean(all_lengths) * 1.15
        final_wid = np.mean(all_widths) * 0.85 if len(all_widths) > 0 else 0
        
        # 마지막 안전장치로 400mm, 200mm를 절대 넘지 않게 고정
        final_len = min(final_len, 400.0)
        final_wid = min(final_wid, 200.0)
        
        print(f"✅ [{video_base_name}] 강제 분석 완료")
        print(f"   -> 발 길이: {final_len:.1f}mm | 발볼: {final_wid:.1f}mm")
    else:
        print(f"⚠️ [{video_base_name}] 인식 데이터 없음")
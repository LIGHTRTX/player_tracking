import os
import cv2
import numpy as np
np.float = float  # hotfix for boxmot config bug
from ultralytics import YOLO
from boxmot import create_tracker
import torch

BROADCAST_VIDEO = r"C:\Users\LIGHTRQX\Desktop\player_tracking\dataset\broadcast.mp4"
TACTICAM_VIDEO = r"C:\Users\LIGHTRQX\Desktop\player_tracking\dataset\tacticam.mp4"
MODEL_PATH = r"C:\Users\LIGHTRQX\Desktop\player_tracking\model\best.pt"
BOT_SORT_CONFIG = r"C:\Users\LIGHTRQX\Desktop\player_tracking\botsort.yaml"
OUTPUT_DIR = r"C:\Users\LIGHTRQX\Desktop\player_tracking\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_players(video_path, output_dir, model_path, conf_thres=0.4):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thres)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                txt_name = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:05d}.txt")
                with open(txt_name, "a") as f:
                    f.write(f"{frame_idx},{x1},{y1},{x2},{y2},{conf:.4f},{cls}\n")
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_frame_{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()

def frames_to_video(frames_dir, output_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frame_files:
        return
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for fname in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, fname))
        out.write(frame)
    out.release()

def run_botsort(video_path, detection_dir, output_img_dir, output_txt_dir, model_name):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    tracker = create_tracker(
        tracker_type="botsort",
        tracker_config=BOT_SORT_CONFIG,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        txt_file = os.path.join(detection_dir, f"{model_name}_frame_{frame_idx:05d}.txt")
        detections = []
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split(',')))
                    detections.append(parts[1:6])
        detections = np.array(detections) if detections else np.empty((0, 5))
        dummy_class_ids = np.zeros((detections.shape[0], 1))
        detections_input = np.hstack((detections, dummy_class_ids)) if detections.size > 0 else np.empty((0, 6))
        tracks = tracker.update(detections_input, frame)
        frame_save_path = os.path.join(output_img_dir, f"{frame_idx:05d}.jpg")
        txt_save_path = os.path.join(output_txt_dir, f"{frame_idx:05d}.txt")
        with open(txt_save_path, "w") as f:
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                conf = track[4] if len(track) > 5 else 1.0
                f.write(f"{x1},{y1},{x2},{y2},{conf:.4f},0,{track_id}\n")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(frame_save_path, frame)
        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    detect_players(BROADCAST_VIDEO, os.path.join(OUTPUT_DIR, "broadcast_detections"), MODEL_PATH)
    detect_players(TACTICAM_VIDEO, os.path.join(OUTPUT_DIR, "tacticam_detections"), MODEL_PATH)
    frames_to_video(
        os.path.join(OUTPUT_DIR, "broadcast_detections"),
        os.path.join(OUTPUT_DIR, "broadcast_detected.mp4")
    )
    frames_to_video(
        os.path.join(OUTPUT_DIR, "tacticam_detections"),
        os.path.join(OUTPUT_DIR, "tacticam_detected.mp4")
    )
    run_botsort(
        video_path=os.path.join(OUTPUT_DIR, "broadcast_detected.mp4"),
        detection_dir=os.path.join(OUTPUT_DIR, "broadcast_detections"),
        output_img_dir=os.path.join(OUTPUT_DIR, "broadcast_tracks"),
        output_txt_dir=os.path.join(OUTPUT_DIR, "broadcast_tracks", "track_data"),
        model_name="broadcast"
    )
    run_botsort(
        video_path=os.path.join(OUTPUT_DIR, "tacticam_detected.mp4"),
        detection_dir=os.path.join(OUTPUT_DIR, "tacticam_detections"),
        output_img_dir=os.path.join(OUTPUT_DIR, "tacticam_tracks"),
        output_txt_dir=os.path.join(OUTPUT_DIR, "tacticam_tracks", "track_data"),
        model_name="tacticam"
    )
    frames_to_video(
        os.path.join(OUTPUT_DIR, "broadcast_tracks"),
        os.path.join(OUTPUT_DIR, "broadcast_tracked.mp4")
    )
    frames_to_video(
        os.path.join(OUTPUT_DIR, "tacticam_tracks"),
        os.path.join(OUTPUT_DIR, "tacticam_tracked.mp4")
    )

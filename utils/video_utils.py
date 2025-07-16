import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path, fps=30):
    """
    Save frames to a video file
    :param output_video_frames: List of video frames
    :param output_video_path: Output video path
    :param fps: Frames per second
    """
    if not output_video_frames:
        print("No frames to save!")
        return

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Get frame dimensions
    height, width, _ = output_video_frames[0].shape
    size = (width, height)
    
    # Determine fourcc code based on file extension
    if output_video_path.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Use DIVX for AVI files
    elif output_video_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for MP4 files
    else:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # Default to DIVX
    
    # Create video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # Write frames
    for frame in output_video_frames:
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_video_path}")
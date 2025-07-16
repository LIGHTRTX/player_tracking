from ultralytics import YOLO
import os
import shutil

input_video = r"C:\Users\LIGHTRQX\Desktop\player_tracking\dataset\15sec_input_720p.mp4"
model_path = r"C:\Users\LIGHTRQX\Desktop\player_tracking\model\best.pt"
output_dir = r"C:\Users\LIGHTRQX\Desktop\player_tracking"
output_video = os.path.join(output_dir, "output_video.mp4")

# Run YOLO prediction and save the annotated video
model = YOLO(model_path)
results = model.predict(input_video, save=True)

print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)

# Find the YOLO output video (usually runs/detect/predict/video.mp4)
# The output directory is recorded in results.save_dir
yolo_output_dir = results.save_dir  # e.g. runs/detect/predict
yolo_output_video = os.path.join(yolo_output_dir, "video.mp4")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Move/rename the output video
if os.path.exists(yolo_output_video):
    shutil.move(yolo_output_video, output_video)
    print(f"Output video saved at: {output_video}")
else:
    print("Could not find YOLO output video!")
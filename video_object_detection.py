import cv2
from cap_from_youtube import cap_from_youtube
from yolov8 import YOLOv8
from datetime import datetime
import time
import os

# Initialize YOLOv8 model
model_path = "models/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize video
video_url = 'https://youtu.be/Snyg0RqpVxY'
cap = cap_from_youtube(video_url, resolution='720p')
start_time = 0  # skip the first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Verbose video info
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Information:\n - FPS: {video_fps}\n - Resolution: {video_width}x{video_height}")

# Define the target FPS (e.g., 5 FPS) for processing
target_fps = 5
frame_interval = int(video_fps / target_fps)  # Process every 'frame_interval' frames

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Process only if it's the correct frame based on the FPS limit
    if frame_count % frame_interval == 0:
        # Track inference time
        inference_start = time.time()
        
        # Run object detection
        boxes, scores, class_ids = yolov8_detector(frame)
        
        # Measure inference time
        inference_time = time.time() - inference_start
        print(f"Inference time for frame {frame_count}: {inference_time:.4f} seconds")

        # Draw detections on the frame
        combined_img = yolov8_detector.draw_detections(frame)

        # Generate a timestamped filename for the output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"video_detected_objects_{timestamp}_{frame_count}.jpg")
        
        # Save the image with detected objects
        cv2.imwrite(output_path, combined_img)
        print(f"Frame {frame_count} processed and saved to {output_path}")

    # Increment frame count
    frame_count += 1

    # Wait briefly to simulate real-time processing at the target FPS
    time.sleep(1 / target_fps)

# Release video capture
cap.release()
print("Video processing completed.")

import cv2
from cap_from_youtube import cap_from_youtube
from yolov8 import YOLOv8
from datetime import datetime
import os

# Initialize YOLOv8 model
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Setup video capture from YouTube
video_url = 'https://youtu.be/Snyg0RqpVxY'
cap = cap_from_youtube(video_url, resolution='720p')
start_time = 5  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

while cap.isOpened():
    # Press 'q' key to stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # Draw detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_path = os.path.join(output_dir, f"detected_objects_{timestamp}.jpg")

    # Save the image with detected objects
    cv2.imwrite(output_path, combined_img)
    print(f"Image saved to {output_path}")

print("Video processing completed.")

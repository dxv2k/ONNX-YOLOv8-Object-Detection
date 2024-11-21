import cv2
from yolov8 import YOLOv8
from yolov8.utils import class_names
from datetime import datetime
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv8 object detector
model_path = "models/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.75, iou_thres=0.5)

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

while cap.isOpened():
    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # verbose result 
    print(f"--------------DETECTION RESULT--------------")
    print(boxes)
    print(scores)
    print(class_ids, [class_names[class_id] for class_id in class_ids])
    print(f"--------------------------------------------")

    # Draw detections on the frame
    combined_img = yolov8_detector.draw_detections(frame)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"webcam_objects_{timestamp}.jpg")

    # Save the image with detected objects
    cv2.imwrite(output_path, combined_img)
    print(f"Image saved to {output_path}")

    # Press 'q' to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
print("Video capture stopped.")

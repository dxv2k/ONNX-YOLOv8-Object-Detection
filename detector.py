import cv2
from datetime import datetime
import os
import asyncio
import time
import requests

# Ensure 'yolov8' is correctly installed and accessible
from yolov8 import YOLOv8
from yolov8 import utils

# ----------------------------
# Configuration Parameters
# ----------------------------
MODEL_PATH = "models/yolov8n.onnx"
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.5
OUTPUT_DIR = "output"
TARGET_CLASS_NAME = "person"
DURATION_TIME_IN_SECS = 2
SAMPLING_DURATION = 5  # seconds to capture frames
SLEEP_DURATION = 2  # seconds to sleep between cycles

# Validate CLASS_NAME
if TARGET_CLASS_NAME not in utils.class_names:
    raise ValueError(
        f"Invalid class_name '{TARGET_CLASS_NAME}' is not in {utils.class_names}"
    )

# ----------------------------
# API Server Configuration
# ----------------------------
API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8000/send_alert")

# ----------------------------
# Initialize YOLOv8 Detector
# ----------------------------
detector = YOLOv8(MODEL_PATH, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)

# ----------------------------
# Initialize Webcam
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Shared State for Alerting
# ----------------------------
class DetectionState:
    def __init__(self):
        self.detected_since = None
        self.alert_sent = False


detection_state = DetectionState()


# ----------------------------
# Helper Function to Send Alert
# ----------------------------
def send_alert(message: str):
    try:
        response = requests.post(API_SERVER_URL, json={"message": message})
        response.raise_for_status()
        print(f"Alert sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send alert: {e}")


# ----------------------------
# Detection Function
# ----------------------------
def detect_in_frames(frames):
    for frame in frames:
        boxes, scores, class_ids = detector(frame)
        for class_id, score in zip(class_ids, scores):
            # class_name = detector.class_names.get(class_id, "Unknown")
            class_name = [detector.class_names[class_id] for class_id in class_ids]
            if TARGET_CLASS_NAME in class_name:
                return True
    return False


# ----------------------------
# Main Loop
# ----------------------------
def main():
    global detection_state
    print("Starting YOLOv8 Detection and Alert System...")

    try:
        while True:
            frames = []
            start_time = time.time()
            print(f"Capturing frames for {SAMPLING_DURATION} seconds...")

            # Sampling Phase: Capture frames for SAMPLING_DURATION seconds
            while (time.time() - start_time) < SAMPLING_DURATION:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                frames.append(frame)
                # Optional: Control frame rate if necessary
                # time.sleep(0.1)  # For example, capture at ~10 FPS

            print(f"Captured {len(frames)} frames. Running detection...")

            # Detection Phase: Run detection on all captured frames
            target_detected = detect_in_frames(frames)

            current_time = datetime.now()

            if target_detected:
                if detection_state.detected_since is None:
                    detection_state.detected_since = current_time
                    detection_state.alert_sent = False
                    print(f"[{current_time}] {TARGET_CLASS_NAME} detected.")
                else:
                    elapsed_time = (
                        current_time - detection_state.detected_since
                    ).total_seconds()
                    if (
                        elapsed_time >= DURATION_TIME_IN_SECS
                        and not detection_state.alert_sent
                    ):
                        alert_message = f"Alert: {TARGET_CLASS_NAME} detected for {DURATION_TIME_IN_SECS} seconds."
                        print(f"[{current_time}] {alert_message}")
                        send_alert(alert_message)
                        detection_state.alert_sent = True
            else:
                if detection_state.detected_since is not None:
                    print(f"[{current_time}] {TARGET_CLASS_NAME} no longer detected.")
                detection_state.detected_since = None
                detection_state.alert_sent = False

            # Sleep for SLEEP_DURATION seconds before the next cycle
            print(f"Sleeping for {SLEEP_DURATION} seconds...\n")
            time.sleep(SLEEP_DURATION)

    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()

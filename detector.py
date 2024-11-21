import dotenv

dotenv.load_dotenv()

import cv2
from typing import Literal, Any
from datetime import datetime
import os
import time
import requests

# NOTE: source
from yolov8 import YOLOv8
from yolov8 import utils

# ----------------------------
# Configuration Parameters
# ----------------------------
VERBOSE = True
print("VERBOSE: ",VERBOSE)
MODEL_PATH = "models/yolov8m.onnx"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
OUTPUT_DIR = "output"

TARGET_CLASS_NAME = os.environ.get("TARGET_CLASS_NAME", "dog")
DURATION_TIME_IN_SECS = float(os.environ.get("DURATION_TIME_IN_SECS", 1))
SLEEP_DURATION = float(
    os.environ.get("SLEEP_DURATION", 1)
)  # seconds to sleep between cycles
SAMPLING_DURATION = int(
    os.environ.get("SAMPLING_DURATION", 5)
)  # seconds to capture frames
SAMPLING_RATE_FPS = int(os.environ.get("SAMPLING_RATE_FPS", 5))  # Frames per second

# Validate CLASS_NAME
if TARGET_CLASS_NAME not in utils.class_names:
    raise ValueError(
        f"Invalid class_name '{TARGET_CLASS_NAME}' is not in {utils.class_names}"
    )

# ----------------------------
# API Server Configuration
# ----------------------------
API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8000/send_alert")
API_IMAGE_URL = os.environ.get("API_IMAGE_URL", "http://localhost:8000/send_image")

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


def send_alert_image(image_path: str, caption: str):
    """
    Sends an image to the specified API server with a caption.

    Args:
        image_path (str): Path to the image file to send.
        caption (str): Caption for the image.
    """
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return

    try:
        with open(image_path, "rb") as image_file:
            # Prepare the files dictionary. Specify the filename and content type.
            files = {"file": (os.path.basename(image_path), image_file, "image/png")}

            # Prepare the query parameters
            params = {"caption": caption}

            # Prepare the headers
            headers = {"accept": "application/json"}

            # Make the POST request with files, params, and headers
            response = requests.post(
                API_IMAGE_URL, files=files, params=params, headers=headers
            )

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Optionally, handle the response data
            response_data = response.json()
            print(f"Image sent successfully: {response_data}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to send image: {e}")
    except ValueError:
        print("Failed to parse response as JSON.")


# ----------------------------
# Detection Function
# ----------------------------
def detect_in_frames(frames) -> tuple[Literal[True], Any] | tuple[Literal[False], None]:
    for frame in frames:
        boxes, scores, class_ids = detector(frame)
        for class_id, score in zip(class_ids, scores):
            # Retrieve class name from class_id
            class_name = [detector.class_names[class_id] for class_id in class_ids]
            if TARGET_CLASS_NAME in class_name:
                print(f"--------------ALERT DETAILS--------------")
                print("bbox: ", boxes)
                print("scores: ", scores)
                print(
                    class_ids,
                    [detector.class_names[class_id] for class_id in class_ids],
                )
                print(f"--------------------------------------------")
                return True, frame
            else:
                print(f"--------------DETECTION RESULT--------------")
                print(boxes)
                print(scores)
                print(
                    class_ids,
                    [detector.class_names[class_id] for class_id in class_ids],
                )
                print(f"--------------------------------------------")
    return False, None


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
            if VERBOSE:
                print(
                    f"Capturing frames for {SAMPLING_DURATION} seconds at {SAMPLING_RATE_FPS} FPS..."
                )

            # Calculate interval between frames based on sampling rate
            frame_interval = 1 / SAMPLING_RATE_FPS  # seconds

            # Sampling Phase: Capture frames for SAMPLING_DURATION seconds
            while (time.time() - start_time) < SAMPLING_DURATION:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                frames.append(frame)
                # Sleep to maintain the sampling rate
                time.sleep(frame_interval)

            if VERBOSE:
                print(f"Captured {len(frames)} frames. Running detection...")

            # Detection Phase: Run detection on all captured frames
            target_detected, detected_frame = detect_in_frames(frames)

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

                        # Draw detections on the frame
                        combined_img = detector.draw_detections(frame)

                        # Generate a timestamp for the filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = os.path.join(
                            OUTPUT_DIR, f"alert_{timestamp}.jpg"
                        )

                        # Save the image with detected objects
                        cv2.imwrite(image_filename, combined_img)
                        print(f"Image saved to {image_filename}")
                        # Send the image to the API server with caption
                        send_alert_image(image_filename, caption=alert_message)

                        detection_state.alert_sent = True

                        # NOTE: prev code
                        # alert_message = f"Alert: {TARGET_CLASS_NAME} detected for {DURATION_TIME_IN_SECS} seconds."
                        # print(f"[{current_time}] {alert_message}")
                        # send_alert(alert_message)
                        # detection_state.alert_sent = True
            else:
                if detection_state.detected_since is not None:
                    print(f"[{current_time}] {TARGET_CLASS_NAME} no longer detected.")
                detection_state.detected_since = None
                detection_state.alert_sent = False

            # Sleep for SLEEP_DURATION seconds before the next cycle
            if VERBOSE:
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

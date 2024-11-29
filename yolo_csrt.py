import cv2
import torch
from collections import deque

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Initialize video
cap = cv2.VideoCapture('/Users/adinai/Downloads/Aya_IOT/YoloTracking-main/highway360.mp4 ')

# Define the target frame size
target_width = 320
target_height = 240

# Initialize tracker for each car detected
trackers = []
tracker_labels = []  # Store labels for each tracker
frame_idx = 0
detection_interval = 30  # Interval for re-running detection and refreshing trackers
tracker_history = deque(maxlen=10)  # Store last locations to assist re-detection

# Define the should_redetect function
def should_redetect(trackers, history):
    return len(trackers) == 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the target dimensions
    frame = cv2.resize(frame, (target_width, target_height))

    # Update trackers if they are already initialized
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = tracker_labels[i]
            # Adjust font size here
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            # Remove the tracker and its label
            trackers.pop(i)
            tracker_labels.pop(i)

    # Run detection if trackers are lost or at an interval
    if frame_idx % detection_interval == 0 or should_redetect(trackers, tracker_history):
        trackers.clear()
        tracker_labels.clear()  # Clear existing labels
        results = model(frame)
        for det in results.xyxy[0]:
            if det[4] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, det[:4])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers.append(tracker)
                class_id = int(det[5])
                label = model.names[class_id]
                tracker_labels.append(label)

    cv2.imshow('Frame', frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Initialize video
cap = cv2.VideoCapture('/Users/irmuun/Documents/Aya_IOT/YoloTracking-main/highway360.mp4')

# Initialize tracker for each car detected
trackers = []
frame_idx = 0
detection_interval = 30  # Detect every N frames

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Resize the frame to improve speed (optional)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to 640x480

    # Update trackers if they are already initialized
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Every N frames, detect objects and reinitialize trackers
    if len(trackers) == 0 or frame_idx % detection_interval == 0:
        results = model(frame_resized)  # Use the resized frame for detection
        detections = results.xyxy[0]  # Get all detections for the current frame

        # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
        # This helps filter out redundant or irrelevant detections
        boxes = detections[:, :4].cpu().numpy()  # Get bounding box coordinates
        scores = detections[:, 4].cpu().numpy()  # Get detection scores
        classes = detections[:, 5].cpu().numpy()  # Get detected class indices
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONFIDENCE_THRESHOLD, 0.4)

        # Process the results after NMS
        for i in indices.flatten():
            if classes[i] == 2:  # Check for 'car' class (usually class 2)
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerKCF_create()
                bbox = (int(x1), int(y1), int(w), int(h))
                tracker.init(frame, bbox)
                trackers.append(tracker)

    # Display the result in a window
    cv2.imshow('Frame', frame)

    # Quit the video feed when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

# Initialize video
cap = cv2.VideoCapture('/Users/adinai/Downloads/Aya_IOT/YoloTracking-main/highway360.mp4 ')

# Initialize tracker for each car detected
trackers = []
frame_idx = 0
detection_interval = 30  # Detect and reinitialize trackers every N frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Update trackers if they are already initialized
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Every N frames, detect objects and reinitialize trackers
    if len(trackers) == 0 or frame_idx % detection_interval == 0:  # Run detection periodically
        results = model(frame)  # Detect objects using YOLO
        for det in results.xyxy[0]:
            if det[4] >= 0.6 and det[-1] == 2:  # Confidence threshold and class index for 'car'
                x1, y1, x2, y2 = map(int, det[:4])
                w, h = x2 - x1, y2 - y1
                tracker = cv2.TrackerCSRT_create()  # Use CSRT for better accuracy
                bbox = (x1, y1, w, h)
                tracker.init(frame, bbox)
                trackers.append(tracker)

    # Display the result in a window
    cv2.imshow('Frame', frame)

    # Quit the video feed when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


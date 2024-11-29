import torch
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to perform object detection
def process_frame(frame, model):
    results = model(frame)
    for det in results.xyxy[0]:  # detections for this frame
        if det[-1] == 2:  # Assuming 2 is the class index for 'car'
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

# Load your video
cap = cv2.VideoCapture('/Users/adinai/Downloads/Aya_IOT/YoloTracking-main/highway360.mp4 ')

# Initialize the ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)
futures = []

# Main loop to read frames
frame_id = 0
skip_frames = 5  # Number of frames to skip

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to speed up the processing
    if frame_id % (skip_frames + 1) == 0:
        # Submit frames for processing
        future = executor.submit(process_frame, frame.copy(), model)
        futures.append(future)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Get results from completed futures and display
    for future in as_completed(futures):
        processed_frame = future.result()
        cv2.imshow('Processed Frame', processed_frame)

    # Remove completed futures from the list
    futures = [f for f in futures if not f.done()]

    frame_id += 1

    if cv2.waitKey(1) == ord('q'):
        break

# Wait for all threads to complete
executor.shutdown(wait=True)

# Clean up
cap.release()
cv2.destroyAllWindows()


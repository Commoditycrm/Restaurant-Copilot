import numpy as np
from ultralytics import YOLO
import cv2

video_path = "44949c52-0613-4967-8911-f33a6f6221a3.mp4"

# Load YOLO model (Replace with your trained model path)
model = YOLO("yolo12l.pt")  # Update with your actual model path

# Open video file for processing
cap = cv2.VideoCapture(video_path)

# Define entrance line points [(start_x, start_y), (end_x, end_y)] - ADJUST THESE VALUES
LINE_START = (1175, 610)  
LINE_END =  (1035, 620)  

# Tracking setup
previous_positions = {}
entry_count = 0
exit_count = 0

# Video writer setup
output_path = "output_customer_counting.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, 3, (frame_width, frame_height))

frame_index = 0
skip_frames = max(1, fps // 3)  # Process 3 frames per second

def calculate_cross(x, y):
    """Calculate cross product for point (x,y) relative to the counting line."""
    return (LINE_END[0] - LINE_START[0]) * (y - LINE_START[1]) - \
           (LINE_END[1] - LINE_START[1]) * (x - LINE_START[0])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % skip_frames != 0:
        continue  # Skip frames to reduce load

    results = model(frame)
    new_positions = {}

    for result in results:
        detections = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(detections):
            if class_ids[i] == 0 and confidences[i] > 0.5:
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                new_positions[i] = (cx, cy)

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {i}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Check crossing condition
                if i in previous_positions:
                    prev_cx, prev_cy = previous_positions[i]

                    # Calculate cross products
                    prev_cross = calculate_cross(prev_cx, prev_cy)
                    current_cross = calculate_cross(cx, cy)

                    # Detect crossing direction
                    if prev_cross * current_cross < 0:
                        # Determine entry/exit based on direction
                        if prev_cross < 0 and current_cross > 0:
                            entry_count += 1
                        else:
                            exit_count += 1

    # Update tracking
    previous_positions = new_positions.copy()

    # Draw counting line
    cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f"Entries: {entry_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Exits: {exit_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write processed frame to output video
    out.write(frame)

cap.release()
out.release()

print(f"Processed video saved at: {output_path}")
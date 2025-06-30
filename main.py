import cv2
from ultralytics import YOLO
import json
import numpy as np
import random

# Load AOI coordinates
with open("area_config.json", "r") as f:
    config = json.load(f)
aoi = np.array(config["AOI"], dtype=np.int32)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or higher for better accuracy

# Load video
video_path = "videos/road_123.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Unable to open video.")
    exit()

# Define vehicle classes
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Assign unique colors to IDs
id_colors = {}
def get_color(track_id):
    if track_id not in id_colors:
        id_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return id_colors[track_id]

# AOI check function
def is_inside_aoi(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

# Set to store unique IDs inside AOI
unique_vehicle_ids_in_aoi = set()

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 tracking
    results = model.track(frame, persist=True)[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_classes:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if is_inside_aoi(cx, cy, aoi):
                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id is not None:
                        unique_vehicle_ids_in_aoi.add(track_id)

                    # Draw box and info
                    color = get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Draw AOI
    cv2.polylines(frame, [aoi], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display total unique vehicle count
    cv2.putText(frame, f'Total Vehicles Passed in AOI: {len(unique_vehicle_ids_in_aoi)}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Smart Traffic Management", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

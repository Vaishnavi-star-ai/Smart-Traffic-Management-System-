import cv2
import torch
import time
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# ✅ Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Load YOLOv8 model
model = YOLO('yolov8n.pt').to(device)

# ✅ Open video file
cap = cv2.VideoCapture('road_trafficone.mp4')

# ✅ Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ✅ Define entry and exit line positions
y_line_entry = int(frame_height * 0.4)
y_line_exit = int(frame_height * 0.85)
real_distance_meters = 10  # Distance between lines in meters

# ✅ Initialize tracking helpers
timestamps = {}
speed_history = defaultdict(lambda: deque(maxlen=5))
last_speed_display = {}
unique_vehicle_ids = set()
frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Draw lines
    cv2.line(frame, (0, y_line_entry), (frame_width, y_line_entry), (0, 0, 255), 3)
    cv2.line(frame, (0, y_line_exit), (frame_width, y_line_exit), (255, 0, 0), 3)

    # Current timestamp on frame
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

    # ✅ Detect and track vehicles
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck

    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = model.names.get(class_idx, "Unknown")

            unique_vehicle_ids.add(track_id)

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            # Entry timestamp
            if track_id not in timestamps and cy > y_line_entry:
                timestamps[track_id] = time.monotonic()

            # Exit timestamp and speed calculation
            if track_id in timestamps and cy > y_line_exit:
                elapsed = time.monotonic() - timestamps[track_id]
                adjusted_time = max(elapsed / frame_skip, 0.001)
                speed_kmh = (real_distance_meters / adjusted_time) * 3.6

                if 10 < speed_kmh < 120:
                    speed_history[track_id].append(speed_kmh)
                    avg_speed = int(np.mean(speed_history[track_id]))
                    last_speed_display[track_id] = avg_speed

                del timestamps[track_id]

            # Display speed
            if track_id in last_speed_display:
                cv2.putText(frame, f"{last_speed_display[track_id]} Km/h", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display total vehicle count
    cv2.putText(frame, f"Total Vehicles: {len(unique_vehicle_ids)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # Show frame
    cv2.imshow("🚗 Vehicle Speed Detection", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"✅ Total Unique Vehicles Detected: {len(unique_vehicle_ids)}")

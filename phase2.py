import cv2
import json
import numpy as np
import random
import time
from collections import deque
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Load AOI config
with open("area_config.json", "r") as f:
    config = json.load(f)
aoi = np.array(config["AOI"], dtype=np.int32)

# Video and model setup
cap = cv2.VideoCapture("videos/road_123.mp4")
if not cap.isOpened():
    print("❌ Unable to open video.")
    exit()

# Settings
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
vehicle_weights = {'car': 1, 'motorcycle': 0.5, 'bus': 2, 'truck': 2}
id_colors = {}
vehicle_tracks = {}
unique_vehicle_ids_in_aoi = set()
MAX_TRACK_LENGTH = 30
gap_threshold = 40
stop_line = [(300, 400), (600, 400)]
stop_line_y = 400
green_time = 10
last_check = 0
counting = False
detect_time = 5
time_limit = 2 * 60  # 2 minutes
start_time = time.time()

# Helper functions
def get_color(track_id):
    if track_id not in id_colors:
        id_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return id_colors[track_id]

def is_inside_aoi(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

def is_gap_clear(cx, cy, others, threshold=gap_threshold):
    return all(not (abs(cx - ox) < 50 and 0 < oy - cy < threshold) for ox, oy in others)

def infer_turn(track):
    if len(track) < 2:
        return "unknown"
    dx = track[-1][0] - track[0][0]
    dy = track[-1][1] - track[0][1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "straight"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    elapsed = now - start_time
    remaining = max(0, int(time_limit - elapsed))
    active_centers = []
    counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}

    # Start detection period
    if now - last_check > green_time:
        counting = True
        last_check = now

    results = model.track(frame, persist=True)[0]
    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name not in vehicle_classes:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = int(box.id[0]) if box.id is not None else None

            if track_id is not None and is_inside_aoi(cx, cy, aoi):
                active_centers.append((cx, cy))
                unique_vehicle_ids_in_aoi.add(track_id)
                vehicle_tracks.setdefault(track_id, deque(maxlen=MAX_TRACK_LENGTH)).append((cx, cy))
                direction = infer_turn(vehicle_tracks[track_id])

                if cy < stop_line_y and not is_gap_clear(cx, cy, active_centers):
                    continue

                color = get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 20), 0, 0.5, color, 2)
                cv2.putText(frame, f"Turn: {direction}", (x1, y1 - 5), 0, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                pts = list(vehicle_tracks[track_id])
                for i in range(1, len(pts)):
                    if is_inside_aoi(*pts[i], aoi) and is_inside_aoi(*pts[i - 1], aoi):
                        cv2.line(frame, pts[i - 1], pts[i], color, 2)

                if counting:
                    counts[class_name] += 1

    # Adjust green signal duration after detect_time
    if counting and now - last_check > detect_time:
        total_weight = sum(vehicle_weights[cls] * count for cls, count in counts.items())
        lanes = 1  # you can make this dynamic
        green_time = int(max(10, min((total_weight * 2) / (lanes + 1), 60)))
        counting = False

    # Display overlays
    cv2.polylines(frame, [aoi], True, (255, 0, 0), 2)
    cv2.line(frame, stop_line[0], stop_line[1], (0, 255, 255), 2)
    cv2.putText(frame, "Stop Line", (stop_line[0][0], stop_line[0][1] - 10), 0, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Green Time: {green_time}s", (20, 30), 0, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Time Left: {remaining//60:02}:{remaining%60:02}", (20, 60), 0, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Vehicles in AOI: {len(unique_vehicle_ids_in_aoi)}", (20, 90), 0, 0.8, (0, 255, 255), 2)

    if elapsed >= time_limit:
        rate = len(unique_vehicle_ids_in_aoi) / (time_limit / 60)
        print("✅ Time limit reached!")
        print(f"🚗 Total unique vehicles in AOI: {len(unique_vehicle_ids_in_aoi)}")
        print(f"📈 Traffic Rate: {rate:.2f} vehicles/min")
        cv2.putText(frame, f"Traffic Rate: {rate:.2f} vehicles/min", (20, 120), 0, 0.8, (0, 255, 0), 2)
        cv2.imshow("Smart Traffic Management", frame)
        cv2.waitKey(0)
        break

    cv2.imshow("Smart Traffic Management", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

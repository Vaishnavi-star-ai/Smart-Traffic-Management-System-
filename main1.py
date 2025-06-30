import cv2
import time
import json
import numpy as np
import pygame
from ultralytics import YOLO
from threading import Thread
from datetime import datetime

# Constants
DETECTION_TIME = 8
TOTAL_CYCLE_TIME = 120
MIN_GREEN_TIME = 7
MAX_GREEN_TIME = 30
YELLOW_TIME = 5

# GUI setup
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
FRAME_SIZE = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Smart Traffic Signal System")
font = pygame.font.SysFont("Arial", 24)
count_font = pygame.font.SysFont("Arial", 20)  # Font for vehicle counts

class VehicleDetector:
    def __init__(self, video_path, aoi_config, direction_label):
        self.direction = direction_label
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(video_path)
        with open(aoi_config, "r") as f:
            config = json.load(f)
        self.aoi = np.array(config["AOI"], dtype=np.int32)
        self.vehicle_weights = {'car': 2, 'motorcycle': 1, 'bus': 3, 'truck': 3, 'bicycle': 1, 'auto rickshaw': 2}
        self.current_counts = {cls: 0 for cls in self.vehicle_weights}
        self.frame_counts = {cls: 0 for cls in self.vehicle_weights}  # Per-frame counts
        self.total_detected = 0
        self.signal_status = '🔴'
        self.frame = None
        self.vehicle_ids = {}
        self.vehicle_id_counter = 0

    def detect_in_frame(self, frame):
        results = self.model(frame)[0]
        detections = []
        new_ids = {}
        frame_counts = {cls: 0 for cls in self.vehicle_weights}  # Reset per-frame counts
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            if cls_name in self.vehicle_weights:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(self.aoi, (cx, cy), False) >= 0:
                    bbox = (x1, y1, x2, y2)
                    match_found = False
                    for old_bbox, v_id in self.vehicle_ids.items():
                        ox1, oy1, ox2, oy2 = old_bbox
                        if abs(ox1 - x1) < 30 and abs(oy1 - y1) < 30:
                            new_ids[bbox] = v_id
                            match_found = True
                            break
                    if not match_found:
                        self.vehicle_id_counter += 1
                        new_ids[bbox] = self.vehicle_id_counter
                    detections.append((cls_name, bbox, new_ids[bbox]))
                    frame_counts[cls_name] += 1  # Increment per-frame count
        self.vehicle_ids = new_ids
        self.frame_counts = frame_counts
        self.total_detected = sum(frame_counts.values())  # Total vehicles in current frame
        return detections

    def count_vehicles(self, duration=DETECTION_TIME):
        self.current_counts = {cls: 0 for cls in self.vehicle_weights}
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.frame = frame.copy()
            detections = self.detect_in_frame(self.frame)
            for cls, _, _ in detections:
                self.current_counts[cls] += 1

    def get_weighted_score(self):
        return sum(self.vehicle_weights[cls] * self.current_counts[cls] for cls in self.current_counts)

    def get_display_frame(self):
        if self.frame is None:
            return np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        overlay = self.frame.copy()
        detections = self.detect_in_frame(overlay)

        cv2.polylines(overlay, [self.aoi], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(overlay, f"{self.direction}: {self.signal_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if self.signal_status == "🟢" else (255, 255, 0) if self.signal_status == "🟡" else (0, 0, 255), 2)

        for cls, bbox, vid in detections:
            x1, y1, x2, y2 = bbox
            label = f"#{vid} {cls}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        y = 60
        cv2.putText(overlay, f"Total Vehicles: {self.total_detected}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(overlay, "Per-Frame Counts:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        for cls, count in self.frame_counts.items():
            y += 20
            if count > 0:
                cv2.putText(overlay, f"{cls}: {count}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        y += 25
        cv2.putText(overlay, "Cycle Counts:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        for cls, count in self.current_counts.items():
            y += 20
            if count > 0:
                cv2.putText(overlay, f"{cls}: {count}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        return cv2.resize(overlay, FRAME_SIZE)

def draw_gui(detectors, remaining_cycle_time, cycle_label):
    frames = [cv2.cvtColor(d.get_display_frame(), cv2.COLOR_BGR2RGB) for d in detectors]
    surfaces = [pygame.surfarray.make_surface(np.rot90(f)) for f in frames]
    screen.blit(surfaces[0], (0, 0))
    screen.blit(surfaces[1], (FRAME_SIZE[0], 0))
    screen.blit(surfaces[2], (0, FRAME_SIZE[1]))
    screen.blit(surfaces[3], (FRAME_SIZE[0], FRAME_SIZE[1]))

    time_text = f"Cycle Time Left: {remaining_cycle_time}s"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_surf = font.render(time_text, True, (255, 255, 0))
    clock_surf = font.render(now, True, (255, 255, 0))
    cycle_surf = font.render(cycle_label, True, (0, 255, 255))

    screen.blit(time_surf, (10, WINDOW_HEIGHT - 60))
    screen.blit(clock_surf, (10, WINDOW_HEIGHT - 30))
    screen.blit(cycle_surf, (WINDOW_WIDTH - 200, WINDOW_HEIGHT - 60))
    pygame.display.update()

def get_cycle_label(cycle_number):
    return f"Cycle: {cycle_number}"

if __name__ == "__main__":
    north = VehicleDetector("videos/road_123.mp4", "aoi_north.json", "North")
    east = VehicleDetector("videos/road_traffic.mp4", "aoi_east.json", "East")
    south = VehicleDetector("videos/road_123.mp4", "aoi_south.json", "South")
    west = VehicleDetector("videos/road_traffic.mp4", "aoi_west.json", "West")
    detectors = [north, east, south, west]
    fixed_order = ['North', 'East', 'South', 'West']
    dir_map = {d.direction: d for d in detectors}
    cycle_number = 1

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            scores = {}
            threads = []
            for detector in detectors:
                t = Thread(target=detector.count_vehicles)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            for detector in detectors:
                scores[detector.direction] = detector.get_weighted_score()

            remaining_time = TOTAL_CYCLE_TIME
            cycle_label = get_cycle_label(cycle_number)

            for idx, dir_label in enumerate(fixed_order):
                if remaining_time <= 0:
                    break
                detector = dir_map[dir_label]
                next_idx = (idx + 1) % len(fixed_order)
                next_detector = dir_map[fixed_order[next_idx]]

                total_vehicles = sum(detector.current_counts.values())
                if total_vehicles > 15:
                    green_time = min(MAX_GREEN_TIME, remaining_time)
                else:
                    green_time = (total_vehicles / 15) * MAX_GREEN_TIME
                    green_time = max(MIN_GREEN_TIME, min(green_time, remaining_time))

                remaining_time -= green_time

                for d in detectors:
                    d.signal_status = "🟢" if d == detector else "🔴"

                for sec in range(int(green_time), 0, -1):  # Ensure integer green_time
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    # Update frames and counts for all detectors
                    for d in detectors:
                        ret, frame = d.cap.read()
                        if not ret:
                            d.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        d.frame = frame.copy()
                        d.detect_in_frame(d.frame)
                    draw_gui(detectors, remaining_time + sec, cycle_label)
                    print(f"🟢 {dir_label} - {sec} sec", end="\r")
                    time.sleep(1)

                detector.signal_status = "🟡"
                # Update frames during yellow phase
                for sec in range(YELLOW_TIME, 0, -1):
                    for d in detectors:
                        ret, frame = d.cap.read()
                        if not ret:
                            d.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        d.frame = frame.copy()
                        d.detect_in_frame(d.frame)
                    draw_gui(detectors, remaining_time, cycle_label)
                    print(f"🟡 {dir_label} - {sec} sec", end="\r")
                    time.sleep(1)

                next_detector.count_vehicles(duration=YELLOW_TIME)
                new_score = next_detector.get_weighted_score()
                scores[fixed_order[next_idx]] = max(MIN_GREEN_TIME, min(new_score, 30))

                detector.signal_status = "🔴"

            cycle_number += 1

    except KeyboardInterrupt:
        print("\n🚓 Traffic system stopped.")
        pygame.quit()
        cv2.destroyAllWindows()
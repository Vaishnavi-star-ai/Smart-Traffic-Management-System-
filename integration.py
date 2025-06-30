import cv2
import time
import json
import numpy as np
import pygame
from ultralytics import YOLO
from threading import Thread
from collections import defaultdict
from datetime import datetime

# Constants
DETECTION_TIME = 8
TOTAL_CYCLE_TIME = 120
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 30

# GUI setup
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
FRAME_SIZE = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Smart Traffic Signal GUI")
font = pygame.font.SysFont("Arial", 24)

class VehicleDetector:
    def __init__(self, video_path, aoi_config, direction_label):
        self.direction = direction_label
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(video_path)
        with open(aoi_config, "r") as f:
            config = json.load(f)
        self.aoi = np.array(config["AOI"], dtype=np.int32)
        self.vehicle_weights = {'car': 1, 'motorcycle': 1, 'bus': 2, 'truck': 2, 'bicycle': 1}
        self.current_counts = {cls: 0 for cls in self.vehicle_weights}
        self.signal_status = '🔴'
        self.frame = None
        self.vehicle_id_counter = 0
        self.vehicle_ids = {}

    def detect_in_frame(self, frame):
        results = self.model(frame)[0]
        detections = []
        new_ids = {}
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name}-{new_ids[bbox]}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
        self.vehicle_ids = new_ids
        return detections

    def count_vehicles(self):
        self.current_counts = {cls: 0 for cls in self.vehicle_weights}
        start_time = time.time()
        while time.time() - start_time < DETECTION_TIME:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.frame = frame.copy()
            detections = self.detect_in_frame(self.frame)
            for cls, _, _ in detections:
                self.current_counts[cls] += 1
        print(f"[{self.direction}] Counts: {self.current_counts}")

    def get_weighted_score(self):
        return sum(self.vehicle_weights[cls] * self.current_counts[cls] for cls in self.current_counts)

    def get_display_frame(self):
        if self.frame is None:
            return np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        overlay = self.frame.copy()
        cv2.polylines(overlay, [self.aoi], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(overlay, f"{self.direction}: {self.signal_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if self.signal_status == "🟢" else (0, 0, 255), 2)
        return cv2.resize(overlay, FRAME_SIZE)

class TrafficSystem:
    def __init__(self, detectors):
        self.detectors = detectors
        self.fixed_order = ['North', 'East', 'South', 'West']

    def gather_all_weights(self):
        scores = {}
        def detect_and_score(detector):
            detector.count_vehicles()
            scores[detector.direction] = detector.get_weighted_score()

        threads = []
        for detector in self.detectors:
            t = Thread(target=detect_and_score, args=(detector,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
        return scores

def draw_gui(detectors, remaining_cycle_time, cycle_label):
    frames = [cv2.cvtColor(d.get_display_frame(), cv2.COLOR_BGR2RGB) for d in detectors]
    surfaces = [pygame.surfarray.make_surface(np.rot90(f)) for f in frames]
    screen.blit(surfaces[0], (0, 0))  # North
    screen.blit(surfaces[1], (FRAME_SIZE[0], 0))  # East
    screen.blit(surfaces[2], (0, FRAME_SIZE[1]))  # South
    screen.blit(surfaces[3], (FRAME_SIZE[0], FRAME_SIZE[1]))  # West

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
    system = TrafficSystem(detectors)

    cycle_number = 1

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            scores = system.gather_all_weights()
            remaining_time = TOTAL_CYCLE_TIME
            cycle_label = get_cycle_label(cycle_number)

            for dir_label in system.fixed_order:
                if remaining_time <= 0:
                    break
                green_time = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, int(scores.get(dir_label, 10))))
                green_time = min(green_time, remaining_time)
                remaining_time -= green_time

                for d in detectors:
                    d.signal_status = "🟢" if d.direction == dir_label else "🔴"

                for sec in range(green_time, 0, -1):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    draw_gui(detectors, remaining_time + sec, cycle_label)
                    print(f"🟢 {dir_label} - {sec} sec", end="\r")
                    time.sleep(1)

                for d in detectors:
                    if d.direction == dir_label:
                        d.signal_status = "🟡"
                draw_gui(detectors, remaining_time, cycle_label)
                time.sleep(2)
                for d in detectors:
                    if d.direction == dir_label:
                        d.signal_status = "🔴"

            cycle_number += 1

    except KeyboardInterrupt:
        print("\n🛑 Traffic system stopped.")
        pygame.quit()
        cv2.destroyAllWindows()

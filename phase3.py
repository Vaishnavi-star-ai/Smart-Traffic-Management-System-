import cv2
import json
import numpy as np
import random
import time
import math
import pygame
import sys
from collections import deque
from ultralytics import YOLO
from threading import Thread

# Initialize pygame for signal display
pygame.init()
signal_display = pygame.display.set_mode((200, 200))
pygame.display.set_caption("Traffic Signal")

# Constants
DEFAULT_RED = 150
DEFAULT_YELLOW = 5
DEFAULT_GREEN = 20
DEFAULT_MIN = 10
DEFAULT_MAX = 60
DETECTION_TIME = 5
SIMULATION_TIME = 120  # 2 minutes

class TrafficSignal:
    def __init__(self, red, yellow, green, min_time, max_time):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.min_time = min_time
        self.max_time = max_time
        self.signal_text = str(green)
        self.total_green_time = 0
        self.default_green = green
        
    def reset(self):
        self.green = self.default_green
        self.yellow = DEFAULT_YELLOW
        self.red = DEFAULT_RED
        self.signal_text = str(self.green)

class VehicleDetector:
    def __init__(self):
        # Initialize YOLO model
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            sys.exit(1)
        
        # Initialize AOI with error handling
        self.aoi = self._initialize_aoi()
        
        # Initialize video capture
        self.cap = self._initialize_video_capture()
        
        # Vehicle tracking parameters
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
        self.vehicle_weights = {'car': 1, 'motorcycle': 0.5, 'bus': 2, 'truck': 2}
        self.id_colors = {}
        self.vehicle_tracks = {}
        self.unique_vehicle_ids_in_aoi = set()
        self.MAX_TRACK_LENGTH = 30
        self.gap_threshold = 40
        self.stop_line = [(300, 400), (600, 400)]
        self.stop_line_y = 400
        
        # Detection parameters
        self.current_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.counting = False
        self.last_check = 0
        self.start_time = time.time()
        self.running = True
    
    def _initialize_aoi(self):
        """Initialize Area of Interest with proper error handling"""
        default_aoi = np.array([[300,400], [600,400], [600,600], [300,600]], dtype=np.int32)
        
        try:
            with open("area_config.json", "r") as f:
                config = json.load(f)
                if "AOI" in config:
                    return np.array(config["AOI"], dtype=np.int32)
                elif "AOIs" in config and len(config["AOIs"]) > 0:
                    return np.array(config["AOIs"][0]["points"], dtype=np.int32)
                else:
                    print("⚠️ No valid AOI configuration found, using defaults")
                    return default_aoi
        except FileNotFoundError:
            print("⚠️ area_config.json not found, using default AOI")
            return default_aoi
        except Exception as e:
            print(f"⚠️ Error loading AOI config: {e}, using defaults")
            return default_aoi
    
    def _initialize_video_capture(self):
        """Initialize video capture with error handling"""
        try:
            cap = cv2.VideoCapture("videos/road_123.mp4")
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            return cap
        except Exception as e:
            print(f"❌ Error initializing video capture: {e}")
            sys.exit(1)

    def get_color(self, track_id):
        if track_id not in self.id_colors:
            self.id_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
        return self.id_colors[track_id]

    def is_inside_aoi(self, x, y):
        return cv2.pointPolygonTest(self.aoi, (int(x), int(y)), False) >= 0

    def is_gap_clear(self, cx, cy, others, threshold=None):
        threshold = threshold or self.gap_threshold
        return all(not (abs(cx - ox) < 50 and 0 < oy - cy < threshold) 
                  for ox, oy in others)

    def infer_turn(self, track):
        if len(track) < 2:
            return "unknown"
        dx = track[-1][0] - track[0][0]
        dy = track[-1][1] - track[0][1]
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "straight"

    def process_frame(self):
        if not self.running:
            return None, None
            
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return None, None
        
        now = time.time()
        active_centers = []
        self.current_counts = {k: 0 for k in self.current_counts}
        
        if now - self.last_check > self.green_time:
            self.counting = True
            self.last_check = now
        
        results = self.model.track(frame, persist=True)[0]
        
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                if class_name not in self.vehicle_classes:
                    continue
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_id = int(box.id[0]) if box.id is not None else None
                
                if track_id is not None and self.is_inside_aoi(cx, cy):
                    active_centers.append((cx, cy))
                    self.unique_vehicle_ids_in_aoi.add(track_id)
                    self.vehicle_tracks.setdefault(track_id, deque(maxlen=self.MAX_TRACK_LENGTH)).append((cx, cy))
                    direction = self.infer_turn(self.vehicle_tracks[track_id])
                    
                    if cy < self.stop_line_y and not self.is_gap_clear(cx, cy, active_centers):
                        continue
                    
                    color = self.get_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 20), 0, 0.5, color, 2)
                    cv2.putText(frame, f"Turn: {direction}", (x1, y1 - 5), 0, 0.5, color, 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    
                    pts = list(self.vehicle_tracks[track_id])
                    for i in range(1, len(pts)):
                        if self.is_inside_aoi(*pts[i]) and self.is_inside_aoi(*pts[i - 1]):
                            cv2.line(frame, pts[i - 1], pts[i], color, 2)
                    
                    if self.counting:
                        self.current_counts[class_name] += 1
        
        cv2.polylines(frame, [self.aoi], True, (255, 0, 0), 2)
        cv2.line(frame, self.stop_line[0], self.stop_line[1], (0, 255, 255), 2)
        cv2.putText(frame, "Stop Line", (self.stop_line[0][0], self.stop_line[0][1] - 10), 0, 0.6, (0, 255, 255), 2)
        
        return frame, self.current_counts

    def calculate_green_time(self):
        total_weight = sum(self.vehicle_weights[cls] * count 
                          for cls, count in self.current_counts.items())
        lanes = 2
        green_time = int(max(DEFAULT_MIN, min((total_weight * 2) / (lanes + 1), DEFAULT_MAX)))
        return green_time

    def run_detection(self):
        while self.running:
            frame, counts = self.process_frame()
            if frame is None:
                break
            
            elapsed = time.time() - self.start_time
            remaining = max(0, int(SIMULATION_TIME - elapsed))
            
            cv2.putText(frame, f"Green Time: {self.green_time}s", (20, 30), 0, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Left: {remaining//60:02}:{remaining%60:02}", (20, 60), 0, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Vehicles in AOI: {len(self.unique_vehicle_ids_in_aoi)}", (20, 90), 0, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Smart Traffic Management", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if elapsed >= SIMULATION_TIME:
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

class TrafficSystem:
    def __init__(self):
        self.signals = [
            TrafficSignal(0, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MIN, DEFAULT_MAX),
            TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MIN, DEFAULT_MAX),
            TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MIN, DEFAULT_MAX),
            TrafficSignal(DEFAULT_RED, DEFAULT_YELLOW, DEFAULT_GREEN, DEFAULT_MIN, DEFAULT_MAX)
        ]
        
        self.current_green = 0
        self.next_green = 1
        self.current_yellow = False
        self.time_elapsed = 0
        self.green_time = DEFAULT_GREEN
        
        self.detector = VehicleDetector()
        self.detector.green_time = self.green_time
        self.running = True
        
    def run_signal_cycle(self):
        while self.running and self.time_elapsed < SIMULATION_TIME:
            # Green phase
            while self.signals[self.current_green].green > 0 and self.running:
                self.update_signal_text(self.current_green)
                self.update_signal_timers()
                
                if self.signals[self.next_green].red == DETECTION_TIME:
                    self.adjust_green_time()
                
                time.sleep(1)
                self.time_elapsed += 1
                if self.time_elapsed >= SIMULATION_TIME:
                    self.running = False
                    break
            
            if not self.running:
                break
                
            # Yellow phase
            self.current_yellow = True
            self.update_signal_text(self.current_green)
            while self.signals[self.current_green].yellow > 0 and self.running:
                self.update_signal_timers()
                time.sleep(1)
                self.time_elapsed += 1
                if self.time_elapsed >= SIMULATION_TIME:
                    self.running = False
                    break
            
            if not self.running:
                break
                
            # Transition to next signal
            self.current_yellow = False
            self.signals[self.current_green].reset()
            self.current_green = self.next_green
            self.next_green = (self.current_green + 1) % 4
            self.signals[self.next_green].red = (
                self.signals[self.current_green].yellow + 
                self.signals[self.current_green].green
            )
    
    def adjust_green_time(self):
        counts = self.detector.current_counts
        total_weight = (
            counts['car'] * 1 + 
            counts['motorcycle'] * 0.5 + 
            counts['bus'] * 2 + 
            counts['truck'] * 2
        )
        
        lanes = 2
        new_green = math.ceil(total_weight / (lanes + 1))
        new_green = max(DEFAULT_MIN, min(DEFAULT_MAX, new_green))
        
        self.signals[self.next_green].green = new_green
        self.signals[self.next_green].default_green = new_green
        self.green_time = new_green
        self.detector.green_time = new_green
        
        print(f"Adjusted green time to {new_green}s based on traffic")
    
    def update_signal_timers(self):
        for i in range(4):
            if i == self.current_green:
                if self.current_yellow:
                    self.signals[i].yellow -= 1
                    self.signals[i].signal_text = str(self.signals[i].yellow) if self.signals[i].yellow > 0 else "STOP"
                else:
                    self.signals[i].green -= 1
                    self.signals[i].total_green_time += 1
                    self.signals[i].signal_text = str(self.signals[i].green) if self.signals[i].green > 0 else "SLOW"
            else:
                self.signals[i].red -= 1
                self.signals[i].signal_text = str(self.signals[i].red) if self.signals[i].red <= 10 else "---"
    
    def update_signal_text(self, signal_index):
        signal = self.signals[signal_index]
        if signal.green > 0 and not self.current_yellow:
            signal.signal_text = str(signal.green)
        elif signal.yellow > 0 and self.current_yellow:
            signal.signal_text = str(signal.yellow)
        else:
            signal.signal_text = "---"
    
    def display_signal(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
            
            signal_display.fill((0, 0, 0))
            
            if self.current_yellow:
                pygame.draw.circle(signal_display, (255, 255, 0), (100, 70), 30)
                text = f"YELLOW {self.signals[self.current_green].yellow}"
            elif self.signals[self.current_green].green > 0:
                pygame.draw.circle(signal_display, (0, 255,
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os

# Global variables
points = []
current_direction = "North"
directions = ["North", "East", "South", "West"]
aoi_dict = {}
video_paths = {
    "North": "videos/road_123.mp4",
    "East": "videos/road_traffic.mp4",
    "South": "videos/road_123.mp4",
    "West": "videos/road_traffic.mp4"
}

def load_video_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))
        return ImageTk.PhotoImage(Image.fromarray(frame))
    return None

def click(event):
    global points
    if len(points) < 4:
        points.append((event.x, event.y))
        canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="red")
        canvas.update()
    if len(points) == 4:
        canvas.create_polygon(points, outline="green", fill="")
        canvas.update()
        aoi_dict[current_direction] = points
        points = []
        next_direction()

def next_direction():
    global current_direction, points
    if directions:
        current_direction = directions.pop(0)
        label.config(text=f"Click 4 points for {current_direction}")
        video_path = video_paths.get(current_direction)
        frame_image = load_video_frame(video_path)
        if frame_image:
            canvas.delete("all")
            canvas.config(width=640, height=360)
            canvas.create_image(0, 0, anchor=tk.NW, image=frame_image)
            canvas.image = frame_image  # Keep a reference
        else:
            canvas.delete("all")
            canvas.create_text(320, 180, text=f"No video for {current_direction}", fill="white")
    else:
        save_aoi()
        root.destroy()

def save_aoi():
    if aoi_dict:
        with open("aoi_all_directions.json", "w") as f:
            json.dump(aoi_dict, f, indent=4)
        print("✅ AOI coordinates saved to aoi_all_directions.json")
    else:
        print("❌ No AOI data to save!")

def undo(event):
    global points
    if points:
        points.pop()
        canvas.delete("all")
        frame_image = load_video_frame(video_paths[current_direction])
        if frame_image:
            canvas.create_image(0, 0, anchor=tk.NW, image=frame_image)
            canvas.image = frame_image
        for p in points:
            canvas.create_oval(p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5, fill="red")
        canvas.update()

# Setup GUI
root = tk.Tk()
root.title("Set AOI Coordinates")
canvas = tk.Canvas(root, width=640, height=360, bg="black")
canvas.pack()
label = tk.Label(root, text=f"Click 4 points for {current_direction}", fg="white", bg="black")
label.pack()

# Bind events
canvas.bind("<Button-1>", click)  # Left-click to add points
canvas.bind("<Button-3>", undo)   # Right-click to undo last point

# Start with the first direction
next_direction()

root.mainloop()
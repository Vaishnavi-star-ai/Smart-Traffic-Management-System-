import cv2
import json
import os

videos = {
    "North": "videos/road_123.mp4",
    "East": "videos/road_traffic.mp4",
    "South": "videos/road_123.mp4",
    "West": "videos/road_traffic.mp4"
}

output_dir = "."

def draw_polygon(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

def draw_aoi_for_direction(direction, video_path):
    global drawing, points
    points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Failed to load video for {direction}")
        return

    clone = frame.copy()
    cv2.namedWindow(f"Draw AOI - {direction}")
    cv2.setMouseCallback(f"Draw AOI - {direction}", draw_polygon)

    while True:
        temp = clone.copy()
        for point in points:
            cv2.circle(temp, point, 4, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(temp, [np.array(points)], False, (0, 255, 0), 2)

        cv2.imshow(f"Draw AOI - {direction}", temp)
        key = cv2.waitKey(1)

        if key == 13:  # Enter to finish
            break
        elif key == 27:  # ESC to cancel
            points = []
            break

    cv2.destroyAllWindows()

    if len(points) >= 3:
        json_path = os.path.join(output_dir, f"aoi_{direction.lower()}.json")
        with open(json_path, "w") as f:
            json.dump({"AOI": points}, f)
        print(f"✅ AOI for {direction} saved to {json_path}")
    else:
        print(f"❌ Not enough points to form AOI for {direction}!")

if __name__ == "__main__":
    import numpy as np
    for direction, video_path in videos.items():
        draw_aoi_for_direction(direction, video_path)

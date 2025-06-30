import cv2
import json
import numpy as np

# === CONFIG ===
video_paths = {
    "North": "videos/road_123.mp4",
    "East": "videos/road_traffic.mp4",
    "South": "videos/road_123.mp4",
    "West": "videos/road_traffic.mp4"
}
output_json_file = "aoi_all_directions.json"
resize_to = (640, 360)  # Match the detection code's resize dimensions

aoi_data = {}

# === Mouse Handling ===
aoi_points = []
drawing_done = False

def draw_points(event, x, y, flags, param):
    global aoi_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        aoi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if aoi_points:
            aoi_points.pop()
    elif event == cv2.EVENT_MBUTTONDOWN:
        drawing_done = True

def draw_aoi_for_direction(direction, video_path):
    global aoi_points, drawing_done
    print(f"\n🎯 Drawing AOI for {direction}...")

    aoi_points = []
    drawing_done = False

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Couldn't load video: {video_path}")
        return

    frame = cv2.resize(frame, resize_to)
    clone = frame.copy()

    cv2.namedWindow(f"Draw AOI: {direction}")
    cv2.setMouseCallback(f"Draw AOI: {direction}", draw_points)

    while True:
        temp_frame = clone.copy()
        if len(aoi_points) > 0:
            cv2.polylines(temp_frame, [np.array(aoi_points)], isClosed=True, color=(0, 255, 0), thickness=2)
            for i, point in enumerate(aoi_points):
                cv2.circle(temp_frame, point, 4, (255, 0, 0), -1)
                cv2.putText(temp_frame, str(i+1), (point[0]+5, point[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(f"Draw AOI: {direction}", temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or drawing_done:  # ESC or middle click
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(aoi_points) >= 3:
        aoi_data[direction] = aoi_points.copy()
        print(f"✅ AOI for {direction} saved.")
    else:
        print(f"⚠️ AOI for {direction} not saved — need at least 3 points.")

def main():
    for direction, video in video_paths.items():
        draw_aoi_for_direction(direction, video)

    if len(aoi_data) == 4:
        with open(output_json_file, 'w') as f:
            json.dump(aoi_data, f, indent=4)
        print(f"\n🎉 All AOIs saved to {output_json_file}")
    else:
        print("\n⚠️ Not all AOIs were defined. JSON file not saved.")

if __name__ == "__main__":
    main()

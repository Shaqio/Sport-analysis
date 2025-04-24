# Sport-analysis
!pip install ultralytics deep_sort_realtime opencv-python matplotlib --quiet

#packs
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from IPython.display import display, clear_output
from PIL import Image
import os

#import files
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # uploaded file name
#
yolo_model = YOLO('yolov8n.pt')  # Use yolov8s.
tracker = DeepSort(max_age=30)

#

#Utility Functions
def draw_text(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

def show_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    display(img)
    clear_output(wait=True)

  #
  #Initialize Variables
cap = cv2.VideoCapture(video_path)
frame_count = 0
max_frames = 300
prev_positions = {}
total_distance = {}
last_possessor = None
pass_events = []

#
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output path and writer
output_path = '/content/processed_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
max_frames = 300  # Adjust for longer videos
prev_positions = {}
total_distance = {}

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]
    detections = []

    for r in results.boxes:
        if int(r.cls[0]) == 0:  # Only 'person' class
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            detections.append(((x1, y1, x2, y2), conf, 'person'))

    # Update DeepSort tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if tid in prev_positions:
            px, py = prev_positions[tid]
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            total_distance[tid] = total_distance.get(tid, 0) + dist
            cv2.line(frame, (px, py), (cx, cy), (0, 255, 0), 2)

        prev_positions[tid] = (cx, cy)
        draw_text(frame, f"Player {tid}", cx - 10, cy - 10)

    out.write(frame)  #Save the frame to the output video
    show_frame(frame)
    frame_count += 1

cap.release()
out.release()

#
from google.colab import files
files.download(output_path)

#
#Summary of Distance Covered
print("\n🏃 Distance Covered (in pixels):")
for tid, dist in total_distance.items():
    print(f"Player {tid}: {dist:.2f} px")

#Summary of Passes
print("\n🔁 Passes Detected:")
for i, (p1, p2) in enumerate(pass_events, 1):
    print(f"{i}. Player {p1} ➡️ Player {p2}")
#
# Store all player center points
heatmap_points = []
#
heatmap_points.append((cx, cy))
#
from google.colab import drive
drive.mount('/content/drive')
#
import matplotlib.pyplot as plt

# Separate x and y coordinates
xs, ys = zip(*heatmap_points)

# Create 2D histogram (heatmap)
heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(64, 64))

# Normalize and display the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
plt.title("Player Movement Heatmap")
plt.xlabel("Field Width (pixels)")
plt.ylabel("Field Height (pixels)")
plt.colorbar(label='Frequency')
plt.show()


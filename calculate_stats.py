import cv2
import math
import numpy as np
from ultralytics import YOLO

# Load model on CPU
model = YOLO("yolov8n.pt").to("cpu")

TARGET_ID = 40
video_path = "test_match.mp4"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30

# --- STATS CALCULATION VARIABLES ---
last_position = None
total_distance_pixels = 0
speeds_kh = []

# Scaling Factor: Estimates how many pixels equal 1 real-world meter on the pitch.
# For a standard broadcast view, roughly 20-25 pixels equal 1 meter.
PIXELS_PER_METER = 22.0 

print(f"📈 Extracting athletic performance stats for Player ID {TARGET_ID}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False, device="cpu")

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            # Isolate our target player
            if class_id == 0 and track_id == TARGET_ID:
                x1, y1, x2, y2 = map(int, box)
                
                # Get current position (bottom center center of box)
                current_position = (int((x1 + x2) / 2), y2)
                
                if last_position is not None:
                    # Distance Formula: sqrt((x2-x1)^2 + (y2-y1)^2)
                    pixel_distance = math.sqrt(
                        (current_position[0] - last_position[0])**2 + 
                        (current_position[1] - last_position[1])**2
                    )
                    
                    # Accumulate total distance
                    total_distance_pixels += pixel_distance
                    
                    # Calculate live speed for this specific frame transition
                    meters_moved = pixel_distance / PIXELS_PER_METER
                    seconds_elapsed = 1.0 / fps
                    
                    # Speed = Distance / Time (meters per second)
                    meters_per_second = meters_moved / seconds_elapsed
                    
                    # Convert meters/sec to kilometers per hour (multiply by 3.6)
                    speed_kh = meters_per_second * 3.6
                    
                    # Filter out sudden extreme tracker glitch jumps (> 40 km/h is humanly impossible)
                    if speed_kh < 40:
                        speeds_kh.append(speed_kh)
                
                last_position = current_position

cap.release()

print("\n=== FINAL ATHLETIC PERFORMANCE PROFILE ===")

if speeds_kh:
    # 1. Calculate Distance Covered
    total_meters_covered = total_distance_pixels / PIXELS_PER_METER
    
    # 2. Extract Top Speed
    top_speed = max(speeds_kh)
    
    # 3. Calculate Average Speed
    avg_speed = sum(speeds_kh) / len(speeds_kh)
    
    # 4. Calculate Stamina Index (Percentage of match spent actively running vs standing/walking)
    # High-intensity running is typically classified as anything above 12 km/h
    high_intensity_frames = sum(1 for s in speeds_kh if s > 12.0)
    stamina_index = (high_intensity_frames / len(speeds_kh)) * 100

    print(f"🏃‍♂️ Total Distance Covered: {total_meters_covered:.2f} meters")
    print(f"⚡ Top Sprint Speed: {top_speed:.1f} km/h")
    print(f"🏃‍♂️ Average Match Velocity: {avg_speed:.1f} km/h")
    print(f"🔋 Physical Stamina Index: {stamina_index:.1f}/100")
    
else:
    print("❌ Could not accumulate enough movement vectors to compile tracking metrics.")
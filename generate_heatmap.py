import cv2
import numpy as np
from ultralytics import YOLO

# Load model on CPU
model = YOLO("yolov8n.pt").to("cpu")

# Target ID locked from Step 1
TARGET_ID = 40
video_path = "test_match.mp4"

cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a blank matrix to accumulate tracking coordinates
heatmap_matrix = np.zeros((height, width), dtype=np.float32)

print(f"🏃‍♂️ Processing movement matrix for Player ID {TARGET_ID}...")

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
            # Check ONLY for our target player
            if class_id == 0 and track_id == TARGET_ID:
                x1, y1, x2, y2 = map(int, box)
                
                # Get the center bottom coordinate (where their feet touch the pitch)
                cx = int((x1 + x2) / 2)
                cy = y2 
                
                # Accumulate their presence on the map by drawing a small point
                cv2.circle(heatmap_matrix, (cx, cy), 20, 1, -1)

cap.release()

print("📊 Compiling tracking matrix into visual heatmap overlay...")

# Normalize data to image range (0-255)
if np.max(heatmap_matrix) > 0:
    normalized = cv2.normalize(heatmap_matrix, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    
    # Blur the intersections to create smooth heat gradients instead of harsh dots
    blurred = cv2.GaussianBlur(normalized, (41, 41), 0)
    
    # Convert data intensity to a Jet Colormap (Blue = cold, Red = hot activity)
    color_heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    
    # Load the background football pitch image
    pitch_bg = cv2.imread("2d_football_pitch.png")
    if pitch_bg is not None:
        pitch_bg = cv2.resize(pitch_bg, (width, height))
        # Blend the heatmap layer transparently over the field image
        final_output = cv2.addWeighted(pitch_bg, 0.6, color_heatmap, 0.4, 0)
    else:
        print("⚠️ '2d_football_pitch.png' not found. Saving raw heatmap instead.")
        final_output = color_heatmap

    # Save the final image asset
    cv2.imwrite("player_40_heatmap.png", final_output)
    print("✅ Success! Your tracking heatmap has been saved as 'player_40_heatmap.png'.")
else:
    print("❌ Error: Player ID 40 was not found consistently enough to plot data maps.")
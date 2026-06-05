import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Initialize YOLO and the EasyOCR reader (configured for English/Numbers)
model = YOLO("yolov8n.pt").to("cpu")
reader = easyocr.Reader(['en'], gpu=False)

video_path = "test_match.mp4"
TARGET_JERSEY_NUMBER = "10"  # Mbappé's number

cap = cv2.VideoCapture(video_path)
target_track_id = None
frame_count = 0

print(f"⚡ Jersey Detection Engine Started. Searching for No. {TARGET_JERSEY_NUMBER}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success or target_track_id is not None:
        break

    frame_count += 1
    
    # Run YOLO tracking frame-by-frame
    results = model.track(frame, persist=True, verbose=False, device="cpu")

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id == 0:  # Player
                x1, y1, x2, y2 = map(int, box)
                h = y2 - y1
                
                # Crop the upper body/torso area where the jersey number is printed
                torso_crop = frame[y1 + int(h*0.15):y1 + int(h*0.65), x1:x2]
                
                if torso_crop.size > 0:
                    # Convert to grayscale to make reading numbers easier for the AI
                    gray_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Run text recognition on the torso crop
                    ocr_results = reader.readtext(gray_crop)
                    
                    for (bbox, text, prob) in ocr_results:
                        # Clean up the text string (remove accidental spaces)
                        cleaned_text = "".join(text.split())
                        
                        if cleaned_text == TARGET_JERSEY_NUMBER and prob > 0.35:
                            target_track_id = track_id
                            print(f"\n🎯 TARGET PLAYER LOCATED VIA JERSEY NUMBER!")
                            print(f"Confirmed No. {TARGET_JERSEY_NUMBER} at Frame {frame_count}.")
                            print(f"YOLO Tracking ID locked onto: {target_track_id}")
                            break
                    
                    if target_track_id is not None:
                        break

cap.release()

if target_track_id is None:
    print(f"\n❌ Finished scanning. Jersey No. {TARGET_JERSEY_NUMBER} was not picked up clearly by the text reader.")
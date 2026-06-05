import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "test_match.mp4"
cap = cv2.get_video_channel_rect(video_path) if hasattr(cv2, 'get_video_channel_rect') else cv2.VideoCapture(video_path)

POSSESSION_THRESHOLD_PIXELS = 85  

# Dictionary to save team assignments permanently for each Track ID
# This prevents a player from flickering teams between frames
track_team_memory = {}

try:
    font_main = ImageFont.truetype("arial.ttf", 20)
    font_bold = ImageFont.truetype("arialbd.ttf", 22)
except IOError:
    font_main = ImageFont.load_default()
    font_bold = ImageFont.load_default()

def detect_team_by_jersey(player_crop):
    """
    Analyzes the jersey crop area and determines if it leans towards
    a Red-dominant jersey or a Blue-dominant jersey.
    """
    if player_crop.size == 0:
        return "UNKNOWN"
        
    # Convert crop to HSV color space
    hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
    
    # Define broad masks for Red jerseys (handles two wraps on Hue spectrum)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Define broad mask for Blue jerseys
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    
    if red_pixels > blue_pixels and red_pixels > 20:
        return "TEAM_RED"
    elif blue_pixels > red_pixels and blue_pixels > 20:
        return "TEAM_BLUE"
    return "TEAM_OTHER"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False)

    players = {}
    ball_pos = None

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Step 5a: Analyze team jersey colors
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if class_id == 0:  # Player
                # Check if we already recognized this player's team in a past frame
                if track_id not in track_team_memory:
                    # Crop the upper torso area of the player to target the jersey
                    height = y2 - y1
                    jersey_crop = frame[y1 + int(height*0.1):y1 + int(height*0.5), x1:x2]
                    
                    team = detect_team_by_jersey(jersey_crop)
                    if team != "UNKNOWN":
                        track_team_memory[track_id] = team
                    else:
                        track_team_memory[track_id] = "TEAM_OTHER"
                
                assigned_team = track_team_memory.get(track_id, "TEAM_OTHER")
                players[track_id] = (center_x, center_y, x1, y1, x2, y2, assigned_team)

                # Set UI indicator colors based on team assignment
                if assigned_team == "TEAM_RED":
                    color = (0, 0, 255)       # Red dot
                elif assigned_team == "TEAM_BLUE":
                    color = (255, 0, 0)       # Blue dot
                else:
                    color = (128, 128, 128)   # Grey for Referees/Unassigned
                    
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

            elif class_id == 32:  # Ball
                ball_pos = (center_x, center_y)
                cv2.circle(frame, (center_x, center_y), 6, (255, 255, 0), -1)

        # Convert Frame to PIL Image to update tactical HUD text
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Top Analytical HUD Panel
        draw.rectangle([10, 10, 560, 65], fill=(15, 23, 42, 220), outline=(51, 65, 85, 255), width=2)
        draw.ellipse([25, 31, 37, 43], fill=(0, 255, 127, 255))
        draw.text((48, 24), "TACTICAL AI ENGINE :", font=font_main, fill=(148, 163, 184))

        status_text = "CONTESTED BALL"
        status_color = (226, 232, 240)

        # Calculate Tactical Possession By Team
        if ball_pos is not None and players:
            bx, by = ball_pos
            closest_player_id = None
            min_distance = float('inf')

            for p_id, (px, py, *_, p_team) in players.items():
                distance = math.sqrt((bx - px)**2 + (by - py)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_player_id = p_id

            if min_distance <= POSSESSION_THRESHOLD_PIXELS:
                p_data = players[closest_player_id]
                active_team = p_data[6]
                
                if active_team == "TEAM_RED":
                    status_text = f"RED TEAM POSSESSION (P{closest_player_id})"
                    status_color = (255, 100, 100)
                elif active_team == "TEAM_BLUE":
                    status_text = f"BLUE TEAM POSSESSION (P{closest_player_id})"
                    status_color = (100, 149, 237)
                else:
                    status_text = f"POSSESSION: PLAYER {closest_player_id}"
                    status_color = (249, 115, 22)

                # Draw Target corner brackets on active player
                px1, py1, px2, py2 = p_data[2], p_data[3], p_data[4], p_data[5]
                length = 15
                draw.line([(px1, py1), (px1 + length, py1)], fill=status_color, width=3)
                draw.line([(px1, py1), (px1, py1 + length)], fill=status_color, width=3)
                draw.line([(px2, py2), (px2 - length, py2)], fill=status_color, width=3)
                draw.line([(px2, py2), (px2, py2 - length)], fill=status_color, width=3)

        draw.text((265, 22), status_text, font=font_bold, fill=status_color)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Advanced Tactical AI Dashboard", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
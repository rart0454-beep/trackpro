import os
import cv2
import math
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from deepface import DeepFace
from groq import Groq

app = FastAPI(title="TrackPro AI RockSolid Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://172.24.160.1:5500",  # ← Add your actual machine IP
        "http://172.24.160.1:8000",
        "*"  # Temporarily allow all during dev/debugging
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/heatmaps", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 💡 Make sure your actual Groq key string is inside the quotes!
GROQ_CLIENT = Groq(api_key="GROQ_API_KEY")
model = YOLO("yolov8n.pt").to("cpu")

def generate_ai_scout_report(idx, dist, top_spd, avg_spd, stamina):
    try:
        prompt = f"""
        Analyze these football tracking metrics for Player Target #{idx}:
        - Distance Covered: {dist} meters
        - Top Speed: {top_spd} km/h
        - Avg Velocity: {avg_spd} km/h
        - Intensity Index: {stamina}/100

        Write a professional scouting profile without using hash marks (###). Use simple double asterisks for labels.
        Format EXACTLY like this text template layout:
        **TACTICAL ARCHETYPE**: [Archetype summary based on speed]
        
        **KEY PERFORMANCE STRENGTHS**:
        • Burst Output: Hitting {top_spd} km/h allows them to exploit transition spaces.
        • Sustained Pace: Maintaining {avg_spd} km/h pressures defensive lines.
        
        **TACTICAL LIMITATIONS & RISK FACTORS**:
        • Recovery Phase: A {stamina}/100 intensity rating dictates structural positioning.
        """
        completion = GROQ_CLIENT.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return "AI Tactical Scout Report compilation delayed."

@app.post("/analyze-multiple-players/")
async def analyze_multiple_players(
    video: UploadFile = File(...),
    face_images: list[UploadFile] = File(...)
):

    video_path = os.path.join("uploads", video.filename)
    with open(video_path, "wb") as f: 
        f.write(await video.read())

    saved_face_paths = []
    for idx, face_img in enumerate(face_images):
        path = os.path.join("uploads", f"target_{idx}_{face_img.filename}")
        with open(path, "wb") as f: 
            f.write(await face_img.read())
        saved_face_paths.append(path)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30

    target_map = {}
    success, initial_frame = cap.read()
    
    if success:
        results = model.track(initial_frame, persist=True, verbose=False, device="cpu")
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for face_idx, face_path in enumerate(saved_face_paths):
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id == 0:
                        x1, y1, x2, y2 = map(int, box)
                        crop = initial_frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            try:
                                res = DeepFace.verify(img1_path=face_path, img2_path=crop, model_name="VGG-Face", enforce_detection=False)
                                if res["verified"] or res["distance"] < 0.65:
                                    target_map[track_id] = face_idx
                                    break
                            except Exception: pass

    if not target_map:
        for i in range(len(saved_face_paths)):
            target_map[i + 1] = i

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tracking_data = {}
    for track_id, face_idx in target_map.items():
        tracking_data[track_id] = {
            "face_index": face_idx,
            "heatmap_matrix": np.zeros((height, width), dtype=np.float32),
            "last_position": None,
            "speeds_kh": [],
            "total_distance_pixels": 0
        }

    PIXELS_PER_METER = 22.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, verbose=False, device="cpu")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id == 0 and track_id in tracking_data:
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    p = tracking_data[track_id]
                    cv2.circle(p["heatmap_matrix"], (cx, cy), 20, 1, -1)
                    
                    if p["last_position"] is not None:
                        pixel_dist = math.sqrt((cx - p["last_position"][0])**2 + (cy - p["last_position"][1])**2)
                        p["total_distance_pixels"] += pixel_dist
                        speed_kh = ((pixel_dist / PIXELS_PER_METER) / (1.0 / fps)) * 3.6
                        if speed_kh < 45: p["speeds_kh"].append(speed_kh)
                        
                    p["last_position"] = (cx, cy)
    cap.release()

    output_profiles = []
    
    # 💡 SAFE HEATMAP GENERATION ENGINE
    # Try reading the image asset locally
    pitch_bg = cv2.imread("2d_football_pitch.png")
    
    # CRITICAL FALLBACK: If missing, build a clean digital pitch array so it NEVER crashes
    if pitch_bg is None:
        print("⚠️ Warning: '2d_football_pitch.png' not found. Generating digital canvas fallback layer...")
        pitch_bg = np.zeros((height, width, 3), dtype=np.uint8)
        pitch_bg[:] = (24, 43, 20) # Modern minimalist deep green background hex color
        # Draw a sleek center circle line geometry
        cv2.circle(pitch_bg, (int(width/2), int(height/2)), 60, (40, 60, 35), 2)
        cv2.rectangle(pitch_bg, (0, 0), (width, height), (40, 60, 35), 4)

    for track_id, data in tracking_data.items():
        if not data["speeds_kh"]: continue

        total_meters = round(data["total_distance_pixels"] / PIXELS_PER_METER, 2)
        top_speed = round(max(data["speeds_kh"]), 1)
        avg_speed = round(sum(data["speeds_kh"]) / len(data["speeds_kh"]), 1)
        stamina_index = round((sum(1 for s in data["speeds_kh"] if s > 12.0) / len(data["speeds_kh"])) * 100, 1)

        ai_report = generate_ai_scout_report(data["face_index"] + 1, total_meters, top_speed, avg_speed, stamina_index)

        heatmap_filename = f"heatmap_target_{track_id}.png"
        heatmap_save_path = os.path.join("static", "heatmaps", heatmap_filename)
        
        # Safe visualization blending matrix
        if np.max(data["heatmap_matrix"]) > 0:
            normalized = cv2.normalize(data["heatmap_matrix"], None, 0, 255, cv2.NORM_MINMAX)
            blurred = cv2.GaussianBlur(np.uint8(normalized), (41, 41), 0)
            color_heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            
            resized_bg = cv2.resize(pitch_bg, (width, height))
            final_output = cv2.addWeighted(resized_bg, 0.6, color_heatmap, 0.4, 0)
            cv2.imwrite(heatmap_save_path, final_output)
        else:
            cv2.imwrite(heatmap_save_path, cv2.resize(pitch_bg, (width, height)))

        output_profiles.append({
            "target_number": data["face_index"] + 1,
            "heatmap_url": f"http://127.0.0.1:8000/static/heatmaps/{heatmap_filename}",
            "analytics": {
                "distance_covered_meters": total_meters,
                "top_speed_kmh": top_speed,
                "average_speed_kmh": avg_speed,
                "stamina_rating": stamina_index
            },
            "ai_scout_review": ai_report
        })

    return {"status": "success", "players": sorted(output_profiles, key=lambda x: x["target_number"])}
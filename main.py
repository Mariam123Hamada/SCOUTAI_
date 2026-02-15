import cv2
import numpy as np
import time
import math
import os
import json
import mediapipe as mp
from pipeline import ConeDetection, PlayerAnalysis, SpeedAnalysis
from utils import draw_analytics_panel


BASE_DIR = os.getcwd()
model_path = os.path.join(BASE_DIR, "models", "best.pt")
video_path = os.path.join(BASE_DIR, "Data", "1_505.mp4")

if not os.path.exists(video_path):
    print(f"Error: Could not find video at {video_path}")


detector = ConeDetection(model_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 60

analysis = PlayerAnalysis(ratio_px2meter=5.0, fps=fps)
speed_analyzer = SpeedAnalysis(ratio_px2meter=5.0)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

frame_count = 0
start_time = time.time()

static_cones = None
gate_start_time = None
gate_speeds = []
gate_radius = 40
previous_gate_speed = None

c1_time = None
c2_time = None

turn_start_time = None
turn_efficiencies = []
turn_radius = 60

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
        )

    detections = detector.get_detections_with_colors(frame)
    ball_pos = None
    cone_positions = []

    for (cx, cy), label in detections.items():
        if label.lower() == "football":
            ball_pos = (cx, cy)
            cv2.circle(frame, (cx, cy), 8, (0, 165, 255), -1)
        else:
            cone_positions.append((cx, cy))
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    
    if static_cones is None and len(cone_positions) >= 2:
        static_cones = cone_positions[:2]
        analysis.calibration(static_cones)
        print(f"[INFO] Speed Gate Cones: {static_cones}")

    if not analysis.m_per_pixel and len(cone_positions) >= 2:
        analysis.calibration(cone_positions[:2])

    
    left_foot, right_foot = analysis.get_feet_positions(frame)
    lean_angle = analysis.get_body_lean(frame)

    
    if ball_pos and left_foot and right_foot:
        touch_detected = analysis.detect_real_touch(ball_pos, left_foot, right_foot)
        if touch_detected:
            cv2.putText(frame, "BALL TOUCHED!", (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

   
    gate_speed_text = previous_gate_speed
    if static_cones and ball_pos:
        cone1, cone2 = static_cones
        cv2.circle(frame, cone1, gate_radius, (255, 255, 255), 2)
        cv2.circle(frame, cone2, gate_radius, (255, 255, 255), 2)

        dist_to_c1 = math.dist(ball_pos, cone1)
        dist_to_c2 = math.dist(ball_pos, cone2)

        if dist_to_c1 < gate_radius and c1_time is None:
            c1_time = time.time()
        if dist_to_c2 < gate_radius and c1_time is not None:
            c2_time = time.time()
            speed_result = speed_analyzer.detect_speed(frame, cone1, cone2, c1_time, c2_time)
            if speed_result:
                speed_mps, speed_kmph = speed_result
                gate_speeds.append(speed_mps)
                gate_speed_text = speed_mps
                print(f"[INFO] Gate Speed: {speed_mps:.2f} m/s ({speed_kmph:.2f} km/h)")
            c1_time, c2_time = None, None
            previous_gate_speed = gate_speed_text

    
    if turn_cone := (min(cone_positions, key=lambda c: math.dist(c, ball_pos)) if ball_pos and cone_positions else None):
        dist_to_turn_cone = math.dist(ball_pos, turn_cone)
        if dist_to_turn_cone < turn_radius and turn_start_time is None:
            turn_start_time = time.time()
        if dist_to_turn_cone >= turn_radius and turn_start_time is not None:
            turn_duration = time.time() - turn_start_time
            if 0.01 < turn_duration < 50.0:
                turn_efficiencies.append(turn_duration)
            turn_start_time = None

    current_turn_eff = turn_efficiencies[-1] if turn_efficiencies else None

    
    frame = draw_analytics_panel(frame, lean_angle, analysis.touch_count,
                                 gate_speed_text, current_turn_eff)

    cv2.imshow("ScoutAI Pro Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Final Report ---
end_time = time.time()
cap.release()
cv2.destroyAllWindows()

drill_duration = frame_count / fps
peak_gate_speed = max(gate_speeds) if gate_speeds else 0
avg_gate_speed = np.mean(gate_speeds) if gate_speeds else 0

print("\n--- FINAL SCOUT REPORT ---")
print(f"Total Drill Time: {drill_duration:.2f}s")
print(f"Peak Gate Speed: {peak_gate_speed:.2f} m/s")
print(f"Average Gate Speed: {avg_gate_speed:.2f} m/s")
print(f"Total Ball Touches: {analysis.touch_count}")
print(f"Measured Drill Time: {end_time - start_time:.2f}s")

scout_report = {
    "Total Drill Time": f"{drill_duration:.2f}s",
    "Peak Gate Speed": f"{peak_gate_speed:.2f} m/s",
    "Average Gate Speed": f"{avg_gate_speed:.2f} m/s",
    "Total Ball Touches": analysis.touch_count,
    "Measured Drill Time": f"{end_time - start_time:.2f}s"
}

with open("scout_report.json", "w") as f:
    json.dump(scout_report, f, indent=4)

print("\n[INFO] Scout report saved to scout_report.json")

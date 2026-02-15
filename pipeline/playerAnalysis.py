import cv2
import math
import time
import mediapipe as mp

class PlayerAnalysis:
    def __init__(self, ratio_px2meter=5.0, fps=60, boundary_px=150):
        # Calibration
        self.dist_between_cones = ratio_px2meter
        self.fps = fps if fps > 0 else 60
        self.m_per_pixel = None

        # Ball tracking
        self.prev_ball_pos = None
        self.prev_velocity = 0

        # Touch detection
        self.touch_count = 0
        self.last_touch_time = 0

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # --------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------
    def calibration(self, cone_centers):
        if len(cone_centers) >= 2:
            c1, c2 = cone_centers[0], cone_centers[1]
            pixel_dist = math.dist(c1, c2)

            if pixel_dist > 0:
                self.m_per_pixel = self.dist_between_cones / pixel_dist
                print(f"[CALIBRATION SUCCESS] 1px = {self.m_per_pixel:.6f} meters")

    # --------------------------------------------------
    # GET FEET POSITIONS
    # --------------------------------------------------
    def get_feet_positions(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None

        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Left ankle (31)
        left_ankle = landmarks[31]
        left_pos = (int(left_ankle.x * w), int(left_ankle.y * h))

        # Right ankle (32)
        right_ankle = landmarks[32]
        right_pos = (int(right_ankle.x * w), int(right_ankle.y * h))

        return left_pos, right_pos

    # --------------------------------------------------
    # REAL TOUCH DETECTION
    # --------------------------------------------------
    def detect_real_touch(self, ball_pos, left_foot, right_foot,
                          foot_threshold=50, velocity_threshold=9.0, cooldown_frames=8):
        """
        Detects a real ball touch using:
        - Ball proximity to either foot
        - Significant velocity change
        - Cooldown frames to avoid multiple counts in consecutive frames
        """
        if ball_pos is None:
            return False

        now = time.time()

        # First frame, just initialize
        if self.prev_ball_pos is None:
            self.prev_ball_pos = ball_pos
            return False

        # Calculate ball movement since last frame
        px_dist = math.dist(ball_pos, self.prev_ball_pos)
        current_velocity = px_dist
        velocity_change = abs(current_velocity - self.prev_velocity)

        # Check if either foot is close enough
        foot_near = False
        if left_foot and math.dist(ball_pos, left_foot) < foot_threshold:
            foot_near = True
        elif right_foot and math.dist(ball_pos, right_foot) < foot_threshold:
            foot_near = True

        # Detect touch
        touch_detected = False
        if foot_near and velocity_change > velocity_threshold and (now - self.last_touch_time > cooldown_frames / self.fps):
            touch_detected = True
            self.touch_count += 1
            self.last_touch_time = now

        # Update previous velocity and position
        self.prev_velocity = current_velocity
        self.prev_ball_pos = ball_pos

        return touch_detected

    # --------------------------------------------------
    # BODY LEAN
    # --------------------------------------------------
    def get_body_lean(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return 0

        landmarks = results.pose_landmarks.landmark

        sh_x = (landmarks[11].x + landmarks[12].x) / 2
        sh_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2

        dx = sh_x - hip_x
        dy = sh_y - hip_y

        angle = abs(math.degrees(math.atan2(dx, dy)))

        return angle

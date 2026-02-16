import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ---------------- LOAD MODEL ----------------
pose_model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- SETTINGS ----------------

MOTION_WINDOW = 20
ARM_SPEED_THRESHOLD = 40
ACC_THRESHOLD = 20
SUSTAIN_FIGHT_FRAMES = 4

FATIGUE_SPEED_THRESHOLD = 10      # very low movement
FATIGUE_FRAMES = 40               # sustained low movement
SHOULDER_DROP_THRESHOLD = 25      # shoulder drop pixels

prev_keypoints = None
prev_arm_speed = 0

arm_speed_buffer = deque(maxlen=MOTION_WINDOW)

fight_counter = 0
fatigue_counter = 0

baseline_shoulder_y = None

# ---------------- FUNCTIONS ----------------

def compute_arm_speed(curr, prev):
    arm_points = [7,8,9,10]  # elbows + wrists
    return np.mean(np.linalg.norm(curr[arm_points] - prev[arm_points], axis=1))

def draw_banner(frame, text, color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 90), color, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, text, (40,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)

# ---------------- MAIN LOOP ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    alert_fight = False
    alert_fatigue = False

    results = pose_model(frame)

    if results[0].keypoints is not None:
        frame = results[0].plot()
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:
            kpts = keypoints[0]  # single person only

            # Shoulder indices (5,6)
            shoulder_y = np.mean([kpts[5][1], kpts[6][1]])

            if baseline_shoulder_y is None:
                baseline_shoulder_y = shoulder_y

            if prev_keypoints is not None:

                arm_speed = compute_arm_speed(kpts, prev_keypoints)
                arm_speed_buffer.append(arm_speed)

                avg_speed = np.mean(arm_speed_buffer)
                acceleration = arm_speed - prev_arm_speed

                # ---------------- FIGHT ----------------
                if avg_speed > ARM_SPEED_THRESHOLD or acceleration > ACC_THRESHOLD:
                    fight_counter += 1
                else:
                    fight_counter = 0

                if fight_counter > SUSTAIN_FIGHT_FRAMES:
                    alert_fight = True
                    fatigue_counter = 0  # reset fatigue if fighting

                # ---------------- FATIGUE ----------------
                shoulder_drop = shoulder_y - baseline_shoulder_y

                if avg_speed < FATIGUE_SPEED_THRESHOLD and shoulder_drop > SHOULDER_DROP_THRESHOLD:
                    fatigue_counter += 1
                else:
                    fatigue_counter = 0

                if fatigue_counter > FATIGUE_FRAMES:
                    alert_fatigue = True

                prev_arm_speed = arm_speed

            prev_keypoints = kpts

    # ---------------- DISPLAY ----------------

    if alert_fight:
        draw_banner(frame, "⚠ FIGHT DETECTED", (0,0,255))

    elif alert_fatigue:
        draw_banner(frame, "😫 FATIGUE DETECTED", (0,165,255))

    cv2.imshow("Fight + Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

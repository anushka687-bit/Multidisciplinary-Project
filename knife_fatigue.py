import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
import torch
from ultralytics import YOLO

# ---------------- YOLO KNIFE MODEL ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/detect/train3/weights/best.pt")
model.to(device)

# ---------------- MEDIAPIPE SETUP ----------------
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected")
    exit()

# ---------------- VARIABLES ----------------
EYE_CLOSED_FRAMES = 0
fatigue_timer = 0
FALLEN = False
knife_timer = 0

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face.process(rgb)
    pose_results = pose.process(rgb)

    h, w, _ = frame.shape

    # =====================================================
    # ---------------- KNIFE DETECTION --------------------
    # =====================================================
    results = model(frame, conf=0.5, verbose=False)[0]
    knife_detected = False

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # If single-class knife model
        if cls == 0:
            knife_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.putText(frame,
                        f"KNIFE {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

    if knife_detected:
        knife_timer = time.time() + 3
        winsound.Beep(1200, 300)

    if time.time() < knife_timer:
        cv2.putText(frame, "KNIFE DETECTED",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # =====================================================
    # ---------------- FATIGUE DETECTION ------------------
    # =====================================================
    if face_results.multi_face_landmarks:
        lm = face_results.multi_face_landmarks[0].landmark

        LEFT_EYE = np.array([[lm[i].x*w, lm[i].y*h] for i in [33,160,158,133,153,144]])
        RIGHT_EYE = np.array([[lm[i].x*w, lm[i].y*h] for i in [362,385,387,263,373,380]])

        ear = (eye_aspect_ratio(LEFT_EYE) +
               eye_aspect_ratio(RIGHT_EYE)) / 2

        if ear < 0.22:
            EYE_CLOSED_FRAMES += 1
        else:
            EYE_CLOSED_FRAMES = 0

        if EYE_CLOSED_FRAMES > 20:
            fatigue_timer = time.time() + 3
            winsound.Beep(1000, 300)

    if time.time() < fatigue_timer:
        cv2.putText(frame, "FATIGUE DETECTED", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # =====================================================
    # ---------------- FALL DETECTION ---------------------
    # =====================================================
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]

        sh_x, sh_y = int(shoulder.x * w), int(shoulder.y * h)
        hip_x, hip_y = int(hip.x * w), int(hip.y * h)

        dx = sh_x - hip_x
        dy = sh_y - hip_y
        angle = abs(math.degrees(math.atan2(dy, dx)))

        if angle < 20:
            if not FALLEN:
                winsound.Beep(1500, 400)
            FALLEN = True

        elif angle > 45:
            FALLEN = False

    if FALLEN:
        cv2.putText(frame, "FALL DETECTED", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # ---------------- DISPLAY ----------------
    cv2.imshow("Fatigue + Fall + Knife Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound   # Windows alarm

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

    # ---------- FATIGUE DETECTION ----------
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

    # ---------- FALL DETECTION (PERSISTENT) ----------
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]

        sh_x, sh_y = int(shoulder.x * w), int(shoulder.y * h)
        hip_x, hip_y = int(hip.x * w), int(hip.y * h)

        dx = sh_x - hip_x
        dy = sh_y - hip_y
        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Fallen if near horizontal
        if angle < 30 or angle > 150:
            if not FALLEN:
                winsound.Beep(1500, 400)
            FALLEN = True

        # Standing if near vertical
        elif 60 < angle < 120:
            FALLEN = False


    if FALLEN:
        cv2.putText(frame, "FALL DETECTED", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # ---------- DISPLAY ----------
    cv2.imshow("Fatigue & Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
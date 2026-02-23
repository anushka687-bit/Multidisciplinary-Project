import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
from ultralytics import YOLO

# --------------------------------------------------
# LOAD YOLO MODEL (Person Detection)
# --------------------------------------------------
model = YOLO("yolov8n.pt")

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()

# --------------------------------------------------
# VIDEO SOURCE (CCTV or Webcam)
# --------------------------------------------------
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/users/havis/Downloads/testfight.mp4")

if not cap.isOpened():
    print("Video not detected")
    exit()

# --------------------------------------------------
# VARIABLES
# --------------------------------------------------
prev_gray = None
fight_frames = 0
last_fight_beep = 0

MOTION_THRESHOLD = 450000
FRAME_THRESHOLD = 5
FIGHT_COOLDOWN = 3

EYE_CLOSED_FRAMES = 0
fatigue_timer = 0

FALLEN = False
fall_counter = 0
previous_hip_y = None
previous_body_height = None

# --------------------------------------------------
# EAR FUNCTION
# --------------------------------------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # ==================================================
    # 1️⃣ FIGHT DETECTION (UNCHANGED)
    # ==================================================
    results = model(frame, conf=0.4, verbose=False)[0]

    person_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:

        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.sum(diff)

        if person_count >= 2 and motion_score > MOTION_THRESHOLD:
            fight_frames += 1
        else:
            fight_frames = max(0, fight_frames - 1)

        if fight_frames >= FRAME_THRESHOLD:

            cv2.putText(frame, "FIGHT DETECTED",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            if time.time() - last_fight_beep > FIGHT_COOLDOWN:
                for _ in range(3):
                    winsound.Beep(1200, 600)
                    time.sleep(0.1)
                last_fight_beep = time.time()

    prev_gray = gray

    # ==================================================
    # 2️⃣ FATIGUE DETECTION (UNCHANGED)
    # ==================================================
    face_results = face.process(rgb)

    if face_results.multi_face_landmarks:
        lm = face_results.multi_face_landmarks[0].landmark

        LEFT_EYE = np.array([[lm[i].x*w, lm[i].y*h]
                             for i in [33,160,158,133,153,144]])
        RIGHT_EYE = np.array([[lm[i].x*w, lm[i].y*h]
                              for i in [362,385,387,263,373,380]])

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
        cv2.putText(frame, "FATIGUE DETECTED",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    # ==================================================
    # 3️⃣ ADVANCED FALL DETECTION (CCTV + SLIDE SAFE)
    # ==================================================
    pose_results = pose.process(rgb)

    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark

        left_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

        sh_y = int((left_sh.y + right_sh.y)/2 * h)
        hip_y = int((left_hip.y + right_hip.y)/2 * h)

        sh_x = int((left_sh.x + right_sh.x)/2 * w)
        hip_x = int((left_hip.x + right_hip.x)/2 * w)

        dx = sh_x - hip_x
        dy = sh_y - hip_y
        angle = abs(math.degrees(math.atan2(dy, dx)))

        body_height = abs(sh_y - hip_y)

        if previous_hip_y is not None:
            hip_velocity = hip_y - previous_hip_y
        else:
            hip_velocity = 0
        previous_hip_y = hip_y

        if previous_body_height is not None:
            collapse_ratio = body_height / (previous_body_height + 1)
        else:
            collapse_ratio = 1
        previous_body_height = body_height

        horizontal_body = (angle < 60 or angle > 120)
        fast_drop = hip_velocity > 8
        gradual_slide = hip_velocity > 3 and collapse_ratio < 0.85
        very_low_body = body_height < 35

        if horizontal_body or fast_drop or gradual_slide or very_low_body:
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        if fall_counter >= 5:
            if not FALLEN:
                winsound.Beep(1500, 1500)
            FALLEN = True

        if 80 < angle < 100 and body_height > 80:
            FALLEN = False
            fall_counter = 0

    if FALLEN:
        cv2.putText(frame, "FALL DETECTED",
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    # ==================================================
    # DISPLAY
    # ==================================================
    cv2.imshow("AI Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()
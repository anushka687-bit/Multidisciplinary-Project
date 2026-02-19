import cv2
import numpy as np
import torch
import mediapipe as mp
import math
import time
import winsound
from ultralytics import YOLO

# --------------------------------------------------
# DEVICE SETUP
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = YOLO("yolov8n.pt")  # Nano = faster
model.to(device)

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face = mp_face.FaceMesh(
    refine_landmarks=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5
)

# --------------------------------------------------
# VIDEO SOURCE
# --------------------------------------------------
#cap = cv2.VideoCapture("C:/Users/havis/Downloads/india1.mp4")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Video not detected")
    exit()

# --------------------------------------------------
# VARIABLES
# --------------------------------------------------
track_history = {}
FIGHT_DISTANCE = 120
VELOCITY_THRESHOLD = 12

EYE_CLOSED_FRAMES = 0
fatigue_timer = 0
FALLEN = False

last_beep_time = 0
BEEP_COOLDOWN = 3

frame_count = 0

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def safe_beep(freq=1000, duration=300):
    global last_beep_time
    if time.time() - last_beep_time > BEEP_COOLDOWN:
        winsound.Beep(freq, duration)
        last_beep_time = time.time()


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (480, 360))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    frame_count += 1

    # ==================================================
    # 1️⃣ FIGHT DETECTION (YOLO Tracking)
    # ==================================================
    results = model.track(
        frame,
        persist=True,
        conf=0.4,
        verbose=False,
        half=(device == "cuda")
    )[0]

    if results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            if int(cls) == 0:  # person only

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # Store history
                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))

                if len(track_history[track_id]) > 5:
                    track_history[track_id].pop(0)

    # ---- Fight Logic ----
    fight_detected = False

    ids_list = list(track_history.keys())

    if len(ids_list) >= 2:

        for i in range(len(ids_list)):
            for j in range(i + 1, len(ids_list)):

                p1 = track_history[ids_list[i]]
                p2 = track_history[ids_list[j]]

                if len(p1) >= 2 and len(p2) >= 2:

                    v1 = np.linalg.norm(np.array(p1[-1]) - np.array(p1[-2]))
                    v2 = np.linalg.norm(np.array(p2[-1]) - np.array(p2[-2]))

                    dist = np.linalg.norm(
                        np.array(p1[-1]) - np.array(p2[-1])
                    )

                    if v1 > VELOCITY_THRESHOLD and v2 > VELOCITY_THRESHOLD and dist < FIGHT_DISTANCE:
                        fight_detected = True

    if fight_detected:
        cv2.putText(frame, "FIGHT DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2)
        safe_beep(1200, 300)

    # ==================================================
    # 2️⃣ FATIGUE DETECTION (every 2 frames)
    # ==================================================
    if frame_count % 2 == 0:

        face_results = face.process(rgb)

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark

            LEFT = np.array([[lm[i].x*w, lm[i].y*h]
                             for i in [33,160,158,133,153,144]])

            RIGHT = np.array([[lm[i].x*w, lm[i].y*h]
                              for i in [362,385,387,263,373,380]])

            ear = (eye_aspect_ratio(LEFT) +
                   eye_aspect_ratio(RIGHT)) / 2

            if ear < 0.22:
                EYE_CLOSED_FRAMES += 1
            else:
                EYE_CLOSED_FRAMES = 0

            if EYE_CLOSED_FRAMES > 15:
                fatigue_timer = time.time() + 2
                safe_beep(1000, 200)

        if time.time() < fatigue_timer:
            cv2.putText(frame, "FATIGUE",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

        # ==================================================
        # 3️⃣ FALL DETECTION
        # ==================================================
        pose_results = pose.process(rgb)

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark

            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]

            dx = (shoulder.x - hip.x) * w
            dy = (shoulder.y - hip.y) * h
            angle = abs(math.degrees(math.atan2(dy, dx)))

            if angle < 30 or angle > 150:
                if not FALLEN:
                    safe_beep(1500, 300)
                FALLEN = True
            elif 60 < angle < 120:
                FALLEN = False

        if FALLEN:
            cv2.putText(frame, "FALL DETECTED",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

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

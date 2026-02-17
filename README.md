# Multidisciplinary-Project
This project is an AI-based real-time monitoring system designed to improve safety in public and private environments using computer vision. The system analyzes live camera footage to detect fatigue, dangerous objects, and violent activity.

🛡️ Multidisciplinary AI Surveillance System

An intelligent computer vision–based surveillance system capable of detecting:

🔪 Knife detection using YOLO

👥 Group fight detection

😴 Fatigue detection using OpenCV

🚨 Fall detection using OpenCV

This system combines deep learning and real-time video processing to enhance safety and automated monitoring.

📌 Features
🔪 Knife Detection

Uses YOLO (You Only Look Once) object detection

Real-time detection from webcam or video feed

Bounding box visualization with confidence score

👥 Group Fight Detection

Detects aggressive group behavior

Tracks multiple persons

Can trigger alerts when abnormal activity is detected

😴 Fatigue Detection

Uses OpenCV

Eye aspect ratio (EAR) or facial landmark detection

Detects prolonged eye closure

Triggers alarm when fatigue threshold is exceeded

🚨 Fall Detection

Detects sudden posture change

Monitors body orientation and motion

Can trigger emergency alert

🛠️ Technologies Used

Python 3.x

OpenCV

YOLO (v5 / v8 depending on implementation)

NumPy

Deep Learning (PyTorch if using YOLOv5/v8)

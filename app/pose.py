"""
Pose detection using MediaPipe Pose.
"""

import mediapipe as mp
import numpy as np
from PIL import Image

mp_pose = mp.solutions.pose

LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}


def detect_pose(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(image_np)

    if not results.pose_landmarks:
        return {"landmarks": {}, "quality_score": 0.0, "detected": False}

    lm = results.pose_landmarks.landmark
    landmarks = {}
    for name, idx in LANDMARKS.items():
        point = lm[idx]
        landmarks[name] = {"x": point.x, "y": point.y, "visibility": point.visibility}

    key_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    quality_score = float(np.mean([landmarks[p]["visibility"] for p in key_points]))

    return {"landmarks": landmarks, "quality_score": quality_score, "detected": True}

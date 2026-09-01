"""
Pose detection using MediaPipe's Tasks API (PoseLandmarker).
"""

import os
import urllib.request

import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = "/tmp/pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

LANDMARK_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def detect_pose(image_path: str, need_segmentation: bool = False) -> dict:
    _ensure_model()

    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    image_np = np.array(pil_image)
    image_width, image_height = pil_image.size

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=need_segmentation,
    )

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return {"landmarks": {}, "quality_score": 0.0, "detected": False}

    lm = result.pose_landmarks[0]
    landmarks = {}
    for name, idx in LANDMARK_INDICES.items():
        point = lm[idx]
        landmarks[name] = {
            "x": point.x,
            "y": point.y,
            "visibility": getattr(point, "visibility", 1.0),
        }

    key_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_ankle", "right_ankle"]
    quality_score = float(np.mean([landmarks[p]["visibility"] for p in key_points]))

    segmentation_mask = None
    if need_segmentation and result.segmentation_masks:
        segmentation_mask = result.segmentation_masks[0].numpy_view()

    return {
        "landmarks": landmarks,
        "quality_score": quality_score,
        "detected": True,
        "image_width": image_width,
        "image_height": image_height,
        "segmentation_mask": segmentation_mask,
    }

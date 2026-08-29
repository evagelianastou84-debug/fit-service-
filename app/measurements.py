"""
Converts MediaPipe's normalized (0-1) pixel-space landmarks into real
centimeter measurements, using the user's stated height as a scale
reference.
"""

from __future__ import annotations


def estimate_measurements(landmarks: dict, height_cm: float) -> dict:
    ls, rs = landmarks["left_shoulder"], landmarks["right_shoulder"]
    lh, rh = landmarks["left_hip"], landmarks["right_hip"]
    la, ra = landmarks["left_ankle"], landmarks["right_ankle"]
    nose = landmarks["nose"]

    ankle_y = (la["y"] + ra["y"]) / 2
    pixel_height_fraction = abs(ankle_y - nose["y"])

    if pixel_height_fraction <= 0:
        raise ValueError("Could not determine a usable body height in the photo")

    scale_cm_per_unit = (height_cm * 1.12) / pixel_height_fraction

    shoulder_width_units = abs(rs["x"] - ls["x"])
    hip_width_units = abs(rh["x"] - lh["x"])

    shoulder_width_cm = shoulder_width_units * scale_cm_per_unit
    hip_width_cm = hip_width_units * scale_cm_per_unit

    waist_width_cm = min(shoulder_width_cm, hip_width_cm) * 0.86

    return {
        "shoulder_width_cm": round(shoulder_width_cm, 1),
        "hip_width_cm": round(hip_width_cm, 1),
        "waist_width_cm": round(waist_width_cm, 1),
        "pixel_height_fraction": round(pixel_height_fraction, 4),
        "scale_cm_per_unit": round(scale_cm_per_unit, 2),
    }

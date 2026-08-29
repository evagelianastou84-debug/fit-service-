"""
Converts MediaPipe's normalized (0-1) landmark positions into real
centimeter measurements, using the user's stated height as a scale
reference.

Important: MediaPipe landmarks are normalized separately per axis —
x is a fraction of image WIDTH, y is a fraction of image HEIGHT. Since
phone photos are rarely square, applying one single scale factor to
both axes (as an earlier version of this code did) produces wildly
wrong widths. We compute the real pixel height first, derive a
cm-per-pixel scale from that, then convert x-axis (width) landmarks
using image_width and y-axis (height) landmarks using image_height,
so both end up in real, comparable centimeters.
"""

from __future__ import annotations


def estimate_measurements(landmarks: dict, height_cm: float, image_width: int, image_height: int) -> dict:
    ls, rs = landmarks["left_shoulder"], landmarks["right_shoulder"]
    lh, rh = landmarks["left_hip"], landmarks["right_hip"]
    la, ra = landmarks["left_ankle"], landmarks["right_ankle"]
    nose = landmarks["nose"]

    # Vertical span of the body in real pixels (nose to ankle midpoint).
    ankle_y_norm = (la["y"] + ra["y"]) / 2
    body_height_px = abs(ankle_y_norm - nose["y"]) * image_height

    if body_height_px <= 0:
        raise ValueError("Could not determine a usable body height in the photo")

    # cm per real pixel, using the person's actual height as reference.
    # +12% accounts for head-top-to-nose and ankle-to-floor offsets not
    # captured by the nose/ankle landmarks themselves.
    cm_per_pixel = (height_cm * 1.12) / body_height_px

    shoulder_width_px = abs(rs["x"] - ls["x"]) * image_width
    hip_width_px = abs(rh["x"] - lh["x"]) * image_width

    shoulder_width_cm = shoulder_width_px * cm_per_pixel
    hip_width_cm = hip_width_px * cm_per_pixel

    waist_width_cm = min(shoulder_width_cm, hip_width_cm) * 0.86

    return {
        "shoulder_width_cm": round(shoulder_width_cm, 1),
        "hip_width_cm": round(hip_width_cm, 1),
        "waist_width_cm": round(waist_width_cm, 1),
        "body_height_px": round(body_height_px, 1),
        "cm_per_pixel": round(cm_per_pixel, 4),
    }

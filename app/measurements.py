"""
Converts MediaPipe's normalized landmark positions into real
centimeter measurements, with two accuracy improvements:

1. Weight (BMI) as a sanity check/blend for the waist estimate, since
   MediaPipe has no direct waist landmark.
2. Optional side-photo depth data to compute real body circumference
   (bust/waist/hip) instead of just front-view width — garment sizing
   is based on circumference, and a single front photo structurally
   cannot capture how "deep" (front-to-back) a body is.
"""

from __future__ import annotations
import math


def estimate_measurements(landmarks: dict, height_cm: float, image_width: int, image_height: int) -> dict:
    ls, rs = landmarks["left_shoulder"], landmarks["right_shoulder"]
    lh, rh = landmarks["left_hip"], landmarks["right_hip"]
    la, ra = landmarks["left_ankle"], landmarks["right_ankle"]
    nose = landmarks["nose"]

    ankle_y_norm = (la["y"] + ra["y"]) / 2
    body_height_px = abs(ankle_y_norm - nose["y"]) * image_height

    if body_height_px <= 0:
        raise ValueError("Could not determine a usable body height in the photo")

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


def apply_weight_correction(measurements: dict, height_cm: float, weight_kg: float | None) -> dict:
    """
    Uses BMI as a sanity check on the landmark-based waist estimate.
    BMI correlates with waist circumference at a population level —
    not a substitute for a real measurement, but a useful correction
    when the two disagree significantly (e.g. photo-based estimate
    looks too slim for a stated weight, or vice versa).
    """
    if not weight_kg or weight_kg <= 0:
        measurements["waist_source"] = "photo_only"
        return measurements

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Rough population-level waist-width-from-BMI estimate (linear
    # approximation, calibrated for adult women — a real product
    # would refine this with actual anthropometric survey data).
    bmi_waist_estimate_cm = 22 + (bmi * 1.05)

    photo_waist_cm = measurements["waist_width_cm"]

    # Blend: trust the photo more (60%) since it's person-specific,
    # but let BMI pull it back toward realistic range (40%).
    blended_waist_cm = (photo_waist_cm * 0.6) + (bmi_waist_estimate_cm * 0.4)

    measurements["waist_width_cm"] = round(blended_waist_cm, 1)
    measurements["waist_source"] = "photo_and_weight_blend"
    measurements["bmi"] = round(bmi, 1)

    return measurements


def estimate_circumference_from_width_and_depth(width_cm: float, depth_cm: float) -> float:
    """
    Approximates body circumference at a given point (bust/waist/hip)
    using the ellipse-perimeter formula, given the front-view width
    and the side-view depth. This is the standard technique real body
    scanning apps use to go from 2D photos to garment-relevant
    circumference measurements.

    Uses Ramanujan's approximation for ellipse perimeter, which is
    accurate to well within photo-based measurement's other error
    margins.
    """
    a = width_cm / 2
    b = depth_cm / 2
    h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
    perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return round(perimeter, 1)


def estimate_depth_from_side_photo(side_landmarks: dict, side_image_width: int, side_cm_per_pixel: float) -> dict:
    """
    From a side-profile photo, estimates front-to-back body depth at
    shoulder/hip level. In a side photo, MediaPipe's shoulder/hip
    landmarks collapse onto roughly the same x-position (since width
    disappears from this angle) — what we actually read here is the
    torso silhouette thickness at those y-levels, approximated from
    the visible landmark spread.
    """
    ls = side_landmarks.get("left_shoulder")
    rs = side_landmarks.get("right_shoulder")
    lh = side_landmarks.get("left_hip")
    rh = side_landmarks.get("right_hip")

    if not all([ls, rs, lh, rh]):
        return {"shoulder_depth_cm": None, "hip_depth_cm": None}

    shoulder_depth_px = abs(rs["x"] - ls["x"]) * side_image_width
    hip_depth_px = abs(rh["x"] - lh["x"]) * side_image_width

    return {
        "shoulder_depth_cm": round(shoulder_depth_px * side_cm_per_pixel, 1),
        "hip_depth_cm": round(hip_depth_px * side_cm_per_pixel, 1),
    }

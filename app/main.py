"""
Fit Service — real pose detection + body measurement estimation.
Run with: uvicorn app.main:app --reload --port 8001
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pose import detect_pose
from app.measurements import (
    estimate_measurements,
    apply_weight_correction,
    estimate_depth_from_side_photo,
    estimate_circumference_from_width_and_depth,
)
from app.body_type import classify_body_type
from app.quality_checks import apply_quality_checks

app = FastAPI(title="Fit Service", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QUALITY_GATE_THRESHOLD = 0.6


@app.get("/health")
def health():
    return {"status": "ok", "service": "fit-service"}


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await upload.read())
        return tmp.name


@app.post("/analyze-photo")
async def analyze_photo(
    photo: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: float | None = Form(None),
    side_photo: UploadFile | None = File(None),
    known_body_type: str | None = Form(None),
):
    """
    photo: required front-facing full-body photo
    height_cm: required, used as the scale reference
    weight_kg: optional — improves waist estimate via BMI blend
    side_photo: optional — enables real circumference estimation
        (bust/waist/hip) instead of front-view width only
    known_body_type: optional — skips body type estimation if provided
    """
    if height_cm <= 0 or height_cm > 250:
        raise HTTPException(status_code=400, detail="height_cm must be a realistic value")

    front_path = await _save_upload(photo)

    try:
        pose_result = detect_pose(front_path)
    finally:
        Path(front_path).unlink(missing_ok=True)

    if not pose_result["detected"]:
        return {
            "detected": False,
            "quality_score": 0.0,
            "mode": "fallback_overlay",
            "message": "No body detected in photo. Ask the user to retake with full body visible.",
        }

    quality_check_result = apply_quality_checks(pose_result["landmarks"], pose_result["quality_score"])
    quality_score = quality_check_result["adjusted_quality_score"]
    mode = "full" if quality_score >= QUALITY_GATE_THRESHOLD else "fallback_overlay"

    try:
        measurements = estimate_measurements(
            pose_result["landmarks"],
            height_cm,
            pose_result["image_width"],
            pose_result["image_height"],
        )
    except ValueError as e:
        return {
            "detected": True,
            "quality_score": quality_score,
            "mode": "fallback_overlay",
            "message": str(e),
        }

    measurements = apply_weight_correction(measurements, height_cm, weight_kg)

    # Optional side photo: upgrades width-only estimates to real
    # circumference using the ellipse-perimeter technique.
    circumference = None
    if side_photo is not None:
        side_path = await _save_upload(side_photo)
        try:
            side_pose = detect_pose(side_path)
        finally:
            Path(side_path).unlink(missing_ok=True)

        if side_pose["detected"]:
            depth = estimate_depth_from_side_photo(
                side_pose["landmarks"],
                side_pose["image_width"],
                measurements["cm_per_pixel"],
            )
            if depth["shoulder_depth_cm"] and depth["hip_depth_cm"]:
                circumference = {
                    "bust_circumference_cm": estimate_circumference_from_width_and_depth(
                        measurements["shoulder_width_cm"], depth["shoulder_depth_cm"]
                    ),
                    "hip_circumference_cm": estimate_circumference_from_width_and_depth(
                        measurements["hip_width_cm"], depth["hip_depth_cm"]
                    ),
                    "source": "front_and_side_photo",
                }

    if known_body_type:
        body_type_result = {"body_type": known_body_type, "confidence": 1.0}
    else:
        body_type_result = classify_body_type(
            measurements["shoulder_width_cm"],
            measurements["waist_width_cm"],
            measurements["hip_width_cm"],
        )

    return {
        "detected": True,
        "quality_score": round(quality_score, 2),
        "quality_notes": quality_check_result["notes"],
        "mode": mode,
        "measurements": measurements,
        "circumference": circumference,
        "body_type": body_type_result["body_type"],
        "body_type_confidence": body_type_result["confidence"],
        "body_type_source": "user_provided" if known_body_type else "estimated",
    }

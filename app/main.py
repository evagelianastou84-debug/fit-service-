"""
Fit Service — real pose detection + body measurement estimation.
Run with: uvicorn app.main:app --reload --port 8001
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pose import detect_pose
from app.measurements import estimate_measurements
from app.body_type import classify_body_type

app = FastAPI(title="Fit Service", version="0.1.0")

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


@app.post("/analyze-photo")
async def analyze_photo(
    photo: UploadFile = File(...),
    height_cm: float = Form(...),
    known_body_type: str | None = Form(None),
):
    if height_cm <= 0 or height_cm > 250:
        raise HTTPException(status_code=400, detail="height_cm must be a realistic value")

    suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await photo.read())
        tmp_path = tmp.name

    try:
        pose_result = detect_pose(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pose_result["detected"]:
        return {
            "detected": False,
            "quality_score": 0.0,
            "mode": "fallback_overlay",
            "message": "No body detected in photo. Ask the user to retake with full body visible.",
        }

    quality_score = pose_result["quality_score"]
    mode = "full" if quality_score >= QUALITY_GATE_THRESHOLD else "fallback_overlay"

    try:
        measurements = estimate_measurements(pose_result["landmarks"], height_cm)
    except ValueError as e:
        return {
            "detected": True,
            "quality_score": quality_score,
            "mode": "fallback_overlay",
            "message": str(e),
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
        "mode": mode,
        "measurements": measurements,
        "body_type": body_type_result["body_type"],
        "body_type_confidence": body_type_result["confidence"],
        "body_type_source": "user_provided" if known_body_type else "estimated",
    }

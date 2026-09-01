"""
Photo quality checks that feed the app's confidence/quality_gate system.
"""


def check_arms_away_from_body(landmarks: dict) -> float:
    shoulder_width = abs(landmarks["right_shoulder"]["x"] - landmarks["left_shoulder"]["x"])
    if shoulder_width <= 0:
        return 1.0

    left_wrist_to_hip = abs(landmarks["left_wrist"]["x"] - landmarks["left_hip"]["x"])
    right_wrist_to_hip = abs(landmarks["right_wrist"]["x"] - landmarks["right_hip"]["x"])
    avg_wrist_offset = (left_wrist_to_hip + right_wrist_to_hip) / 2

    ratio = avg_wrist_offset / shoulder_width

    if ratio < 0.15:
        return 1.0
    elif ratio < 0.35:
        return 0.85
    else:
        return 0.6


def check_facing_camera(landmarks: dict) -> float:
    left_x = landmarks["left_shoulder"]["x"]
    right_x = landmarks["right_shoulder"]["x"]
    shoulder_width = abs(right_x - left_x)
    if shoulder_width <= 0:
        return 1.0

    shoulder_center = (left_x + right_x) / 2
    nose_offset = abs(landmarks["nose"]["x"] - shoulder_center) / shoulder_width

    if nose_offset < 0.15:
        return 1.0
    elif nose_offset < 0.3:
        return 0.85
    else:
        return 0.6


def apply_quality_checks(landmarks: dict, base_quality_score: float) -> dict:
    arms_penalty = check_arms_away_from_body(landmarks)
    facing_penalty = check_facing_camera(landmarks)

    adjusted_score = base_quality_score * arms_penalty * facing_penalty

    notes = []
    if arms_penalty < 1.0:
        notes.append("Arms appear away from the body — keep arms relaxed at your sides for best accuracy.")
    if facing_penalty < 1.0:
        notes.append("Body appears turned away from the camera — face the camera directly for best accuracy.")

    return {
        "adjusted_quality_score": round(adjusted_score, 2),
        "notes": notes,
    }

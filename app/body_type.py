"""
Classifies body type from shoulder/waist/hip measurements into the
5-label scale: Hourglass, Pear, Apple, Rectangle, Oval.
"""

BODY_TYPES = ["Hourglass", "Pear", "Apple", "Rectangle", "Oval"]


def classify_body_type(shoulder_cm: float, waist_cm: float, hip_cm: float) -> dict:
    shoulder_hip_diff = abs(shoulder_cm - hip_cm) / max(shoulder_cm, hip_cm)
    waist_definition = 1 - (waist_cm / min(shoulder_cm, hip_cm))

    if waist_definition > 0.22 and shoulder_hip_diff < 0.08:
        body_type, base_confidence = "Hourglass", 0.85
    elif hip_cm > shoulder_cm * 1.08:
        body_type, base_confidence = "Pear", 0.8
    elif shoulder_cm > hip_cm * 1.08:
        body_type, base_confidence = "Apple", 0.8
    elif waist_definition < 0.10 and shoulder_hip_diff < 0.06:
        body_type, base_confidence = "Rectangle", 0.75
    else:
        body_type, base_confidence = "Oval", 0.65

    boundary_distance = min(
        abs(waist_definition - 0.22),
        abs(shoulder_hip_diff - 0.08),
    )
    confidence = round(min(base_confidence, base_confidence - (0.3 * max(0, 0.05 - boundary_distance))), 2)

    return {"body_type": body_type, "confidence": max(confidence, 0.4)}

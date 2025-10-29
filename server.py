import base64
import math
import time
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Gesture & Expression Detection Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hands_solution = mp.solutions.hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

face_mesh_solution = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

finger_indices = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

gesture_emojis = {
    "airplane": "✈️",
    "open": "🖐️",
    "fist": "👊",
    "thumbsup": "👍",
    "peace": "✌️",
    "love": "🤟",
}

expression_emojis = {
    "tongue": "😛",
    "smile": "😊",
    "surprised": "😮",
    "angry": "😠",
    "wink": "😉",
}

SESSION_HISTORY_LIMIT = 15
SESSION_ACTIVE_LIFETIME = 1.2

sessions_lock = Lock()
processing_lock = Lock()
sessions: Dict[str, Dict[str, object]] = {}


class DetectionRequest(BaseModel):
    image: str
    session_id: str
    sensitivity: Optional[float] = None


def decode_image(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Malformed data URL")
    header, encoded = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(encoded)
    except base64.binascii.Error as exc:
        raise ValueError("Invalid base64 data") from exc
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def vector_length(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def normalize(vec: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vec)
    if length <= 1e-6:
        return np.zeros_like(vec)
    return vec / length


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = vector(b, a)
    cb = vector(b, c)
    length = vector_length(ab) * vector_length(cb)
    if not length:
        return 0.0
    cosine = np.clip(np.dot(ab, cb) / length, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def finger_extension_score(landmarks: List[np.ndarray], finger: str) -> float:
    mcp, pip, dip, tip = finger_indices[finger]
    angle = angle_between(landmarks[mcp], landmarks[pip], landmarks[tip])

    wrist = landmarks[0]
    palm_reference = (
        vector_length(vector(wrist, landmarks[9]))
        + vector_length(vector(wrist, landmarks[13]))
    )
    if palm_reference <= 1e-5:
        palm_reference = 1.0

    tip_distance = vector_length(vector(wrist, landmarks[tip]))
    pip_distance = vector_length(vector(wrist, landmarks[pip]))

    length_ratio = (tip_distance - pip_distance * 0.35) / palm_reference
    length_ratio = float(np.clip(length_ratio, 0.0, 1.2))

    angle_score = np.clip((angle - 110.0) / 70.0, 0.0, 1.0)
    combined = 0.55 * angle_score + 0.45 * min(1.0, length_ratio)
    return float(np.clip(combined, 0.0, 1.0))


def is_finger_extended(landmarks: List[np.ndarray], finger: str) -> bool:
    return finger_extension_score(landmarks, finger) > 0.62


def thumb_orientation(landmarks: List[np.ndarray], handedness: str) -> Dict[str, float]:
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[1]
    wrist = landmarks[0]
    vertical = thumb_tip[1] - thumb_mcp[1]
    horizontal = thumb_tip[0] - wrist[0]
    direction = -horizontal if handedness == "Left" else horizontal
    return {"vertical": vertical, "direction": direction}


def finger_spread(landmarks: List[np.ndarray]) -> float:
    tips = [landmarks[i] for i in (8, 12, 16, 20)]
    distances: List[float] = []
    for i in range(len(tips) - 1):
        for j in range(i + 1, len(tips)):
            distances.append(vector_length(vector(tips[i], tips[j])))

    wrist = landmarks[0]
    reference = vector_length(vector(wrist, landmarks[9])) + vector_length(vector(wrist, landmarks[13]))
    if reference <= 1e-5:
        reference = 1.0
    mean_distance = float(np.mean(distances)) if distances else 0.0
    return mean_distance / reference


def palm_normal(landmarks: List[np.ndarray]) -> np.ndarray:
    wrist = landmarks[0]
    index_base = landmarks[5]
    pinky_base = landmarks[17]
    normal = np.cross(vector(wrist, index_base), vector(wrist, pinky_base))
    return normalize(normal)


def finger_direction(landmarks: List[np.ndarray], finger: str) -> np.ndarray:
    mcp, _, _, tip = finger_indices[finger]
    return normalize(vector(landmarks[mcp], landmarks[tip]))


def palm_facing_camera_score(landmarks: List[np.ndarray]) -> float:
    wrist = landmarks[0]
    tips = [landmarks[i] for i in (8, 12, 16, 20)]
    average_tip_depth = float(np.mean([tip[2] for tip in tips]))
    return wrist[2] - average_tip_depth


def analyze_hands(results, threshold: float) -> List[Dict[str, object]]:
    detections: List[Dict[str, object]] = []
    if not results or not results.multi_hand_landmarks:
        return detections

    handedness_list = []
    if results.multi_handedness:
        handedness_list = [h.classification[0].label for h in results.multi_handedness]

    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = handedness_list[index] if index < len(handedness_list) else "Unknown"
        landmarks = [np.array([lm.x, lm.y, lm.z if lm.z is not None else 0.0], dtype=np.float32)
                     for lm in hand_landmarks.landmark]

        extension_scores = {finger: finger_extension_score(landmarks, finger) for finger in finger_indices}
        extended = {finger: score > 0.62 for finger, score in extension_scores.items()}
        spread = finger_spread(landmarks)
        orientation = thumb_orientation(landmarks, handedness)
        palm_facing_score = palm_facing_camera_score(landmarks)
        normal = palm_normal(landmarks)
        forward_facing = max(0.0, -normal[2])
        sideways_bias = abs(normal[0])

        total_extended = sum(1 for value in extended.values() if value)
        average_extension = float(np.mean(list(extension_scores.values())))
        clenched_score = max(0.0, 1.0 - average_extension)

        hand_candidates: List[Dict[str, object]] = []

        airplane_wings = extended["thumb"] and extended["middle"] and extended["pinky"]
        airplane_folded = (extension_scores["index"] < 0.45) and (extension_scores["ring"] < 0.45)
        if airplane_wings and airplane_folded:
            middle_direction = finger_direction(landmarks, "middle")
            pinky_direction = finger_direction(landmarks, "pinky")
            wing_angle = math.degrees(math.acos(np.clip(np.dot(middle_direction, pinky_direction), -1.0, 1.0)))
            wing_span = (extension_scores["thumb"] + extension_scores["middle"] + extension_scores["pinky"]) / 3.0
            fold_suppression = 1.0 - ((extension_scores["index"] + extension_scores["ring"]) / 2.0)
            wing_balance = np.clip((wing_angle - 15.0) / 45.0, 0.0, 1.0)
            forward_bonus = min(1.0, forward_facing * 1.4 + max(0.0, 0.25 - sideways_bias) * 1.1)
            confidence = min(
                1.0,
                0.35
                + wing_span * 0.35
                + max(0.0, spread - 0.1) * 0.9
                + fold_suppression * 0.3
                + wing_balance * 0.25
                + forward_bonus * 0.2,
            )
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"airplane-{index}",
                    "label": "Airplane Gesture",
                    "emoji": gesture_emojis["airplane"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Thumb, middle, and pinky extended while index and ring fold like airplane wings.",
                })
        elif total_extended >= 4 and spread > 0.16:
            finger_alignment = np.mean([
                max(0.0, np.dot(finger_direction(landmarks, finger), np.array([0.0, -1.0, 0.0])))
                for finger in ("index", "middle", "ring", "pinky")
            ])
            facing_bonus = min(1.0, forward_facing * 1.3 + max(0.0, palm_facing_score) * 3)
            confidence = min(1.0, 0.4 + spread * 1.1 + finger_alignment * 0.5 + facing_bonus * 0.4)
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"open-{index}",
                    "label": "Open Palm",
                    "emoji": gesture_emojis["open"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "All fingers extended with relaxed spacing.",
                })

        if clenched_score > 0.55 and spread < 0.15:
            knuckle_compactness = max(0.0, 0.22 - spread) * 3.2
            confidence = min(1.0, 0.48 + clenched_score * 0.85 + knuckle_compactness)
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"fist-{index}",
                    "label": "Closed Fist",
                    "emoji": gesture_emojis["fist"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "All fingers curled toward the palm.",
                })

        if extended["thumb"] and not any(extended[f] for f in ("index", "middle", "ring", "pinky")):
            vertical = -orientation["vertical"]
            sideways = abs(orientation["direction"])
            sideways_penalty = max(0.0, sideways - 0.05) * 2.2
            forward_penalty = max(0.0, 0.15 - forward_facing) * 1.4
            confidence = min(1.0, 0.52 + max(0.0, vertical) * 2.3 - sideways_penalty - forward_penalty)
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"thumbsup-{index}",
                    "label": "Thumbs Up",
                    "emoji": gesture_emojis["thumbsup"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Thumb extended upward with other fingers curled.",
                })

        if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            index_direction = finger_direction(landmarks, "index")
            middle_direction = finger_direction(landmarks, "middle")
            v_angle = math.degrees(math.acos(np.clip(np.dot(index_direction, middle_direction), -1.0, 1.0)))
            separation = distance_between(landmarks, 8, 12) / max(spread, 1e-5)
            v_quality = np.clip((v_angle - 10.0) / 40.0, 0.0, 1.0)
            confidence = min(1.0, 0.48 + spread * 1.0 + separation * 0.08 + v_quality * 0.6)
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"peace-{index}",
                    "label": "Peace Sign",
                    "emoji": gesture_emojis["peace"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Index and middle fingers extended to form a V shape.",
                })

        if extended["index"] and extended["pinky"] and extended["thumb"] and not extended["middle"] and not extended["ring"]:
            curl_balance = (extension_scores["thumb"] + extension_scores["index"] + extension_scores["pinky"]) / 3.0
            curl_penalty = (extension_scores["middle"] + extension_scores["ring"]) / 2.0
            hook_span = distance_between(landmarks, 8, 20) / max(spread + 1e-5, 1e-5)
            confidence = min(1.0, 0.48 + spread * 0.9 + curl_balance * 0.45 - curl_penalty * 0.65 + min(1.0, hook_span * 0.05))
            if confidence >= threshold:
                hand_candidates.append({
                    "id": f"love-{index}",
                    "label": "Love You Sign",
                    "emoji": gesture_emojis["love"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Thumb, index, and pinky extended with middle and ring fingers folded.",
                })

        if hand_candidates:
            best_candidate = max(hand_candidates, key=lambda item: item.get("confidence", 0.0))
            detections.append(best_candidate)

    return detections


def distance_between(landmarks: List[np.ndarray], a: int, b: int) -> float:
    return vector_length(vector(landmarks[a], landmarks[b]))


def average(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def detection_slot(detection_id: str) -> Optional[str]:
    """Return a slot identifier for mutually exclusive detections.

    Gesture detections encode the tracked hand index in their identifier
    (e.g. ``airplane-0``). When we promote a new detection for that hand we
    should retire any lingering entries associated with the same slot so the
    session state cannot hold conflicting gestures for the same limb.
    """

    parts = detection_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return None


def eye_openness(landmarks: List[np.ndarray], vertical_pair, horizontal_pair) -> float:
    vertical = distance_between(landmarks, vertical_pair[0], vertical_pair[1])
    horizontal = distance_between(landmarks, horizontal_pair[0], horizontal_pair[1])
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def eyebrow_compression(landmarks: List[np.ndarray]) -> float:
    brow_left = distance_between(landmarks, 70, 105)
    brow_right = distance_between(landmarks, 300, 334)
    eye_left = distance_between(landmarks, 159, 145)
    eye_right = distance_between(landmarks, 386, 374)
    ratio_left = eye_left / brow_left if brow_left else 0.0
    ratio_right = eye_right / brow_right if brow_right else 0.0
    return 0.5 - average([ratio_left, ratio_right])


def analyze_face(results, threshold: float) -> List[Dict[str, object]]:
    detections: List[Dict[str, object]] = []
    if not results or not results.multi_face_landmarks:
        return detections

    landmarks = [
        np.array([lm.x, lm.y, lm.z if lm.z is not None else 0.0], dtype=np.float32)
        for lm in results.multi_face_landmarks[0].landmark
    ]

    mouth_width = distance_between(landmarks, 61, 291)
    mouth_height = distance_between(landmarks, 13, 14)
    face_width = distance_between(landmarks, 33, 263)
    upper_lip_to_nose = distance_between(landmarks, 13, 1)

    smile_ratio = mouth_width / face_width if face_width else 0.0
    surprise_ratio = mouth_height / mouth_width if mouth_width else 0.0
    mouth_open = mouth_height / upper_lip_to_nose if upper_lip_to_nose else 0.0

    tongue_depth = (landmarks[17][2] if len(landmarks) > 17 else 0.0) - (landmarks[14][2] if len(landmarks) > 14 else 0.0)
    tongue_visibility = max(0.0, -tongue_depth)

    left_eye = eye_openness(landmarks, (159, 145), (133, 33))
    right_eye = eye_openness(landmarks, (386, 374), (362, 263))

    brow_comp = eyebrow_compression(landmarks)

    face_candidates: List[Dict[str, object]] = []

    if smile_ratio > 0.4 and mouth_open < 0.32:
        smile_tightness = max(0.0, 0.48 - mouth_open)
        confidence = min(1.0, (smile_ratio - 0.4) * 3.2 + smile_tightness)
        if confidence >= threshold:
            face_candidates.append({
                "id": "smile",
                "label": "Bright Smile",
                "emoji": expression_emojis["smile"],
                "confidence": confidence,
                "type": "expression",
                "description": "Mouth corners widened — classic happy smile.",
            })

    if surprise_ratio > 0.3 and mouth_open > 0.26:
        lip_roundness = surprise_ratio * 0.6 + mouth_open * 0.4
        confidence = min(1.0, (lip_roundness - 0.26) * 4.5 + 0.45)
        if confidence >= threshold:
            face_candidates.append({
                "id": "surprised",
                "label": "Surprised",
                "emoji": expression_emojis["surprised"],
                "confidence": confidence,
                "type": "expression",
                "description": "Mouth open with rounded lips indicating surprise.",
            })

    if mouth_open > 0.3 and tongue_visibility > 0.01:
        confidence = min(1.0, 0.55 + tongue_visibility * 55)
        if confidence >= threshold:
            face_candidates.append({
                "id": "tongue",
                "label": "Tongue Out",
                "emoji": expression_emojis["tongue"],
                "confidence": confidence,
                "type": "expression",
                "description": "Tongue visible with playful expression.",
            })

    if brow_comp > 0.04 and mouth_open < 0.22:
        confidence = min(1.0, 0.48 + brow_comp * 8.5)
        if confidence >= threshold:
            face_candidates.append({
                "id": "angry",
                "label": "Angry",
                "emoji": expression_emojis["angry"],
                "confidence": confidence,
                "type": "expression",
                "description": "Brows drawn together signaling anger or focus.",
            })

    eye_asymmetry = abs(left_eye - right_eye)
    if eye_asymmetry > 0.09:
        winking_eye_open = min(left_eye, right_eye)
        confidence = min(1.0, 0.52 + eye_asymmetry * 2.5 + max(0.0, 0.2 - winking_eye_open) * 3)
        if confidence >= threshold:
            face_candidates.append({
                "id": "wink",
                "label": "Playful Wink",
                "emoji": expression_emojis["wink"],
                "confidence": confidence,
                "type": "expression",
                "description": "One eye relaxed while the other remains open.",
            })

    if face_candidates:
        best_candidate = max(face_candidates, key=lambda item: item.get("confidence", 0.0))
        detections.append(best_candidate)

    return detections


def ensure_session(session_id: str, sensitivity: float) -> Dict[str, object]:
    with sessions_lock:
        session = sessions.get(session_id)
        if not session:
            session = {
                "history": deque(maxlen=SESSION_HISTORY_LIMIT),
                "stats": {"gesture": 0, "expression": 0, "confidences": []},
                "active": {},
                "sensitivity": sensitivity,
            }
            sessions[session_id] = session
        else:
            session["sensitivity"] = sensitivity
        return session


def update_session_state(session: Dict[str, object], detections: List[Dict[str, object]]):
    now = time.monotonic()
    active = session["active"]
    history: deque = session["history"]
    stats = session["stats"]

    for detection in detections:
        detection_with_time = detection.copy()
        detection_with_time["timestamp"] = now

        slot = detection_slot(detection["id"])
        if slot is not None:
            conflicting_ids = [
                key
                for key in list(active.keys())
                if key != detection["id"] and detection_slot(key) == slot
            ]
            for key in conflicting_ids:
                active.pop(key, None)

        active[detection["id"]] = detection_with_time

    expired = [key for key, value in active.items() if now - value["timestamp"] > SESSION_ACTIVE_LIFETIME]
    for key in expired:
        active.pop(key, None)

    active_list = sorted(active.values(), key=lambda item: item["confidence"], reverse=True)
    primary = active_list[0] if active_list else None

    if primary:
        last_entry = history[0] if history else None
        if not last_entry or last_entry["label"] != primary["label"]:
            entry = {
                "label": primary["label"],
                "emoji": primary["emoji"],
                "confidence": primary.get("confidence"),
                "type": primary["type"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            history.appendleft(entry)
            stats[primary["type"]] += 1
            confidence_value = primary.get("confidence")
            if isinstance(confidence_value, (int, float)):
                stats["confidences"].append(float(confidence_value))

    average_confidence = average(stats["confidences"])
    return {
        "primary": primary,
        "history": list(history),
        "stats": {
            "gesture": stats["gesture"],
            "expression": stats["expression"],
            "average_confidence": average_confidence if stats["confidences"] else None,
        },
    }


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.post("/api/detect")
async def detect(payload: DetectionRequest):
    sensitivity = payload.sensitivity if payload.sensitivity is not None else 0.7
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    session = ensure_session(payload.session_id, sensitivity)

    try:
        frame = decode_image(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start = time.perf_counter()

    with processing_lock:
        hand_results = hands_solution.process(frame)
        face_results = face_mesh_solution.process(frame)

    threshold = session.get("sensitivity", sensitivity)
    hand_detections = analyze_hands(hand_results, threshold)
    face_detections = analyze_face(face_results, threshold)
    detections = hand_detections + face_detections

    session_state = update_session_state(session, detections)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "primary_detection": session_state["primary"],
        "history": session_state["history"],
        "stats": session_state["stats"],
        "detections": detections,
        "latency_ms": latency_ms,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=4564, reload=True)

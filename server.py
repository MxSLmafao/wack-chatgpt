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


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = vector(b, a)
    cb = vector(b, c)
    length = vector_length(ab) * vector_length(cb)
    if not length:
        return 0.0
    cosine = np.clip(np.dot(ab, cb) / length, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def is_finger_extended(landmarks: List[np.ndarray], finger: str) -> bool:
    mcp, pip, dip, tip = finger_indices[finger]
    angle = angle_between(landmarks[mcp], landmarks[pip], landmarks[tip])
    return angle > 160


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
    return float(np.mean(distances)) if distances else 0.0


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

        extended = {finger: is_finger_extended(landmarks, finger) for finger in finger_indices}
        spread = finger_spread(landmarks)
        orientation = thumb_orientation(landmarks, handedness)

        total_extended = sum(1 for value in extended.values() if value)
        clenched_score = 1 - total_extended / 5.0

        if total_extended == 5 and spread > 0.19:
            confidence = min(1.0, 0.6 + spread)
            if confidence >= threshold:
                detections.append({
                    "id": f"airplane-{index}",
                    "label": "Airplane Gesture",
                    "emoji": gesture_emojis["airplane"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Palm open with fingers stretched wide like airplane wings.",
                })
        elif total_extended == 5:
            confidence = min(1.0, 0.45 + spread * 2)
            if confidence >= threshold:
                detections.append({
                    "id": f"open-{index}",
                    "label": "Open Palm",
                    "emoji": gesture_emojis["open"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "All fingers extended with relaxed spacing.",
                })

        if clenched_score > 0.7:
            confidence = min(1.0, clenched_score + 0.2)
            if confidence >= threshold:
                detections.append({
                    "id": f"fist-{index}",
                    "label": "Closed Fist",
                    "emoji": gesture_emojis["fist"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "All fingers curled toward the palm.",
                })

        if extended["thumb"] and not any(extended[f] for f in ("index", "middle", "ring", "pinky")):
            vertical = -orientation["vertical"]
            confidence = min(1.0, 0.6 + max(0.0, vertical) * 2)
            if confidence >= threshold:
                detections.append({
                    "id": f"thumbsup-{index}",
                    "label": "Thumbs Up",
                    "emoji": gesture_emojis["thumbsup"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Thumb extended upward with other fingers curled.",
                })

        if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            confidence = min(1.0, 0.55 + spread * 1.5)
            if confidence >= threshold:
                detections.append({
                    "id": f"peace-{index}",
                    "label": "Peace Sign",
                    "emoji": gesture_emojis["peace"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Index and middle fingers extended to form a V shape.",
                })

        if extended["index"] and extended["pinky"] and extended["thumb"] and not extended["middle"] and not extended["ring"]:
            confidence = min(1.0, 0.55 + spread * 1.25)
            if confidence >= threshold:
                detections.append({
                    "id": f"love-{index}",
                    "label": "Love You Sign",
                    "emoji": gesture_emojis["love"],
                    "confidence": confidence,
                    "type": "gesture",
                    "description": "Thumb, index, and pinky extended with middle and ring fingers folded.",
                })

    return detections


def distance_between(landmarks: List[np.ndarray], a: int, b: int) -> float:
    return vector_length(vector(landmarks[a], landmarks[b]))


def average(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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

    if smile_ratio > 0.42:
        confidence = min(1.0, (smile_ratio - 0.42) * 3 + 0.5)
        if confidence >= threshold:
            detections.append({
                "id": "smile",
                "label": "Bright Smile",
                "emoji": expression_emojis["smile"],
                "confidence": confidence,
                "type": "expression",
                "description": "Mouth corners widened — classic happy smile.",
            })

    if surprise_ratio > 0.32 and mouth_open > 0.24:
        confidence = min(1.0, (surprise_ratio - 0.32) * 4 + 0.5)
        if confidence >= threshold:
            detections.append({
                "id": "surprised",
                "label": "Surprised",
                "emoji": expression_emojis["surprised"],
                "confidence": confidence,
                "type": "expression",
                "description": "Mouth open with rounded lips indicating surprise.",
            })

    if mouth_open > 0.28 and tongue_visibility > 0.008:
        confidence = min(1.0, 0.6 + tongue_visibility * 60)
        if confidence >= threshold:
            detections.append({
                "id": "tongue",
                "label": "Tongue Out",
                "emoji": expression_emojis["tongue"],
                "confidence": confidence,
                "type": "expression",
                "description": "Tongue visible with playful expression.",
            })

    if brow_comp > 0.04 and mouth_open < 0.2:
        confidence = min(1.0, 0.5 + brow_comp * 8)
        if confidence >= threshold:
            detections.append({
                "id": "angry",
                "label": "Angry",
                "emoji": expression_emojis["angry"],
                "confidence": confidence,
                "type": "expression",
                "description": "Brows drawn together signaling anger or focus.",
            })

    if abs(left_eye - right_eye) > 0.08:
        winking_eye_open = min(left_eye, right_eye)
        confidence = min(1.0, 0.55 + (0.22 - winking_eye_open) * 4)
        if confidence >= threshold:
            detections.append({
                "id": "wink",
                "label": "Playful Wink",
                "emoji": expression_emojis["wink"],
                "confidence": confidence,
                "type": "expression",
                "description": "One eye relaxed while the other remains open.",
            })

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

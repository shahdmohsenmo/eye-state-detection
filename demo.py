"""Live demo: webcam -> MediaPipe Face Landmarker -> eye crops -> MobileNetV3-Small (model v3).

Needs (see README):
    models/eye_state_mnv3s_v3.pt   trained eye-state model
    models/face_landmarker.task    MediaPipe face landmark model

Run:
    python demo.py

The decision uses the CNN (v3) only. EAR is shown on screen for comparison.

Keys:  Q or ESC = quit
"""
import threading
import time
from math import atan2, degrees, hypot
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------- settings
HERE = Path(__file__).resolve().parent
EYE_MODEL = HERE / "models" / "eye_state_mnv3s_v3.pt"
FACE_MODEL = HERE / "models" / "face_landmarker.task"

CAMERA_INDEX = 0        # try 1 if the wrong camera opens
CROP_SIZE = 128
CROP_SCALE = 2.2        # crop side = 2.2 x eye width (same as in Colab)
IMG_SIZE = 112
THRESHOLD = 0.5         # average P(open) below this = eyes closed
SMOOTHING = 0.6         # 0 = no smoothing, closer to 1 = smoother but slower
DROWSY_SECONDS = 1.5    # eyes closed longer than this = drowsiness alert

# Eye corners in the MediaPipe face mesh: (outer, inner)
EYE_CORNERS = {"right": (33, 133), "left": (362, 263)}

# EAR (shown for comparison only): 6 points per eye
RIGHT_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EAR = [362, 385, 387, 263, 373, 380]

# Same preprocessing as validation/test in Colab
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------- models
def load_eye_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)            # 0 = closed, 1 = open
    model.load_state_dict(torch.load(EYE_MODEL, map_location="cpu"))
    return model.eval()


def create_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=FACE_MODEL.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------- eye crops
def extract_eyes(frame_bgr, landmarks):
    """Square crop around each eye, rotated so the eye is horizontal."""
    h, w = frame_bgr.shape[:2]
    eyes = []
    for name, (i1, i2) in EYE_CORNERS.items():
        p1 = np.array([landmarks[i1].x * w, landmarks[i1].y * h])
        p2 = np.array([landmarks[i2].x * w, landmarks[i2].y * h])
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        center = (p1 + p2) / 2
        eye_width = hypot(*(p2 - p1))
        if eye_width < 8:                       # face too far away
            continue
        side = eye_width * CROP_SCALE
        angle = degrees(atan2(p2[1] - p1[1], p2[0] - p1[0]))

        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, CROP_SIZE / side)
        M[0, 2] += CROP_SIZE / 2 - center[0]
        M[1, 2] += CROP_SIZE / 2 - center[1]
        crop = cv2.warpAffine(frame_bgr, M, (CROP_SIZE, CROP_SIZE), borderMode=cv2.BORDER_REPLICATE)

        box = cv2.boxPoints(((float(center[0]), float(center[1])), (side, side), angle)).astype(np.int32)
        eyes.append({"name": name, "crop": crop, "box": box})
    return eyes


def predict_open(model, crops_bgr):
    """Probability of 'open' for each crop."""
    batch = torch.stack([
        transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr
    ])
    with torch.no_grad():
        return torch.softmax(model(batch), dim=1)[:, 1].numpy()


def eye_aspect_ratio(landmarks, idx, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return vertical / (2 * np.linalg.norm(p[0] - p[3]))


# ---------------------------------------------------------------- sound
try:
    import winsound                              # Windows

    def _beep():
        winsound.Beep(1000, 400)
except ImportError:                              # macOS / Linux
    def _beep():
        print("\a", end="", flush=True)


def beep():
    threading.Thread(target=_beep, daemon=True).start()


# ---------------------------------------------------------------- main loop
def main():
    for path in (EYE_MODEL, FACE_MODEL):
        if not path.exists():
            raise SystemExit(f"Missing file: {path}  (see README, 'Setup')")
    print("Loading models...")
    model = load_eye_model()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1.")

    smoothed = 1.0              # start as "open"
    closed = False
    closed_since = None
    blinks = 0
    last_beep = 0.0
    fps = 0.0
    last_time = time.perf_counter()
    start = time.perf_counter()
    last_ts = -1

    with create_landmarker() as landmarker:
        print("Running. Press Q or ESC in the video window to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read from the camera.")
                break

            now = time.perf_counter()
            ts = max(int((now - start) * 1000), last_ts + 1)   # increasing timestamps
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            display = frame.copy()
            height, width = display.shape[:2]
            eyes = extract_eyes(frame, result.face_landmarks[0]) if result.face_landmarks else []

            if eyes:
                p_open = predict_open(model, [e["crop"] for e in eyes])
                p_avg = float(p_open.mean())                         # average of both eyes
                smoothed = SMOOTHING * smoothed + (1 - SMOOTHING) * p_avg

                # Hysteresis: avoids flickering around the threshold
                if not closed and smoothed < THRESHOLD - 0.05:
                    closed, closed_since = True, now
                elif closed and smoothed > THRESHOLD + 0.05:
                    if now - closed_since < DROWSY_SECONDS:
                        blinks += 1                                  # short closure = blink
                    closed, closed_since = False, None

                color = (0, 0, 255) if closed else (0, 255, 0)
                for eye, p in zip(eyes, p_open):
                    cv2.polylines(display, [eye["box"]], True, color, 2)
                    x, y = eye["box"].min(axis=0)
                    cv2.putText(display, f"{p:.2f}", (int(x), int(y) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Eye crops preview (what the model sees), top-right corner
                for i, eye in enumerate(eyes):
                    small = cv2.resize(eye["crop"], (96, 96))
                    x0 = width - 96 * (len(eyes) - i) - 10 * (len(eyes) - i)
                    if x0 >= 0 and 106 <= height:
                        display[10:106, x0:x0 + 96] = small

                state = "CLOSED" if closed else "OPEN"
                cv2.putText(display, f"Eyes: {state}   P(open) = {smoothed:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                lm = result.face_landmarks[0]
                ear = (eye_aspect_ratio(lm, RIGHT_EAR, width, height) +
                       eye_aspect_ratio(lm, LEFT_EAR, width, height)) / 2
                cv2.putText(display, f"Model v3   |   EAR = {ear:.3f} (info only)", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if closed and now - closed_since >= DROWSY_SECONDS:
                    cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 14)
                    cv2.putText(display, "DROWSINESS ALERT!", (20, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    if now - last_beep > 1.0:
                        beep()
                        last_beep = now
            else:
                cv2.putText(display, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            fps = 0.9 * fps + 0.1 / max(1e-6, now - last_time)
            last_time = now
            cv2.putText(display, f"Blinks: {blinks}    FPS: {fps:.0f}", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Eye State Detection - press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Inference-time benchmark of the full pipeline on this computer (CPU), using live webcam frames.

Run from the repository folder (same models as demo.py):
    python benchmark.py

Sit in front of the camera (face visible) until it finishes (~20 seconds).
"""
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

import demo as d

N_WARMUP = 20
N_FRAMES = 300

print("Loading models...")
model = d.load_eye_model()
camera = cv2.VideoCapture(d.CAMERA_INDEX)
if not camera.isOpened():
    raise SystemExit("Could not open the camera.")

times = {"1. MediaPipe face landmarks": [], "2. Eye cropping": [], "3. Pre-processing": [],
         "4. MobileNetV3 (2 eyes)": [], "5. EAR (info only)": [], "TOTAL per frame": []}
ts, frames_done, no_face = 0, 0, 0

with d.create_landmarker() as landmarker:
    print(f"Measuring {N_FRAMES} frames - keep your face in front of the camera...")
    while frames_done < N_WARMUP + N_FRAMES:
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("Could not read from the camera.")
        h, w = frame.shape[:2]
        ts += 33

        t0 = time.perf_counter()
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        t1 = time.perf_counter()
        if not result.face_landmarks:
            no_face += 1
            continue
        lm = result.face_landmarks[0]
        eyes = d.extract_eyes(frame, lm)
        if len(eyes) != 2:
            continue
        t2 = time.perf_counter()
        batch = torch.stack([d.transform(Image.fromarray(cv2.cvtColor(e["crop"], cv2.COLOR_BGR2RGB))) for e in eyes])
        t3 = time.perf_counter()
        with torch.no_grad():
            p_cnn = float(torch.softmax(model(batch), dim=1)[:, 1].mean())
        t4 = time.perf_counter()
        ear = (d.eye_aspect_ratio(lm, d.RIGHT_EAR, w, h) + d.eye_aspect_ratio(lm, d.LEFT_EAR, w, h)) / 2
        t5 = time.perf_counter()

        frames_done += 1
        if frames_done <= N_WARMUP:
            continue
        for key, value in zip(times, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0]):
            times[key].append(value * 1000)
        if frames_done % 50 == 0:
            print(f"   {frames_done - N_WARMUP}/{N_FRAMES} frames")

camera.release()

print("\n" + "=" * 66)
print(f"Device: {platform.processor() or platform.machine()} | CPU threads used by torch: {torch.get_num_threads()}")
print(f"Frame size: {w} x {h} | frames measured: {N_FRAMES} | frames without a face: {no_face}")
print("=" * 66)
print(f"{'Stage':32s} {'mean ms':>9s} {'median ms':>10s} {'95% ms':>8s} {'share':>7s}")
total_mean = np.mean(times["TOTAL per frame"])
for key, values in times.items():
    v = np.array(values)
    share = "" if key.startswith("TOTAL") else f"{v.mean() / total_mean:6.1%}"
    print(f"{key:32s} {v.mean():9.2f} {np.median(v):10.2f} {np.percentile(v, 95):8.2f} {share:>7s}")
print("-" * 66)
print(f"Processing speed: {1000 / total_mean:.1f} FPS  (camera capture time not included)")

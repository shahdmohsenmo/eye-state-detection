"""Measures the time of every pipeline stage on this computer (CPU) with live webcam frames.

Run from the project folder (same files as demo.py):   python benchmark.py
Keep your face in front of the camera until it finishes.
"""
import platform
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import demo as d

N_WARMUP, N_FRAMES = 20, 300

cfg = d.load_config()
model = d.load_eye_model(d.HERE / cfg["model_file"])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
camera = cv2.VideoCapture(cfg["camera_index"])
if not camera.isOpened():
    raise SystemExit("Could not open the camera.")

stages = ["1. MediaPipe face landmarks", "2. Eye cropping", "3. Pre-processing", "4. MobileNetV3 (2 eyes)",
          "5. EAR + tracker", "TOTAL per frame"]
times = {s: [] for s in stages}
tracker = d.DrowsinessTracker(cfg)
ts, done, no_face = 0, 0, 0

with d.create_landmarker(d.HERE / cfg["face_model_file"]) as landmarker:
    print(f"Measuring {N_FRAMES} frames...")
    while done < N_WARMUP + N_FRAMES:
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("Could not read from the camera.")
        h, w = frame.shape[:2]
        ts += 33
        t0 = time.perf_counter()
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        t1 = time.perf_counter()
        if not res.face_landmarks:
            no_face += 1
            continue
        lm = res.face_landmarks[0]
        eyes = d.extract_eyes(frame, lm, cfg["crop_size"], cfg["crop_scale"])
        if len(eyes) != 2:
            continue
        t2 = time.perf_counter()
        batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(e["crop"], cv2.COLOR_BGR2RGB))) for e in eyes])
        t3 = time.perf_counter()
        with torch.no_grad():
            p = float(torch.softmax(model(batch), 1)[:, 1].mean())
        t4 = time.perf_counter()
        (d.eye_aspect_ratio(lm, d.RIGHT_EAR, w, h) + d.eye_aspect_ratio(lm, d.LEFT_EAR, w, h)) / 2
        tracker.update(t4, p)
        tracker.perclos()
        t5 = time.perf_counter()
        done += 1
        if done > N_WARMUP:
            for s, v in zip(stages, [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0]):
                times[s].append(v * 1000)

camera.release()
total = np.mean(times["TOTAL per frame"])
print(f"\nDevice: {platform.processor() or platform.machine()} | torch threads: {torch.get_num_threads()}")
print(f"Frame {w}x{h} | frames measured: {N_FRAMES} | frames without a face: {no_face}")
print(f"{'Stage':30s} {'mean ms':>8s} {'median':>8s} {'95%':>8s} {'share':>7s}")
for s in stages:
    v = np.array(times[s])
    share = "" if s.startswith("TOTAL") else f"{v.mean() / total:6.1%}"
    print(f"{s:30s} {v.mean():8.2f} {np.median(v):8.2f} {np.percentile(v, 95):8.2f} {share:>7s}")
print(f"Processing speed: {1000 / total:.1f} FPS (camera capture not included)")

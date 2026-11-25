"""Live eye-state and drowsiness demo.

webcam -> MediaPipe Face Landmarker -> eye crops -> MobileNetV3-Small -> time-based smoothing
-> OPEN / CLOSED with hysteresis -> blink / microsleep / long-closure events, PERCLOS, no-face alarm
-> per-frame log (CSV) + event summary (JSON) for the fusion layer.

All tuned values (crop scale, thresholds, hysteresis, smoothing time constant) are read from
config/demo_config.json, written by the final notebook from the validation data.

Run:   python demo.py            (Q or Esc to quit)
"""
import csv
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
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

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "config" / "demo_config.json"

DEFAULT_CONFIG = {
    "model_file": "models/eye_state_final.pt",
    "face_model_file": "models/face_landmarker.task",
    "camera_index": 0,
    "img_size": 112,
    "crop_size": 128,
    "crop_scale": 2.2,
    "threshold": 0.5,
    "threshold_close": 0.45,
    "threshold_open": 0.55,
    "smoothing_tau_s": 0.10,
    "blink_max_s": 0.5,
    "microsleep_max_s": 1.5,
    "no_face_alarm_s": 2.0,
    "perclos_window_s": 60.0,
    "perclos_warning": 0.15,
}

EYE_CORNERS = {"right": (33, 133), "left": (362, 263)}
RIGHT_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EAR = [362, 385, 387, 263, 373, 380]


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    else:
        print(f"Warning: {CONFIG_FILE} not found, using default values.")
    return cfg


def load_eye_model(path):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def create_landmarker(path):
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=Path(path).read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_eyes(frame_bgr, landmarks, crop_size, crop_scale):
    h, w = frame_bgr.shape[:2]
    eyes = []
    for name, (i1, i2) in EYE_CORNERS.items():
        p1 = np.array([landmarks[i1].x * w, landmarks[i1].y * h])
        p2 = np.array([landmarks[i2].x * w, landmarks[i2].y * h])
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        center = (p1 + p2) / 2
        eye_width = hypot(*(p2 - p1))
        if eye_width < 8:
            continue
        side = eye_width * crop_scale
        angle = degrees(atan2(p2[1] - p1[1], p2[0] - p1[0]))
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, crop_size / side)
        M[0, 2] += crop_size / 2 - center[0]
        M[1, 2] += crop_size / 2 - center[1]
        crop = cv2.warpAffine(frame_bgr, M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE)
        box = cv2.boxPoints(((float(center[0]), float(center[1])), (side, side), angle)).astype(np.int32)
        eyes.append({"name": name, "crop": crop, "box": box})
    return eyes


def eye_aspect_ratio(landmarks, idx, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in idx]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]))


class DrowsinessTracker:
    """Frame-rate independent eye-state logic.

    smoothing: s += (1 - exp(-dt / tau)) * (p - s)
    state:     OPEN -> CLOSED when s < threshold_close, CLOSED -> OPEN when s > threshold_open
    events:    closure < blink_max_s = blink, < microsleep_max_s = microsleep, longer = long closure
    PERCLOS:   time-weighted share of CLOSED time in the last perclos_window_s seconds (face visible only)
    no face:   ends any running closure without classifying it; alarm after no_face_alarm_s
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.smoothed = None
        self.closed = False
        self.closed_since = None
        self.last_t = None
        self.face_lost_since = None
        self.history = deque()
        self.counts = {"blink": 0, "microsleep": 0, "long_closure": 0, "no_face_alarm": 0}
        self.events = []
        self.no_face_alarm_on = False

    def _add_history(self, t, dt, closed):
        if dt > 0:
            self.history.append((t, dt, closed))
        while self.history and self.history[0][0] < t - self.cfg["perclos_window_s"]:
            self.history.popleft()

    def perclos(self):
        total = sum(d for _, d, _ in self.history)
        return sum(d for _, d, c in self.history if c) / total if total > 0 else 0.0

    def _event(self, kind, t, duration):
        self.counts[kind] += 1
        self.events.append({"event": kind, "time_s": round(t, 3), "duration_s": round(duration, 3)})
        return kind

    def _end_closure(self, t):
        duration = t - self.closed_since
        if duration < self.cfg["blink_max_s"]:
            kind = "blink"
        elif duration < self.cfg["microsleep_max_s"]:
            kind = "microsleep"
        else:
            kind = "long_closure"
        self.closed, self.closed_since = False, None
        return self._event(kind, t, duration)

    def update(self, t, p_open):
        """p_open is None when no face is visible. Returns (state, event, closure_s, alarm)."""
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        event = ""

        if p_open is None:
            if self.face_lost_since is None:
                self.face_lost_since = t
                if self.closed:
                    self.events.append({"event": "closure_interrupted_no_face", "time_s": round(t, 3),
                                        "duration_s": round(t - self.closed_since, 3)})
                self.closed, self.closed_since, self.smoothed = False, None, None
            lost_for = t - self.face_lost_since
            if lost_for >= self.cfg["no_face_alarm_s"] and not self.no_face_alarm_on:
                self.no_face_alarm_on = True
                event = self._event("no_face_alarm", t, lost_for)
            return "NO_FACE", event, 0.0, self.no_face_alarm_on

        self.face_lost_since = None
        self.no_face_alarm_on = False
        if self.smoothed is None:
            self.smoothed = p_open
        else:
            alpha = 1.0 - math.exp(-dt / self.cfg["smoothing_tau_s"]) if dt > 0 else 0.0
            self.smoothed += alpha * (p_open - self.smoothed)

        self._add_history(t, dt, self.closed)
        if not self.closed and self.smoothed < self.cfg["threshold_close"]:
            self.closed, self.closed_since = True, t
        elif self.closed and self.smoothed > self.cfg["threshold_open"]:
            event = self._end_closure(t)

        closure_s = t - self.closed_since if self.closed else 0.0
        alarm = self.closed and closure_s >= self.cfg["microsleep_max_s"]
        return ("CLOSED" if self.closed else "OPEN"), event, closure_s, alarm

    def finish(self, t):
        if self.closed:
            self._end_closure(t)


try:
    import winsound

    def _beep():
        winsound.Beep(1000, 400)
except ImportError:
    def _beep():
        print("\a", end="", flush=True)


def beep():
    threading.Thread(target=_beep, daemon=True).start()


def main():
    cfg = load_config()
    model_path, face_path = HERE / cfg["model_file"], HERE / cfg["face_model_file"]
    for path in (model_path, face_path):
        if not path.exists():
            raise SystemExit(f"Missing file: {path}  (see README)")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = load_eye_model(model_path)
    camera = cv2.VideoCapture(cfg["camera_index"])
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {cfg['camera_index']}.")

    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(log_dir / f"frames_{stamp}.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["timestamp", "t_s", "frame", "face", "p_open_raw", "p_open_smoothed", "ear", "state",
                  "closure_s", "perclos", "event", "alarm"])

    tracker = DrowsinessTracker(cfg)
    start = time.perf_counter()
    last_ts, frame_idx, fps, last_time = -1, 0, 0.0, start
    last_beep = 0.0

    with create_landmarker(face_path) as landmarker:
        print("Running. Press Q or Esc to quit.")
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read from the camera.")
                break
            now = time.perf_counter()
            t = now - start
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            frame_idx += 1

            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            display = frame.copy()
            height, width = display.shape[:2]
            eyes = extract_eyes(frame, result.face_landmarks[0], cfg["crop_size"], cfg["crop_scale"]) \
                if result.face_landmarks else []

            p_raw, ear = None, None
            if len(eyes) == 2:
                batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(e["crop"], cv2.COLOR_BGR2RGB))) for e in eyes])
                with torch.no_grad():
                    p_eyes = torch.softmax(model(batch), dim=1)[:, 1].numpy()
                p_raw = float(p_eyes.mean())
                lm = result.face_landmarks[0]
                ear = (eye_aspect_ratio(lm, RIGHT_EAR, width, height) + eye_aspect_ratio(lm, LEFT_EAR, width, height)) / 2

            state, event, closure_s, alarm = tracker.update(t, p_raw)
            perclos = tracker.perclos()

            if state == "NO_FACE":
                color = (0, 255, 255)
                cv2.putText(display, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if alarm:
                    cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 14)
                    cv2.putText(display, "FACE NOT VISIBLE - CHECK DRIVER!", (20, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                color = (0, 0, 255) if state == "CLOSED" else (0, 255, 0)
                for eye, p in zip(eyes, p_eyes):
                    cv2.polylines(display, [eye["box"]], True, color, 2)
                    x, y = eye["box"].min(axis=0)
                    cv2.putText(display, f"{'OPEN' if p > cfg['threshold'] else 'CLOSED'} {p:.2f}",
                                (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                for i, eye in enumerate(eyes):
                    x0 = width - 106 * (2 - i)
                    if x0 >= 0:
                        display[10:106, x0:x0 + 96] = cv2.resize(eye["crop"], (96, 96))
                cv2.putText(display, f"Eyes: {state}  ({tracker.smoothed:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"EAR {ear:.3f}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if alarm:
                    cv2.rectangle(display, (0, 0), (width - 1, height - 1), (0, 0, 255), 14)
                    cv2.putText(display, f"DROWSINESS ALERT! {closure_s:.1f}s", (20, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

            if alarm and now - last_beep > 1.0:
                beep()
                last_beep = now

            pc_color = (0, 0, 255) if perclos >= cfg["perclos_warning"] else (255, 255, 255)
            cv2.putText(display, f"PERCLOS {perclos * 100:.1f}%", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pc_color, 2)
            c = tracker.counts
            fps = 0.9 * fps + 0.1 / max(1e-6, now - last_time)
            last_time = now
            cv2.putText(display, f"Blinks {c['blink']}  Microsleeps {c['microsleep']}  Long {c['long_closure']}  FPS {fps:.0f}",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            log.writerow([datetime.now().isoformat(timespec="milliseconds"), round(t, 3), frame_idx,
                          int(state != "NO_FACE"),
                          "" if p_raw is None else round(p_raw, 4),
                          "" if tracker.smoothed is None else round(tracker.smoothed, 4),
                          "" if ear is None else round(ear, 4),
                          state, round(closure_s, 3), round(perclos, 4), event, int(alarm)])

            cv2.imshow("Eye State Detection - press Q to quit", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    tracker.finish(time.perf_counter() - start)
    camera.release()
    cv2.destroyAllWindows()
    log_file.close()
    summary = {"session": stamp, "duration_s": round(time.perf_counter() - start, 1), "frames": frame_idx,
               "counts": tracker.counts, "final_perclos": round(tracker.perclos(), 4),
               "events": tracker.events, "config": cfg}
    (log_dir / f"events_{stamp}.json").write_text(json.dumps(summary, indent=2))
    print(f"Logs saved in {log_dir}")


if __name__ == "__main__":
    main()

# Real-Time Eye State Detection for Driver Drowsiness

A webcam frame goes through **MediaPipe Face Landmarker**, both eyes are cropped, and a fine-tuned **MobileNetV3-Small** gives the probability that the eyes are open. A frame-rate-independent tracker turns that into eye state, blink / microsleep / long-closure events, PERCLOS and alarms, and writes a per-frame log for the fusion layer.

```
webcam → MediaPipe (478 landmarks) → 2 eye crops (2.2 × eye width, rotation-aligned) → MobileNetV3-Small
       → time-based smoothing → OPEN / CLOSED with hysteresis → events, PERCLOS, alarms → CSV / JSON log
```

## Folder

```
notebooks/Eye_State_Detection_Final.ipynb   the whole project: data, 5 models, model selection, thresholds, tests, video test
demo.py                            live demo (reads config/demo_config.json)
benchmark.py                       time of every pipeline stage on your computer
requirements.txt
models/        v1.pt … v4.pt, eye_state_final.pt, face_landmarker.task      (written by the notebook)
config/        demo_config.json: crop scale, threshold, hysteresis, timings  (written by the notebook)
results/       tables (.csv), confusion matrices and training curves for every model, video timelines
logs/          per-frame CSV and event JSON (demo sessions and the video test)
README_results.md                  all numbers of the last notebook run       (written by the notebook)
notebooks/history/                 the earlier experiment notebook (v2, CEW, EAR, v3)
```

## How to run

1. Open `Eye_State_Detection_Final.ipynb` in Google Colab, choose a T4 GPU, **Run all**.
   It reads the original files in `eye_project/` (dataset zip, CEW zip, `split.csv`, face model) and writes everything into `eye_project/Eye_State_Detection_Final/`.
   `RETRAIN = False` reuses models already in `models/`; delete a model file (or set `RETRAIN = True`) to train it again.
   FL3D and NITYMED are downloaded from Kaggle with `kagglehub`; if Colab asks, add `KAGGLE_USERNAME` and `KAGGLE_KEY` under Secrets.
2. Copy `models/` and `config/` from the Drive folder next to `demo.py` and run:
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   python demo.py
   ```

## Models

| Model | Training data | Augmentation |
|---|---|---|
| v1 | our dataset | crop, flip, rotation ±10°, brightness/contrast |
| v2 | our dataset | v1 + Gaussian blur + low-resolution simulation |
| cew_only | CEW train split | as v2 |
| v3 | our dataset + CEW train split | as v2 |
| v4 | our dataset + CEW train split + FL3D train drivers | as v2 |

All: MobileNetV3-Small pretrained on ImageNet, last layer 1000 → 2, grayscale 112 × 112, AdamW (lr 3e-4, weight decay 1e-4), cosine schedule, 15 epochs, best epoch on validation, seed 0.
**The final model is the one with the best mean validation accuracy**; test sets are only used for reporting. All numbers are in `README_results.md` after a run.

## Data

| Dataset | Content | Split |
|---|---|---|
| Our dataset | 5,452 eye crops: 4,000 MRL Eye (infrared, 16 persons) + 1,452 colour images | by person (`split.csv`), 0 near-duplicates between train and val/test |
| CEW | 2,423 faces, 100 × 100 px, closed / open | 70% train (15% of it validation) / 30% test |
| FL3D | frames of the NITYMED night-driving videos, labelled alert / microsleep / yawning | by video: every video's frames go to one split only (25 train / 5 validation / 14 test videos; overlap checked = 0). Driver IDs are not in the file names, so the split is by video, not by driver |
| NITYMED | real night-driving videos with microsleeps | system test of the full pipeline (logs, events, timelines) |

FL3D labels are mapped `alert → open`, `microsleep → closed`; `yawning` frames are left out because the eye state is not defined there. Datasets are not included in this repository. CEW is for research use only and must not be redistributed; NITYMED/FL3D must be cited as their pages ask.

## Alarm test on real driving videos

Notebook part 14 (`notebooks/cell_14_split_and_alarm_check.py`, `notebooks/cell_14b_check_false_alarms.py`) compares the system's closures with the FL3D frame labels on the 4 NITYMED test videos (8.7 labelled minutes; `frameN.jpg` = video frame N):

| Measure | Result |
|---|---|
| Labelled microsleeps ≥ 0.5 s detected | 47 / 48 (97.9%), median delay 0.58 s |
| Labelled microsleeps ≥ 1.5 s with an alarm | 20 / 23 (87%), median delay 1.58 s (alarm waits 1.5 s by design) |
| False alarms | 3 in 8.7 min (≈ 21 / hour, rough), all while the driver laughs or yawns with squeezed eyes |
| FL3D split overlap (train/val/test videos) | 0 / 0 / 0 |

## Demo logic (`demo.py`)

| Item | How |
|---|---|
| Crop scale, threshold, hysteresis | read from `config/demo_config.json`, the same values the notebook used for testing; threshold = best balanced accuracy on validation, hysteresis band = thresholds within 0.5 points of the best (±0.02 … ±0.10) |
| Smoothing | time-based: `s += (1 − exp(−dt/τ)) · (p − s)`, τ = 0.10 s, so behaviour does not depend on FPS |
| Events | closure < 0.5 s = blink, 0.5–1.5 s = microsleep, ≥ 1.5 s = long closure (alarm while it lasts); every event is logged with its duration |
| PERCLOS | time-weighted share of closed time in the last 60 s (face visible only); shown on screen, red above 15% (configurable) |
| No face | a closure in progress is ended without counting it; no face for 2 s raises a separate alarm (head dropped, driver out of view) |
| Log | `logs/frames_<time>.csv`: timestamp, t_s, frame, face, p_open_raw, p_open_smoothed, ear, state, closure_s, perclos, event, alarm; `logs/events_<time>.json`: counts, events, config |

## References

1. F. Song, X. Tan, X. Liu, S. Chen. *Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients.* Pattern Recognition, 2014. (CEW)
2. A. Howard et al. *Searching for MobileNetV3.* ICCV 2019.
3. Y. Kartynnik et al. *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.* CVPR Workshops 2019.
4. R. Florez et al. *A CNN-Based Approach for Driver Drowsiness Detection by Real-Time Eye State Identification.* Applied Sciences 13(13), 7849, 2023.
5. T. Soukupová, J. Čech. *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW 2016. (EAR)
6. NITYMED and FL3D: cite the papers listed on their dataset pages.

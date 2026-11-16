# Real-Time Eye State Detection (Open / Closed) for Driver Drowsiness

Graduation project. A webcam frame goes through **MediaPipe Face Landmarker**, both eyes are cropped, and a fine-tuned **MobileNetV3-Small** decides whether each eye is open or closed. If the eyes stay closed longer than 1.5 s, the system raises a drowsiness alarm.

```
Webcam frame → MediaPipe (478 face landmarks) → eye crops (2.2 × eye width, rotation-aligned)
             → MobileNetV3-Small (P(open) per eye, average of both) → blink counter / drowsiness alarm
```

## Results

Final model: **v3**, trained on our dataset + 70% of CEW.

| Method | Trained on | Our test set (971 eye images, unseen persons) | CEW test (726 faces) |
|---|---|---|---|
| CNN v2 | Our data | 99.79% | 86.64% |
| CNN | CEW 70% only | 80.12% | 98.76% |
| **CNN v3 (final)** | **Both** | **99.79%** | **97.25%** |
| EAR baseline (tuned threshold) | No training | needs a full face | 95.32% |

- Closed eyes missed on our test set: **0**
- AUC on CEW test: v3 **0.997**, EAR 0.995
- Each CNN does well on data like its own training data (domain shift); training on both closed the gap.

**Speed** (full pipeline, laptop CPU AMD Ryzen, no GPU): **46.8 ms per frame ≈ 21 FPS**
(MediaPipe 21.5 ms · eye crop 0.5 ms · pre-processing 2.9 ms · MobileNetV3-Small 21.8 ms · EAR 0.1 ms).
Model size: **1.52 M parameters, 6.2 MB**.

## Method

- **Data:** 5,452 pre-cropped eye images (2,726 closed / 2,726 open): 4,000 from the MRL Eye Dataset (16 persons) + 1,452 colour images from a second source.
- **Split by person:** 3,747 train / 734 validation / 971 test. A perceptual-hash check found 0 near-duplicates between train and val/test.
- **Model:** MobileNetV3-Small pretrained on ImageNet, last layer 1000 → 2 classes, grayscale 112 × 112 input, BatchNorm momentum 0.1.
- **Training:** AdamW (lr 3e-4, weight decay 1e-4), cosine schedule, 15 epochs, best checkpoint on validation accuracy. Augmentation: random crop, flip, rotation ±10°, brightness/contrast, blur and low-resolution simulation.
- **External test:** CEW (Closed Eyes in the Wild), full faces through the same MediaPipe pipeline.

## Repository

```
demo.py          live webcam demo (model v3), blink counter + drowsiness alarm
benchmark.py     measures the time of every pipeline stage on your computer
requirements.txt
models/          put the two model files here (see Setup)
notebooks/       Google Colab notebooks: training, testing, CEW, EAR, v3
results/         confusion matrices, training curves, EAR histogram
```

## Setup

```bash
pip install -r requirements.txt
```

Put these two files in `models/`:

- `eye_state_mnv3s_v3.pt`: the trained model (included in this repo, or from the release page)
- `face_landmarker.task`: MediaPipe model, download from
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Run:

```bash
python demo.py
```

Press **Q** or **Esc** to quit. If the wrong camera opens, change `CAMERA_INDEX` in `demo.py`.

## Datasets

The datasets are **not** included in this repository.

- **MRL Eye Dataset:** http://mrl.cs.vsb.cz/eyedataset
- **CEW (Closed Eyes in the Wild):** for research use only and may not be redistributed. Request it from the authors' page: https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html

## References

1. F. Song, X. Tan, X. Liu, S. Chen. *Eyes Closeness Detection from Still Images with Multi-scale Histograms of Principal Oriented Gradients.* Pattern Recognition, 2014. (CEW dataset)
2. A. Howard et al. *Searching for MobileNetV3.* ICCV 2019.
3. Y. Kartynnik et al. *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.* CVPR Workshops 2019. (MediaPipe face mesh)
4. R. Florez et al. *A CNN-Based Approach for Driver Drowsiness Detection by Real-Time Eye State Identification.* Applied Sciences 13(13), 7849, 2023.
5. T. Soukupová, J. Čech. *Real-Time Eye Blink Detection using Facial Landmarks.* CVWW 2016. (EAR)

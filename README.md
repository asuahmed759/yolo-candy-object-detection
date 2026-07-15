# 🍬 Candy Object Detection with YOLO26

A custom object detection model that identifies and localizes 11 different candy brands in images, built with **Ultralytics YOLO26** and trained on a custom-labeled dataset.

## Overview

This project trains a real-time object detection model to recognize candy packages/pieces from images. The model was trained locally on a custom dataset, using GPU acceleration for both training and inference.

## Classes Detected

The model detects 11 candy classes:

- MMs_peanut
- MMs_regular
- airheads
- gummy_worms
- milky_way
- nerds
- skittles
- snickers
- starbust
- three_musketeers
- twizzlers

## Tech Stack

- **Model:** YOLO26 (small variant — `yolo26s.pt`)
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics) (v8.4.95)
- **Backend:** PyTorch 2.5.1 with CUDA 12.1
- **Hardware:** NVIDIA RTX 4050 (Laptop GPU)
- **Language:** Python 3.10

## Dataset

- Custom-labeled candy dataset in YOLO format (`images/` + `labels/` with a `classes.txt` labelmap)
- Split 80/20 into training and validation sets
- Trained at `640x640` resolution

## Training Configuration

| Parameter | Value |
|---|---|
| Model | `yolo26s.pt` |
| Epochs | 60 |
| Image size | 640x640 |
| Device | GPU (CUDA) |

```bash
yolo detect train data=data.yaml model=yolo26s.pt epochs=60 imgsz=640 device=0
```

## Results

| Metric | Score |
|---|---|
| Precision (avg) | ~0.84 |
| Recall (avg) | ~0.70 |
| mAP50 | ~0.74 |
| mAP50-95 | ~0.35 |
| Inference speed | ~3.4 ms/image |

Training and validation loss curves, along with precision/recall/mAP progression, are available in [`runs/detect/train/results.png`](runs/detect/train/results.png).

The confusion matrix ([`runs/detect/train/confusion_matrix.png`](runs/detect/train/confusion_matrix.png)) shows strong performance on most classes (e.g. `airheads`, `three_musketeers`, `skittles`), with `twizzlers` being the weakest class — likely due to fewer training examples for that class.



## Project Structure

```
Yolo_Object_Detection/
├── data.yaml                  # Dataset config (paths + class names)
├── Yolo_object_Detection.ipynb  # Full training notebook
├── Yolo_Candy_Dataset/        # Raw images + YOLO-format labels
├── data/                      # Train/validation split (generated)
│   ├── train/
│   └── validation/
├── runs/detect/train/         # Training outputs
│   ├── weights/
│   │   ├── best.pt            # Best model checkpoint
│   │   └── last.pt
│   ├── results.png            # Training curves
│   └── confusion_matrix.png
└── README.md
```

## How to Run

### 1. Set up environment

```bash
python -m venv objectdetectvenv
objectdetectvenv\Scripts\activate      # Windows
pip install ultralytics
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run inference on your own images

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image_or_folder save=True
```

### 3. Run on a webcam (live detection)

```bash
python yolo_detect.py --model runs/detect/train/weights/best.pt --source usb0 --resolution 1280x720
```

## Future Improvements

- Add more training images for the `twizzlers` class to improve recall
- Export model to ONNX/TFLite for deployment on edge devices (e.g. Raspberry Pi)
- Deploy as a simple Streamlit web app for interactive detection

## Acknowledgements

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)

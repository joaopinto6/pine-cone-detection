# Automated Strobili Detection in UAV Imagery

This repository contains the official codebase for the Master's Thesis: **"Prediction of Forestry Profit using Machine Learning"**. 

It provides an end-to-end deep learning pipeline to detect developing pine cones (strobili) in high-resolution Unmanned Aerial Vehicle (UAV) imagery. Because *Pinus pinea* strobili require three years to mature, early detection at the canopy apices allows forestry managers to accurately forecast harvest yields up to three years in advance.

To solve the "small object detection gap" (detecting ~20px objects inside 5K resolution images), this pipeline integrates **YOLOv8** with **Slicing Aided Hyper Inference (SAHI)**, augmented by a custom heuristic Canopy Region of Interest (ROI) filter to minimize computational overhead.

## 🌟 Key Features

*   **Canopy ROI Filtering:** Uses HSV color-space thresholding and morphological operations to isolate the tree canopy, automatically discarding background tiles (dirt, roads) to accelerate training and reduce false positives.
*   **Smart Tiling & Image-Level Splitting:** Slices 5K images into 640x640 patches while strictly enforcing train/val/test splits at the macro-image level to prevent data leakage.
*   **SAHI Integration:** Employs Slicing Aided Hyper Inference to detect extremely small objects natively without destructive image downscaling.
*   **Unified CLI Controller:** A clean `main.py` entry point to manage indexing, preprocessing, training, and evaluation.
*   **Benchmarking & Visualization:** Standalone scripts to calculate inference latency (FPS) and generate publication-ready detection overlays.

## 📁 Repository Structure

```text
.
├── main.py                     # Main CLI controller for the pipeline
├── requirements.txt            # Python dependencies
├── configs/
│   └── default.yaml            # Hyperparameters for ROI, Tiling, and YOLOv8
├── data/
│   ├── masks/                  # Binary ground-truth masks
│   └── raw/                    # High-res 5472x3648 UAV images
├── fine-tuned-models/          
│   ├── yolov8nano-best.pt      # Pre-trained Nano weights (F1: 0.9868)
│   └── yolov8small-best.pt     # Pre-trained Small weights (F1: 0.9894)
└── src/
    ├── bench_inference_speed.py # Script for latency and throughput metrics
    ├── evaluate.py              # SAHI inference and IoU metrics calculation
    ├── index.py                 # Maps raw images to ground-truth masks
    ├── preprocess.py            # Applies ROI filter, tiles images, and formats labels
    ├── roi.py                   # Classical OpenCV canopy isolation logic
    ├── train.py                 # YOLOv8 fine-tuning with custom augmentations
    └── vis_predictions.py       # Generates overlay images (TP/FP/FN)
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/joaopinto6/pine-cone-detection.git
cd pine-cone-detection
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies include `ultralytics`, `sahi`, `opencv-python`, `numpy`, `pyyaml`, and `torch`)*

## 🚀 Pipeline Usage

The pipeline is controlled sequentially via the `main.py` script. 

### 1. Data Indexing
Pairs the raw images with their corresponding ground-truth masks and generates a `manifest.csv` file.
```bash
python main.py index --images-dir data/raw --masks-dir data/masks --output data/manifest.csv
```

### 2. Preprocessing & Tiling
Reads the manifest, applies the Canopy ROI filter, slices the images into 640x640 overlapping tiles, converts binary masks to YOLO bounding boxes, and generates the final dataset splits.
```bash
python main.py preprocess --manifest data/manifest.csv --config configs/default.yaml --output-root data/yolo_dataset
```

### 3. Model Training (Fine-Tuning)
Trains a YOLOv8 model (Nano or Small) on the preprocessed tiles using heavy photometric and geometric augmentations defined in the configuration file.
```bash
python main.py train --model-size s --batch 16 --epochs 150 --device 0
```
*Note: To train the Nano model, use `--model-size n`.*

### 4. SAHI Evaluation
Evaluates the fine-tuned model against the original, high-resolution test images using SAHI. Computes Precision, Recall, Mean IoU, and F1-score across various Confidence and NMS thresholds.
```bash
python main.py evaluate --model-path fine-tuned-models/yolov8small-best.pt --dataset-scope test
```

## 📊 Visualization and Benchmarking

The `src` directory contains standalone scripts for qualitative analysis and hardware benchmarking.

**Generate Detection Overlays:**
Outputs high-resolution images with color-coded bounding boxes (Green: True Positive, Red: False Positive, Yellow: False Negative).
```bash
python src/vis_predictions.py --model-path fine-tuned-models/yolov8small-best.pt --manifest data/manifest.csv --conf-threshold 0.45
```

**Benchmark Inference Speed:**
Calculates the average inference latency (milliseconds per image) and throughput (FPS) between the Nano and Small architectures on CPU or GPU.
```bash
python src/bench_inference_speed.py --device cpu
```

## 🛠️ Configuration (`default.yaml`)

All pipeline hyperparameters are centralized in `configs/default.yaml`. You can tune the ROI color-space thresholds, tiling overlap, train/val/test split ratios, and YOLOv8 training parameters (e.g., mosaic augmentation, learning rates) directly from this file without altering the source code.

## 📜 License & Citation

This project was developed as part of a Master's Thesis in Computer Science and Engineering at Instituto Superior Técnico (Universidade de Lisboa). 

If you use this code or the provided pre-trained models in your research, please refer to the accompanying thesis document for formal citation details.
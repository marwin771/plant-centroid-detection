# Plant Centroid Detection

This repository implements two approaches for **counting plants and localizing their centroids** in top‑down agricultural images:

1. **Classical Computer Vision (CV) Baseline**
2. **Machine Learning (ML) YOLOv8‑based Detection**

Both pipelines take `.bmp` images as input and output:
- the **number of plants**
- the **(x, y) coordinates of each plant centroid**
- visual outputs showing detections and masks
- a CSV summary of centroids per image

---

## Project Structure

- **cv_baseline** 
  - **green_segmentation_dbscan.py**: Classical computer vision pipeline using green pixel segmentation and DBSCAN clustering.
- **ml_yolo_detection**
  - **train_yolo.py**: Script for training YOLOv8 models on annotated plant datasets.
  - **yolo_inference.py**: Performs inference with a trained YOLOv8 model (`.pt` or `.onnx`).
  - **export_and_quantize.py**: Exports the trained YOLOv8 model to ONNX FP32 and also converts to FP16. Reports model size too for each model.
  - **ML+CV_hybrid.py**: Combines YOLOv8 detections with CV segmentation for improved centroid localization.
  - **yolov8n.pt** YOLOv8 pretrained model used for detection.
  - **runs/detect/train/**: Directory containing the training outputs and checkpoints. Includes both `.pt` and `.onnx` versions of the trained models in **weights/**.
  - **dataset/**: images and labels(bbox) for train, test, val
- **report.pdf**:  
- **requirements.txt**: Python dependencies  
- **README.md**: Project documentation

---

## Problem Overview

Given overhead 2D images of plants, the task is to:

1. **Count** the number of plants in each image  
2. **Predict the centroids** (x, y) of each plant

---

## Classical CV Baseline

The baseline uses:
- **Green pixel segmentation** (HSV thresholding) + morphological clean‑up
- **DBSCAN clustering** to group pixels belonging to the same plant and calculate centroid for each cluster

### Usage

```bash
python green_segmentation_dbscan.py
```

## Outputs
- **output_binary_masks/**: binary masks (white plants on black)
- **output_plants_centroid/**: colored plant clusters + centroids
- **baseline_plant_centroids.csv**: centroid list per image
- It will also print per‑image and average processing times for segmentation, DBSCAN and total

This provides a quick and simple baseline without deep learning.

---

## ML‑Based YOLOv8 Solution

A custom YOLOv8‑nano model was trained on annotated plant data (YOLO format).
The ML pipeline outputs bounding boxes and centroids per image.

### Inference Script

```bash
python yolo_inference.py
```
This script will:
- Run inference with the trained model (`.pt` or `.onnx`)
- Draw bounding boxes and centroids on each image
- Save results in **results/...inference_results/**
- Create a CSV with coordinates for each image
- Print per‑image and average inference time

---

## BONUS ML + CV Hybrid Approach
**ML+CV_hybrid.py**: Combines YOLOv8 object detections with classical computer vision segmentation to improve plant centroid localization.

- **Motivation**: Centroids computed from YOLO bounding boxes are approximate and may not represent the true plant center.
- **Solution**: Apply green color segmentation to the detected bounding box region to generate a mask of the plant. The mask allows precise calculation of the actual centroid.
- **Output**: Refined centroids, visualizations of masks overlaid on images, and CSV files with accurate coordinates.

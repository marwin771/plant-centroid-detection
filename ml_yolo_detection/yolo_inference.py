from ultralytics import YOLO
import cv2
import os
import glob
import csv
import time
import numpy as np

# Load trained YOLOv8-nano model (.pt/.onnx)
model = YOLO("runs/detect/train/weights/best_fp16.onnx")

# Input and output folders
test_folder = "dataset/images/test"
output_folder = "fp16onnx_inference_results"
os.makedirs(output_folder, exist_ok=True)

# Prepare CSV file
csv_path = os.path.join(output_folder, "yolo_test_centroids.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["image", "centroids"])  # header

# Timing lists
times = []

# Loop over test images
for img_path in glob.glob(os.path.join(test_folder, "*.bmp")):
    img = cv2.imread(img_path)

    start_time = time.time()
    # Run inference
    results = model.predict(
        img_path,
        imgsz=640,
        conf=0.25,
        iou=0.3
    )
    end_time = time.time()
    times.append(end_time - start_time)

    # results[0] contains detections for the image
    dets = results[0].boxes  # Boxes object

    centroids = []
    for box in dets:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Compute centroid
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centroids.append((cx, cy))
        # Draw centroid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    # Save visualized image
    out_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(out_path, img)

    # Write to CSV
    writer.writerow([os.path.basename(img_path), centroids])

    print(f"{os.path.basename(img_path)}: {len(centroids)} plants detected | Inference time: {times[-1]:.3f}s")

csv_file.close()

# Average runtime report
print(f"\nProcessed {len(times)} images")
print(f"Average YOLO inference time per image (CPU): {np.mean(times):.3f} seconds")
print(f"All results saved in {output_folder} including centroids CSV.")

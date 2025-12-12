from ultralytics import YOLO
import cv2
import os
import glob
import csv
import numpy as np
import time

# ---------- Green segmentation ----------
def segment_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphology to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

# ---------- Compute centroid ----------
def compute_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cx = int(xs.mean())
    cy = int(ys.mean())
    return cx, cy

# ---------- Main processing with timing ----------
def process_yolo_green(model_path, test_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    mask_folder = os.path.join(output_folder, "binary_masks")
    color_folder = os.path.join(output_folder, "colored_plants")
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    # Load YOLO model
    model = YOLO(model_path)

    # Prepare CSV
    csv_path = os.path.join(output_folder, "ml_green_centroids.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["image", "centroids"])

    yolo_times = []
    green_times = []
    total_times = []

    # Loop over test images
    for img_path in glob.glob(os.path.join(test_folder, "*.bmp")):
        img = cv2.imread(img_path)
        start_total = time.time()

        # YOLO inference timing
        start_yolo = time.time()
        results = model.predict(img_path, imgsz=640, conf=0.25, iou=0.5)
        end_yolo = time.time()
        yolo_times.append(end_yolo - start_yolo)

        dets = results[0].boxes

        # Prepare output images
        binary_mask_total = np.zeros(img.shape[:2], dtype=np.uint8)
        colored_img = np.zeros_like(img)
        centroids_all = []

        start_green = time.time()
        for i, box in enumerate(dets):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            green_mask = segment_green(crop)

            # Compute centroid inside bbox
            centroid = compute_centroid(green_mask)
            if centroid is not None:
                cx_global = centroid[0] + x1
                cy_global = centroid[1] + y1
                centroids_all.append((cx_global, cy_global))

            # Update binary mask (all plants white)
            binary_mask_total[y1:y2, x1:x2][green_mask > 0] = 255

            # Color each plant differently in the colored mask
            color = np.random.randint(50, 255, size=3)
            mask_indices = np.where(green_mask > 0)
            for c in range(3):
                colored_img[y1:y2, x1:x2, c][mask_indices] = color[c]

            # Draw centroid on colored image
            if centroid is not None:
                cv2.circle(colored_img, (cx_global, cy_global), 5, (0, 0, 255), -1)

            # Optional: draw YOLO bbox on original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        end_green = time.time()
        green_times.append(end_green - start_green)

        end_total = time.time()
        total_times.append(end_total - start_total)

        # Save outputs
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(mask_folder, base_name), binary_mask_total)
        cv2.imwrite(os.path.join(color_folder, base_name), colored_img)
        cv2.imwrite(os.path.join(output_folder, "bbox_overlay_" + base_name), img)

        # Save centroids to CSV
        writer.writerow([base_name, centroids_all])
        print(f"{base_name}: {len(centroids_all)} plants detected | Total time: {total_times[-1]:.3f}s")

    csv_file.close()

    # Report average runtimes
    print("\n===== Inference timing summary (CPU) =====")
    print(f"Average YOLO time per image: {np.mean(yolo_times):.3f}s")
    print(f"Average green segmentation + centroid time per image: {np.mean(green_times):.3f}s")
    print(f"Average total time per image: {np.mean(total_times):.3f}s")
    print(f"All results saved in {output_folder}, including centroids CSV.")

# ---------------- Run script ----------------
if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"  # your trained YOLO model
    test_folder = "dataset/images/test"
    output_folder = "hybrid_inference_results"
    process_yolo_green(model_path, test_folder, output_folder)

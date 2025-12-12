import cv2
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN
import time
import csv

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

# ---------- Compute plant centroids ----------
def plant_centroids_from_mask(mask, eps=9, min_samples=1):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [], {}  # empty
    points = np.column_stack((xs, ys))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    centroids = []
    plant_clusters = {}
    for label in unique_labels:
        if label == -1:
            continue  # noise
        cluster_points = points[labels == label]
        cx = cluster_points[:, 0].mean()
        cy = cluster_points[:, 1].mean()
        centroids.append((cx, cy))
        plant_clusters[label] = cluster_points
    return centroids, plant_clusters

# ---------- Save masks ----------
def save_binary_mask(mask, save_path):
    mask_vis = np.zeros_like(mask)
    mask_vis[mask > 0] = 255
    cv2.imwrite(save_path, mask_vis)

def save_colored_plants(mask, plant_clusters, centroids, save_path):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (label, points) in enumerate(plant_clusters.items()):
        color = [int(c) for c in np.random.randint(50, 255, 3)]
        for x, y in points:
            for c in range(3):
                color_img[y, x, c] = color[c]
    # Draw centroids
    for cx, cy in centroids:
        cv2.circle(color_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, color_img)

# ---------- Main processing ----------
def process_folder(input_folder, mask_output_folder, color_output_folder):
    bmp_files = glob.glob(os.path.join(input_folder, '*.bmp'))
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(color_output_folder, exist_ok=True)
    results = {}

    green_times = []
    dbscan_times = []
    total_times = []

    for file in bmp_files:
        start_total = time.time()
        img = cv2.imread(file)

        # Green segmentation timing
        start_green = time.time()
        mask = segment_green(img)
        end_green = time.time()
        green_times.append(end_green - start_green)

        # DBSCAN centroid calculation timing
        start_dbscan = time.time()
        centroids, plant_clusters = plant_centroids_from_mask(mask)
        end_dbscan = time.time()
        dbscan_times.append(end_dbscan - start_dbscan)

        end_total = time.time()
        total_times.append(end_total - start_total)

        results[os.path.basename(file)] = centroids
        print(f"{os.path.basename(file)}: {len(centroids)} plants detected | "
              f"Green seg: {green_times[-1]:.3f}s, "
              f"DBSCAN: {dbscan_times[-1]:.3f}s, "
              f"Total: {total_times[-1]:.3f}s")

        # Save outputs
        mask_path = os.path.join(mask_output_folder, os.path.basename(file))
        color_path = os.path.join(color_output_folder, os.path.basename(file))
        save_binary_mask(mask, mask_path)
        save_colored_plants(mask, plant_clusters, centroids, color_path)

    # Report average runtime
    if bmp_files:
        print("\n===== Average processing times =====")
        print(f"Green segmentation: {np.mean(green_times):.3f}s per image")
        print(f"DBSCAN centroid calc: {np.mean(dbscan_times):.3f}s per image")
        print(f"Total: {np.mean(total_times):.3f}s per image")

    return results

# ---------------- Run script ----------------
if __name__ == "__main__":
    input_folder = "all_thumbnails"
    mask_output_folder = "output_binary_masks"
    color_output_folder = "output_plants_centroid"
    results = process_folder(input_folder, mask_output_folder, color_output_folder)

    # save centroids to CSV
    with open('baseline_plant_centroids.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'centroids'])
        for img_name, c_list in results.items():
            centroids_clean = [(round(float(cx), 1), round(float(cy), 1)) for cx, cy in c_list]
            writer.writerow([img_name, centroids_clean])

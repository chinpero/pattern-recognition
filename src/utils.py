import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


KEYPOINT_NAMES = [
    "TeardropR", "TeardropL", "TiR", "TiL", "FHR", "FHL",
    "tonnisR1", "tonnisR2", "tonnisL1", "tonnisL2",
]


def mask_sensitive_region(image_path: str, output_path: str, box_size: int = 128):
    """Cover the top-left region of an image with a black rectangle."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image[0:box_size, 0:box_size] = 0
    cv2.imwrite(output_path, image)


def save_predictions_csv(
    output_path: str, predictions: List[Tuple[str, List[Tuple[float, float]]]]
):
    header = ["image_path"]
    for name in KEYPOINT_NAMES:
        header.extend([f"{name}_x", f"{name}_y"])
    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for image_path, keypoints in predictions:
            row = [image_path]
            for x, y in keypoints:
                row.extend([x, y])
            writer.writerow(row)


def load_predictions_csv(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load predictions from CSV file."""
    predictions = {}
    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            img = row["image_path"]
            kps = []
            for name in KEYPOINT_NAMES:
                x = float(row.get(f"{name}_x", -1))
                y = float(row.get(f"{name}_y", -1))
                kps.append((x, y))
            predictions[img] = kps
    return predictions


def denormalize_keypoints(
    keypoints: List[Tuple[float, float]], image_shape: Tuple[int, int]
):
    h, w = image_shape
    return [(float(x * w), float(y * h)) for x, y in keypoints]


def keypoint_distance(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Euclidean distance between two keypoints."""
    if pred[0] < 0 or pred[1] < 0 or gt[0] < 0 or gt[1] < 0:
        return float("nan")
    return float(np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2))


def evaluate_keypoints(
    predictions: Dict[str, List[Tuple[float, float]]],
    ground_truths: Dict[str, List[Tuple[float, float]]],
    threshold: float = 10.0,
) -> Dict:
    """Evaluate keypoint predictions against ground truth.

    Returns a dictionary with per-keypoint and overall metrics.
    """
    tp = {name: 0 for name in KEYPOINT_NAMES}
    fp = {name: 0 for name in KEYPOINT_NAMES}
    fn = {name: 0 for name in KEYPOINT_NAMES}
    tn = {name: 0 for name in KEYPOINT_NAMES}
    distances = {name: [] for name in KEYPOINT_NAMES}

    for img_path, pred_kps in predictions.items():
        if img_path not in ground_truths:
            continue
        gt_kps = ground_truths[img_path]
        for i, name in enumerate(KEYPOINT_NAMES):
            pred, gt = pred_kps[i], gt_kps[i]
            gt_present = gt[0] >= 0 and gt[1] >= 0
            pred_present = pred[0] >= 0 and pred[1] >= 0

            if gt_present and pred_present:
                dist = keypoint_distance(pred, gt)
                distances[name].append(dist)
                if dist < threshold:
                    tp[name] += 1
                else:
                    fp[name] += 1
                    fn[name] += 1
            elif gt_present and not pred_present:
                fn[name] += 1
            elif not gt_present and pred_present:
                fp[name] += 1
            else:
                tn[name] += 1

    metrics = {}
    for name in KEYPOINT_NAMES:
        t = tp[name]
        f_p = fp[name]
        f_n = fn[name]
        t_n = tn[name]
        acc = (t + t_n) / (t + f_p + f_n + t_n + 1e-8)
        prec = t / (t + f_p + 1e-8)
        rec = t / (t + f_n + 1e-8)
        spec = t_n / (t_n + f_p + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        mean_dist = np.nanmean(distances[name]) if distances[name] else float("nan")
        metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1": f1,
            "mean_distance": mean_dist,
        }

    all_tp = sum(tp.values())
    all_fp = sum(fp.values())
    all_fn = sum(fn.values())
    all_tn = sum(tn.values())
    all_dists = [d for dists in distances.values() for d in dists if not np.isnan(d)]
    metrics["overall"] = {
        "accuracy": (all_tp + all_tn) / (all_tp + all_fp + all_fn + all_tn + 1e-8),
        "precision": all_tp / (all_tp + all_fp + 1e-8),
        "recall": all_tp / (all_tp + all_fn + 1e-8),
        "specificity": all_tn / (all_tn + all_fp + 1e-8),
        "f1": 2 * all_tp / (2 * all_tp + all_fp + all_fn + 1e-8),
        "mean_distance": float(np.mean(all_dists)) if all_dists else float("nan"),
    }
    return metrics


def visualize_comparison(
    image_path: str,
    gt_keypoints: List[Tuple[float, float]],
    pred_keypoints: List[Tuple[float, float]],
    save_path: str,
):
    """Visualize ground truth vs predicted keypoints on the X-ray image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    for i, (name, gt, pred) in enumerate(
        zip(KEYPOINT_NAMES, gt_keypoints, pred_keypoints)
    ):
        if gt[0] >= 0 and gt[1] >= 0:
            ax.plot(gt[0], gt[1], "rs", markersize=6, markeredgewidth=1.5,
                    markerfacecolor="none")
        if pred[0] >= 0 and pred[1] >= 0:
            ax.plot(pred[0], pred[1], "bo", markersize=6, markerfacecolor="none")

    ax.plot([], [], "rs", markersize=6, markeredgewidth=1.5,
            markerfacecolor="none", label="Ground Truth")
    ax.plot([], [], "bo", markersize=6, markerfacecolor="none",
            label="Prediction")
    ax.legend(loc="lower right")
    ax.set_title("Keypoint Detection: GT vs Prediction")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

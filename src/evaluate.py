import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import parse_labelme_json, KEYPOINT_NAMES
from src.utils import (
    evaluate_keypoints,
    load_predictions_csv,
    visualize_comparison,
)


def load_ground_truths(data_dir: str) -> dict:
    """Load all ground truth keypoints from a LabelMe annotation directory."""
    label_dir = Path(data_dir) / "labels"
    image_dir = Path(data_dir) / "images"
    gts = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for json_path in sorted(label_dir.glob("*.json")):
        pts = parse_labelme_json(json_path)
        kps = []
        for name in KEYPOINT_NAMES:
            pt = pts.get(name, (-1.0, -1.0))
            kps.append((float(pt[0]), float(pt[1])))
        for ext in image_exts:
            img_path = image_dir / (json_path.stem + ext)
            if img_path.exists():
                gts[str(img_path)] = kps
                break
    return gts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DDH keypoint predictions")
    parser.add_argument("--predictions", required=True, help="CSV file from predict.py")
    parser.add_argument("--gt-dir", required=True, help="Directory with ground truth labels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--num-vis", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions_csv(args.predictions)
    gts = load_ground_truths(args.gt_dir)

    print(f"Loaded {len(preds)} predictions, {len(gts)} ground truths")

    common = set(preds.keys()) & set(gts.keys())
    preds_common = {k: preds[k] for k in common}
    gts_common = {k: gts[k] for k in common}
    print(f"Matching samples: {len(common)}")

    metrics = evaluate_keypoints(preds_common, gts_common, args.threshold)

    print("\n=== Per-Keypoint Metrics ===")
    print(f"{'Keypoint':<14} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'F1':>7} {'Dist':>7}")
    print("-" * 56)
    for name in KEYPOINT_NAMES:
        m = metrics[name]
        print(
            f"{name:<14} {m['accuracy']:7.4f} {m['precision']:7.4f} "
            f"{m['recall']:7.4f} {m['specificity']:7.4f} {m['f1']:7.4f} "
            f"{m['mean_distance']:7.2f}"
        )

    overall = metrics["overall"]
    print("-" * 56)
    print(
        f"{'Overall':<14} {overall['accuracy']:7.4f} {overall['precision']:7.4f} "
        f"{overall['recall']:7.4f} {overall['specificity']:7.4f} {overall['f1']:7.4f} "
        f"{overall['mean_distance']:7.2f}"
    )

    with open(output_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2, default=str)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")

    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    sample_paths = sorted(common)[: args.num_vis]
    for img_path in sample_paths:
        name = Path(img_path).stem
        save_path = vis_dir / f"{name}_comparison.png"
        visualize_comparison(img_path, gts_common[img_path], preds_common[img_path],
                             str(save_path))
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()

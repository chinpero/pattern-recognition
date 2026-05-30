import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.model import HeatmapKeypointModel

KEYPOINT_NAMES = [
    "TeardropR", "TeardropL", "TiR", "TiL", "FHR", "FHL",
    "tonnisR1", "tonnisR2", "tonnisL1", "tonnisL2",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict DDH hip keypoints")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--mask-sensitive", action="store_true")
    return parser.parse_args()


def decode_heatmaps(
    heatmaps: np.ndarray, original_shape: Tuple[int, int]
) -> List[Tuple[float, float]]:
    """Decode heatmaps to keypoint coordinates by finding peak positions."""
    h_orig, w_orig = original_shape
    num_kp, h_hm, w_hm = heatmaps.shape
    keypoints = []
    for i in range(num_kp):
        hm = heatmaps[i]
        idx = np.argmax(hm)
        cy, cx = idx // w_hm, idx % w_hm
        if hm[cy, cx] < 0.1:
            keypoints.append((-1.0, -1.0))
            continue
        x = (cx / w_hm) * w_orig
        y = (cy / h_hm) * h_orig
        keypoints.append((float(x), float(y)))
    return keypoints


def save_predictions_csv(
    output_file: str, predictions: List[Tuple[str, List[Tuple[float, float]]]]
):
    header = ["image_path"]
    for name in KEYPOINT_NAMES:
        header.extend([f"{name}_x", f"{name}_y"])
    with open(output_file, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for image_path, keypoints in predictions:
            row = [image_path]
            for x, y in keypoints:
                row.extend([x, y])
            writer.writerow(row)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeatmapKeypointModel(num_keypoints=10, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dir = Path(args.test_dir) / "images"
    image_paths = (
        sorted(test_dir.glob("*.jpg"))
        + sorted(test_dir.glob("*.png"))
        + sorted(test_dir.glob("*.BMP"))
    )
    predictions = []

    with torch.no_grad():
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: unable to read {image_path}")
                continue
            original_shape = image.shape[:2]

            if args.mask_sensitive:
                image[0:128, 0:128] = 0

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (args.image_size, args.image_size))
            image_tensor = (
                torch.from_numpy(image_resized.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            pred_hm = model(image_tensor)
            pred_hm = pred_hm.squeeze(0).cpu().numpy()
            keypoints = decode_heatmaps(pred_hm, original_shape)
            predictions.append((str(image_path), keypoints))

    save_predictions_csv(args.output_file, predictions)
    print(f"Prediction complete. Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

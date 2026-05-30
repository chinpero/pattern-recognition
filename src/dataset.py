import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

KEYPOINT_NAMES = [
    "TeardropR",
    "TeardropL",
    "TiR",
    "TiL",
    "FHR",
    "FHL",
    "tonnisR1",
    "tonnisR2",
    "tonnisL1",
    "tonnisL2",
]


def parse_labelme_json(json_path: Path) -> Dict[str, Tuple[float, float]]:
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    points = {}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in KEYPOINT_NAMES:
            continue
        coords = shape.get("points", [])
        shape_type = shape.get("shape_type", "point")
        if len(coords) == 0:
            continue
        if shape_type == "circle":
            (x1, y1), (x2, y2) = coords[0], coords[1]
            points[label] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        else:
            points[label] = list(coords[0])
    return points


def _gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (cx, cy) with given sigma."""
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap


def build_heatmap_target(
    keypoints: List[Tuple[float, float]],
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    """Build multi-channel heatmap target from keypoint coordinates.

    keypoints are in heatmap-space (pixel coordinates in [0, h) and [0, w)).
    """
    h, w = heatmap_size
    target = np.zeros((len(KEYPOINT_NAMES), h, w), dtype=np.float32)
    mask = np.zeros(len(KEYPOINT_NAMES), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if i >= len(KEYPOINT_NAMES):
            break
        if x < 0 or y < 0:
            continue
        target[i] = _gaussian_heatmap(h, w, x, y, sigma)
        mask[i] = 1.0
    return target, mask


def build_transforms(image_size: int = 512, is_train: bool = True):
    if is_train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
    else:
        return A.Compose(
            [A.Resize(image_size, image_size)],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )


class HipKeypointDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        heatmap_sigma: float = 2.0,
        transform: Optional[A.BasicTransform] = None,
        mask_sensitive: bool = False,
        is_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(
            list((self.data_dir / "images").glob("*.jpg"))
            + list((self.data_dir / "images").glob("*.png"))
            + list((self.data_dir / "images").glob("*.BMP"))
        )
        self.label_dir = self.data_dir / "labels"
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.mask_sensitive = mask_sensitive
        self.is_train = is_train
        self.transform = (
            transform if transform is not None else build_transforms(image_size, is_train)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.label_dir / (image_path.stem + ".json")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        if self.mask_sensitive:
            image = self._mask_sensitive_region(image)

        keypoints = self._load_keypoints(label_path)

        if self.transform is not None:
            # Albumentations expects list of [x, y] with native Python floats
            kps = [[float(p[0]), float(p[1])] for p in keypoints]
            augmented = self.transform(image=image, keypoints=kps)
            image = augmented["image"]
            keypoints = augmented["keypoints"]

        keypoints_flat, vis_mask = self._format_keypoints(keypoints, image.shape[:2])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image).float()

        heatmap_h = self.image_size // 4
        heatmap_w = self.image_size // 4
        heatmap_keypoints = []
        for i in range(0, len(keypoints_flat), 2):
            x_norm = keypoints_flat[i]
            y_norm = keypoints_flat[i + 1]
            if vis_mask[i] > 0:
                heatmap_keypoints.append(
                    (x_norm * heatmap_w, y_norm * heatmap_h)
                )
            else:
                heatmap_keypoints.append((-1.0, -1.0))

        heatmap_target, hm_mask = build_heatmap_target(
            heatmap_keypoints, (heatmap_h, heatmap_w), self.heatmap_sigma
        )

        return {
            "image": image_tensor,
            "heatmap": torch.from_numpy(heatmap_target).float(),
            "hm_mask": torch.from_numpy(hm_mask).float(),
            "image_path": str(image_path),
        }

    def _load_keypoints(self, label_path: Path) -> List[Tuple[float, float]]:
        if not label_path.exists():
            return [[-1.0, -1.0] for _ in KEYPOINT_NAMES]
        points = parse_labelme_json(label_path)
        keypoints = []
        for name in KEYPOINT_NAMES:
            if name in points:
                keypoints.append(points[name])
            else:
                keypoints.append([-1.0, -1.0])
        return keypoints

    def _format_keypoints(
        self, keypoints: List[Tuple[float, float]], image_shape: Tuple[int, int]
    ):
        h, w = image_shape
        flattened = []
        mask = []
        for x, y in keypoints:
            if x < 0 or y < 0:
                flattened.extend([0.0, 0.0])
                mask.extend([0.0, 0.0])
            else:
                flattened.extend([x / w, y / h])
                mask.extend([1.0, 1.0])
        return flattened, mask

    @staticmethod
    def _mask_sensitive_region(image: np.ndarray, box_size: int = 128) -> np.ndarray:
        image = image.copy()
        image[0:box_size, 0:box_size] = 0
        return image

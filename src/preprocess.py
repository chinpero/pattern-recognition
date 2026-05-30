"""Preprocess DDH dataset: mask sensitive regions and optionally anonymize filenames."""

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2


def anonymize_filename(original_name: str, keep_ext: bool = True) -> str:
    stem = Path(original_name).stem
    ext = Path(original_name).suffix if keep_ext else ""
    digest = hashlib.md5(stem.encode()).hexdigest()[:12]
    return f"{digest}{ext}"


def process_directory(
    input_dir: str,
    output_dir: str,
    box_size: int = 128,
    anonymize: bool = True,
):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_images = out_path / "images"
    out_labels = out_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    in_images = in_path / "images"
    in_labels = in_path / "labels"
    name_map = {}

    for ext in ("*.jpg", "*.png", "*.bmp", "*.BMP", "*.JPG", "*.PNG"):
        for img in in_images.glob(ext):
            new_stem = anonymize_filename(img.name) if anonymize else img.stem
            name_map[img.stem] = new_stem

            new_img = out_images / f"{new_stem}{img.suffix}"
            image = cv2.imread(str(img))
            if image is None:
                print(f"Warning: unable to read {img}")
                continue
            image[0:box_size, 0:box_size] = 0
            cv2.imwrite(str(new_img), image)

            label_file = in_labels / f"{img.stem}.json"
            if label_file.exists():
                shutil.copy2(str(label_file), str(out_labels / f"{new_stem}.json"))

    print(f"Processed {len(name_map)} images: {input_dir} -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess DDH dataset")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-size", type=int, default=128)
    parser.add_argument("--no-anonymize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    process_directory(
        args.input_dir,
        args.output_dir,
        box_size=args.box_size,
        anonymize=not args.no_anonymize,
    )


if __name__ == "__main__":
    main()

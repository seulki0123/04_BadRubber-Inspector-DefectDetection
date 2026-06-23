import argparse
import math
import os
from typing import Iterator, List, Sequence, Tuple

import cv2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch-size must be a positive integer")
    return parsed


def _list_images(img_dir: str) -> List[str]:
    return [
        os.path.join(img_dir, img_name)
        for img_name in os.listdir(img_dir)
        if img_name.endswith(".jpg")
    ]


def _image_id_from_path(img_path: str) -> str:
    return os.path.splitext(os.path.basename(img_path))[0]


def _iter_batches(items: Sequence, batch_size: int) -> Iterator[Tuple[int, int]]:
    total = len(items)
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect detection on a folder of images.")
    parser.add_argument("--img-dir", default="./tests/dot", help="input image directory")
    parser.add_argument(
        "--dst-dir",
        default="./tests/dot_results",
        help="directory to save visualize images",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=10, help="batch size")
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="save per-image artifacts under --artifacts-dir",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="./tests/dot_artifacts",
        help="directory to save artifacts",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="skip visualize image saving",
    )
    parser.add_argument(
        "--save-heatmap-npy",
        action="store_true",
        help="save raw float32 anomaly map as heatmap.npy (post-resize, fg-masked)",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="save per-image AnomalyCLIP intermediates (patch_features etc.) to intermediates/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from defect_detection import Detector
    from defect_detection.utils.artifact_saver import save_batch_artifacts

    imgs_path = _list_images(args.img_dir)
    imgs = [cv2.imread(img_path) for img_path in imgs_path]

    os.makedirs(args.dst_dir, exist_ok=True)
    if args.save_artifacts:
        os.makedirs(args.artifacts_dir, exist_ok=True)

    detector = Detector()
    if args.save_intermediates:
        detector.anomaly_extractor.capture_intermediates = True

    total_images = len(imgs)
    if total_images == 0:
        print("No .jpg images found.")
        return 0

    num_batches = int(math.ceil(total_images / float(args.batch_size)))
    saved_viz_count = 0
    saved_artifacts_count = 0

    for batch_index, (start, end) in enumerate(_iter_batches(imgs, args.batch_size), start=1):
        batch_imgs = imgs[start:end]
        batch_paths = imgs_path[start:end]
        batch_ids = [_image_id_from_path(img_path) for img_path in batch_paths]

        print(f"batch {batch_index}/{num_batches}: running {len(batch_imgs)} images")
        results = detector.detect(batch_imgs)

        if not args.no_viz:
            for img_path, result in zip(batch_paths, results):
                image_name = os.path.basename(img_path)
                out_path = os.path.join(args.dst_dir, image_name)
                if cv2.imwrite(out_path, result.visualize()):
                    saved_viz_count += 1
                else:
                    raise IOError(f"Failed to write visualize image: {out_path}")

        if args.save_artifacts:
            saved_dirs = save_batch_artifacts(
                args.artifacts_dir,
                batch_imgs,
                batch_ids,
                results,
                anomaly_extractor=detector.anomaly_extractor if args.save_intermediates else None,
                save_intermediates=args.save_intermediates,
                save_heatmap_npy=args.save_heatmap_npy,
            )
            saved_artifacts_count += len(saved_dirs)

        print(
            f"batch {batch_index}/{num_batches}: processed {end}/{total_images}, "
            f"viz_saved={saved_viz_count}, artifacts_saved={saved_artifacts_count}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

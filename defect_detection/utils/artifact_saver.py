import json
import os
from typing import Any, List, Optional, Sequence

import cv2
import numpy as np

from defect_detection.detect.crop import (
    build_filename,
    compute_crop_bbox,
    crop_image,
    denormalize_bbox_xyxy_n,
    process_segmentations,
)
from defect_detection.utils import compute_image_hash


def _safe_path_part(s: str) -> str:
    if s is None:
        return "unknown"

    value = str(s)
    if value == "":
        return "unknown"
    value = value.replace("/", "_").replace("\\", "_").replace("\x00", "_")
    value = value.strip().strip(".")
    if not value or value in {".", ".."}:
        return "unknown"

    return value[:120] or "unknown"


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _as_region_list(value: Any) -> List[Any]:
    if value is None:
        return []
    regions = getattr(value, "regions", None)
    if regions is not None:
        return list(regions)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _classification_to_metadata(classification: Any) -> Any:
    if classification is None:
        return None

    return {
        "class_id": getattr(classification, "class_id", None),
        "class_name": getattr(classification, "class_name", None),
        "confidence": getattr(classification, "confidence", None),
        "is_pass": getattr(classification, "is_pass", None),
    }


def _segmentations_to_metadata(seg_list: Sequence[Any], bbox_scaled, image_shape):
    filtered = [seg for seg in seg_list if seg is not None]
    if not filtered:
        return [], False

    _, segmentations, _, _, _ = process_segmentations(
        filtered,
        bbox_scaled,
        image_shape,
    )

    clean_segmentations = [
        {
            "class_id": seg.get("class_id"),
            "class_name": seg.get("class_name"),
            "confidence": seg.get("confidence"),
            "area": seg.get("area"),
            "polygon": seg.get("polygon"),
        }
        for seg in segmentations
    ]

    return clean_segmentations, True


def save_detection_artifacts(
    out_root: str,
    image: np.ndarray,
    image_id: str,
    anomaly_item,
    anomaly_cls_item,
    segmentation_item,
    *,
    save_heatmap_image: bool = True,
    save_heatmap_npy: bool = False,
    save_crops: bool = True,
    save_metadata: bool = True,
    save_intermediates: bool = False,
    intermediates: Optional[dict] = None,
    polygon_scale: float = 2.0,
) -> str:
    """Save detection artifacts for one image.

    Contract:
    - `anomaly_item` must provide `.map` and `.regions` (or an iterable of regions).
    - Each region must follow the `AnomalyRegion` interface, including `bboxes_xyxy_n`
      and `confidence`-style attributes used downstream.
    - Missing required attributes are treated as contract violations and should raise
      `AttributeError` immediately; no defensive fallback is applied.
    """
    image_id = _safe_path_part(image_id)
    out_dir = os.path.abspath(os.path.join(out_root, image_id))
    os.makedirs(out_dir, exist_ok=True)

    crops_dir = os.path.join(out_dir, "crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    H, W = image.shape[:2]
    anomaly_map = np.asarray(getattr(anomaly_item, "map"))
    anomaly_regions = _as_region_list(anomaly_item.regions)
    anomaly_cls_regions = _as_region_list(anomaly_cls_item)
    segmentation_regions = _as_region_list(segmentation_item) if segmentation_item is not None else []

    heatmap_file = None
    heatmap_npy_file = None

    if save_heatmap_image or save_heatmap_npy:
        heatmap = np.clip(anomaly_map.astype(np.float32), 0.0, 1.0)
        if save_heatmap_image:
            heatmap_u8 = (heatmap * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heatmap_file = "heatmap.jpg"
            heatmap_path = os.path.join(out_dir, heatmap_file)
            if not cv2.imwrite(heatmap_path, heatmap_bgr):
                raise IOError(f"Failed to write heatmap image: {heatmap_path}")
        if save_heatmap_npy:
            heatmap_npy_file = "heatmap.npy"
            heatmap_npy_path = os.path.join(out_dir, heatmap_npy_file)
            try:
                np.save(heatmap_npy_path, heatmap)
            except OSError as e:
                raise IOError(f"Failed to write heatmap npy: {heatmap_npy_path}: {e}") from e

    if save_intermediates and intermediates:
        intermediates_dir = os.path.join(out_dir, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)

        anomaly_map_pre_resize = np.asarray(intermediates["anomaly_map_pre_resize"])
        image_features = np.asarray(intermediates["image_features"])
        patch_features = list(intermediates["patch_features"])

        np.save(
            os.path.join(intermediates_dir, "anomaly_map_pre_resize.npy"),
            anomaly_map_pre_resize,
        )
        np.save(
            os.path.join(intermediates_dir, "image_features.npy"),
            image_features,
        )
        for idx, patch_feature in enumerate(patch_features):
            np.save(
                os.path.join(intermediates_dir, f"patch_features_layer{idx:02d}.npy"),
                np.asarray(patch_feature),
            )

    metadata = None
    if save_metadata:
        metadata = {
            "image": {
                "file": f"{image_id}.jpg",
                "hash": compute_image_hash(image),
                "resolution": [W, H],
            },
            "heatmap": {
                "file": heatmap_file,
                "npy": heatmap_npy_file,
                "shape": list(anomaly_map.shape[:2]),
            },
            "crops": [],
        }

    for idx, anomaly_region in enumerate(anomaly_regions):
        if anomaly_region is None:
            continue

        bbox_scaled = compute_crop_bbox(
            anomaly_region,
            H,
            W,
            scale=polygon_scale,
        )
        if bbox_scaled is None:
            continue

        crop_img = crop_image(image, bbox_scaled)
        crop_h, crop_w = crop_img.shape[:2]
        crop_hash = compute_image_hash(crop_img) if save_metadata else None

        bbox_original = denormalize_bbox_xyxy_n(
            anomaly_region.bboxes_xyxy_n,
            H,
            W,
        )

        classification = anomaly_cls_regions[idx] if idx < len(anomaly_cls_regions) else None
        seg_list = segmentation_regions[idx] if idx < len(segmentation_regions) else []
        segmentations, seg_exist = _segmentations_to_metadata(
            seg_list,
            bbox_scaled,
            (H, W),
        )

        crop_file = None
        if save_crops:
            class_name = getattr(classification, "class_name", None) if classification is not None else None
            crop_file = build_filename(f"{idx:04d}", _safe_path_part(class_name or "none")) + ".jpg"
            crop_path = os.path.join(crops_dir, crop_file)
            if not cv2.imwrite(crop_path, crop_img):
                raise IOError(f"Failed to write crop image: {crop_path}")

        if metadata is not None:
            metadata["crops"].append(
                {
                    "file": crop_file,
                    "hash": crop_hash,
                    "xyxy": list(bbox_scaled),
                    "xyxy_original": list(bbox_original),
                    "resolution": [crop_w, crop_h],
                    "area": int(crop_w * crop_h),
                    "classification": _classification_to_metadata(classification),
                    "segmentations": segmentations,
                    "seg_exist": bool(seg_exist),
                }
            )

    if metadata is not None:
        metadata_path = os.path.join(out_dir, "metadata.json")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)
        except OSError as e:
            raise IOError(f"Failed to write metadata: {metadata_path}: {e}") from e

    return out_dir


def save_batch_artifacts(
    out_root: str,
    images: Sequence[np.ndarray],
    image_ids: Sequence[str],
    detector_output,
    anomaly_extractor: Optional[Any] = None,
    save_intermediates: bool = False,
    **kwargs,
) -> List[str]:
    if len(images) != len(image_ids):
        raise ValueError("images and image_ids length mismatch")

    if len(detector_output) != len(images):
        raise ValueError("detector_output and images length mismatch")

    batch_intermediates = None
    if save_intermediates:
        if anomaly_extractor is None or getattr(anomaly_extractor, "_last_intermediates", None) is None:
            raise ValueError("save_intermediates requires anomaly_extractor._last_intermediates")
        batch_intermediates = anomaly_extractor._last_intermediates
        if len(batch_intermediates) != len(images):
            raise ValueError("anomaly_extractor._last_intermediates and images length mismatch")

    saved_dirs: List[str] = []
    for idx, (image, image_id) in enumerate(zip(images, image_ids)):
        batch_item = detector_output[idx]
        intermediates = batch_intermediates[idx] if batch_intermediates is not None else None
        saved_dirs.append(
            save_detection_artifacts(
                out_root=out_root,
                image=image,
                image_id=image_id,
                anomaly_item=batch_item.anomaly,
                anomaly_cls_item=batch_item.anomaly_cls,
                segmentation_item=batch_item.segmentation,
                save_intermediates=save_intermediates,
                intermediates=intermediates,
                **kwargs,
            )
        )

    return saved_dirs

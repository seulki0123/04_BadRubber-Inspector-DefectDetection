from typing import Optional

import cv2
import numpy as np


def _largest_filled_contour(mask: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros_like(mask, dtype=np.uint8), None

    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, [contour], 1)
    return filled, contour


def _odd_kernel_size(size: float) -> int:
    size = max(1, round(size))
    return size if size % 2 else size + 1


def _normalized_polygon(
    contour: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    polygon = contour.reshape(-1, 2).astype(np.float32)
    polygon /= [max(width, 1), max(height, 1)]
    return np.clip(polygon, 0, 1)


def _ellipse_kernel(size: float) -> np.ndarray:
    size = _odd_kernel_size(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def make_core_hull_mask(
    mask: np.ndarray,
    resize_scale: float,
    mask_threshold: float,
    erode_kernel_size: int,
    restore_dilate_scale: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    H, W = mask.shape[:2]
    work_w = max(1, round(W * resize_scale))
    work_h = max(1, round(H * resize_scale))

    work_mask = cv2.resize(
        mask.astype(np.float32),
        (work_w, work_h),
        interpolation=cv2.INTER_AREA,
    )

    binary_mask, binary_contour = _largest_filled_contour(
        (work_mask >= mask_threshold).astype(np.uint8)
    )

    if binary_contour is None:
        return work_mask.astype(np.float32), None

    eroded_mask = cv2.erode(binary_mask, _ellipse_kernel(erode_kernel_size))
    _, eroded_contour = _largest_filled_contour(eroded_mask)

    if eroded_contour is None:
        output_mask = cv2.resize(binary_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        return output_mask.astype(np.float32), _normalized_polygon(binary_contour, work_w, work_h)

    epsilon = 0.1 * cv2.arcLength(eroded_contour, True)
    hull = cv2.convexHull(cv2.approxPolyDP(eroded_contour, epsilon, True))

    hull_mask = np.zeros_like(binary_mask)
    cv2.fillPoly(hull_mask, [hull], 1)

    restored_mask = cv2.dilate(
        hull_mask,
        _ellipse_kernel(erode_kernel_size * restore_dilate_scale),
    )
    restored_mask, restored_contour = _largest_filled_contour(restored_mask)

    if restored_contour is None:
        return work_mask.astype(np.float32), None

    output_mask = cv2.resize(restored_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return output_mask.astype(np.float32), _normalized_polygon(restored_contour, work_w, work_h)

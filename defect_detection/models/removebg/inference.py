from typing import Optional, Sequence, Tuple, List

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from defect_detection.outputs.removebg.output import ForegroundMaskOutput


class BackgroundRemover:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int,
        mask_refine_mode: str = "raw",
        mask_shrink_px: int = 10,
        mask_morph_kernel: int = 100,
        quad_epsilon_min: float = 0.01,
        quad_epsilon_max: float = 0.08,
        quad_epsilon_steps: int = 30,
        blur_kernel: int = 101,
        blur_threshold: float = 0.3,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.mask_refine_mode = str(mask_refine_mode or "raw").lower()
        self.mask_shrink_px = max(0, int(mask_shrink_px))
        self.mask_morph_kernel = max(1, int(mask_morph_kernel))
        self.quad_epsilon_min = max(0.0, float(quad_epsilon_min))
        self.quad_epsilon_max = max(self.quad_epsilon_min, float(quad_epsilon_max))
        self.quad_epsilon_steps = max(1, int(quad_epsilon_steps))
        self.blur_kernel = max(1, int(blur_kernel))
        self.blur_threshold = float(blur_threshold)
        self._warmup()

    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(range(10), desc="Warm up YOLO segmentation model for background remover"):
            dummy_images = [
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            _ = self.model(dummy_images, imgsz=self.imgsz, verbose=False)

    def _parse_yolo_segmentation(
        self,
        results,
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:

        masks_batch = []
        polygons_batch = []

        for r in results:
            H, W = r.orig_shape

            if r.masks is None:
                masks_batch.append(np.ones((H, W), dtype=np.float32))
                polygons_batch.append([])
                continue

            # --- binary mask (pixel space) ---
            fg = r.masks.data.any(dim=0).float().cpu().numpy()
            fg_resized = cv2.resize(
                fg,
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            masks_batch.append(fg_resized.astype(np.float32))

            # --- normalized polygons (0~1) ---
            polys_n = []
            for poly_n in r.masks.xyn:   # already normalized
                poly_n = np.asarray(poly_n, dtype=np.float32)
                poly_n = np.clip(poly_n, 0.0, 1.0)
                polys_n.append(poly_n)

            polygons_batch.append(polys_n)

        return np.stack(masks_batch), polygons_batch


    def infer(
        self,
        images: Sequence[np.ndarray],
    ) -> ForegroundMaskOutput:
        results = self.model(images, imgsz=self.imgsz, verbose=False)
        masks, polygons_n = self._parse_yolo_segmentation(results)
        if self.mask_refine_mode == "blur":
            masks, polygons_n = self._make_blurred_contour_masks(masks, polygons_n)
        elif self.mask_refine_mode == "quad":
            masks, polygons_n = self._make_shrunk_quad_masks(masks, polygons_n)
        forground_images = self._apply_background_removal(images, masks)
        return ForegroundMaskOutput(
            masks=masks,
            polygons_n=polygons_n,
            images=forground_images,
        )

    def _apply_background_removal(
        self,
        images: Sequence[np.ndarray],
        masks: np.ndarray,
    ) -> List[np.ndarray]:

        fg_images = []

        for img, mask in zip(images, masks):
            # mask: float32 (0~1) → uint8 (0 or 255)
            mask_bin = (mask > 0.5).astype(np.uint8) * 255

            # 3채널로 확장
            mask_3ch = cv2.merge([mask_bin, mask_bin, mask_bin])

            # foreground만 남기기
            fg = cv2.bitwise_and(img, mask_3ch)

            fg = cv2.bitwise_and(img, mask_3ch)
            fg_images.append(fg)

        return fg_images

    def _make_shrunk_quad_masks(
        self,
        masks: np.ndarray,
        polygons_n: List[List[np.ndarray]],
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        quad_masks = []
        quad_polygons_n = []

        for mask, raw_polygon_n in zip(masks, polygons_n):
            quad_mask, polygon_n = self._make_shrunk_quad_mask(mask)
            if polygon_n is None:
                quad_masks.append(mask.astype(np.float32))
                quad_polygons_n.append(raw_polygon_n)
            else:
                quad_masks.append(quad_mask)
                quad_polygons_n.append([polygon_n])

        return np.stack(quad_masks).astype(np.float32), quad_polygons_n

    def _make_blurred_contour_masks(
        self,
        masks: np.ndarray,
        polygons_n: List[List[np.ndarray]],
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        blurred_masks = []
        blurred_polygons_n = []

        for mask, raw_polygon_n in zip(masks, polygons_n):
            blurred_mask, polygon_n = self._make_blurred_contour_mask(mask)
            if polygon_n is None:
                blurred_masks.append(mask.astype(np.float32))
                blurred_polygons_n.append(raw_polygon_n)
            else:
                blurred_masks.append(blurred_mask)
                blurred_polygons_n.append([polygon_n])

        return np.stack(blurred_masks).astype(np.float32), blurred_polygons_n

    def _make_blurred_contour_mask(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        H, W = mask.shape[:2]
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(
            mask.astype(np.float32),
            (kernel_size, kernel_size),
            0,
        )
        mask_bin = (blurred >= self.blur_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return mask.astype(np.float32), None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 0:
            return mask.astype(np.float32), None

        contour_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(contour_mask, [contour.astype(np.int32)], 1)

        polygon = contour.reshape(-1, 2).astype(np.float32)
        polygon[:, 0] /= max(W, 1)
        polygon[:, 1] /= max(H, 1)
        polygon = np.clip(polygon, 0.0, 1.0)

        return contour_mask.astype(np.float32), polygon

    def _make_shrunk_quad_mask(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        H, W = mask.shape[:2]
        mask_bin = (mask > 0.5).astype(np.uint8) * 255

        kernel_size = self.mask_morph_kernel
        if kernel_size > 1:
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size),
            )
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return np.zeros((H, W), dtype=np.float32), None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 0:
            return np.zeros((H, W), dtype=np.float32), None

        polygon = self._find_quad_points(contour)
        if polygon is None:
            return mask.astype(np.float32), None

        polygon = self._shrink_polygon_points(polygon, float(self.mask_shrink_px))
        polygon[:, 0] = np.clip(polygon[:, 0], 0, W - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, H - 1)

        quad_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(quad_mask, [np.round(polygon).astype(np.int32)], 1)

        polygon_n = polygon.astype(np.float32)
        polygon_n[:, 0] /= max(W, 1)
        polygon_n[:, 1] /= max(H, 1)
        polygon_n = np.clip(polygon_n, 0.0, 1.0)

        return quad_mask.astype(np.float32), polygon_n

    def _find_quad_points(
        self,
        contour: np.ndarray,
    ) -> Optional[np.ndarray]:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        for epsilon_ratio in np.linspace(
            self.quad_epsilon_min,
            self.quad_epsilon_max,
            self.quad_epsilon_steps,
        ):
            approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        return None

    def _shrink_polygon_points(
        self,
        polygon: np.ndarray,
        shrink_px: float,
    ) -> np.ndarray:
        if shrink_px <= 0:
            return polygon.astype(np.float32)

        center = polygon.astype(np.float32).mean(axis=0)
        shrunk = polygon.astype(np.float32).copy()

        for i, point in enumerate(shrunk):
            vector = center - point
            distance = float(np.linalg.norm(vector))
            if distance <= 1.0:
                continue

            move_px = min(shrink_px, distance - 1.0)
            shrunk[i] = point + vector / distance * move_px

        return shrunk

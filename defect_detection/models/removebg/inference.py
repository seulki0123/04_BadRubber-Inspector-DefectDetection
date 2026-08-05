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
        use_blur_mask: bool = False,
        blur_kernel: int = 101,
        blur_threshold: float = 0.3,
        blur_resize_scale: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.device = device
        self.use_blur_mask = bool(use_blur_mask)
        self.blur_kernel = max(1, int(blur_kernel))
        self.blur_threshold = float(blur_threshold)
        self.blur_resize_scale = min(1.0, max(0.05, float(blur_resize_scale)))
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
            _ = self.model(
                dummy_images,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

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
        results = self.model(
            images,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        masks, polygons_n = self._parse_yolo_segmentation(results)
        if self.use_blur_mask:
            masks, polygons_n = self._make_blurred_contour_masks(masks, polygons_n)
        forground_images = self._apply_background_removal(images, masks)
        return ForegroundMaskOutput(masks=masks, polygons_n=polygons_n, images=forground_images)

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
        scale = self.blur_resize_scale
        work_w = max(1, int(round(W * scale)))
        work_h = max(1, int(round(H * scale)))
        work_mask = cv2.resize(
            mask.astype(np.float32),
            (work_w, work_h),
            interpolation=cv2.INTER_AREA,
        )

        kernel_size = max(1, int(round(self.blur_kernel * scale)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(
            work_mask,
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

        contour_mask_small = np.zeros((work_h, work_w), dtype=np.uint8)
        cv2.fillPoly(contour_mask_small, [contour.astype(np.int32)], 1)
        contour_mask = cv2.resize(
            contour_mask_small,
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )

        polygon = contour.reshape(-1, 2).astype(np.float32)
        polygon[:, 0] /= max(work_w, 1)
        polygon[:, 1] /= max(work_h, 1)
        polygon = np.clip(polygon, 0.0, 1.0)

        return contour_mask.astype(np.float32), polygon

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

            fg_images.append(fg)

        return fg_images

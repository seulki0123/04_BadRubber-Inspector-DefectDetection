from typing import Optional, Sequence

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from defect_detection.outputs.removebg.output import ForegroundMaskOutput
from .post_process import make_core_hull_mask


class BackgroundRemover:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int,
        postprocess: bool = True,
        core_hull_resize_scale: float = 0.25,
        core_hull_mask_threshold: float = 0.9,
        core_hull_erode_kernel: int = 81,
        core_hull_restore_dilate_scale: float = 0.95,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.device = device
        self.postprocess = postprocess
        self.core_hull_resize_scale = min(1.0, max(0.05, float(core_hull_resize_scale)))
        self.core_hull_mask_threshold = float(core_hull_mask_threshold)
        self.core_hull_erode_kernel = max(1, int(core_hull_erode_kernel))
        self.core_hull_restore_dilate_scale = max(0.0, float(core_hull_restore_dilate_scale))
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
    ) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        masks, polygons = [], []

        for r in results:
            H, W = r.orig_shape

            if r.masks is None:
                masks.append(np.ones((H, W), dtype=np.float32))
                polygons.append([])
                continue

            masks.append(
                cv2.resize(
                    r.masks.data.any(dim=0).float().cpu().numpy(),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)
            )

            polygons.append([
                np.clip(
                    np.asarray(p, dtype=np.float32),
                    0.0,
                    1.0,
                )
                for p in r.masks.xyn
            ])

        return np.stack(masks), polygons

    def _make_core_hull_masks(
        self,
        masks: np.ndarray,
        polygons_n: list[list[np.ndarray]],
    ) -> tuple[np.ndarray, list[list[np.ndarray]]]:
        refined_masks = []
        refined_polygons = []

        for mask, original_polygons in zip(masks, polygons_n):
            refined_mask, refined_polygon = make_core_hull_mask(
                mask,
                self.core_hull_resize_scale,
                self.core_hull_mask_threshold,
                self.core_hull_erode_kernel,
                self.core_hull_restore_dilate_scale,
            )

            refined_masks.append(
                refined_mask if refined_polygon is not None
                else mask.astype(np.float32)
            )
            refined_polygons.append(
                [refined_polygon] if refined_polygon is not None
                else original_polygons
            )

        return np.stack(refined_masks).astype(np.float32), refined_polygons

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

        if self.postprocess:
            masks, polygons_n = self._make_core_hull_masks(
                masks,
                polygons_n,
            )

        foreground_images = self._apply_background_removal(
            images,
            masks,
        )

        return ForegroundMaskOutput(
            masks=masks,
            polygons_n=polygons_n,
            images=foreground_images,
        )

    def _apply_background_removal(
        self,
        images: Sequence[np.ndarray],
        masks: np.ndarray,
    ) -> list[np.ndarray]:
        fg_images = []

        for image, mask in zip(images, masks):
            mask_8bit = (mask > 0.5).astype(np.uint8) * 255
            mask_3ch = cv2.merge([mask_8bit, mask_8bit, mask_8bit])

            fg_images.append(
                cv2.bitwise_and(image, mask_3ch)
            )

        return fg_images

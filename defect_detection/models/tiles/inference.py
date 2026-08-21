"""Anomaly candidate extractor that uses fixed mask ROI tiles."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from defect_detection.outputs import AnomalyCLIPOutput
from defect_detection.outputs.anomalyclip import AnomalyRegion

from .geometry import GridTileBox, make_multi_grid_bboxes_from_mask


ImageNP = np.ndarray
BatchImageNP = Sequence[ImageNP]


class TiledAnomalyExtractor:
    """
    AnomalyCLIP-compatible extractor that emits 3x3/4x4 tile candidates.

    It does not run an anomaly model. It uses the foreground mask as the crop ROI,
    then emits overlapping tile boxes as ``AnomalyRegion`` items so the existing
    classify/segment pipeline can consume them unchanged.
    """

    def __init__(
        self,
        *,
        grids: Sequence[int] = (3, 4),
        overlap: float = 0.25,
        pad_ratio: float = 0.05,
        score: float = 1.0,
        score_threshold: float = 0.0,
        area_threshold: int = 0,
        name: str = "tiles",
    ) -> None:
        self.grids = tuple(int(grid) for grid in grids)
        self.overlap = float(overlap)
        self.pad_ratio = float(pad_ratio)
        self.score = float(score)
        self.score_threshold = float(score_threshold)
        self.area_threshold = int(area_threshold)
        self.name = name

    def infer(
        self,
        imgs_np: Union[ImageNP, BatchImageNP],
        foreground_masks: Optional[np.ndarray] = None,
    ) -> AnomalyCLIPOutput:
        images = self._normalize_images(imgs_np)
        masks = self._normalize_masks(foreground_masks, images)

        maps_np = np.stack(
            [mask.astype(np.float32, copy=False) for mask in masks],
            axis=0,
        )
        out = AnomalyCLIPOutput(
            maps=maps_np,
            score_threshold=self.score_threshold,
            area_threshold=self.area_threshold,
            source=self.name,
            extract_regions=False,
        )

        batch_regions = [
            self._tiles_to_regions(image.shape[:2], mask)
            for image, mask in zip(images, masks)
        ]
        object.__setattr__(out, "batch_regions", batch_regions)
        object.__setattr__(out, "global_scores", [self.score] * len(images))
        return out

    @staticmethod
    def _normalize_images(imgs_np: Union[ImageNP, BatchImageNP]) -> List[ImageNP]:
        if isinstance(imgs_np, np.ndarray):
            if imgs_np.ndim == 3:
                return [imgs_np]
            if imgs_np.ndim == 4:
                return [imgs_np[i] for i in range(imgs_np.shape[0])]
            raise ValueError("Unsupported image shape")
        if not isinstance(imgs_np, (list, tuple)):
            raise TypeError("imgs_np must be an image or a sequence of images")
        return list(imgs_np)

    @staticmethod
    def _normalize_masks(
        foreground_masks: Optional[np.ndarray],
        images: Sequence[ImageNP],
    ) -> List[np.ndarray]:
        if foreground_masks is None:
            return [
                np.ones(image.shape[:2], dtype=np.uint8)
                for image in images
            ]

        masks = np.asarray(foreground_masks)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]
        if masks.ndim != 3:
            raise ValueError(
                f"foreground_masks must have shape (H, W) or (B, H, W), got {masks.shape}"
            )
        if masks.shape[0] != len(images):
            raise ValueError(
                f"len(images) ({len(images)}) != foreground_masks batch ({masks.shape[0]})"
            )
        for idx, (mask, image) in enumerate(zip(masks, images)):
            if mask.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"foreground_masks[{idx}] shape {mask.shape[:2]} != image shape {image.shape[:2]}"
                )
        return [(mask > 0.5).astype(np.uint8) for mask in masks]

    def _tiles_to_regions(
        self,
        image_shape: Tuple[int, int],
        mask: np.ndarray,
    ) -> List[AnomalyRegion]:
        _, tiles = make_multi_grid_bboxes_from_mask(
            mask,
            grids=self.grids,
            overlap=self.overlap,
            pad_ratio=self.pad_ratio,
            score=self.score,
        )
        return [self._tile_to_region(tile, image_shape) for tile in tiles]

    def _tile_to_region(
        self,
        tile: GridTileBox,
        image_shape: Tuple[int, int],
    ) -> AnomalyRegion:
        H, W = image_shape[:2]
        x1, y1, x2, y2 = tile.xyxy
        polygon = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        )
        polygon_n = polygon.astype(np.float32)
        polygon_n[:, 0] = np.clip(polygon_n[:, 0] / W, 0.0, 1.0)
        polygon_n[:, 1] = np.clip(polygon_n[:, 1] / H, 0.0, 1.0)
        bbox_n = tile.xyxy_n(image_shape)
        area = float(tile.box.area)

        region = AnomalyRegion(
            polygon=polygon,
            polygon_n=polygon_n,
            bboxes_xyxy=(x1, y1, x2, y2),
            bboxes_xyxy_n=bbox_n,
            confidence=float(tile.box.score),
            area=area,
            area_n=area / float(H * W),
            source=self.name,
            is_pass=False,
        )
        region.class_id = -1
        region.class_name = f"tile_{tile.grid}x{tile.grid}"
        region.color = (0, 255, 255) if tile.grid == 3 else (255, 255, 0)
        return region

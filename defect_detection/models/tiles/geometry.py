"""Mask ROI based overlapping tile bbox helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TileBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True)
class GridTileBox:
    grid: int
    row: int
    col: int
    box: TileBox

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.box.xyxy

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return self.box.xywh

    def xyxy_n(self, image_shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
        H, W = image_shape[:2]
        x1, y1, x2, y2 = self.xyxy
        return (
            max(0.0, min(1.0, x1 / W)),
            max(0.0, min(1.0, y1 / H)),
            max(0.0, min(1.0, x2 / W)),
            max(0.0, min(1.0, y2 / H)),
        )

    def as_dict(self, image_shape: Tuple[int, int] | None = None) -> dict:
        data = {
            "grid": self.grid,
            "row": self.row,
            "col": self.col,
            "xyxy": self.xyxy,
            "xywh": self.xywh,
            "score": self.box.score,
            "area": self.box.area,
        }
        if image_shape is not None:
            data["xyxy_n"] = self.xyxy_n(image_shape)
        return data


def crop_rect_from_mask(
    mask: np.ndarray,
    *,
    pad_ratio: float = 0.05,
) -> Tuple[int, int, int, int]:
    """Return inclusive (x1, y1, x2, y2) crop rect from a binary/float mask."""
    H, W = mask.shape[:2]
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, W - 1, H - 1)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw, bh = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)
    return (x1, y1, x2, y2)


def make_overlapping_grid_bboxes(
    crop_rect: Tuple[int, int, int, int],
    *,
    grid: int = 3,
    overlap: float = 0.25,
    score: float = 1.0,
) -> List[TileBox]:
    """
    Build grid x grid boxes covering the crop rect in original image coordinates.

    ``overlap`` is the adjacent tile overlap ratio by tile width/height.
    """
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    if not 0.0 <= overlap < 0.5:
        raise ValueError(f"overlap must be in [0, 0.5), got {overlap}")

    x1, y1, x2, y2 = crop_rect
    if x2 < x1 or y2 < y1:
        raise ValueError(f"invalid crop_rect: {crop_rect}")

    cw = max(1, x2 - x1 + 1)
    ch = max(1, y2 - y1 + 1)
    denom = 1.0 + (grid - 1) * (1.0 - overlap)
    tw = cw / denom
    th = ch / denom
    sx = tw * (1.0 - overlap)
    sy = th * (1.0 - overlap)

    boxes: List[TileBox] = []
    for row in range(grid):
        for col in range(grid):
            tx1 = int(round(x1 + col * sx))
            ty1 = int(round(y1 + row * sy))
            tx2 = int(round(tx1 + tw))
            ty2 = int(round(ty1 + th))
            tx1 = max(x1, min(tx1, x2))
            ty1 = max(y1, min(ty1, y2))
            tx2 = max(tx1 + 1, min(tx2, x2 + 1))
            ty2 = max(ty1 + 1, min(ty2, y2 + 1))
            w, h = tx2 - tx1, ty2 - ty1
            boxes.append(TileBox(tx1, ty1, w, h, score))
    return boxes


def make_overlapping_grid_tile_bboxes(
    crop_rect: Tuple[int, int, int, int],
    *,
    grid: int = 3,
    overlap: float = 0.25,
    score: float = 1.0,
) -> List[GridTileBox]:
    boxes = make_overlapping_grid_bboxes(
        crop_rect,
        grid=grid,
        overlap=overlap,
        score=score,
    )
    return [
        GridTileBox(grid=grid, row=idx // grid, col=idx % grid, box=box)
        for idx, box in enumerate(boxes)
    ]


def make_multi_grid_bboxes(
    crop_rect: Tuple[int, int, int, int],
    *,
    grids: Sequence[int] = (3, 4),
    overlap: float = 0.25,
    score: float = 1.0,
) -> List[GridTileBox]:
    tiles: List[GridTileBox] = []
    for grid in grids:
        tiles.extend(
            make_overlapping_grid_tile_bboxes(
                crop_rect,
                grid=int(grid),
                overlap=overlap,
                score=score,
            )
        )
    return tiles


def make_multi_grid_bboxes_from_mask(
    mask: np.ndarray,
    *,
    grids: Sequence[int] = (3, 4),
    overlap: float = 0.25,
    pad_ratio: float = 0.05,
    score: float = 1.0,
) -> Tuple[Tuple[int, int, int, int], List[GridTileBox]]:
    crop_rect = crop_rect_from_mask(mask, pad_ratio=pad_ratio)
    return crop_rect, make_multi_grid_bboxes(
        crop_rect,
        grids=grids,
        overlap=overlap,
        score=score,
    )

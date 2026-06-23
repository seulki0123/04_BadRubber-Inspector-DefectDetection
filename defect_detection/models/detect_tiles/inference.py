from typing import Optional, Sequence, List

import numpy as np
import tqdm
from ultralytics import YOLO

from defect_detection.outputs import AnomalyCLIPOutput


class TiledObjectDetector:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int = 2048,
        threshold: float = 0.3,
        name: str = "tiled_yolo",
        tile_overlap_x: float = 0.0,
        tile_overlap_y: float = 0.0,
        roi_left: float = 0.10,
        roi_right: float = 0.10,
        roi_top: float = 0.0,
        roi_bottom: float = 0.0,
    ) -> None:
        self.model = YOLO(checkpoint_path)

        self.imgsz = imgsz
        self.threshold = threshold
        self.name = name

        self.tile_overlap_x = tile_overlap_x
        self.tile_overlap_y = tile_overlap_y

        self.roi_left = roi_left
        self.roi_right = roi_right
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom

        self._warmup()

    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(
            range(10),
            desc="Warm up tiled YOLO detection model",
        ):
            dummy_images = [
                np.zeros(
                    (self.imgsz, self.imgsz, 3),
                    dtype=np.uint8,
                )
                for _ in range(batch_size)
            ]

            _ = self.model(
                dummy_images,
                imgsz=self.imgsz,
                verbose=False,
            )

    @staticmethod
    def _apply_roi(
        img: np.ndarray,
        roi_left: float,
        roi_right: float,
        roi_top: float,
        roi_bottom: float,
    ):
        h, w = img.shape[:2]

        x0 = max(
            0,
            int(w * min(max(roi_left, 0.0), 0.49)),
        )

        x1 = min(
            w,
            int(
                w
                * (
                    1.0
                    - min(max(roi_right, 0.0), 0.49)
                )
            ),
        )

        y0 = max(
            0,
            int(h * min(max(roi_top, 0.0), 0.49)),
        )

        y1 = min(
            h,
            int(
                h
                * (
                    1.0
                    - min(max(roi_bottom, 0.0), 0.49)
                )
            ),
        )

        if x0 >= x1 or y0 >= y1:
            return img, 0, 0

        return img[y0:y1, x0:x1], x0, y0

    @staticmethod
    def _tile_starts(
        length: int,
        tile_sz: int,
        step: int,
    ) -> List[int]:
        if length <= tile_sz:
            return [0]

        starts = list(
            range(
                0,
                length - tile_sz,
                step,
            )
        )

        last = length - tile_sz

        if not starts or starts[-1] < last:
            starts.append(last)

        return starts
        
    @staticmethod
    def _apply_bale_crop(
        img: np.ndarray,
        foreground_mask: Optional[np.ndarray],
        x_offset: int,
        y_offset: int,
    ):
        if foreground_mask is None:
            raise ValueError(
                "foreground_mask must be provided for TiledObjectDetector."
            )

        roi_mask = foreground_mask[
            y_offset:y_offset + img.shape[0],
            x_offset:x_offset + img.shape[1],
        ]

        ys, xs = np.where(roi_mask > 0.5)

        if len(xs) == 0 or len(ys) == 0:
            return img, x_offset, y_offset

        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1

        return (
            img[y0:y1, x0:x1],
            x_offset + x0,
            y_offset + y0,
        )

    def _infer_single_map(
        self,
        img: np.ndarray,
        conf_threshold: float,
        foreground_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        h, w = img.shape[:2]

        amap = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        roi_img, roi_x0, roi_y0 = self._apply_roi(
            img,
            self.roi_left,
            self.roi_right,
            self.roi_top,
            self.roi_bottom,
        )

        roi_img, roi_x0, roi_y0 = self._apply_bale_crop(
            roi_img,
            foreground_mask,
            roi_x0,
            roi_y0,
        )

        roi_h, roi_w = roi_img.shape[:2]

        step_x = max(
            1,
            int(
                self.imgsz
                * (1.0 - self.tile_overlap_x)
            ),
        )

        step_y = max(
            1,
            int(
                self.imgsz
                * (1.0 - self.tile_overlap_y)
            ),
        )

        xs = self._tile_starts(
            roi_w,
            self.imgsz,
            step_x,
        )

        ys = self._tile_starts(
            roi_h,
            self.imgsz,
            step_y,
        )

        for y0 in ys:
            for x0 in xs:

                x1 = min(
                    x0 + self.imgsz,
                    roi_w,
                )

                y1 = min(
                    y0 + self.imgsz,
                    roi_h,
                )

                tile = roi_img[
                    y0:y1,
                    x0:x1,
                ]

                th, tw = tile.shape[:2]

                #
                # padding
                #
                if (
                    th < self.imgsz
                    or tw < self.imgsz
                ):
                    padded = np.zeros(
                        (
                            self.imgsz,
                            self.imgsz,
                            3,
                        ),
                        dtype=np.uint8,
                    )

                    padded[:th, :tw] = tile
                    tile = padded

                result = self.model(
                    [tile],
                    imgsz=self.imgsz,
                    verbose=False,
                )[0]

                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for (bx1, by1, bx2, by2), score in zip(
                    boxes,
                    scores,
                ):
                    if score < conf_threshold:
                        continue

                    #
                    # padding 영역 시작 detection 제거
                    #
                    if bx1 >= tw or by1 >= th:
                        continue

                    bx1 = int(
                        np.clip(
                            bx1 + x0 + roi_x0,
                            0,
                            w,
                        )
                    )

                    bx2 = int(
                        np.clip(
                            bx2 + x0 + roi_x0,
                            0,
                            w,
                        )
                    )

                    by1 = int(
                        np.clip(
                            by1 + y0 + roi_y0,
                            0,
                            h,
                        )
                    )

                    by2 = int(
                        np.clip(
                            by2 + y0 + roi_y0,
                            0,
                            h,
                        )
                    )

                    amap[
                        by1:by2,
                        bx1:bx2,
                    ] = np.maximum(
                        amap[
                            by1:by2,
                            bx1:bx2,
                        ],
                        score,
                    )

        return amap

    def infer(
        self,
        images: Sequence[np.ndarray],
        conf_thresholds: Optional[
            Sequence[Optional[float]]
        ] = None,
        foreground_masks: Optional[np.ndarray] = None,
    ):
    
        if foreground_masks is None:
            raise ValueError(
                f"{self.name}: foreground_masks must be provided."
            )
            
        if (
            conf_thresholds is not None
            and len(images) != len(conf_thresholds)
        ):
            raise ValueError(
                f"{self.name}: "
                f"len(images) ({len(images)}) must equal "
                f"len(conf_thresholds) ({len(conf_thresholds)})"
            )

        maps = []

        for idx, img in enumerate(images):
            conf_threshold = self.threshold

            if (
                conf_thresholds is not None
                and conf_thresholds[idx] is not None
            ):
                conf_threshold = float(
                    conf_thresholds[idx]
                )

            maps.append(
                self._infer_single_map(
                    img,
                    conf_threshold,
                    foreground_masks[idx],
                )
            )

        maps = np.stack(maps)

        return AnomalyCLIPOutput(
            maps=maps,
            score_threshold=0.0,
            area_threshold=0,
            source=self.name,
        )
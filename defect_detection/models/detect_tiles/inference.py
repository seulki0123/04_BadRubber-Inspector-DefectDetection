from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, List, Tuple, Union

import numpy as np
import tqdm
from ultralytics import YOLO

from defect_detection.outputs import AnomalyCLIPOutputOldVersion


_BG_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}


class TiledObjectDetector:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int = 2048,
        threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: Union[str, Sequence[str], None] = None,
        name: str = "tiled_yolo",
        tile_overlap_x: float = 0.0,
        tile_overlap_y: float = 0.0,
        roi_left: float = 0.10,
        roi_right: float = 0.10,
        roi_top: float = 0.0,
        roi_bottom: float = 0.0,
        bg_color: str = "black",
        bg_margin_top: int = 0,
        bg_margin_bottom: int = 0,
        bg_margin_left: int = 0,
        bg_margin_right: int = 0,
        bg_min_pixel: int = 8,
    ) -> None:
        self.devices = (
            list(device) if isinstance(device, (list, tuple)) else [device]
        ) or [None]
        self.model = YOLO(checkpoint_path)
        self.models = [self.model] + [
            YOLO(checkpoint_path) for _ in range(max(0, len(self.devices) - 1))
        ]

        self.imgsz = imgsz
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.device = self.devices[0]
        self.name = name

        self.tile_overlap_x = tile_overlap_x
        self.tile_overlap_y = tile_overlap_y

        self.roi_left = roi_left
        self.roi_right = roi_right
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom

        self.bg_color = bg_color
        self.bg_margin_top = bg_margin_top
        self.bg_margin_bottom = bg_margin_bottom
        self.bg_margin_left = bg_margin_left
        self.bg_margin_right = bg_margin_right
        self.bg_min_pixel = bg_min_pixel

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

            for model, device in zip(self.models, self.devices):
                _ = model.predict(
                    dummy_images,
                    imgsz=self.imgsz,
                    conf=self.threshold,
                    iou=self.iou_threshold,
                    device=device,
                    verbose=False,
                )

    @staticmethod
    def _apply_roi(
        img: np.ndarray,
        roi_left: float,
        roi_right: float,
        roi_top: float,
        roi_bottom: float,
    ) -> Tuple[np.ndarray, int, int]:
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
    def _nms_numpy(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float,
    ) -> np.ndarray:
        """
        detect_dot.py와 동일한 class-agnostic NMS.
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []

        while len(order):
            i = int(order[0])
            keep.append(i)

            if len(order) == 1:
                break

            rest = order[1:]

            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])

            inter = (
                np.maximum(0.0, ix2 - ix1)
                * np.maximum(0.0, iy2 - iy1)
            )

            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            order = rest[iou <= iou_thresh]

        return np.array(keep, dtype=np.int64)

    @staticmethod
    def _mask_to_bool(
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        foreground_mask가 0/1, 0~1, 0~255 어느 형태여도 동작하게 변환.
        detect_dot.py의 remover mask는 0~255 기준으로 mask > 127을 사용한다.
        """
        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask[:, :, 0]
            else:
                mask = mask.max(axis=2)

        if mask.dtype == np.bool_:
            return mask

        if mask.size == 0:
            return mask.astype(bool)

        max_value = float(np.nanmax(mask))
        threshold = 0.5 if max_value <= 1.0 else 127.0

        return mask > threshold

    @staticmethod
    def _extract_roi_mask(
        foreground_mask: np.ndarray,
        roi_h: int,
        roi_w: int,
        x_offset: int,
        y_offset: int,
    ) -> np.ndarray:
        """
        foreground_mask가 원본 이미지 기준 mask이면 ROI 위치만 slicing.
        이미 ROI 크기와 같은 mask이면 그대로 사용.
        """
        mask_h, mask_w = foreground_mask.shape[:2]

        if mask_h == roi_h and mask_w == roi_w:
            return foreground_mask

        y1 = y_offset + roi_h
        x1 = x_offset + roi_w

        roi_mask = foreground_mask[
            y_offset:y1,
            x_offset:x1,
        ]

        if roi_mask.shape[:2] != (roi_h, roi_w):
            raise ValueError(
                "foreground_mask shape does not match image/ROI. "
                f"mask_shape={foreground_mask.shape[:2]}, "
                f"roi_shape={(roi_h, roi_w)}, "
                f"offset={(x_offset, y_offset)}"
            )

        return roi_mask

    def _apply_bale_crop(
        self,
        img: np.ndarray,
        foreground_mask: Optional[np.ndarray],
        x_offset: int,
        y_offset: int,
    ) -> Tuple[np.ndarray, int, int]:
        """
        detect_dot.py의 apply_bale_crop() 흐름을 foreground_mask 기반으로 맞춘 버전.

        1. foreground mask에서 Bale 영역 bbox 계산
        2. margin 적용
        3. 이미지 면과 Bale 면 간격이 bg_min_pixel 이하이면 해당 면 margin 무시
        4. bg_color != "none"이면 mask 외부를 지정 색으로 채움
        """
        if foreground_mask is None:
            raise ValueError(
                "foreground_mask must be provided for TiledObjectDetector."
            )

        roi_h, roi_w = img.shape[:2]

        roi_mask = self._extract_roi_mask(
            foreground_mask=foreground_mask,
            roi_h=roi_h,
            roi_w=roi_w,
            x_offset=x_offset,
            y_offset=y_offset,
        )

        fg_mask = self._mask_to_bool(roi_mask)

        ys, xs = np.where(fg_mask)

        if len(xs) == 0 or len(ys) == 0:
            return img, x_offset, y_offset

        bx1 = int(xs.min())
        bx2 = int(xs.max()) + 1
        by1 = int(ys.min())
        by2 = int(ys.max()) + 1

        h, w = img.shape[:2]

        left = (
            bx1
            if bx1 <= self.bg_min_pixel
            else max(0, bx1 - self.bg_margin_left)
        )

        top = (
            by1
            if by1 <= self.bg_min_pixel
            else max(0, by1 - self.bg_margin_top)
        )

        right = (
            bx2
            if (w - bx2) <= self.bg_min_pixel
            else min(w, bx2 + self.bg_margin_right)
        )

        bottom = (
            by2
            if (h - by2) <= self.bg_min_pixel
            else min(h, by2 + self.bg_margin_bottom)
        )

        if right <= left or bottom <= top:
            return img, x_offset, y_offset

        crop = img[
            top:bottom,
            left:right,
        ]

        if self.bg_color != "none":
            if self.bg_color not in _BG_COLORS:
                raise ValueError(
                    f"Unsupported bg_color: {self.bg_color}. "
                    "Use one of: none, black, white, gray."
                )

            sub_mask = fg_mask[
                top:bottom,
                left:right,
            ]

            bg = np.empty_like(crop)
            bg[...] = np.array(
                _BG_COLORS[self.bg_color],
                dtype=crop.dtype,
            )

            crop = np.where(
                sub_mask[:, :, None],
                crop,
                bg,
            ).astype(np.uint8)

        return (
            crop,
            x_offset + left,
            y_offset + top,
        )

    def _run_single_patch(
        self,
        patch: np.ndarray,
        conf_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        YOLO 추론 1회.
        detect_dot.py처럼 conf/iou를 model.predict()에 직접 전달한다.
        """
        result = self.model.predict(
            [patch],
            imgsz=self.imgsz,
            conf=conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)

        return boxes, scores, cls_ids

    @staticmethod
    def _empty_boxes() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    @staticmethod
    def _result_to_arrays(result) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if result.boxes is None or len(result.boxes) == 0:
            return TiledObjectDetector._empty_boxes()

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = result.boxes.cls.cpu().numpy().astype(np.int32)

        return boxes, scores, cls_ids

    def _predict_patches(
        self,
        patches: Sequence[np.ndarray],
        conf_threshold: float,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(patches) == 0:
            return []

        if len(self.models) == 1 or len(patches) <= 1:
            results = self.model.predict(
                patches,
                imgsz=self.imgsz,
                conf=conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            return [self._result_to_arrays(result) for result in results]

        outputs = [self._empty_boxes() for _ in patches]

        def _run_group(group_idx: int) -> None:
            indices = list(range(group_idx, len(patches), len(self.models)))
            if not indices:
                return

            results = self.models[group_idx].predict(
                [patches[idx] for idx in indices],
                imgsz=self.imgsz,
                conf=conf_threshold,
                iou=self.iou_threshold,
                device=self.devices[group_idx],
                verbose=False,
            )
            for idx, result in zip(indices, results):
                outputs[idx] = self._result_to_arrays(result)

        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            list(executor.map(_run_group, range(len(self.models))))

        return outputs

    def _build_tiles(
        self,
        img: np.ndarray,
        conf_threshold: float,
        foreground_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, int, int, List[Tuple[np.ndarray, int, int, int, int]]]:
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

        h, w = roi_img.shape[:2]

        if h <= self.imgsz and w <= self.imgsz:
            return roi_img, roi_x0, roi_y0, [(roi_img, 0, 0, w, h)]

        overlap_x = min(max(self.tile_overlap_x, 0.0), 0.5)
        overlap_y = min(max(self.tile_overlap_y, 0.0), 0.5)

        step_x = max(1, int(self.imgsz * (1.0 - overlap_x)))
        step_y = max(1, int(self.imgsz * (1.0 - overlap_y)))

        xs = self._tile_starts(w, self.imgsz, step_x)
        ys = self._tile_starts(h, self.imgsz, step_y)

        tiles = []
        for y0 in ys:
            y1 = min(y0 + self.imgsz, h)
            for x0 in xs:
                x1 = min(x0 + self.imgsz, w)
                tile = roi_img[y0:y1, x0:x1]
                th, tw = tile.shape[:2]

                if th < self.imgsz or tw < self.imgsz:
                    padded = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                    padded[:th, :tw] = tile
                    tile = padded

                tiles.append((tile, x0, y0, tw, th))

        return roi_img, roi_x0, roi_y0, tiles

    def _restore_tile_boxes(
        self,
        tile_results: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        tiles: Sequence[Tuple[np.ndarray, int, int, int, int]],
        roi_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = roi_shape
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_cls_ids: List[np.ndarray] = []

        for (_, x0, y0, tw, th), (boxes, scores, cls_ids) in zip(
            tiles,
            tile_results,
        ):
            if len(boxes) == 0:
                continue

            valid_mask = (
                (boxes[:, 0] < tw)
                & (boxes[:, 1] < th)
            )

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            cls_ids = cls_ids[valid_mask]

            if len(boxes) == 0:
                continue

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] + x0, 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] + y0, 0, h)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls_ids.append(cls_ids)

        if not all_boxes:
            return self._empty_boxes()

        all_boxes_np = np.concatenate(all_boxes, axis=0)
        all_scores_np = np.concatenate(all_scores, axis=0)
        all_cls_ids_np = np.concatenate(all_cls_ids, axis=0)

        keep = self._nms_numpy(
            boxes=all_boxes_np,
            scores=all_scores_np,
            iou_thresh=self.iou_threshold,
        )

        return (
            all_boxes_np[keep],
            all_scores_np[keep],
            all_cls_ids_np[keep],
        )

    @staticmethod
    def _boxes_to_map(
        img_shape: Tuple[int, int],
        boxes: np.ndarray,
        scores: np.ndarray,
        roi_x0: int,
        roi_y0: int,
    ) -> np.ndarray:
        h, w = img_shape
        amap = np.zeros((h, w), dtype=np.float32)

        if len(boxes) == 0:
            return amap

        boxes = boxes.copy()
        boxes[:, [0, 2]] += roi_x0
        boxes[:, [1, 3]] += roi_y0
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        for (bx1, by1, bx2, by2), score in zip(boxes, scores):
            bx1 = int(np.clip(bx1, 0, w))
            bx2 = int(np.clip(bx2, 0, w))
            by1 = int(np.clip(by1, 0, h))
            by2 = int(np.clip(by2, 0, h))

            if bx2 <= bx1 or by2 <= by1:
                continue

            amap[by1:by2, bx1:bx2] = np.maximum(
                amap[by1:by2, bx1:bx2],
                float(score),
            )

        return amap

    def _infer_tiled_boxes(
        self,
        img: np.ndarray,
        conf_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        detect_dot.py의 infer_tiled()와 같은 방식.

        - ROI/Bale crop 이후 이미지가 imgsz 이하이면 일반 YOLO 추론
        - imgsz보다 크면 tile 분할
        - padding 영역에서 시작한 detection 제거
        - tile 좌표를 crop 이미지 좌표로 복원
        - 전체 tile 결과에 class-agnostic NMS 적용
        """
        h, w = img.shape[:2]

        if h <= self.imgsz and w <= self.imgsz:
            return self._run_single_patch(
                patch=img,
                conf_threshold=conf_threshold,
            )

        overlap_x = min(max(self.tile_overlap_x, 0.0), 0.5)
        overlap_y = min(max(self.tile_overlap_y, 0.0), 0.5)

        step_x = max(
            1,
            int(self.imgsz * (1.0 - overlap_x)),
        )

        step_y = max(
            1,
            int(self.imgsz * (1.0 - overlap_y)),
        )

        xs = self._tile_starts(
            w,
            self.imgsz,
            step_x,
        )

        ys = self._tile_starts(
            h,
            self.imgsz,
            step_y,
        )

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_cls_ids: List[np.ndarray] = []

        for y0 in ys:
            y1 = min(
                y0 + self.imgsz,
                h,
            )

            for x0 in xs:
                x1 = min(
                    x0 + self.imgsz,
                    w,
                )

                tile = img[
                    y0:y1,
                    x0:x1,
                ]

                th, tw = tile.shape[:2]

                if th < self.imgsz or tw < self.imgsz:
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

                boxes, scores, cls_ids = self._run_single_patch(
                    patch=tile,
                    conf_threshold=conf_threshold,
                )

                if len(boxes) == 0:
                    continue

                # detect_dot.py와 동일하게 padding 영역에서 시작한 bbox 제거
                valid_mask = (
                    (boxes[:, 0] < tw)
                    & (boxes[:, 1] < th)
                )

                boxes = boxes[valid_mask]
                scores = scores[valid_mask]
                cls_ids = cls_ids[valid_mask]

                if len(boxes) == 0:
                    continue

                boxes[:, [0, 2]] = np.clip(
                    boxes[:, [0, 2]] + x0,
                    0,
                    w,
                )

                boxes[:, [1, 3]] = np.clip(
                    boxes[:, [1, 3]] + y0,
                    0,
                    h,
                )

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_cls_ids.append(cls_ids)

        if not all_boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        all_boxes_np = np.concatenate(
            all_boxes,
            axis=0,
        )

        all_scores_np = np.concatenate(
            all_scores,
            axis=0,
        )

        all_cls_ids_np = np.concatenate(
            all_cls_ids,
            axis=0,
        )

        keep = self._nms_numpy(
            boxes=all_boxes_np,
            scores=all_scores_np,
            iou_thresh=self.iou_threshold,
        )

        return (
            all_boxes_np[keep],
            all_scores_np[keep],
            all_cls_ids_np[keep],
        )

    def _infer_single_map(
        self,
        img: np.ndarray,
        conf_threshold: float,
        foreground_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        h, w = img.shape[:2]

        amap = np.zeros(
            (h, w),
            dtype=np.float32,
        )

        # 1. ROI crop
        roi_img, roi_x0, roi_y0 = self._apply_roi(
            img,
            self.roi_left,
            self.roi_right,
            self.roi_top,
            self.roi_bottom,
        )

        # 2. Bale crop + mask 외부 bg_color 채움
        roi_img, roi_x0, roi_y0 = self._apply_bale_crop(
            roi_img,
            foreground_mask,
            roi_x0,
            roi_y0,
        )

        # 3. detect_dot.py 방식의 tiled YOLO
        boxes, scores, _cls_ids = self._infer_tiled_boxes(
            img=roi_img,
            conf_threshold=conf_threshold,
        )

        if len(boxes) == 0:
            return amap

        # 4. crop 좌표를 원본 이미지 좌표로 복원
        boxes[:, [0, 2]] += roi_x0
        boxes[:, [1, 3]] += roi_y0

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            w,
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            h,
        )

        # 5. bbox를 anomaly map으로 변환
        for (bx1, by1, bx2, by2), score in zip(
            boxes,
            scores,
        ):
            bx1 = int(np.clip(bx1, 0, w))
            bx2 = int(np.clip(bx2, 0, w))
            by1 = int(np.clip(by1, 0, h))
            by2 = int(np.clip(by2, 0, h))

            if bx2 <= bx1 or by2 <= by1:
                continue

            amap[
                by1:by2,
                bx1:bx2,
            ] = np.maximum(
                amap[
                    by1:by2,
                    bx1:bx2,
                ],
                float(score),
            )

        return amap

    def infer(
        self,
        images: Sequence[np.ndarray],
        conf_thresholds: Optional[
            Sequence[Optional[float]]
        ] = None,
        foreground_masks: Optional[Sequence[np.ndarray]] = None,
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

        # images가 1장이고 foreground_masks가 HxW 또는 HxWxC 단일 mask로 들어온 경우 대응
        masks = foreground_masks
        if isinstance(foreground_masks, np.ndarray):
            if (
                len(images) == 1
                and foreground_masks.ndim in (2, 3)
                and foreground_masks.shape[0] != 1
            ):
                masks = [foreground_masks]

        if len(images) != len(masks):
            raise ValueError(
                f"{self.name}: "
                f"len(images) ({len(images)}) must equal "
                f"len(foreground_masks) ({len(masks)})"
            )

        tile_patches = []
        tile_meta = []
        side_infos = []

        for idx, img in enumerate(images):
            conf_threshold = self.threshold

            if (
                conf_thresholds is not None
                and conf_thresholds[idx] is not None
            ):
                conf_threshold = float(
                    conf_thresholds[idx]
                )

            roi_img, roi_x0, roi_y0, tiles = self._build_tiles(
                img=img,
                conf_threshold=conf_threshold,
                foreground_mask=masks[idx],
            )
            side_infos.append((img.shape[:2], roi_img.shape[:2], roi_x0, roi_y0))

            for tile_idx, (tile, x0, y0, tw, th) in enumerate(tiles):
                tile_patches.append(tile)
                tile_meta.append((idx, tile_idx, x0, y0, tw, th, conf_threshold))

        tile_results = [self._empty_boxes() for _ in tile_patches]
        thresholds = sorted({meta[-1] for meta in tile_meta})
        for threshold in thresholds:
            indices = [
                idx
                for idx, meta in enumerate(tile_meta)
                if meta[-1] == threshold
            ]
            if not indices:
                continue

            results = self._predict_patches(
                [tile_patches[idx] for idx in indices],
                threshold,
            )
            for idx, result in zip(indices, results):
                tile_results[idx] = result

        results_by_side = [[] for _ in images]
        tiles_by_side = [[] for _ in images]
        for meta, result in zip(tile_meta, tile_results):
            side_idx, _tile_idx, x0, y0, tw, th, _threshold = meta
            results_by_side[side_idx].append(result)
            tiles_by_side[side_idx].append((None, x0, y0, tw, th))

        maps = []
        for side_idx, img in enumerate(images):
            img_shape, roi_shape, roi_x0, roi_y0 = side_infos[side_idx]
            boxes, scores, _cls_ids = self._restore_tile_boxes(
                results_by_side[side_idx],
                tiles_by_side[side_idx],
                roi_shape,
            )
            maps.append(
                self._boxes_to_map(
                    img_shape,
                    boxes,
                    scores,
                    roi_x0,
                    roi_y0,
                )
            )

        maps = np.stack(
            maps,
            axis=0,
        )

        return AnomalyCLIPOutputOldVersion(
            maps=maps,
            score_threshold=0.0,
            area_threshold=0,
            source=self.name,
        )

from typing import Any, List, Sequence, Tuple, Dict

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO

from defect_detection.outputs import Segmentation


class Segmenter:
    """Multi-checkpoint YOLO segmenter.

    Each entry in `models` is a dict describing one checkpoint:
        {
            "checkpoint": str,                # weights path
            "imgsz": int,                     # inference image size
            "threshold": float,               # model-level default threshold
            "classes": {                      # this model's class table
                <yolo_cls_id>: {
                    "name": str,
                    "color": tuple | None,
                    "pass": bool,             # True -> skip this class for this model
                    "conf": float,             # optional per-class threshold
                    "description": str,       # (optional)
                },
                ...
            },
        }

    Per-model `classes`:
        - Classes omitted from the dict are treated as `pass=True` (skipped).
        - Same `name` across models is unified in the final output -- their
          polygons are merged together via mask union.
    """

    def __init__(self, models: List[Dict[str, Any]]) -> None:
        if not models:
            raise ValueError("Segmenter requires at least one model config.")

        self.models: List[Dict[str, Any]] = []
        for cfg in models:
            if "checkpoint" not in cfg:
                raise ValueError(
                    f"Segmenter model config missing 'checkpoint': {cfg!r}"
                )
            self.models.append(
                {
                    "model": YOLO(cfg["checkpoint"]),
                    "imgsz": int(cfg.get("imgsz", 640)),
                    "threshold": float(cfg.get("threshold", 0.5)),
                    "classes": cfg.get("classes") or {},
                }
            )

        self._warmup()

    def _warmup(self, batch_size: int = 1) -> None:
        for idx, m in enumerate(self.models):
            for _ in tqdm.tqdm(
                range(5),
                desc=f"Warm up YOLO segmenter [{idx + 1}/{len(self.models)}]",
            ):
                dummy = [np.zeros((m["imgsz"], m["imgsz"], 3), np.uint8)]
                _ = m["model"](dummy, imgsz=m["imgsz"], verbose=False)

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
        offsets: Sequence[Tuple[int, int, int, int]],  # x1, y1, W, H
        full_w: int,
        full_h: int,
    ) -> List[Segmentation]:
        if len(patches) == 0:
            return []

        # Polygons unified by class name across all models.
        class_polys: Dict[str, List[Tuple[np.ndarray, float, float]]] = {}
        # name -> {"class_id", "name", "color"}, assigned in first-seen order.
        unified_classes: Dict[str, Dict[str, Any]] = {}

        for m in self.models:
            classes_cfg: Dict[int, Dict[str, Any]] = m["classes"]
            threshold: float = m["threshold"]

            results = m["model"](
                patches,
                imgsz=m["imgsz"],
                verbose=False,
            )

            for r, (x1, y1, W, H) in zip(results, offsets):

                if r.masks is None:
                    continue

                for mask, cls_id, conf in zip(
                    r.masks.xy,
                    r.boxes.cls,
                    r.boxes.conf,
                ):
                    conf = float(conf)

                    cls_id = int(cls_id)

                    cls_info = classes_cfg.get(cls_id)
                    if cls_info is None or cls_info.get("pass", False):
                        # this model opted out of this class
                        continue

                    class_threshold = float(cls_info.get("conf", threshold))
                    if conf < class_threshold:
                        continue

                    name = cls_info["name"]
                    if name not in unified_classes:
                        unified_classes[name] = {
                            "class_id": len(unified_classes),
                            "name": name,
                            "color": cls_info.get("color") or (0, 0, 255),
                        }

                    polygon_patch = np.array(mask)

                    polygon_global = polygon_patch.copy()
                    polygon_global[:, 0] += x1
                    polygon_global[:, 1] += y1
                    polygon_global = polygon_global.astype(np.float32)

                    area = cv2.contourArea(polygon_global)

                    class_polys.setdefault(name, []).append(
                        (polygon_global, conf, area)
                    )

        return self._merge_polygons_by_class(
            class_polys,
            unified_classes,
            full_w,
            full_h,
        )

    def _merge_polygons_by_class(
        self,
        class_polys: Dict[str, List[Tuple[np.ndarray, float, float]]],
        unified_classes: Dict[str, Dict[str, Any]],
        W: int,
        H: int,
    ) -> List[Segmentation]:

        region_segments: List[Segmentation] = []

        for cls_name, polys in class_polys.items():
            cls_info = unified_classes[cls_name]

            mask = np.zeros((H, W), dtype=np.uint8)

            # 1. mask 생성 (global 그대로)
            for poly, _, _ in polys:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

            # 2. morphology (optional)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 3. contour 추출
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                contour = contour.squeeze(1).astype(np.float32)

                if contour.shape[0] < 3:
                    continue

                # 4. 포함 polygon 찾기
                merged_confs = []
                merged_areas = []

                for poly, conf, area in polys:
                    cx, cy = poly.mean(axis=0)

                    if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                        merged_confs.append(conf)
                        merged_areas.append(area)

                if len(merged_confs) == 0:
                    continue

                merged_confs = np.array(merged_confs)
                merged_areas = np.array(merged_areas)

                # 5. confidence (area-weighted)
                confidence = float(
                    np.sum(merged_confs * merged_areas)
                    / np.sum(merged_areas)
                )

                # 6. bbox
                xmin, ymin = contour.min(axis=0)
                xmax, ymax = contour.max(axis=0)

                bbox_xyxy = (
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax),
                )

                # 7. normalize (전체 기준)
                polygon_n = contour.copy()
                polygon_n[:, 0] /= W
                polygon_n[:, 1] /= H

                xmin_n, ymin_n = polygon_n.min(axis=0)
                xmax_n, ymax_n = polygon_n.max(axis=0)

                bboxes_xyxy_n = (
                    max(0.0, min(1.0, float(xmin_n))),
                    max(0.0, min(1.0, float(ymin_n))),
                    max(0.0, min(1.0, float(xmax_n))),
                    max(0.0, min(1.0, float(ymax_n))),
                )

                # 8. area
                area = cv2.contourArea(contour)
                area_n = area / float(W * H)

                region_segments.append(
                    Segmentation(
                        polygon=contour,
                        polygon_n=polygon_n,
                        bboxes_xyxy=bbox_xyxy,
                        bboxes_xyxy_n=bboxes_xyxy_n,
                        confidence=confidence,
                        area=area,
                        area_n=area_n,
                        class_id=cls_info["class_id"],
                        class_name=cls_name,
                        color=cls_info.get("color") or (0, 0, 255),
                    )
                )

        return region_segments

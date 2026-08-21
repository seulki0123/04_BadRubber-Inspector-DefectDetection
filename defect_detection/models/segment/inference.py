import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
            "target": "crop" | "full",        # optional; default "crop"
            "device": str | None,              # e.g. "cuda:0", "cuda:1", "cpu"
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
        self.last_debug = {}
        for cfg in models:
            if "checkpoint" not in cfg:
                raise ValueError(
                    f"Segmenter model config missing 'checkpoint': {cfg!r}"
                )
            target = str(cfg.get("target", "crop")).lower()
            if target not in {"crop", "full"}:
                raise ValueError(
                    f"Segmenter model target must be 'crop' or 'full', got {target!r}"
                )
            self.models.append(
                {
                    "model": YOLO(cfg["checkpoint"]),
                    "imgsz": int(cfg.get("imgsz", 640)),
                    "threshold": float(cfg.get("threshold", 0.5)),
                    "target": target,
                    "classes": cfg.get("classes") or {},
                    "device": cfg.get("device"),
                    "predict_batch_size": max(1, int(cfg.get("predict_batch_size", 32))),
                }
            )
        self.has_full_target = any(m["target"] == "full" for m in self.models)

        self._warmup()

    def _warmup(self, batch_size: int = 1) -> None:
        for idx, m in enumerate(self.models):
            for _ in tqdm.tqdm(
                range(5),
                desc=f"Warm up YOLO segmenter [{idx + 1}/{len(self.models)}]",
            ):
                dummy = [np.zeros((m["imgsz"], m["imgsz"], 3), np.uint8)]
                _ = m["model"](
                    dummy,
                    imgsz=m["imgsz"],
                    device=m["device"],
                    verbose=False,
                )

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
        offsets: Sequence[Tuple[int, int, int, int]],  # x1, y1, W, H
        full_w: int,
        full_h: int,
        full_image: Optional[np.ndarray] = None,
    ) -> List[Segmentation]:
        if len(patches) == 0 and not self.has_full_target:
            return []

        per_image = self.infer_patches_by_image(
            patches,
            [(0, x1, y1, W, H) for x1, y1, W, H in offsets],
            {0: (full_w, full_h)},
            full_images={0: full_image} if full_image is not None else None,
        )
        return per_image.get(0, [])

    def infer_patches_by_image(
        self,
        patches: Sequence[np.ndarray],
        offsets: Sequence[Tuple[int, int, int, int, int]],  # b_idx, x1, y1, W, H
        image_shapes: Dict[int, Tuple[int, int]],  # b_idx -> (W, H)
        full_images: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, List[Segmentation]]:
        full_images = full_images or {}
        if len(patches) == 0 and not (self.has_full_target and full_images):
            self.last_debug = {
                "patches": 0,
                "model_chunks": [],
                "wall_ms": 0.0,
                "speed_ms": {},
            }
            return {}

        class_polys_by_batch: Dict[int, Dict[str, List[Tuple[np.ndarray, float, float]]]] = {}
        unified_classes_by_batch: Dict[int, Dict[str, Dict[str, Any]]] = {}
        started = time.perf_counter()
        model_chunks = []
        speed_totals: Dict[str, float] = {}

        for m in self.models:
            classes_cfg: Dict[int, Dict[str, Any]] = m["classes"]
            threshold: float = m["threshold"]
            batch_size = m["predict_batch_size"]
            if m["target"] == "full":
                model_inputs = [
                    (b_idx, image)
                    for b_idx, image in full_images.items()
                    if b_idx in image_shapes
                ]
            else:
                model_inputs = list(zip(offsets, patches))

            for start in range(0, len(model_inputs), batch_size):
                chunk_items = model_inputs[start : start + batch_size]
                if m["target"] == "full":
                    chunk = [image for _, image in chunk_items]
                    meta_chunk = [
                        (b_idx, 0, 0, image_shapes[b_idx][0], image_shapes[b_idx][1])
                        for b_idx, _ in chunk_items
                    ]
                else:
                    meta_chunk = [meta for meta, _ in chunk_items]
                    chunk = [patch for _, patch in chunk_items]

                if not chunk:
                    continue

                model_chunks.append(len(chunk))
                results = m["model"](
                    chunk,
                    imgsz=m["imgsz"],
                    device=m["device"],
                    verbose=False,
                )
                for result in results:
                    for key, value in getattr(result, "speed", {}).items():
                        speed_totals[key] = speed_totals.get(key, 0.0) + float(value)

                for r, (b_idx, x1, y1, W, H) in zip(results, meta_chunk):

                    if r.masks is None:
                        continue

                    class_polys = class_polys_by_batch.setdefault(b_idx, {})
                    unified_classes = unified_classes_by_batch.setdefault(b_idx, {})

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

        self.last_debug = {
            "patches": len(patches),
            "model_chunks": model_chunks,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "speed_ms": speed_totals,
        }

        return {
            b_idx: self._merge_polygons_by_class(
                class_polys_by_batch.get(b_idx, {}),
                unified_classes_by_batch.get(b_idx, {}),
                W,
                H,
            )
            for b_idx, (W, H) in image_shapes.items()
        }

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

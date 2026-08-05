from typing import Optional, Sequence, Tuple, List

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO
from defect_detection.outputs import AnomalyCLIPOutputOldVersion


class ObjectDetector:
    def __init__(
        self,
        checkpoint_path: str,
        imgsz: int,
        threshold: float,
        name: str,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.threshold = threshold
        self.device = device
        self._warmup()
        self.name = name

    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(range(10), desc="Warm up YOLO detection model"):
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

    def infer(
        self,
        images: Sequence[np.ndarray],
        conf_thresholds: Optional[Sequence[Optional[float]]] = None,
    ):
        if conf_thresholds is not None and len(images) != len(conf_thresholds):
            raise ValueError(
                f"{self.name}: len(images) ({len(images)}) must equal "
                f"len(conf_thresholds) ({len(conf_thresholds)})"
            )

        results = self.model(
            images,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        maps = self._yolo_to_maps(results, images, conf_thresholds=conf_thresholds)

        return AnomalyCLIPOutputOldVersion(
            maps=maps,
            score_threshold=0.0,
            area_threshold=0,  # bbox는 작을 수 있으니 낮게
            source=self.name,
        )

    def _yolo_to_maps(
        self,
        results,
        images,
        conf_thresholds: Optional[Sequence[Optional[float]]] = None,
    ):
        maps = []

        for idx, (result, img) in enumerate(zip(results, images)):
            h, w = img.shape[:2]
            amap = np.zeros((h, w), dtype=np.float32)
            conf_threshold = self.threshold
            if conf_thresholds is not None and conf_thresholds[idx] is not None:
                conf_threshold = float(conf_thresholds[idx])

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), score in zip(boxes, scores):
                    if score < conf_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # bbox 영역에 score 채우기
                    amap[y1:y2, x1:x2] = np.maximum(
                        amap[y1:y2, x1:x2], score
                    )

            maps.append(amap)

        return np.stack(maps)

from typing import Sequence, Tuple, List

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO
from defect_detection.outputs import AnomalyCLIPOutput


class ObjectDetector:
    def __init__(self, checkpoint_path: str, imgsz: int, threshold: float, name: str) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.threshold = threshold
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
            _ = self.model(dummy_images, imgsz=self.imgsz, verbose=False)

    def infer(
        self,
        images: Sequence[np.ndarray],
    ):
        results = self.model(images, imgsz=self.imgsz, verbose=False)

        maps = self._yolo_to_maps(results, images)

        return AnomalyCLIPOutput(
            maps=maps,
            score_threshold=self.threshold,
            area_threshold=0,  # bbox는 작을 수 있으니 낮게
            super_area_threshold=None,
            source=self.name,
        )

    def _yolo_to_maps(self, results, images):
        maps = []

        for result, img in zip(results, images):
            h, w = img.shape[:2]
            amap = np.zeros((h, w), dtype=np.float32)

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), score in zip(boxes, scores):
                    if score < self.threshold:
                        continue

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # bbox 영역에 score 채우기
                    amap[y1:y2, x1:x2] = np.maximum(
                        amap[y1:y2, x1:x2], score
                    )

            maps.append(amap)

        return np.stack(maps)
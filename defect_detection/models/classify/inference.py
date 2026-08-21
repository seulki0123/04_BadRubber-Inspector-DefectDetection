import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import tqdm
from ultralytics import YOLO

from defect_detection.outputs import Classification


class Classifier:
    def __init__(
        self,
        checkpoint_path: str,
        classes: Dict[int, Dict[str, Any]],
        imgsz: int = 32,
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
        predict_batch_size: int = 32,
    ) -> None:
        self.model = YOLO(checkpoint_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.device = device
        self.predict_batch_size = max(1, int(predict_batch_size))
        self.last_debug = {}
        self._warmup()

    def _warmup(self, batch_size: int = 1) -> None:
        for _ in tqdm.tqdm(range(5), desc="Warm up YOLO classification model"):
            dummy = [np.zeros((self.imgsz, self.imgsz, 3), np.uint8)]
            _ = self.model(
                dummy,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
    ) -> List[Classification]:

        if len(patches) == 0:
            self.last_debug = {
                "patches": 0,
                "chunks": 0,
                "batch_size": self.predict_batch_size,
                "wall_ms": 0.0,
                "speed_ms": {},
            }
            return []

        started = time.perf_counter()
        results = []
        speed_totals: Dict[str, float] = {}
        chunk_sizes = []

        for start in range(0, len(patches), self.predict_batch_size):
            chunk = patches[start : start + self.predict_batch_size]
            chunk_sizes.append(len(chunk))
            chunk_results = self.model(
                chunk,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            results.extend(chunk_results)
            for result in chunk_results:
                for key, value in getattr(result, "speed", {}).items():
                    speed_totals[key] = speed_totals.get(key, 0.0) + float(value)

        self.last_debug = {
            "patches": len(patches),
            "chunks": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "batch_size": self.predict_batch_size,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "speed_ms": speed_totals,
        }

        outputs: List[Classification] = []

        for r in results:
            if r.probs is None:
                outputs.append(
                    Classification(
                        class_id=-1,
                        class_name="unknown",
                        confidence=0.0,
                        is_pass=True,
                        color=(0, 0, 0),
                    )
                )
                continue

            cls_id = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            class_name = self.classes[cls_id]["name"]
            is_pass = self.classes[cls_id]["pass"] or conf < self.conf_threshold
            color = self.classes[cls_id]["color"] if not is_pass else (0, 0, 0)

            outputs.append(
                Classification(
                    class_id=cls_id,
                    confidence=conf,
                    class_name=class_name,
                    is_pass=is_pass,
                    color=color,
                )
            )

        return outputs

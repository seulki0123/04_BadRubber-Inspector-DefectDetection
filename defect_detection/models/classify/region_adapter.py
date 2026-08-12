from collections import Counter
from typing import List, Tuple
import numpy as np

from defect_detection.outputs import RegionClassificationOutput, Classification
from defect_detection.outputs.anomalyclip import AnomalyCLIPOutput
from defect_detection.utils import scale_bbox_xyxy_n
from .inference import Classifier


class RegionClassifierAdapter:
    """
    AnomalyCLIPOutput → RegionClassificationOutput
    """

    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.last_debug = {}

    def infer(
        self,
        images: List[np.ndarray],
        anomaly: AnomalyCLIPOutput,
    ) -> RegionClassificationOutput:

        patches = []
        mapping: List[Tuple[int, int]] = []
        source_counts = Counter()
        region_count = 0

        for b_idx, (img, regions) in enumerate(zip(images, anomaly.batch_regions)):
            H, W = img.shape[:2]
            region_count += len(regions)

            for r_idx, region in enumerate(regions):
                source_counts[region.source] += 1
                scale = 10.0 if "dot" in region.source else 2.0
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(region.bboxes_xyxy_n, scale=scale)

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                mapping.append((b_idx, r_idx, region.is_pass))

        self.last_debug = {
            "regions": region_count,
            "patches": len(patches),
            "source_counts": dict(source_counts),
        }

        results = self.classifier.infer_patches(patches)

        batch_out = [
            [None] * len(regions)
            for regions in anomaly.batch_regions
        ]

        for (b_idx, r_idx, is_pass), cls in zip(mapping, results):
            region_cls = cls if not is_pass else None
            batch_out[b_idx][r_idx] = region_cls

        for b in range(len(batch_out)):
            for r in range(len(batch_out[b])):
                if batch_out[b][r] is None:
                    batch_out[b][r] = Classification(
                        class_id=-1,
                        class_name="unknown",
                        confidence=0.0,
                        is_pass=True,
                        color=(0, 0, 0),
                    )

        return RegionClassificationOutput(batch_out)

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from defect_detection.outputs import SegmentationOutput, Segmentation
from defect_detection.outputs.anomalyclip import AnomalyCLIPOutput
from defect_detection.outputs.classify import RegionClassificationOutput
from defect_detection.utils import scale_bbox_xyxy_n
from .inference import Segmenter


class RegionSegmenterAdapter:
    """
    AnomalyCLIPOutput → SegmentationOutput
    [B][R][S] (B: batch size, R: region size, S: segmentation size)
    """

    def __init__(self, segmenter: Segmenter):
        self.segmenter = segmenter
        self.last_debug = {}

    def infer(
        self,
        images: List[np.ndarray],
        anomaly: AnomalyCLIPOutput,
        classifications: RegionClassificationOutput,
    ) -> SegmentationOutput:

        patches: List[np.ndarray] = []
        offsets: List[Tuple[int, int, int, int, int]] = []
        patches_by_batch: Dict[int, int] = defaultdict(int)
        regions_with_patch: Dict[int, Set[int]] = defaultdict(set)
        region_count = 0
        pass_count = 0
        image_shapes = {
            b_idx: (img.shape[1], img.shape[0])
            for b_idx, img in enumerate(images)
        }

        # 1. collect patches (배치 이미지별로 묶음)
        for b_idx, (img, regions) in enumerate(
            zip(images, anomaly.batch_regions)
        ):
            H, W = img.shape[:2]
            region_count += len(regions)

            for r_idx, (region, region_cls) in enumerate(
                zip(regions, classifications[b_idx].regions)
            ):
                if region_cls.is_pass:
                    pass_count += 1
                    continue

                scale = 1.0 if "tiles" in region.source else 2.0
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(
                    region.bboxes_xyxy_n, scale=scale
                )

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches.append(patch)
                offsets.append((b_idx, x1, y1, W, H))
                patches_by_batch[b_idx] += 1
                regions_with_patch[b_idx].add(r_idx)

        # 2. segmentation inference — 이미지(배치 인덱스) 단위로 병합·좌표계 유지
        patch_count = len(patches)
        self.last_debug = {
            "regions": region_count,
            "pass_regions": pass_count,
            "candidate_regions": region_count - pass_count,
            "patches": patch_count,
            "patches_by_batch": dict(patches_by_batch),
        }

        merged_by_batch = self.segmenter.infer_patches_by_image(
            patches,
            offsets,
            image_shapes,
        )
        self.last_debug["model"] = getattr(self.segmenter, "last_debug", {})

        # 3. create [B][R][S] structure
        batch_out: List[List[List[Segmentation]]] = []

        for b_idx in range(len(images)):
            num_regions = len(anomaly.batch_regions[b_idx])
            batch_out.append([[] for _ in range(num_regions)])

        # # 4. 같은 이미지에서 나온 region 대통합 결과를 해당 region들에 동일하게 부여
        # for b_idx, r_set in regions_with_patch.items():
        #     segs = merged_by_batch[b_idx]
        #     for r_idx in r_set:
        #         batch_out[b_idx][r_idx] = segs

        # 4. 이미지 단위 대통합 seg는 region 인덱스가 가장 작은 칸 하나에만 둠 (중복 카운트 방지)
        for b_idx, r_set in regions_with_patch.items():
            r_primary = min(r_set)
            batch_out[b_idx][r_primary] = merged_by_batch[b_idx]
            
        return SegmentationOutput(batch_out)

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

    def infer(
        self,
        images: List[np.ndarray],
        anomaly: AnomalyCLIPOutput,
        classifications: RegionClassificationOutput,
    ) -> SegmentationOutput:

        patches_by_batch: Dict[int, List[np.ndarray]] = defaultdict(list)
        offsets_by_batch: Dict[int, List[Tuple[int, int, int, int]]] = (
            defaultdict(list)
        )
        regions_with_patch: Dict[int, Set[int]] = defaultdict(set)

        # 1. collect patches (배치 이미지별로 묶음)
        for b_idx, (img, regions) in enumerate(
            zip(images, anomaly.batch_regions)
        ):
            H, W = img.shape[:2]

            for r_idx, (region, region_cls) in enumerate(
                zip(regions, classifications[b_idx].regions)
            ):
                if region_cls.is_pass:
                    continue

                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(
                    region.bboxes_xyxy_n, scale=2.0
                )

                x1, y1 = int(x1n * W), int(y1n * H)
                x2, y2 = int(x2n * W), int(y2n * H)

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                patches_by_batch[b_idx].append(patch)
                offsets_by_batch[b_idx].append((x1, y1, W, H))
                regions_with_patch[b_idx].add(r_idx)

        # 2. segmentation inference — 이미지(배치 인덱스) 단위로 병합·좌표계 유지
        merged_by_batch: Dict[int, List[Segmentation]] = {}
        batch_indices = set(patches_by_batch)
        if self.segmenter.has_full_target:
            batch_indices.update(range(len(images)))

        for b_idx in batch_indices:
            H, W = images[b_idx].shape[:2]
            merged_by_batch[b_idx] = self.segmenter.infer_patches(
                patches_by_batch.get(b_idx, []),
                offsets_by_batch.get(b_idx, []),
                full_w=W,
                full_h=H,
                full_image=images[b_idx],
            )

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
        for b_idx, segs in merged_by_batch.items():
            if not segs:
                continue

            r_set = regions_with_patch.get(b_idx)
            if r_set:
                r_primary = min(r_set)
                batch_out[b_idx][r_primary] = segs
            elif self.segmenter.has_full_target:
                batch_out[b_idx].append(segs)
            
        return SegmentationOutput(batch_out)

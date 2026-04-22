from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import cv2
import numpy as np

from .colors import HeatmapColorRanges

@dataclass
class AnomalyRegion:
    __slots__ = (
        "polygon",
        "polygon_n",
        "bboxes_xyxy",
        "bboxes_xyxy_n",
        "confidence",
        "area",
        "area_n",
        "color",
        "class_id",
        "class_name",
        "source",
        "is_pass",
    )

    polygon: np.ndarray
    polygon_n: np.ndarray            # (N, 2) float32 normalized
    bboxes_xyxy: Tuple[int, int, int, int]
    bboxes_xyxy_n: Tuple[float, float, float, float]
    confidence: float
    area: float
    area_n: float
    source: str
    is_pass: bool

@dataclass
class AnomalyCLIPBatchItem:
    __slots__ = (
        "map",
        "regions",
        "super_regions",
        "global_score",
        "source",
    )
    map: np.ndarray
    regions: List[AnomalyRegion]
    super_regions: List[AnomalyRegion]
    global_score: float
    source: str

@dataclass
class AnomalyCLIPOutput:
    __slots__ = (
        "maps",
        "score_threshold",
        "area_threshold",
        "super_area_threshold",
        "batch_regions",
        "super_batch_regions",
        "global_scores",
        "source",
    )

    # declare only fields that should be declared as dataclass fields
    maps: np.ndarray                 # (B, H, W)
    score_threshold: float
    area_threshold: float
    # super defect(= red) region 으로 인정할 최소 픽셀 크기.
    # None 이면 area_threshold 와 동일하게 동작.
    # NOTE: __slots__ 와 dataclass default 충돌 때문에 기본값을 두지 않는다.
    # 호출부에서 명시적으로 None 을 전달해야 한다.
    super_area_threshold: Optional[float]
    source: str

    def __post_init__(self):
        self._validate_inputs()

        batch_regions, super_batch_regions, global_scores = self._extract_regions_batch()

        # create slots-only attributes here
        object.__setattr__(self, "batch_regions", batch_regions)
        object.__setattr__(self, "super_batch_regions", super_batch_regions)
        object.__setattr__(self, "global_scores", global_scores)

    def _validate_inputs(self):

        if not isinstance(self.maps, np.ndarray):
            raise TypeError("maps must be np.ndarray")

        if self.maps.ndim != 3:
            raise ValueError("maps must have shape (B, H, W)")

        if not isinstance(self.score_threshold, (float, int)):
            raise TypeError("score_threshold must be float")

        if not isinstance(self.area_threshold, (float, int)):
            raise TypeError("area_threshold must be float")

        if self.super_area_threshold is not None and not isinstance(
            self.super_area_threshold, (float, int)
        ):
            raise TypeError("super_area_threshold must be float or None")

    def _extract_regions_batch(
        self,
    ) -> Tuple[List[List[AnomalyRegion]], List[List[AnomalyRegion]], List[float]]:

        regions_batch = []
        super_regions_batch = []
        global_scores_batch = []

        for amap in self.maps:
            regions, super_regions = self._extract_regions_single(amap)
            global_score = self._compute_global_score(amap)
            print(global_score)
            # global_score = 0.0

            regions_batch.append(regions)
            super_regions_batch.append(super_regions)
            global_scores_batch.append(global_score)

        return regions_batch, super_regions_batch, global_scores_batch

    def _extract_regions_single(
        self,
        amap: np.ndarray,
    ) -> Tuple[List[AnomalyRegion], List[AnomalyRegion]]:

        H, W = amap.shape
        raw = np.asarray(amap, dtype=np.float32)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.clip(raw, 0.0, None)

        # Normalize to [0, 1] for colormap-based region extraction.
        # The raw anomaly map can have arbitrary small values (e.g. max ~0.2)
        # that would map entirely to "blue" in JET without normalization,
        # producing zero detected regions.
        positive = raw[raw > 0]
        if positive.size == 0:
            return [], []

        lo = float(np.percentile(positive, 5.0))
        hi = float(np.percentile(positive, 99.5))
        if hi <= lo:
            lo = float(positive.min())
            hi = float(positive.max())

        if hi <= lo:
            return [], []

        amap_norm = np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        amap_norm[raw <= 0] = 0.0

        # Build raw-value score mask using score_threshold mapped back
        # through the same normalization (so threshold 0.25 means ~top 25%
        # of the positive value range, not absolute raw value).
        norm_threshold = float(max(self.score_threshold, 0.0))
        score_mask = (amap_norm >= norm_threshold).astype(np.uint8) * 255

        color_ranges = HeatmapColorRanges(amap_norm)

        # 시각화에서 파란색(차가운 영역)을 제외한 모든 영역을 일반 region 으로 추출.
        binary = color_ranges.non_blue()
        binary = cv2.bitwise_and(binary, score_mask)
        regions = self._regions_from_binary(
            binary, amap_norm, H, W, area_threshold=self.area_threshold,
        )

        # "가장 뜨거운" red 영역을 super defect 로 별도 추출.
        # super defect 는 일정 크기(= super_area_threshold) 이상부터만 인정.
        super_area_threshold = (
            self.super_area_threshold
            if self.super_area_threshold is not None
            else self.area_threshold
        )
        super_binary = cv2.bitwise_and(color_ranges.red(), score_mask)
        super_regions = self._regions_from_binary(
            super_binary, amap_norm, H, W, area_threshold=super_area_threshold,
        )

        return regions, super_regions

    def _regions_from_binary(
        self,
        binary: np.ndarray,
        amap_norm: np.ndarray,
        H: int,
        W: int,
        area_threshold: Optional[float] = None,
    ) -> List[AnomalyRegion]:

        if binary is None or not np.any(binary):
            return []

        if area_threshold is None:
            area_threshold = self.area_threshold

        # 작은 끊김/구멍을 줄여 bbox 누락 완화
        kernel_close = np.ones((5, 5), dtype=np.uint8)
        kernel_open = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < area_threshold:
                continue
            area_n = area / float(H * W)

            polygon = cnt.reshape(-1, 2).astype(np.int32)

            polygon_n = polygon.astype(np.float32)
            polygon_n[:, 0] = np.clip(polygon_n[:, 0] / W, 0.0, 1.0)
            polygon_n[:, 1] = np.clip(polygon_n[:, 1] / H, 0.0, 1.0)

            x, y, w, h = cv2.boundingRect(cnt)
            bbox_xyxy = (x, y, x + w, y + h)

            bboxes_xyxy_n = (
                max(0.0, min(1.0, x / W)),
                max(0.0, min(1.0, y / H)),
                max(0.0, min(1.0, (x + w) / W)),
                max(0.0, min(1.0, (y + h) / H)),
            )

            score = self._compute_polygon_score(amap_norm, polygon)

            regions.append(
                AnomalyRegion(
                    polygon=polygon,
                    polygon_n=polygon_n,
                    bboxes_xyxy=bbox_xyxy,
                    bboxes_xyxy_n=bboxes_xyxy_n,
                    confidence=score,
                    area=area,
                    area_n=area_n,
                    source=self.source,
                    is_pass=False,
                )
            )

        return regions

    def _compute_polygon_score(
        self,
        anomaly_map: np.ndarray,
        polygon: np.ndarray,
        mode: str = "mean",
    ) -> float:

        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)

        values = anomaly_map[mask == 1]

        if values.size == 0:
            return 0.0

        if mode == "mean":
            return float(values.mean())

        if mode == "max":
            return float(values.max())

        raise ValueError(mode)

    def _compute_global_score(
        self,
        anomaly_map: np.ndarray,
        mode: str = "topk",
        topk_ratio: float = 0.05,
    ) -> float:

        flat = anomaly_map.reshape(-1)

        if mode == "mean":
            return float(flat.mean())

        if mode == "max":
            return float(flat.max())

        if mode == "topk":
            k = max(1, int(len(flat) * topk_ratio))
            return float(np.partition(flat, -k)[-k:].mean())

        raise ValueError(mode)

    def __len__(self):
        return self.maps.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield AnomalyCLIPBatchItem(
                map=self.maps[i],
                regions=self.batch_regions[i],
                super_regions=self.super_batch_regions[i],
                global_score=self.global_scores[i],
                source=self.source,
            )

    def __getitem__(self, idx):
        return AnomalyCLIPBatchItem(
            map=self.maps[idx],
            regions=self.batch_regions[idx],
            super_regions=self.super_batch_regions[idx],
            global_score=self.global_scores[idx],
            source=self.source,
        )

def merge_anomlay_outputs(outputs: List[AnomalyCLIPOutput]) -> AnomalyCLIPOutput:
    assert len(outputs) > 0

    batch_size = len(outputs[0].maps)

    dummy_maps = np.zeros_like(outputs[0].maps, dtype=np.float32)

    merged = AnomalyCLIPOutput(
        maps=dummy_maps,
        score_threshold=0.0,
        area_threshold=0.0,
        super_area_threshold=None,
        source="merged",
    )

    # 안전하게 override
    new_regions = [[] for _ in range(batch_size)]
    new_super_regions = [[] for _ in range(batch_size)]

    for out in outputs:
        if out is None:
            continue
        for i in range(batch_size):
            new_regions[i].extend(out.batch_regions[i])
            # super_batch_regions 는 AnomalyCLIP 계열에만 존재하므로 안전하게 접근
            super_regions = getattr(out, "super_batch_regions", None)
            if super_regions is not None:
                new_super_regions[i].extend(super_regions[i])

    object.__setattr__(merged, "batch_regions", new_regions)
    object.__setattr__(merged, "super_batch_regions", new_super_regions)

    return merged

def filter_by_cluster(anomaly, cluster_output):
    new_regions = []

    for regions, cluster_regions in zip(anomaly.batch_regions, cluster_output.batch):
        updated = []

        for r, c in zip(regions, cluster_regions):
            r.is_pass = c.is_pass
            r.class_name = c.class_name
            updated.append(r)

        new_regions.append(updated)

    object.__setattr__(anomaly, "batch_regions", new_regions)
    return anomaly
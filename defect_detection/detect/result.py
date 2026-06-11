from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Sequence
import numpy as np

from defect_detection.outputs import (
    ForegroundMaskOutput,
    AnomalyCLIPOutput,
    RegionClassificationOutput,
    ClassificationBatchItem,
    SegmentationOutput,
    SegmentationBatchItem,
    AnomalyCLIPBatchItem,
    ForegroundMaskBatchItem,
    PatchcoreOutput,
    PatchcoreBatchItem,
)
from .visualize import visualize


# ---------------------------------
# Batch Item
# ---------------------------------

@dataclass
class DetectorBatchItem:
    image: np.ndarray
    foreground: ForegroundMaskBatchItem
    anomaly: AnomalyCLIPBatchItem
    anomaly_cls: ClassificationBatchItem
    segmentation: SegmentationBatchItem
    segmentation_cls: ClassificationBatchItem
    patchcore: Optional[PatchcoreBatchItem] = None
    show: Dict[str, Any] = field(default_factory=dict)

    @property
    def regions(self):
        return self.anomaly.batch_regions

    def visualize(self) -> np.ndarray:
        show = self.show or {}
        return visualize(
            image=self.image,
            foreground=self.foreground,
            anomaly=self.anomaly,
            anomaly_cls=self.anomaly_cls,
            segmentation=self.segmentation,
            segmentation_cls=self.segmentation_cls,
            patchcore=self.patchcore,
            show_foreground=show.get("foreground", False),
            show_anomaly_map=show.get("anomaly_map", False),
            show_anomaly_score=show.get("anomaly_score", False),
            show_anomaly_regions_polygon=show.get("anomaly_regions_polygon", False),
            show_anomaly_regions_bbox=show.get("anomaly_regions_bbox", False),
            show_segmentation_regions_polygon=show.get("segmentation_regions_polygon", False),
            show_segmentation_regions_bbox=show.get("segmentation_regions_bbox", False),
            show_patchcore=show.get("patchcore", False),
            show_pass_classes=show.get("show_pass_classes", False),
        )


# ---------------------------------
# Output
# ---------------------------------

@dataclass
class DetectorOutput:
    images: Sequence[np.ndarray]
    foreground: ForegroundMaskOutput
    anomaly: AnomalyCLIPOutput
    anomaly_cls: ClassificationBatchItem
    segmentation: SegmentationOutput
    segmentation_cls: ClassificationBatchItem
    patchcore: Optional[PatchcoreOutput] = None
    show: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        B = len(self.images)

        if len(self.foreground) != B:
            raise ValueError("Foreground batch size mismatch")

        if len(self.anomaly) != B:
            raise ValueError("Anomaly batch size mismatch")

        if len(self.anomaly_cls) != B:
            raise ValueError("Classification batch size mismatch")

        if self.segmentation is not None:
            if len(self.segmentation) != B:
                raise ValueError("Segmentation batch size mismatch")

        if self.segmentation_cls is not None:
            if len(self.segmentation_cls) != B:
                raise ValueError("Segmentation classification batch size mismatch")

        if self.patchcore is not None:
            if len(self.patchcore) != B:
                raise ValueError("Patchcore batch size mismatch")

    def __len__(self):
        return len(self.images)

    def __iter__(self) -> Iterator[DetectorBatchItem]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> DetectorBatchItem:
        return DetectorBatchItem(
            image=self.images[idx],
            foreground=self.foreground[idx],
            anomaly=self.anomaly[idx],
            anomaly_cls=self.anomaly_cls[idx],
            segmentation=self.segmentation[idx] if self.segmentation is not None else None,
            segmentation_cls=self.segmentation_cls[idx] if self.segmentation_cls is not None else None,
            patchcore=self.patchcore[idx] if self.patchcore is not None else None,
            show=self.show,
        )
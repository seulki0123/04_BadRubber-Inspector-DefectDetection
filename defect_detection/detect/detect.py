import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from defect_detection.models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter, ObjectDetector, TiledObjectDetector, Cluster, PatchcoreDetector
from defect_detection.outputs import RegionClassificationOutput, ClassificationBatchItem, Classification, merge_anomlay_outputs, filter_by_cluster, merge_cls_outputs
from defect_detection.utils import load_config, random_color
from .result import DetectorOutput
from .visualize import draw_normalized_polygons

class Detector:
    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config
        self.show = config.get("show") or {}

        self.anomaly_extractor = AnomalyCLIPInference(
            checkpoint_path=config["anomalyclip"]["checkpoint"],
            imgsz=config["anomalyclip"]["imgsz"],
            score_threshold=config["anomalyclip"]["threshold"],
            area_threshold=config["anomalyclip"]["min_area"],
        )

        self.bgremover = BackgroundRemover(
            checkpoint_path=config["bgremover"]["checkpoint"],
            imgsz=config["bgremover"]["imgsz"],
            use_blur_mask=config["bgremover"]["use_blur_mask"],
            blur_kernel=config["bgremover"]["blur_kernel"],
            blur_threshold=config["bgremover"]["blur_threshold"],
            blur_resize_scale=config["bgremover"]["blur_resize_scale"],
        )

        if config['anomaly_cluster'] is not None:
            self.region_anomaly_cluster = RegionClassifierAdapter(
                Cluster(
                checkpoints_path=config["anomaly_cluster"]["checkpoints_path"],
                threshold=config["anomaly_cluster"]["threshold"],
                classes=config["anomaly_cluster"]["classes"],
                )
            )
        else:
            self.region_anomaly_cluster = None

        if config['dot_detector1'] is not None:
            self.dot_detector1 = ObjectDetector(
                checkpoint_path=config["dot_detector1"]["checkpoint"],
                imgsz=config["dot_detector1"]["imgsz"],
                threshold=config["dot_detector1"]["threshold"],
                name="dot_detector1",
            )
        else:
            self.dot_detector1 = None

        if config['dot_detector2'] is not None:
            self.dot_detector2 = ObjectDetector(
                checkpoint_path=config["dot_detector2"]["checkpoint"],
                imgsz=config["dot_detector2"]["imgsz"],
                threshold=config["dot_detector2"]["threshold"],
                name="dot_detector2",
            )
        else:
            self.dot_detector2 = None

        if config['tile_detector'] is not None:
            self.tile_detector = TiledObjectDetector(
                checkpoint_path=config["tile_detector"]["checkpoint"],
                imgsz=config["tile_detector"]["imgsz"],
                threshold=config["tile_detector"]["threshold"],
                name="tile_detector",
                tile_overlap_x=config["tile_detector"]["tile_overlap_x"],
                tile_overlap_y=config["tile_detector"]["tile_overlap_y"],
                roi_left=config["tile_detector"]["roi_left"],
                roi_right=config["tile_detector"]["roi_right"],
                roi_top=config["tile_detector"]["roi_top"],
                roi_bottom=config["tile_detector"]["roi_bottom"],
            )
        else:
            self.tile_detector = None

        if config['dot_cluster'] is not None:
            self.region_dot_cluster = RegionClassifierAdapter(
                Cluster(
                checkpoints_path=config["dot_cluster"]["checkpoints_path"],
                threshold=config["dot_cluster"]["threshold"],
                classes=config["dot_cluster"]["classes"],
                )
            )
        else:
            self.region_dot_cluster = None

        if config.get('dot_classifier') is not None:
            self.region_dot_classifier = RegionClassifierAdapter(
                Classifier(
                checkpoint_path=config["dot_classifier"]["checkpoint"],
                imgsz=config["dot_classifier"]["imgsz"],
                conf_threshold=config["dot_classifier"]["threshold"],
                classes=config["dot_classifier"]["classes"],
                )
            )
        else:
            self.region_dot_classifier = None

        if config['classifier'] is not None:
            self.region_classifier = RegionClassifierAdapter(
                Classifier(
                checkpoint_path=config["classifier"]["checkpoint"],
                imgsz=config["classifier"]["imgsz"],
                conf_threshold=config["classifier"]["threshold"],
                classes=config["classifier"]["classes"],
                )
            )
        else:
            self.region_classifier = None

        if config.get('patchcore') is not None:
            self.patchcore = PatchcoreDetector(
                checkpoint_path=config["patchcore"]["checkpoint"],
                backbone_path=config["patchcore"]["backbone"],
                holdout_path=config["patchcore"]["holdout"],
                score_threshold=config["patchcore"]["threshold"],
                imgsz=config["patchcore"]["imgsz"],
            )
        else:
            self.patchcore = None

        if config['segmenter'] is not None:
            seg_cfg = config["segmenter"]
            # backward-compat: single-model dict -> wrap into list
            seg_models = [seg_cfg] if isinstance(seg_cfg, dict) else list(seg_cfg)
            self.region_segmenter = RegionSegmenterAdapter(
                Segmenter(models=seg_models)
            )
        else:
            self.region_segmenter = None

    # ---------------------------------
    # Main API
    # ---------------------------------

    def detect(
        self,
        images: List[np.ndarray],
        dot_confs: Optional[Sequence[Optional[float]]] = None,
        patchcore_active: Optional[Sequence[Optional[bool]]] = None,
    ) -> DetectorOutput:
        t0 = time.time()
        if dot_confs is not None and len(images) != len(dot_confs):
            raise ValueError(
                f"len(images) ({len(images)}) must equal len(dot_confs) ({len(dot_confs)})"
            )
        if patchcore_active is not None and len(images) != len(patchcore_active):
            raise ValueError(
                f"len(images) ({len(images)}) must equal len(patchcore_active) ({len(patchcore_active)})"
            )

        # read images
        # images = [cv2.imread(p) for p in imgs_path]
        t1 = time.time()

        foreground = self.bgremover.infer(images)
        t2 = time.time()

        # detect anomaly regions
        anomaly = self.anomaly_extractor.infer(images, foreground.masks)
        t3 = time.time()

        # cluster anomaly regions
        anomaly_clusters = self.region_anomaly_cluster.infer(images, anomaly) if self.region_anomaly_cluster is not None else None
        anomaly = filter_by_cluster(anomaly, anomaly_clusters) if anomaly_clusters is not None else anomaly
        t4 = time.time()

        # classify anomaly regions
        anomaly_cls = self.region_classifier.infer(images, anomaly) if self.region_classifier is not None else anomaly_clusters
        t5 = time.time()

        # segment anomaly regions
        segmentation = self.region_segmenter.infer(foreground.images, anomaly, anomaly_cls) if self.region_segmenter is not None else None
        t6 = time.time()

        # reclassify segmented regions
        segmentation_cls = [ClassificationBatchItem(regions=[]) for _ in range(len(images))] if segmentation is not None else None
        t7 = time.time()

        # (Optional, Independent from Anomaly) dot detection
        dot1 = self.dot_detector1.infer(foreground.images, conf_thresholds=dot_confs) if self.dot_detector1 is not None else None
        t8 = time.time()

        dot2 = self.dot_detector2.infer(foreground.images, conf_thresholds=dot_confs) if self.dot_detector2 is not None else None
        t9 = time.time()

        dot3 = self.tile_detector.infer(images, conf_thresholds=dot_confs, foreground_masks=foreground.masks) if self.tile_detector is not None else None
        t10 = time.time()

        merged_dot = merge_anomlay_outputs([x for x in (dot1, dot2) if x is not None]) if any(x is not None for x in (dot1, dot2)) else None
        t11 = time.time()

        # TODO: 점이물 하드 코딩, 추후 개선
        dot_clusters = self.region_dot_cluster.infer(images, merged_dot) if self.region_dot_cluster is not None else (RegionClassificationOutput([[Classification(class_id=-1, class_name="foreign", confidence=float(r.confidence), is_pass=False, color=(0, 0, 255)) for r in regions] for regions in merged_dot.batch_regions]) if merged_dot is not None else None)
        merged_dot = filter_by_cluster(merged_dot, dot_clusters) if dot_clusters is not None else merged_dot
        t12 = time.time()

        dot_cls = self.region_dot_classifier.infer(images, merged_dot) if self.region_dot_classifier is not None else dot_clusters
        t13 = time.time()
        
        tile_cls = RegionClassificationOutput([[Classification(class_id=-1, class_name="foreign", confidence=float(r.confidence), is_pass=False, color=(0, 0, 255)) for r in regions] for regions in dot3.batch_regions]) if dot3 is not None else None
        
        # Merge Anomaly's and Dot Detection's Classifications
        merged_anomaly = merge_anomlay_outputs([anomaly, merged_dot, dot3])
        merged_cls = merge_cls_outputs([anomaly_cls, dot_cls, tile_cls])
        t14 = time.time()

        patchcore = self.patchcore.infer(images, foreground.masks, active_by_side=patchcore_active) if self.patchcore is not None else None
        t15 = time.time()

        # TODO:
        # Merge Segmentation and Dot Detection

        print(f"load images          : {(t1  - t0 ) * 1000:.1f} ms")
        print(f"foreground           : {(t2  - t1 ) * 1000:.1f} ms")
        print(f"anomaly              : {(t3  - t2 ) * 1000:.1f} ms")
        print(f"anomaly_cluster      : {(t4  - t3 ) * 1000:.1f} ms")
        print(f"classification       : {(t5  - t4 ) * 1000:.1f} ms")
        print(f"segmentation         : {(t6  - t5 ) * 1000:.1f} ms")
        print(f"segmentation_cls     : {(t7  - t6 ) * 1000:.1f} ms")

        print(f"dot1                 : {(t8  - t7 ) * 1000:.1f} ms")
        print(f"dot2                 : {(t9  - t8 ) * 1000:.1f} ms")
        print(f"tile_detector        : {(t10 - t9 ) * 1000:.1f} ms")

        print(f"merge_dot            : {(t11 - t10) * 1000:.1f} ms")
        print(f"dot_cluster          : {(t12 - t11) * 1000:.1f} ms")
        print(f"dot_classification   : {(t13 - t12) * 1000:.1f} ms")

        print(f"merge_final          : {(t14 - t13) * 1000:.1f} ms")
        print(f"patchcore            : {(t15 - t14) * 1000:.1f} ms")

        print(f"image count          : {len(images)}")
        print(f"total                : {(t15 - t0) * 1000:.1f} ms")

        return DetectorOutput(
            images=images,
            foreground=foreground,
            anomaly=merged_anomaly,
            anomaly_cls=merged_cls,
            segmentation=segmentation,
            segmentation_cls=segmentation_cls,
            patchcore=patchcore,
            show=self.show,
        )
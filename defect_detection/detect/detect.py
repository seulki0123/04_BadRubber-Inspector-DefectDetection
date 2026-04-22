import time
from typing import List, Tuple

import cv2
import numpy as np

from defect_detection.models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter, ObjectDetector, Cluster
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

        if config['segmenter'] is not None:
            self.region_segmenter = RegionSegmenterAdapter(
                Segmenter(
                checkpoint_path=config["segmenter"]["checkpoint"],
                imgsz=config["segmenter"]["imgsz"],
                conf_threshold=config["segmenter"]["threshold"],
                classes=config["segmenter"]["classes"],
                )
            )
        else:
            self.region_segmenter = None

    # ---------------------------------
    # Main API
    # ---------------------------------

    def detect(self, images: List[np.ndarray]) -> DetectorOutput:
        t0 = time.time()

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
        dot1 = self.dot_detector1.infer(foreground.images) if self.dot_detector1 is not None else None
        dot2 = self.dot_detector2.infer(foreground.images) if self.dot_detector2 is not None else None
        merged_dot = merge_anomlay_outputs([dot1, dot2]) if dot1 is not None else None
        t8 = time.time()

        dot_clusters = self.region_dot_cluster.infer(images, merged_dot) if self.region_dot_cluster is not None else (RegionClassificationOutput([[Classification(class_id=-1, class_name=r.source, confidence=float(r.confidence), is_pass=False, color=(0, 0, 255)) for r in regions] for regions in merged_dot.batch_regions]) if merged_dot is not None else None)
        merged_dot = filter_by_cluster(merged_dot, dot_clusters) if dot_clusters is not None else merged_dot
        t9 = time.time()
        
        # Merge Anomaly's and Dot Detection's Classifications
        merged_anomaly = merge_anomlay_outputs([anomaly, merged_dot])
        merged_cls = merge_cls_outputs([anomaly_cls, dot_clusters])
        t10 = time.time()

        # TODO:
        # Merge Segmentation and Dot Detection


        print(f"load images: {(t1-t0)*1000}ms")
        print(f"foreground: {(t2-t1)*1000}ms")
        print(f"anomaly: {(t3-t2)*1000}ms")
        print(f"anomaly_cluster: {(t4-t3)*1000}ms")
        print(f"classification: {(t5-t4)*1000}ms")
        print(f"segmentation: {(t6-t5)*1000}ms")
        print(f"segmentation_cls: {(t7-t6)*1000}ms")
        print(f"dot1: {(t8-t7)*1000}ms")
        print(f"dot2: {(t9-t8)*1000}ms")
        print(f"dot_cluster: {(t10-t9)*1000}ms")
        print(f"image count: {len(images)}")
        print(f"total: {(t10-t0)*1000}ms")

        return DetectorOutput(
            images=images,
            foreground=foreground,
            anomaly=merged_anomaly,
            anomaly_cls=merged_cls,
            segmentation=segmentation,
            segmentation_cls=segmentation_cls,
            show=self.show,
        )
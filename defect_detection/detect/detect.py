import time
from typing import List, Tuple

import cv2
import numpy as np

from defect_detection.models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter, ObjectDetector, Cluster
from defect_detection.outputs import RegionClassificationOutput, ClassificationBatchItem, merge_anomlay_outputs, filter_by_cluster, merge_cls_outputs, Segmentation
from defect_detection.utils import load_config, random_color
from .result import DetectorOutput
from .visualize import draw_normalized_polygons

SUPER_DEFECT_CLASS_NAME = "etc"
SUPER_DEFECT_CLASS_ID = -1
SUPER_DEFECT_COLOR = (0, 0, 255)
SUPER_DEFECT_COVERAGE_THRESHOLD = 0.8
# super defect(= red region) 로 인정할 최소 픽셀 크기.
# TODO: 임시 상수. 안정화되면 config/profile 로 이동.
SUPER_DEFECT_MIN_AREA = 30000

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
            super_area_threshold=SUPER_DEFECT_MIN_AREA,
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

        # super defect(= AnomalyCLIP 의 red_high region)는 무조건 검출되어야 한다.
        # 기존 segmentation 이 super defect 를 80% 이상 커버하지 못하면 segmentation 에 추가.
        if segmentation is not None:
            segmentation = self._inject_super_defects(
                images,
                anomaly,
                segmentation,
                coverage_threshold=SUPER_DEFECT_COVERAGE_THRESHOLD,
            )

        t6 = time.time()

        # reclassify segmented regions
        segmentation_cls = [ClassificationBatchItem(regions=[]) for _ in range(len(images))] if segmentation is not None else None
        t7 = time.time()

        # (Optional, Independent from Anomaly) dot detection
        dot1 = self.dot_detector1.infer(foreground.images) if self.dot_detector1 is not None else None
        dot2 = self.dot_detector2.infer(foreground.images) if self.dot_detector2 is not None else None
        merged_dot = merge_anomlay_outputs([dot1, dot2]) if dot1 is not None else None
        t8 = time.time()

        dot_clusters = self.region_dot_cluster.infer(images, merged_dot) if self.region_dot_cluster is not None else None
        merged_dot = filter_by_cluster(merged_dot, dot_clusters) if dot_clusters is not None else merged_dot
        t9 = time.time()
        
        # Merge Anomaly's and Dot Detection's Classifications
        # merged_anomaly = merge_anomlay_outputs([anomaly, merged_dot])
        # merged_cls = merge_cls_outputs([anomaly_cls, dot_clusters])
        merged_anomaly = anomaly
        merged_cls = anomaly_cls
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

    # ---------------------------------
    # Super defect injection
    # ---------------------------------

    def _inject_super_defects(
        self,
        images: List[np.ndarray],
        anomaly,
        segmentation,
        coverage_threshold: float = 0.8,
    ):
        """Super defect (AnomalyCLIP red_high region) 중 기존 segmentation 이
        `coverage_threshold` 이상 커버하지 못하는 것들을 segmentation 에 강제 추가한다.

        - "1번(super defect) 기준" overlap 비율 = intersection / super_defect_area
        - 기존 segmentation 중 하나라도 기준을 만족하면 커버된 것으로 보고 스킵.
        - 커버 안 된 super defect 들은 새 region 슬롯으로 추가한다.
        """
        super_batch = getattr(anomaly, "super_batch_regions", None)
        if not super_batch:
            return segmentation

        for b_idx, img in enumerate(images):
            if b_idx >= len(super_batch):
                break

            super_regions = super_batch[b_idx]
            if not super_regions:
                continue

            H, W = img.shape[:2]

            # 해당 batch 의 모든 segmentation 을 flatten
            existing_segs: List[Segmentation] = []
            for r_segs in segmentation.batch_regions[b_idx]:
                existing_segs.extend(r_segs)

            # segmentation 마스크를 미리 렌더링해두고 재사용
            seg_masks = []
            for s in existing_segs:
                m = np.zeros((H, W), dtype=np.uint8)
                poly = np.asarray(s.polygon, dtype=np.int32)
                if poly.size == 0:
                    seg_masks.append(None)
                    continue
                cv2.fillPoly(m, [poly.reshape(-1, 2)], 1)
                seg_masks.append(m)

            added_segs: List[Segmentation] = []
            for sr in super_regions:
                sr_poly = np.asarray(sr.polygon, dtype=np.int32)
                if sr_poly.size == 0:
                    continue

                sr_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(sr_mask, [sr_poly.reshape(-1, 2)], 1)
                sr_area = int(sr_mask.sum())
                if sr_area == 0:
                    continue

                covered = False
                for s_mask in seg_masks:
                    if s_mask is None:
                        continue
                    inter = int(np.logical_and(sr_mask, s_mask).sum())
                    if inter / sr_area >= coverage_threshold:
                        covered = True
                        break

                if covered:
                    continue

                added_segs.append(self._super_region_to_segmentation(sr))

            if added_segs:
                # [B][R][S] 구조에서 새 region 슬롯으로 추가
                segmentation.batch_regions[b_idx].append(added_segs)

        return segmentation

    def _super_region_to_segmentation(self, region) -> Segmentation:
        return Segmentation(
            polygon=region.polygon,
            polygon_n=region.polygon_n,
            bboxes_xyxy=region.bboxes_xyxy,
            bboxes_xyxy_n=region.bboxes_xyxy_n,
            confidence=float(region.confidence),
            area=float(region.area),
            area_n=float(region.area_n),
            class_id=SUPER_DEFECT_CLASS_ID,
            class_name=SUPER_DEFECT_CLASS_NAME,
            color=SUPER_DEFECT_COLOR,
        )
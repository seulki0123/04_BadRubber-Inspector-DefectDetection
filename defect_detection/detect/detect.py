from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from defect_detection.models import AnomalyCLIPInference, BackgroundRemover, Classifier, RegionClassifierAdapter, Segmenter, RegionSegmenterAdapter, ObjectDetector, TiledObjectDetector, TiledAnomalyExtractor, Cluster, PatchcoreDetector
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

        anomaly_extractor_cfg = config.get("anomaly_extractor") or {}
        anomalyclip_cfg = config.get("anomalyclip") or {}
        if anomaly_extractor_cfg.get("use_tiles", False):
            self.anomaly_extractor = TiledAnomalyExtractor(
                grids=anomaly_extractor_cfg.get("grids", (3, 4)),
                overlap=anomaly_extractor_cfg.get("overlap", 0.25),
                pad_ratio=anomaly_extractor_cfg.get("pad_ratio", 0.05),
                score=anomaly_extractor_cfg.get("score", 1.0),
                score_threshold=anomaly_extractor_cfg.get(
                    "score_threshold",
                    anomalyclip_cfg.get("threshold", 0.0),
                ),
                area_threshold=anomaly_extractor_cfg.get(
                    "area_threshold",
                    anomalyclip_cfg.get("min_area", 0),
                ),
                name=anomaly_extractor_cfg.get("name", "tiles"),
            )
        else:
            self.anomaly_extractor = AnomalyCLIPInference(
                checkpoint_path=anomalyclip_cfg["checkpoint"],
                imgsz=anomalyclip_cfg["imgsz"],
                score_threshold=anomalyclip_cfg["threshold"],
                area_threshold=anomalyclip_cfg["min_area"],
                device=anomalyclip_cfg.get("device"),
            )

        self.bgremover = BackgroundRemover(
            checkpoint_path=config["bgremover"]["checkpoint"],
            imgsz=config["bgremover"]["imgsz"],
            use_blur_mask=config["bgremover"]["use_blur_mask"],
            blur_kernel=config["bgremover"]["blur_kernel"],
            blur_threshold=config["bgremover"]["blur_threshold"],
            blur_resize_scale=config["bgremover"]["blur_resize_scale"],
            device=config["bgremover"].get("device"),
        )

        if config['anomaly_cluster'] is not None:
            self.region_anomaly_cluster = RegionClassifierAdapter(
                Cluster(
                checkpoints_path=config["anomaly_cluster"]["checkpoints_path"],
                threshold=config["anomaly_cluster"]["threshold"],
                classes=config["anomaly_cluster"]["classes"],
                device=config["anomaly_cluster"].get("device"),
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
                device=config["dot_detector1"].get("device"),
            )
        else:
            self.dot_detector1 = None

        if config['dot_detector2'] is not None:
            self.dot_detector2 = ObjectDetector(
                checkpoint_path=config["dot_detector2"]["checkpoint"],
                imgsz=config["dot_detector2"]["imgsz"],
                threshold=config["dot_detector2"]["threshold"],
                name="dot_detector2",
                device=config["dot_detector2"].get("device"),
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
                device=config["tile_detector"].get("devices",config["tile_detector"].get("device"),),
            )
        else:
            self.tile_detector = None

        if config['dot_cluster'] is not None:
            self.region_dot_cluster = RegionClassifierAdapter(
                Cluster(
                checkpoints_path=config["dot_cluster"]["checkpoints_path"],
                threshold=config["dot_cluster"]["threshold"],
                classes=config["dot_cluster"]["classes"],
                device=config["dot_cluster"].get("device"),
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
                device=config["dot_classifier"].get("device"),
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
                device=config["classifier"].get("device"),
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
                device=config["patchcore"].get("device"),
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

    @staticmethod
    def _timed_call(fn, *args, **kwargs):
        started = time.time()
        result = fn(*args, **kwargs)
        return result, time.time() - started

    def _run_anomaly_pipeline(self, images, foreground):
        started = time.time()
        anomaly = self.anomaly_extractor.infer(images, foreground.masks)
        anomaly_clusters = self.region_anomaly_cluster.infer(images, anomaly) if self.region_anomaly_cluster is not None else None
        anomaly = filter_by_cluster(anomaly, anomaly_clusters) if anomaly_clusters is not None else anomaly
        anomaly_cls = self.region_classifier.infer(images, anomaly) if self.region_classifier is not None else anomaly_clusters
        segmentation = self.region_segmenter.infer(foreground.images, anomaly, anomaly_cls) if self.region_segmenter is not None else None
        segmentation_cls = [ClassificationBatchItem(regions=[]) for _ in range(len(images))] if segmentation is not None else None
        return anomaly, anomaly_cls, segmentation, segmentation_cls, time.time() - started

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

        parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            anomaly_task = executor.submit(self._run_anomaly_pipeline, images, foreground)
            dot1_task = executor.submit(self._timed_call, self.dot_detector1.infer, foreground.images, conf_thresholds=dot_confs) if self.dot_detector1 is not None else None
            dot2_task = executor.submit(self._timed_call, self.dot_detector2.infer, foreground.images, conf_thresholds=dot_confs) if self.dot_detector2 is not None else None
            tile_task = executor.submit(self._timed_call, self.tile_detector.infer, images, conf_thresholds=dot_confs, foreground_masks=foreground.masks) if self.tile_detector is not None else None
            patchcore_task = executor.submit(self._timed_call, self.patchcore.infer, images, foreground.masks, active_by_side=patchcore_active) if self.patchcore is not None else None

            # (Optional, Independent from Anomaly) dot detection
            dot1, dot1_time = dot1_task.result() if dot1_task is not None else (None, 0.0)
            dot2, dot2_time = dot2_task.result() if dot2_task is not None else (None, 0.0)

            merge_dot_start = time.time()
            merged_dot = merge_anomlay_outputs([x for x in (dot1, dot2) if x is not None]) if any(x is not None for x in (dot1, dot2)) else None
            merge_dot_time = time.time() - merge_dot_start

            # TODO: 점이물 하드 코딩, 추후 개선
            dot_cluster_start = time.time()
            dot_clusters = self.region_dot_cluster.infer(images, merged_dot) if self.region_dot_cluster is not None else (RegionClassificationOutput([[Classification(class_id=-1, class_name="foreign", confidence=float(r.confidence), is_pass=False, color=(0, 0, 255)) for r in regions] for regions in merged_dot.batch_regions]) if merged_dot is not None else None)
            merged_dot = filter_by_cluster(merged_dot, dot_clusters) if dot_clusters is not None else merged_dot
            dot_cluster_time = time.time() - dot_cluster_start

            dot_classification_start = time.time()
            dot_cls = self.region_dot_classifier.infer(images, merged_dot) if self.region_dot_classifier is not None else dot_clusters
            dot_classification_time = time.time() - dot_classification_start

            anomaly, anomaly_cls, segmentation, segmentation_cls, anomaly_time = anomaly_task.result()
            dot3, tile_time = tile_task.result() if tile_task is not None else (None, 0.0)
            patchcore, patchcore_time = patchcore_task.result() if patchcore_task is not None else (None, 0.0)

        tile_cls = RegionClassificationOutput([[Classification(class_id=-1, class_name="foreign", confidence=float(r.confidence), is_pass=False, color=(0, 0, 255)) for r in regions] for regions in dot3.batch_regions]) if dot3 is not None else None

        merge_final_start = time.time()
        merged_anomaly = merge_anomlay_outputs([anomaly, merged_dot, dot3])
        merged_cls = merge_cls_outputs([anomaly_cls, dot_cls, tile_cls])
        merge_final_time = time.time() - merge_final_start
        parallel_time = time.time() - parallel_start
        t15 = time.time()

        # TODO:
        # Merge Segmentation and Dot Detection

        print(f"load images          : {(t1  - t0 ) * 1000:.1f} ms")
        print(f"foreground           : {(t2  - t1 ) * 1000:.1f} ms")
        print(f"anomaly pipeline     : {anomaly_time * 1000:.1f} ms")

        print(f"dot1                 : {dot1_time * 1000:.1f} ms")
        print(f"dot2                 : {dot2_time * 1000:.1f} ms")
        print(f"tile_detector        : {tile_time * 1000:.1f} ms")

        print(f"merge_dot            : {merge_dot_time * 1000:.1f} ms")
        print(f"dot_cluster          : {dot_cluster_time * 1000:.1f} ms")
        print(f"dot_classification   : {dot_classification_time * 1000:.1f} ms")

        print(f"merge_final          : {merge_final_time * 1000:.1f} ms")
        print(f"patchcore            : {patchcore_time * 1000:.1f} ms")
        print(f"parallel wall        : {parallel_time * 1000:.1f} ms")

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

from typing import Sequence, Optional
import numpy as np
import cv2

from defect_detection.outputs import ForegroundMaskBatchItem, AnomalyCLIPBatchItem, ClassificationBatchItem, SegmentationBatchItem


def visualize(
    image: np.ndarray,
    foreground: ForegroundMaskBatchItem,
    anomaly: AnomalyCLIPBatchItem,
    anomaly_cls: ClassificationBatchItem,
    segmentation: SegmentationBatchItem,
    segmentation_cls: ClassificationBatchItem,
    show_foreground: bool = True,
    show_anomaly_map: bool = True,
    show_anomaly_score: bool = True,
    show_anomaly_regions_polygon: bool = True,
    show_anomaly_regions_bbox: bool = True,
    show_segmentation_regions_polygon: bool = True,
    show_segmentation_regions_bbox: bool = True,
    show_pass_classes: bool = True,
) -> np.ndarray:

    vis_img = image.copy()
    
    # draw foreground
    if show_foreground:
        vis_img = draw_normalized_polygons(
            image=vis_img,
            polygons_n=foreground.polygons_n,
            colors=[(0, 255, 0) for _ in range(len(foreground.polygons_n))],
            thickness=5,
        )

    # draw anomaly map
    if show_anomaly_map:
        vis_img = draw_anomaly_map(
            image=vis_img,
            anomaly_map=anomaly.map,
            alpha=0.3,
        )
    
    # draw anomaly score
    if show_anomaly_score:
        vis_img = draw_text(
            image=vis_img,
            text=f"Global anomaly: {anomaly.global_score:.3f}",
            position="top_left",
            bg_color=None,
            text_color=(255, 255, 255),
            font_scale=3,
            font_thickness=5,
        )

    # draw anomaly regions
    if show_anomaly_regions_polygon:
        vis_img = draw_normalized_polygons(
            image=vis_img,
            polygons_n=[region.polygon_n for region in anomaly.regions],
            # labels=[region.class_name for region in anomaly.regions],
            colors=[(255, 255, 255) if region.is_pass else (0, 0, 255) for region in anomaly.regions],
            is_draw=[not region.is_pass for region in anomaly_cls.regions] if not show_pass_classes else None,
            thickness=5,
        )

    # draw anomaly region bboxes
    if show_anomaly_regions_bbox:
        vis_img = draw_bboxes_xyxyn(
            image=vis_img,
            bboxes_xyxyn=[region.bboxes_xyxy_n for region in anomaly.regions],
            labels=[f"{region.class_name} {region.confidence:.2f}" for region in anomaly_cls.regions],
            colors=[region.color for region in anomaly_cls.regions],
            is_draw=[not region.is_pass for region in anomaly_cls.regions] if not show_pass_classes else None,
            thickness=5,
            font_scale=2,
            font_thickness=2,
        )

    # TODO: 점이물 하드 코딩, 추후 개선
    else:
        vis_img = draw_bboxes_xyxyn(
            image=vis_img,
            bboxes_xyxyn=[region.bboxes_xyxy_n for idx, region in enumerate(anomaly.regions) if anomaly_cls.regions[idx].class_name == "foreign"],
            labels=[f"{region.class_name} {region.confidence:.2f}" for region in anomaly_cls.regions if region.class_name == "foreign"],
            colors=[region.color for region in anomaly_cls.regions if region.class_name == "foreign"],
            is_draw=[not region.is_pass for region in anomaly_cls.regions if region.class_name == "foreign"] if not show_pass_classes else None,
            thickness=5,
            font_scale=2,
            font_thickness=2,
        )
    
    # draw segmentation regions
    if show_segmentation_regions_polygon:
        seg_regions = [seg for region in segmentation.regions for seg in region]
        vis_img = draw_normalized_polygons(
            image=vis_img,
            polygons_n=[region.polygon_n for region in seg_regions],
            # labels=[f"{region.class_name} {region.confidence:.2f}" for region in seg_regions],
            colors=[region.color for region in seg_regions],
            thickness=5,
            font_scale=2,
            font_thickness=2,
        )

    # draw segmentation region bboxes
    if show_segmentation_regions_bbox:
        seg_regions = [seg for region in segmentation.regions for seg in region]

        vis_img = draw_bboxes_xyxyn(
            image=vis_img,
            bboxes_xyxyn=[seg.bboxes_xyxy_n for seg in seg_regions],
            labels=[f"{seg.class_name} {seg.confidence:.2f}" for seg in seg_regions],
            colors=[seg.color for seg in seg_regions],
            thickness=5,
            font_scale=2,
            font_thickness=2,
        )

        # Dot detections live in anomaly outputs. When anomaly bbox drawing is off,
        # keep them visible in segmentation mode by drawing dot bboxes here.
        if not show_anomaly_regions_bbox:
            dot_pairs = [
                (region, cls)
                for region, cls in zip(anomaly.regions, anomaly_cls.regions)
                if str(getattr(region, "source", "")).startswith("dot_detector")
                and (show_pass_classes or not cls.is_pass)
            ]
            if dot_pairs:
                vis_img = draw_bboxes_xyxyn(
                    image=vis_img,
                    bboxes_xyxyn=[region.bboxes_xyxy_n for region, _ in dot_pairs],
                    labels=[f"{cls.class_name} {cls.confidence:.2f}" for _, cls in dot_pairs],
                    colors=[cls.color for _, cls in dot_pairs],
                    thickness=5,
                    font_scale=2,
                    font_thickness=2,
                )

    return vis_img

def draw_normalized_polygons(
    image: np.ndarray,
    polygons_n: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[tuple]] = None,
    is_draw: Optional[Sequence[bool]] = None,
    thickness: int = 2,
    font_scale: float = 2,
    font_thickness: int = 1,
    closed: bool = True,
    alpha: float = 0.6,
) -> np.ndarray:

    vis_img = image.copy()
    overlay = vis_img.copy()
    H, W = vis_img.shape[:2]

    if colors is None:
        colors = [(0, 255, 0)] * len(polygons_n)

    for i, poly_n in enumerate(polygons_n):
        if is_draw is not None and not is_draw[i]:
            continue

        if poly_n.size == 0:
            continue

        color = colors[i] if i < len(colors) else (0, 255, 0)

        poly_px = poly_n.astype(np.float32).copy()
        poly_px[:, 0] *= W
        poly_px[:, 1] *= H
        poly_px = poly_px.astype(np.int32)

        cv2.polylines(
            overlay,
            [poly_px],
            isClosed=closed,
            color=color,
            thickness=thickness,
        )

    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    if labels is not None:
        for i, poly_n in enumerate(polygons_n):
            if is_draw is not None and not is_draw[i]:
                continue
            if poly_n.size == 0 or i >= len(labels):
                continue
            poly_px = poly_n.copy()
            poly_px[:, 0] *= W
            poly_px[:, 1] *= H
            poly_px = poly_px.astype(np.int32)

            vis_img = draw_text(
                image=vis_img,
                text=str(labels[i]),
                origin=tuple(poly_px[0]),
                anchor_mode="bottom",
                text_color=colors[i],
                bg_color=None,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )

    return vis_img

def draw_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:

    amap = np.clip(anomaly_map, 0.0, 1.0)

    heatmap = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mask = amap > 0  # (H, W) bool

    image[mask] = (
        image[mask] * (1 - alpha)
        + heatmap[mask] * alpha
    ).astype(np.uint8)

    return image

def draw_bboxes_xyxyn(
    image: np.ndarray,
    bboxes_xyxyn: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[tuple]] = None,
    is_draw: Optional[Sequence[bool]] = None,
    thickness: int = 2,
    font_scale: float = 2,
    font_thickness: int = 1,
    alpha: float = 0.6,
) -> np.ndarray:

    vis_img = image.copy()
    overlay = vis_img.copy()
    H, W = vis_img.shape[:2]
    
    if colors is None:
        colors = [(0, 0, 255)] * len(bboxes_xyxyn)

    for i, box in enumerate(bboxes_xyxyn):
        if is_draw is not None and not is_draw[i]:
            continue

        x1n, y1n, x2n, y2n = box

        x1 = int(x1n * W)
        y1 = int(y1n * H)
        x2 = int(x2n * W)
        y2 = int(y2n * H)

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        color = colors[i]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)

    if labels:
        for i, box in enumerate(bboxes_xyxyn):
            if is_draw is not None and not is_draw[i]:
                continue
            if i >= len(labels):
                continue
            x1 = int(box[0] * W)
            y1 = int(box[1] * H)

            color = colors[i]
            
            vis_img = draw_text(
                image=vis_img,
                text=str(labels[i]),
                origin=(x1, y1),
                anchor_mode="top",
                text_color=color,
                bg_color=None,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )

    return vis_img

def draw_text(
    image: np.ndarray,
    text: str,
    origin: Optional[tuple] = None,      # (x, y) anchor
    position: Optional[str] = None,      # screen-based position (top_left, etc.)
    margin: int = 0,
    font_scale: float = 2,
    font_thickness: int = 1,
    text_color: tuple = (255, 255, 255),
    bg_color: Optional[tuple] = None,    # None means no background
    alpha: float = 0.6,
    anchor_mode: str = "top",            # origin-based position (top/bottom/center)
) -> np.ndarray:
    """
    Unified text drawing function.

    Usage:
    1) position → screen-based position
    2) origin → specific coordinate-based
    """

    vis = image.copy()
    H, W = vis.shape[:2]

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )

    box_w = tw + 8
    box_h = th + baseline + 8

    if position is not None:
        if position == "top_left":
            x1 = margin
            y1 = margin
        elif position == "top_right":
            x1 = W - box_w - margin
            y1 = margin
        elif position == "bottom_left":
            x1 = margin
            y1 = H - box_h - margin
        elif position == "bottom_right":
            x1 = W - box_w - margin
            y1 = H - box_h - margin
        elif position == "center":
            x1 = (W - box_w) // 2
            y1 = (H - box_h) // 2
        else:
            raise ValueError(f"Unknown position: {position}")

    elif origin is not None:
        x, y = origin

        if anchor_mode == "top":
            x1 = x
            y1 = y - box_h - margin
        elif anchor_mode == "bottom":
            x1 = x
            y1 = y + margin
        elif anchor_mode == "center":
            x1 = x - box_w // 2
            y1 = y - box_h // 2
        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

    else:
        raise ValueError("Either position or origin must be provided")

    x1 = max(0, min(W - box_w, x1))
    y1 = max(0, min(H - box_h, y1))
    x2 = x1 + box_w
    y2 = y1 + box_h

    if bg_color is not None:
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    text_x = x1 + 4
    text_y = y1 + box_h - baseline - 4

    cv2.putText(
        vis,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return vis
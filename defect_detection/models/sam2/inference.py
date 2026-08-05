from typing import List, Optional

import cv2
import tqdm
import torch
import numpy as np

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("sam2 is not installed")
from defect_detection.outputs import SAM2Output, SAM2Region, AnomalyRegion
from defect_detection.utils import mask_to_polygon, normalize_polygon, scale_bbox_xyxy_n, random_color
from defect_detection.detect.visualize import draw_normalized_polygons


class SAM2Inference:

    def __init__(
        self,
        checkpoint_path: str,
        config_name: str,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.5,
        points_per_side: int = 1,
        imgsz: int = 96,  # warmup dummy size
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        sam2_model = build_sam2(
            f"configs/{config_name}",
            checkpoint_path,
            device=self.device,
        )

        self.predictor = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

        self.imgsz = imgsz
        self._warmup()

    def _warmup(self, n_iters: int = 5):

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)

        for _ in tqdm.tqdm(range(n_iters), desc="Warm up SAM2 model"):
            _ = self.predictor.generate(dummy)

    def infer(
        self,
        images: List[np.ndarray],
        anomaly_regions: List[List[AnomalyRegion]],
        visualize: bool = False,
    ) -> SAM2Output:

        batch_out: List[List[List[SAM2Region]]] = []
        vis_images = [] if visualize else None

        for idx, (img, regions) in enumerate(zip(images, anomaly_regions)):
            vis_img = img.copy() if visualize else None

            H, W = img.shape[:2]
            image_out: List[List[SAM2Region]] = []

            for region in regions:

                # 1. scale bbox
                x1n, y1n, x2n, y2n = scale_bbox_xyxy_n(
                    region.bboxes_xyxy_n,
                    scale=2.0,
                )

                x1 = int(x1n * W)
                y1 = int(y1n * H)
                x2 = int(x2n * W)
                y2 = int(y2n * H)

                # 안전성 체크
                if x2 <= x1 or y2 <= y1:
                    image_out.append([])
                    continue

                crop = img[y1:y2, x1:x2]

                if crop.shape[0] < 16 or crop.shape[1] < 16:
                    image_out.append([])
                    continue

                # 2. run SAM2
                anns = self.predictor.generate(crop)

                region_out: List[SAM2Region] = []

                for ann in anns:
                    mask = ann["segmentation"].astype(np.uint8)
                    area = float(ann["area"])
                    predicted_iou = float(ann["predicted_iou"])
                    stability_score = float(ann["stability_score"])

                    polygon = mask_to_polygon(mask)

                    if polygon is None or len(polygon) < 3:
                        continue

                    # 3. restore crop to original coordinates
                    polygon[:, 0] += x1
                    polygon[:, 1] += y1

                    polygon_n = normalize_polygon(polygon, H, W)

                    if visualize:
                        color = random_color(bright=False)

                        vis_img = draw_normalized_polygons(
                            image=vis_img,
                            polygons_n=[polygon_n],
                            labels=[f"{predicted_iou:.2f}_{stability_score:.2f}"],
                            colors=[color],
                            thickness=1,
                            font_scale=0.3,
                        )

                    region_out.append(
                        SAM2Region(
                            polygon_n=polygon_n,
                            area=area,
                            predicted_iou=predicted_iou,
                            stability_score=stability_score,
                        )
                    )

                image_out.append(region_out)

            if visualize:
                cv2.imwrite(f"./tests/sam2/sam2_crop_{idx}.jpg", vis_img)

            batch_out.append(image_out)

        return SAM2Output(batch=batch_out)

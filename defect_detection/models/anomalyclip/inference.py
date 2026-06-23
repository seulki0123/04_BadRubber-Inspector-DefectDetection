import time
from typing import List, Tuple, Sequence, Optional, Union

import cv2
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from argparse import Namespace

from defect_detection.outputs import AnomalyCLIPOutput
from . import AnomalyCLIP_lib
from .utils import get_transform, setup_seed
from .prompt_ensemble import AnomalyCLIP_PromptLearner


ImageNP = np.ndarray  # (H, W, 3)
BatchImageNP = Sequence[ImageNP]
AnomalyMap = np.ndarray  # (B, H, W)
ScoreArray = np.ndarray  # (B,)


class AnomalyCLIPInference:
    def __init__(
        self,
        checkpoint_path: str,
        features_list: Optional[List[int]] = None,
        imgsz: int = 518,
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        feature_map_layer: Tuple[int, ...] = (0, 1, 2, 3),
        sigma: int = 4,
        device: Optional[str] = None,
        DPAM_layer: int = 20,
        score_threshold: float = 0.25,
        area_threshold: int = 300,
        name: str = "anomalyclip",
    ) -> None:
        setup_seed(10)

        self.checkpoint_path = checkpoint_path
        self.features_list = features_list if features_list is not None else [24]
        self.imgsz = imgsz
        self.depth = depth
        self.n_ctx = n_ctx
        self.t_n_ctx = t_n_ctx
        self.feature_map_layer = feature_map_layer
        self.sigma = sigma
        self.DPAM_layer = DPAM_layer
        self.score_threshold = score_threshold
        self.area_threshold = area_threshold
        self.name = name

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.prompt_learner: Optional[AnomalyCLIP_PromptLearner] = None
        self.text_features: Optional[torch.Tensor] = None

        self._load_model_and_prompt_learner()
        self._build_text_features()
        self.capture_intermediates = False    # 기본 off — 메모리·시간 오버헤드 없음
        self._last_intermediates = None       # infer() 호출마다 per-image dict 리스트로 채움
        self._warmup()

    # ------------------------------------------------
    # Model & Prompt
    # ------------------------------------------------
    def _warmup(
        self,
        batch_size: int = 1,
    ) -> None:
        for _ in tqdm.tqdm(range(10), desc="Warm up AnomalyCLIP model"):
            dummy_images = [
                np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                for _ in range(batch_size)
            ]
            _ = self.infer(dummy_images, foreground_masks=None)


    def _load_model_and_prompt_learner(self) -> None:
        params = {
            "Prompt_length": self.n_ctx,
            "learnabel_text_embedding_depth": self.depth,
            "learnabel_text_embedding_length": self.t_n_ctx,
        }

        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px",
            device=self.device,
            design_details=params,
            download_root="./checkpoints/defect/anomalyclip",
        )
        model.eval()

        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])

        prompt_learner.to(self.device)
        model.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=self.DPAM_layer)

        self.model = model
        self.prompt_learner = prompt_learner

    def _build_text_features(self) -> None:
        prompts, tokenized_prompts, compound_prompts_text = self.prompt_learner(
            cls_id=None
        )

        text_features: torch.Tensor = self.model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()

        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2), dim=1
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features

    # ------------------------------------------------
    # Image
    # ------------------------------------------------
    def _load_and_preprocess_images(
        self,
        imgs_np: Union[ImageNP, BatchImageNP],
    ) -> torch.Tensor:
        if isinstance(imgs_np, np.ndarray):

            # single image (H, W, 3)
            if imgs_np.ndim == 3:
                imgs_np = [imgs_np]

            # batch image (B, H, W, 3)
            elif imgs_np.ndim == 4:
                imgs_np = [imgs_np[i] for i in range(imgs_np.shape[0])]

            else:
                raise ValueError("Unsupported image shape")

        elif not isinstance(imgs_np, (list, tuple)):
            raise TypeError("imgs_np must be an image or a sequence of images")

        preprocess, _ = get_transform(Namespace(image_size=self.imgsz))

        tensors: List[torch.Tensor] = []
        for img_np in imgs_np:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_np)
            tensors.append(preprocess(img_pil))

        return torch.stack(tensors).to(self.device)

    # ------------------------------------------------
    # Anomaly Map
    # ------------------------------------------------
    def _compute_anomaly_maps(
        self,
        patch_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        anomaly_maps: List[torch.Tensor] = []

        for idx, patch_feature in enumerate(patch_features):
            if idx < self.feature_map_layer[0]:
                continue

            patch_feature = patch_feature / patch_feature.norm(
                dim=-1, keepdim=True
            )

            similarity, _ = AnomalyCLIP_lib.compute_similarity(
                patch_feature, self.text_features[0]
            )

            similarity_map = AnomalyCLIP_lib.get_similarity_map(
                similarity[:, 1:, :], self.imgsz
            )

            anomaly_map = (
                similarity_map[..., 1] + 1 - similarity_map[..., 0]
            ) / 2.0

            anomaly_maps.append(anomaly_map)

        anomaly_maps = torch.stack(anomaly_maps).sum(dim=0)
        anomaly_maps = gaussian_filter(
            anomaly_maps.detach().cpu().numpy(),
            sigma=(0, self.sigma, self.sigma),
        )

        return torch.from_numpy(anomaly_maps).to(self.device)

    def _compute_image_scores(
        self,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        text_features: torch.Tensor = self.text_features.squeeze(0)  # (2, D)

        logits: torch.Tensor = image_features @ text_features.T  # (B, 2)
        probs: torch.Tensor = (logits / 0.07).softmax(dim=-1)

        return probs[:, 1].detach()

    def _resize_to_original(
        self,
        anomaly_maps: torch.Tensor,
        original_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Resize anomaly maps (B, H, W) → original image sizes
        """
        resized_maps = []

        for i, (h, w) in enumerate(original_sizes):
            resized = F.interpolate(
                anomaly_maps[i].unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            resized_maps.append(resized.squeeze())

        return torch.stack(resized_maps)

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------
    def infer(
        self,
        imgs_np: Union[ImageNP, BatchImageNP],
        foreground_masks: Optional[np.ndarray] = None,
    ) -> AnomalyCLIPOutput:
        if isinstance(imgs_np, np.ndarray) and imgs_np.ndim == 3:
            original_sizes = [imgs_np.shape[:2]]
        else:
            original_sizes = [img.shape[:2] for img in imgs_np]

        imgs_tensor = self._load_and_preprocess_images(imgs_np)

        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(
                imgs_tensor, self.features_list, DPAM_layer=self.DPAM_layer
            )

        image_abnormal_probs = self._compute_image_scores(image_features)

        anomaly_maps = self._compute_anomaly_maps(patch_features)
        resized_maps = self._resize_to_original(anomaly_maps, original_sizes)

        maps_np = resized_maps.cpu().numpy().astype(np.float32)
        if foreground_masks is not None:
            m = np.asarray(foreground_masks, dtype=np.float32)
            if m.ndim == 2:
                m = m[np.newaxis, :, :]
            if m.shape != maps_np.shape:
                raise ValueError(
                    f"foreground_masks shape {m.shape} != maps shape {maps_np.shape}"
                )
            maps_np = maps_np * m

        out = AnomalyCLIPOutput(
            maps=maps_np,
            score_threshold=self.score_threshold,
            area_threshold=self.area_threshold,
            source=self.name,
        )
        probs_list = [float(x) for x in image_abnormal_probs.cpu().numpy().reshape(-1)]
        object.__setattr__(out, "global_scores", probs_list)

        if self.capture_intermediates:
            pf_np = [pf.detach().cpu().numpy() for pf in patch_features]   # list of (B, N, D)
            imf_np = image_features.detach().cpu().numpy()                  # (B, D)
            amap_pre_np = anomaly_maps.detach().cpu().numpy()               # (B, H_in, W_in) pre-resize
            batch_sz = imf_np.shape[0]
            self._last_intermediates = [
                {
                    "patch_features": [pf[i] for pf in pf_np],
                    "image_features": imf_np[i],
                    "anomaly_map_pre_resize": amap_pre_np[i],
                }
                for i in range(batch_sz)
            ]
        else:
            self._last_intermediates = None
        return out

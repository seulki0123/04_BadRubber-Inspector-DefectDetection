"""PatchCore 메모리뱅크 기반 전역 이상점수 검출기.

`참고/01_notebooks/infer_nbr.py` 의 추론 로직을 그대로 옮긴 것이다.
학습된 PatchCore 메모리뱅크(bank_*.pt)로 원본 이미지 1장의 전역 이상점수(raw)를
계산하고, z-score = (raw - mu) / sd 가 `score_threshold`(= infer_nbr.py 의 `--z`)
이상이면 "비정상적으로 높은" 것으로 보고 'etc' 클래스로 검출한다.

    입력 : 원본 이미지 (List[np.ndarray], BGR)
    출력 : PatchcoreOutput  (anomaly/segmentation 과 섞지 않는 독립 출력)
           점수가 임계 이상인 이미지마다
             class_id=9999, class_name='etc', confidence=raw score,
             bboxes_xyxy=이미지 가운데 고정 박스
"""
import json
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F

from defect_detection.outputs import (
    Patchcore,
    PatchcoreOutput,
    ETC_CLASS_ID,
    ETC_CLASS_NAME,
)


# infer_nbr.py 와 동일한 ImageNet 정규화 상수
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PatchcoreDetector:
    NN_CHUNK = 2048

    def __init__(
        self,
        checkpoint_path: str,
        holdout_path: str,
        score_threshold: Optional[float],
        imgsz: int,
        device: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"patchcore: 모델(메모리뱅크) 파일 없음: {checkpoint_path}")
        if not os.path.exists(holdout_path):
            raise FileNotFoundError(f"patchcore: holdout(합격선) 파일 없음: {holdout_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # score_threshold == infer_nbr.py 의 args.z (z-score 합격선)
        # None 이면 holdout json 의 p99(suggest_thr_p99) 를 합격선으로 사용
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.color = color

        try:
            m = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # 구버전 torch (weights_only 인자 없음)
            m = torch.load(checkpoint_path, map_location="cpu")

        with open(holdout_path, encoding="utf-8") as f:
            hold = json.load(f)
        self.mu = float(hold["score_mean"])
        sd = float(hold["score_std"])
        self.sd = sd if sd > 1e-9 else 1.0
        # --z, --threshold 둘 다 미지정 시 fallback (infer_nbr.py 와 동일)
        self.thr_p99 = float(hold.get("suggest_thr_p99", self.mu + 3 * self.sd))

        self.btype = m["btype"]
        self.imgsz = int(imgsz)
        self._build_backbone(m)
        self.bank = m["memory_bank"].to(torch.float32).to(self.device)

        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

    # ------------------------------------------------------------------
    # 모델 로딩 / 임베딩 (infer_nbr.py load_model/embed 와 동일)
    # ------------------------------------------------------------------
    def _build_backbone(self, m: dict) -> None:
        if self.btype == "cnn":
            self.backbone = (
                timm.create_model(
                    m["backbone"],
                    pretrained=True,
                    features_only=True,
                    out_indices=tuple(m["layers"]),
                )
                .eval()
                .to(self.device)
            )
            self.npref = None
        else:
            self.backbone = (
                timm.create_model(
                    m["backbone"],
                    pretrained=True,
                    img_size=self.imgsz,
                    dynamic_img_size=True,
                )
                .eval()
                .to(self.device)
            )
            self.npref = int(self.backbone.num_prefix_tokens)

    @torch.no_grad()
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.btype == "cnn":
            fs = self.backbone(x)
            fs = [F.avg_pool2d(f, 3, 1, 1) for f in fs]
            ref = fs[0].shape[-2:]
            fs = [
                F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
                for f in fs
            ]
            f = torch.cat(fs, 1)
            B, C, H, W = f.shape
            return f.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t = self.backbone.forward_features(x)
        t = t[:, self.npref:, :]
        B, N, C = t.shape
        H = W = int(round(N ** 0.5))
        g = t.reshape(B, H, W, C).permute(0, 3, 1, 2)
        g = F.avg_pool2d(g, 3, 1, 1)
        return g.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def _patch_min(self, p: torch.Tensor) -> torch.Tensor:
        p = p.to(torch.float32)
        out = []
        for s in range(0, p.shape[0], self.NN_CHUNK):
            out.append(torch.cdist(p[s:s + self.NN_CHUNK], self.bank).min(1).values)
        return torch.cat(out)

    def _reduce_score(
        self,
        d: torch.Tensor,
        mask: Optional[np.ndarray],
    ) -> float:
        """패치별 최소거리 d(=(P,)) 중 최대값을 이미지 점수로 반환.

        foreground mask 가 주어지면 배경 패치는 제외하고 foreground 패치들에
        대해서만 max 를 취해 배경 score 가 섞이지 않게 한다.
        """
        if mask is None:
            return float(d.max())

        p = int(d.shape[0])
        g = int(round(p ** 0.5))
        if g * g != p:
            # 패치 그리드(정사각형) 복원 불가 → 마스킹 생략
            return float(d.max())

        # 원본 해상도 mask → 패치 그리드(g x g) 로 축소 (NEAREST 로 0/1 유지)
        m = cv2.resize(
            mask.astype(np.float32), (g, g), interpolation=cv2.INTER_NEAREST
        )
        fg = torch.from_numpy(m > 0.5).reshape(-1).to(d.device)
        if not bool(fg.any()):
            # foreground 패치가 하나도 없으면 전체 패치로 fallback
            return float(d.max())
        return float(d[fg].max())

    @torch.no_grad()
    def _raw_scores(
        self,
        images: Sequence[np.ndarray],
        foreground_masks: Optional[Sequence[np.ndarray]] = None,
    ) -> List[float]:
        tens = []
        for img in images:
            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(im).float().permute(2, 0, 1) / 255.0
            tens.append(t)
        xb = torch.stack(tens).to(self.device)
        xb = (xb - self._mean) / self._std
        pb = self._embed(xb)
        return [
            self._reduce_score(
                self._patch_min(pb[j]),
                None if foreground_masks is None else foreground_masks[j],
            )
            for j in range(pb.shape[0])
        ]

    @staticmethod
    def _center_bbox(
        img: np.ndarray,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
        H, W = img.shape[:2]
        xyxy_n = (0.1, 0.05, 0.9, 0.95)
        xyxy = (
            int(xyxy_n[0] * W),
            int(xyxy_n[1] * H),
            int(xyxy_n[2] * W),
            int(xyxy_n[3] * H),
        )
        return xyxy, xyxy_n

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def infer(
        self,
        images: Sequence[np.ndarray],
        foreground_masks: Optional[Sequence[np.ndarray]] = None,
        active_by_side: Optional[Sequence[Optional[bool]]] = None,
    ) -> PatchcoreOutput:
        # 합격선 결정 (infer_nbr.py 와 동일)
        #   score_threshold(=args.z) 지정 시 → raw 환산: thr = mu + z * sd
        #   미지정(None) 시            → json 의 p99(suggest_thr_p99) fallback
        if self.score_threshold is not None:
            thr = self.mu + self.score_threshold * self.sd
        else:
            thr = self.thr_p99

        # side 별 활성화 여부 결정 (None=지정 안 됨 → 활성)
        def _is_active(i: int) -> bool:
            if active_by_side is None or active_by_side[i] is None:
                return True
            return bool(active_by_side[i])

        active_idx = [i for i in range(len(images)) if _is_active(i)]

        # 활성화된 이미지만 모델에 통과시켜 점수 계산 (시간 이득)
        scores = [0.0] * len(images)
        if active_idx:
            active_images = [images[i] for i in active_idx]
            active_masks = (
                None if foreground_masks is None
                else [foreground_masks[i] for i in active_idx]
            )
            active_scores = self._raw_scores(active_images, active_masks)
            for i, s in zip(active_idx, active_scores):
                scores[i] = float(s)

        batch: List[List[Patchcore]] = []
        for idx, img in enumerate(images):
            regions: List[Patchcore] = []
            raw = scores[idx]
            if _is_active(idx) and raw >= thr:  # 활성화된 side 이면서 점수가 비정상적으로 높음 → 'etc'
                xyxy, xyxy_n = self._center_bbox(img)
                regions.append(
                    Patchcore(
                        class_id=ETC_CLASS_ID,
                        class_name=ETC_CLASS_NAME,
                        confidence=float(raw),
                        bboxes_xyxy=xyxy,
                        bboxes_xyxy_n=xyxy_n,
                        is_pass=False,
                        color=self.color,
                    )
                )
            batch.append(regions)

        return PatchcoreOutput(batch=batch, scores=[float(s) for s in scores])

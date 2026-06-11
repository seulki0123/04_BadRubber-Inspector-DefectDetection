"""PatchCore 전역 이상점수 검출 결과.

`참고/01_notebooks/infer_nbr.py` 처럼 원본 이미지 1장의 전역 이상점수(raw)를
계산해, 점수가 임계 이상이면 'etc' 로 판정한 결과를 담는다.

다른 파이프라인(anomaly/segmentation)과 섞지 않고 독립된 출력으로 유지한다.

    PatchcoreOutput     : 배치 전체  (List[PatchcoreBatchItem])
    PatchcoreBatchItem  : 이미지 1장 (raw score + Patchcore 0~1개)
    Patchcore           : 검출 1건   (class_id / class_name / confidence / bboxes_xyxy)
"""
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple


# 점수가 임계 이상일 때 부여하는 'etc' 검출 식별값
ETC_CLASS_ID = 9999
ETC_CLASS_NAME = "etc"


# ---------------------------------
# Single detection
# ---------------------------------

@dataclass(frozen=True)
class Patchcore:
    class_id: int
    class_name: str
    confidence: float
    bboxes_xyxy: Tuple[int, int, int, int]
    bboxes_xyxy_n: Tuple[float, float, float, float]
    is_pass: bool = False
    color: Tuple[int, int, int] = (0, 0, 255)


# ---------------------------------
# Per-image batch item
# ---------------------------------

@dataclass(frozen=True)
class PatchcoreBatchItem:
    regions: List[Patchcore]          # 0개(정상) 또는 1개('etc')
    score: float                      # 이미지 전역 raw 이상점수

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self) -> Iterator[Patchcore]:
        return iter(self.regions)


# ---------------------------------
# Batch output
# ---------------------------------

@dataclass
class PatchcoreOutput:
    batch: List[List[Patchcore]]      # [B][0..1]
    scores: List[float]               # [B]

    def __post_init__(self):
        if len(self.batch) != len(self.scores):
            raise ValueError("PatchcoreOutput: batch/scores length mismatch")

    def __len__(self) -> int:
        return len(self.batch)

    def __iter__(self) -> Iterator[PatchcoreBatchItem]:
        for regions, score in zip(self.batch, self.scores):
            yield PatchcoreBatchItem(regions, score)

    def __getitem__(self, idx: int) -> PatchcoreBatchItem:
        return PatchcoreBatchItem(self.batch[idx], self.scores[idx])

import os
import glob
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
import numpy as np
from defect_detection.outputs import Classification
from .dinov2_embed import get_embedding, get_embeddings_batch

class Cluster:
    def __init__(
        self,
        checkpoints_path: str,
        threshold: float,
        classes: Dict[str, Dict[str, Any]],
    ):
        (
            self.embeddings,
            self.labels,
            self.paths,
        ) = self._load_database(db_path=checkpoints_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for cluster")
        self.device = device
        self.embeddings = self.embeddings.to(self.device).half()
        self.threshold = threshold
        self.classes = classes
        self._warmup()
        
    def _warmup(self):
        dummy = [np.zeros((224,224,3), dtype=np.uint8)] * 8

        with torch.inference_mode():
            queries = get_embeddings_batch(dummy).to(self.device)

            if self.device.type == "cuda":
                queries = queries.half()

            _ = torch.matmul(queries, self.embeddings.T)

    def infer_patches(
        self,
        patches: Sequence[np.ndarray],
    ) -> List[Classification]:

        if len(patches) == 0:
            return []

        results = self._infer(patches)

        outputs: List[Classification] = []

        for r in results:
            outputs.append(
                Classification(
                    class_id=r["class_id"],
                    confidence=r["confidence"],
                    class_name=r["class_name"],
                    is_pass=r["is_pass"],
                    color=r["color"],
                )
            )
        return outputs

    def _infer(self, images, topk=5, sim_threshold=0.35):
        outputs = []

        with torch.inference_mode():
            queries = get_embeddings_batch(images)
            queries = queries.to(self.device, non_blocking=True)

            # FP16 맞추기 (GPU일 때)
            if self.device.type == "cuda":
                queries = queries.half()

            sims = torch.matmul(queries, self.embeddings.T)

            max_sims, _ = sims.max(dim=1)
            top_vals, top_idx = torch.topk(sims, topk, dim=1)

        max_sims = max_sims.cpu().numpy()
        top_idx = top_idx.cpu().numpy()
        top_vals = top_vals.cpu().numpy()

        for i, image in enumerate(images):
            h, w = image.shape[:2]

            if max_sims[i] < sim_threshold:
                pred_score = 0.0
                class_id = -1
                class_name = "unknown"
                color = (0, 0, 255)
                is_pass = False
            else:
                idxs = top_idx[i]
                scores = top_vals[i]

                labels = [self.labels[j] for j in idxs]
                pred, pred_score = self._weighted_vote(labels, scores)

                class_information = self.classes[pred]
                class_id = class_information["class_id"]
                class_name = class_information["name"]
                color = class_information["color"]
                is_pass = class_information["pass"] or pred_score < self.threshold
            
            outputs.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(pred_score),
                "is_pass": is_pass,
                "color": color,
            })

        return outputs

    def _infer_one(self, image, topk=5, sim_threshold=0.35):
        import time
        t0 = time.time()

        query = get_embedding(image)
        query = query.to(self.device).half()

        sims = torch.matmul(self.embeddings, query)

        max_sim = sims.max().item()

        if max_sim < sim_threshold:
            return "unknown_under_max_sim", [], [], [], 0.0

        top_vals, top_idx = torch.topk(sims, topk)
        top_idx = top_idx.cpu().tolist()
        top_labels = [self.labels[i] for i in top_idx]
        top_paths = [self.paths[i] for i in top_idx]
        top_scores = top_vals.cpu().tolist()

        pred, pred_score = self._weighted_vote(top_labels, top_scores)
        t1 = time.time()
        print(f"inference time: {(t1-t0)*1000}ms")
        return pred, top_labels, top_paths, top_scores, pred_score

    def _weighted_vote(self, top_labels, top_scores):
        return top_labels[0], top_scores[0]

    def _load_database(self, db_path):

        db = torch.load(db_path, map_location="cpu")

        embeddings = db["embeddings"]
        labels = db["labels"]
        paths = db["paths"]

        return embeddings, labels, paths
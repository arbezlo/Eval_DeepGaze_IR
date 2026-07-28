from __future__ import annotations

from typing import Optional, Dict, Callable

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from helpers.models import RoiBox
import numpy as np

class Metrics:
    def __init__(self, epsilon: float = 1e-12) -> None:
        self.epsilon = epsilon
        self.registry: Dict[str, Dict[str, Callable]] = {
            "similarity": {
                "cc": self.pearson_correlation,
                "sim": self.sim,
                "kl": self.kl_div_oriented,
            },
            "predictivity": {
                "nss": self.nss,
                "auc_judd": self.auc_judd,
                "ig": self.information_gain,
            },
            # "roi": {
            #     "mass": self.roi_mass,
            #     "density": self.roi_density,
            #     "ratio": self.roi_ratio,
            #     "normalized_ratio": self.roi_normalized_ratio,
            # }
        }

    # ------------------------------------------------------------------
    # HELPER
    # ------------------------------------------------------------------

    def _as_binary(self, arr: np.ndarray) -> np.ndarray:
        return (arr > 0).astype(np.uint8)

    def _as_density(self, arr: np.ndarray) -> np.ndarray:
        arr = np.maximum(arr, 0.0).astype(np.float64)
        total = arr.sum()
        if total <= self.epsilon:
            arr = np.ones_like(arr, dtype=np.float64)
            total = arr.sum()
        return arr / (total + self.epsilon)

    # ------------------------------------------------------------------
    # SIMILARITY METRICS
    # ------------------------------------------------------------------

    def pearson_correlation(self, map_a: np.ndarray, map_b: np.ndarray) -> float:
        a, b = map_a.ravel(), map_b.ravel()
        if np.std(a) < self.epsilon or np.std(b) < self.epsilon:
            return float("nan")
        return float(pearsonr(a, b).statistic)

    def sim(self, map_a: np.ndarray, map_b: np.ndarray) -> float:
        return float(np.minimum(self._as_density(map_a), self._as_density(map_b)).sum())

    def kl_div(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        p, q = self._as_density(reference), self._as_density(prediction)
        return float(np.sum(p * np.log((p + self.epsilon) / (q + self.epsilon))))
    
    def kl_div_oriented(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        p, q = self._as_density(reference), self._as_density(prediction)
        return -float(np.sum(p * np.log((p + self.epsilon) / (q + self.epsilon))))


    # ------------------------------------------------------------------
    # PREDICTIVITY METRICS
    # ------------------------------------------------------------------

    def nss(self, fixation_map: np.ndarray, saliency_map: np.ndarray) -> float:
        fixation_binary = self._as_binary(fixation_map)
        if fixation_binary.sum() == 0:
            return float("nan")
        s = saliency_map.astype(np.float64)
        s = (s - s.mean()) / (s.std() + self.epsilon)
        return float(s[fixation_binary > 0].mean())
    

    def auc_judd(self, fixation_map: np.ndarray, saliency_map: np.ndarray) -> float:
        fixation_binary = self._as_binary(fixation_map).ravel()
        if fixation_binary.sum() == 0:
            return float("nan")
        sal = saliency_map.ravel()
        return float(roc_auc_score(fixation_binary, sal))
    

    def log_likelihood(self, fixation_map: np.ndarray, saliency_map: np.ndarray) -> float:
        fixation_counts = fixation_map.astype(np.float64)
        n_fix = fixation_counts.sum()
        if n_fix <= self.epsilon:
            return float("nan")
        p = self._as_density(saliency_map)
        return float(np.sum(fixation_counts * np.log(p + self.epsilon)) / n_fix)

    def information_gain(self, fixation_map: np.ndarray, saliency_map: np.ndarray, baseline: Optional[np.ndarray] = None) -> float:
        """Log-likelihood gain over a baseline (uniform if None). Units: nats/fixation."""
        model_ll = self.log_likelihood(fixation_map, saliency_map)
        if np.isnan(model_ll):
            return float("nan")
        if baseline is None:
            baseline = np.ones_like(saliency_map, dtype=np.float64)
        return float(model_ll - self.log_likelihood(fixation_map, baseline))

    # ------------------------------------------------------------------
    # ROI METRICS
    # ------------------------------------------------------------------

    

    def _roi_mask(self, rois: list[RoiBox], saliency_map: np.ndarray) -> np.ndarray:
        h, w = saliency_map.shape
        mask = np.zeros((h, w), dtype=bool)

        for roi in rois:
            x_min = max(0, int(roi.x_min))
            y_min = max(0, int(roi.y_min))
            x_max = min(w, int(roi.x_max))
            y_max = min(h, int(roi.y_max))

            if x_max <= x_min or y_max <= y_min:
                continue

            mask[y_min:y_max, x_min:x_max] = True

        return mask


    def roi_mass(self, saliency_map: np.ndarray, rois: list[RoiBox]) -> float:
        mask = self._roi_mask(rois, saliency_map)
        return float(np.sum(saliency_map[mask], dtype=np.float64))


    def roi_ratio(self, saliency_map: np.ndarray, rois: list[RoiBox]) -> float:
        total_mass = float(np.sum(saliency_map, dtype=np.float64))
        if total_mass <= self.epsilon:
            return float("nan")

        return float(self.roi_mass(saliency_map, rois) / (total_mass + self.epsilon))


    def roi_density(self, saliency_map: np.ndarray, rois: list[RoiBox]) -> float:
        mask = self._roi_mask(rois, saliency_map)
        area = int(mask.sum())

        if area <= 0:
            return float("nan")

        return float(self.roi_mass(saliency_map, rois) / (area + self.epsilon))


    def roi_normalized_ratio(self, saliency_map: np.ndarray, rois: list[RoiBox]) -> float:
        mask = self._roi_mask(rois, saliency_map)

        roi_area = int(mask.sum())
        image_area = int(saliency_map.size)
        total_mass = float(np.sum(saliency_map, dtype=np.float64))

        if roi_area <= 0 or image_area <= 0 or total_mass <= self.epsilon:
            return float("nan")

        roi_mass_ratio = self.roi_mass(saliency_map, rois) / (total_mass + self.epsilon)
        roi_area_ratio = roi_area / (image_area + self.epsilon)

        return float(roi_mass_ratio / (roi_area_ratio + self.epsilon))
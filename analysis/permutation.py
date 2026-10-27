"""
Fast split-half permutation test of Comparator.run_expe1.

Same metrics as analysis/metrics.py, computed faster:
- The Gaussian blur is linear, so the saliency map of a group of participants is the sum of
  their blurred maps. Each participant is blurred once per stem, not 16 times per permutation.
- What each metric needs from a map (density, log-density, z-scores, sorted values) is
  computed once per group, then reused for every pair the group is part of.
- AUC-Judd uses the rank formula (Mann-Whitney U, ties averaged), which gives the same value
  as sklearn's roc_auc_score.
- The functions are module-level so stems can run in parallel processes.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Order of the metrics in the score vectors. Must match Metrics.registry.
METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("similarity", "cc"),
    ("similarity", "sim"),
    ("similarity", "kl"),
    ("predictivity", "nss"),
    ("predictivity", "auc_judd"),
    ("predictivity", "ig"),
)

# Sparse participant fixation map: (flat pixel indices, fixation counts).
SparseMap = tuple[np.ndarray, np.ndarray]


def to_sparse(fixation_map: np.ndarray) -> SparseMap:
    flat = fixation_map.ravel()
    idx = np.flatnonzero(flat)
    return idx, flat[idx].astype(np.float64)


class _Group:
    """One group of participants: its fixations and the quantities used by the metrics."""

    def __init__(self, sparse_maps, blurred, members, eps):
        n_pixels = blurred.shape[1]

        # Fixation map of the group (sparse).
        idx = np.concatenate([sparse_maps[m][0] for m in members])
        counts = np.concatenate([sparse_maps[m][1] for m in members])
        self.fix_idx, inverse = np.unique(idx, return_inverse=True)
        self.fix_counts = np.bincount(inverse, weights=counts)
        self.n_fix = float(self.fix_counts.sum())

        # Saliency map = blur(sum of fixation maps) = sum of blurred maps, normalised to 1.
        s = blurred[members[0]].astype(np.float64)
        for m in members[1:]:
            s += blurred[m]
        total = s.sum()
        if total > 0:
            s /= total
        self.s = s

        # nss / cc
        self.mean = s.mean()
        self.std = s.std()
        if self.std < eps:
            self.u = None  # Pearson undefined
        else:
            u = s - self.mean
            u /= np.sqrt(u @ u)
            self.u = u

        # sim / kl / ig: density as in Metrics._as_density
        p = np.maximum(s, 0.0)
        p_total = p.sum()
        if p_total <= eps:
            p = np.ones(n_pixels, dtype=np.float64)
            p_total = float(n_pixels)
        p /= p_total + eps
        self.p = p
        self.logp = np.log(p + eps)
        self.p_logp = float(p @ self.logp)

        # auc_judd, in float32 like Builder.build_saliency_map (same ties as the original)
        self.s32 = s.astype(np.float32)
        self.sorted_s32 = np.sort(self.s32)


def _nanmean2(a: float, b: float) -> float:
    if np.isnan(a):
        return b
    if np.isnan(b):
        return a
    return 0.5 * (a + b)


def _nss(fix: _Group, sal: _Group, eps: float) -> float:
    if fix.fix_idx.size == 0:
        return float("nan")
    return float((sal.s[fix.fix_idx].mean() - sal.mean) / (sal.std + eps))


def _auc_judd(fix: _Group, sal: _Group) -> float:
    k = fix.fix_idx.size
    n_pixels = sal.s.size
    if k == 0 or k == n_pixels:
        return float("nan")
    values = sal.s32[fix.fix_idx]
    below = np.searchsorted(sal.sorted_s32, values, side="left")
    below_or_equal = np.searchsorted(sal.sorted_s32, values, side="right")
    rank_sum = 0.5 * float(np.sum(below + below_or_equal + 1))  # average 1-based ranks
    u_stat = rank_sum - k * (k + 1) / 2.0
    return u_stat / (k * (n_pixels - k))


def _ig(fix: _Group, sal: _Group, eps: float) -> float:
    if fix.n_fix <= eps:
        return float("nan")
    n_pixels = sal.s.size
    log_likelihood = float(fix.fix_counts @ sal.logp[fix.fix_idx]) / fix.n_fix
    log_likelihood_uniform = np.log(1.0 / (n_pixels + eps) + eps)
    return log_likelihood - log_likelihood_uniform


def score_pair(a: _Group, b: _Group, eps: float) -> np.ndarray:
    """Same values as Comparator._compute_human_scores, in METRIC_KEYS order."""
    cc = float("nan") if a.u is None or b.u is None else float(np.clip(a.u @ b.u, -1.0, 1.0))
    sim = float(np.minimum(a.p, b.p).sum())
    kl_ab = -(a.p_logp - float(a.p @ b.logp))
    kl_ba = -(b.p_logp - float(b.p @ a.logp))
    return np.array([
        cc,
        sim,
        _nanmean2(kl_ab, kl_ba),
        _nanmean2(_nss(a, b, eps), _nss(b, a, eps)),
        _nanmean2(_auc_judd(a, b), _auc_judd(b, a)),
        _nanmean2(_ig(a, b, eps), _ig(b, a, eps)),
    ])


def _blur_all(sparse_maps: list[SparseMap], shape: tuple[int, int], sigma: float) -> np.ndarray:
    blurred = np.empty((len(sparse_maps), shape[0] * shape[1]), dtype=np.float32)
    dense = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for i, (idx, counts) in enumerate(sparse_maps):
        dense[:] = 0.0
        dense[idx] = counts
        blurred[i] = gaussian_filter(dense.reshape(shape), sigma=float(sigma)).ravel()
    return blurred


def permutation_scores(
    visible_maps: list[SparseMap],
    infrared_maps: list[SparseMap],
    shape: tuple[int, int],
    sigma: float,
    n_permutations: int,
    seed: int | None,
    eps: float,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Split-half permutation test of one stem.

    Each permutation splits the visible participants into two disjoint groups V1, V2 and the
    infrared ones into I1, I2 (n participants per group), then computes
        intra_VIS = score(V1, V2), intra_IR = score(I1, I2),
        inter     = mean of score(V1, I1), score(V2, I2), score(V1, I2), score(V2, I1),
        delta_VIS = intra_VIS - inter, delta_IR = intra_IR - inter.

    Returns {modality: {"intra": ..., "inter": ..., "delta": ...}} for modality in VIS, IR,
    each array of shape (n_permutations, n_metrics), columns in METRIC_KEYS order
    ("inter" is the same array for both modalities).
    """
    n_visible, n_infrared = len(visible_maps), len(infrared_maps)
    n = min(n_visible // 2, n_infrared // 2)
    if n == 0:
        raise ValueError(
            f"Need at least 2 participants per modality, got {n_visible} visible "
            f"and {n_infrared} infrared."
        )

    blurred_visible = _blur_all(visible_maps, shape, sigma)
    blurred_infrared = _blur_all(infrared_maps, shape, sigma)

    rng = np.random.default_rng(seed)
    size = (n_permutations, len(METRIC_KEYS))
    inter_all = np.empty(size)
    intra_all = {"VIS": np.empty(size), "IR": np.empty(size)}

    for k in range(n_permutations):
        # Same random draws, in the same order, as the original implementation.
        visible_indices = rng.permutation(n_visible)
        infrared_indices = rng.permutation(n_infrared)

        v1 = _Group(visible_maps, blurred_visible, visible_indices[:n], eps)
        v2 = _Group(visible_maps, blurred_visible, visible_indices[n:2 * n], eps)
        i1 = _Group(infrared_maps, blurred_infrared, infrared_indices[:n], eps)
        i2 = _Group(infrared_maps, blurred_infrared, infrared_indices[n:2 * n], eps)

        intra_all["VIS"][k] = score_pair(v1, v2, eps)
        intra_all["IR"][k] = score_pair(i1, i2, eps)
        inter_all[k] = 0.25 * (
            score_pair(v1, i1, eps)
            + score_pair(v2, i2, eps)
            + score_pair(v1, i2, eps)
            + score_pair(v2, i1, eps)
        )

    return {
        modality: {"intra": intra, "inter": inter_all, "delta": intra - inter_all}
        for modality, intra in intra_all.items()
    }

#!/usr/bin/env python3
"""
Gold-standard fixation density and explainable information gain for one image.

Follows Kümmerer & Bethge (2023), "Predicting Visual Fixations", Annu. Rev. Vis. Sci. 9:269-91,
Supplemental Material section 4 ("Building a Good Gold Standard Model for Spatial Fixation Density").

For a held-out subject s on image I, the gold standard is the mixture

    p_gold^(s)(x) = w_kde * KDE_sigma(x | fixations of all subjects != s on I)
                  + w_cb  * p_centerbias(x)     # KDE over fixations on *other* images
                  + w_u   * 1 / (W * H)         # uniform

with the bandwidth sigma and the weights chosen to maximise this leave-one-subject-out
log-likelihood. The explainable information gain is then

    IG_gold = mean_i [ log2 p_gold(x_i) - log2 p_centerbias(x_i) ]        (bit / fixation)

and a model p_hat explains IGE = IG(p_hat || centerbias) / IG_gold of it. For a dataset,
average the IGs per image and then over images, and take the ratio of those averages
(not the mean of per-image ratios).

Implementation notes
  * The KDEs are evaluated in closed form at the fixation locations rather than on a pixel
    grid. Kernels are truncated to the image and the sum is renormalised,
        KDE(x) = sum_j N(x; x_j, sigma^2 I) / sum_j mass_j,   mass_j = kernel mass inside the image,
    which is the continuous version of "blur the fixation histogram, then renormalise".
  * Densities are per pixel, so the uniform model is 1 / (W * H) and the log-likelihood
    relative to uniform is mean(log2 p) + log2(W * H).
  * The center bias lives in normalised coordinates ([0, 1]^2) so that images of
    different sizes can be pooled.
  * The paper fits sigma and the weights once for the whole dataset (pass a list of images
    to fit_gold_standard). Fitting them on a single image works, but the result is noisier
    and slightly optimistic, since the same fixations choose and score the hyperparameters.

Usage with your own data
    img = ImageFixations(x=fx, y=fy, subject=subj, width=W, height=H)
    # fixations from all OTHER images, divided by their own image's width / height
    cb = fit_centerbias(other_nx, other_ny, other_image_ids)
    gs = fit_gold_standard(img, cb)
    print(explainable_information_gain(img, gs, cb))
    print(model_information_gain(img, model_log_density, cb))   # optional: score a model

Run this file directly for a demo on synthetic data.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr

LN2 = np.log(2.0)
COMPONENTS = ("kde", "centerbias", "uniform")


# --------------------------------------------------------------------------- data

@dataclass
class ImageFixations:
    """All fixations recorded on one image.

    x, y          : fixation coordinates in pixels, 0-based; x = column, y = row
                    (pixel (r, c) covers [c, c+1) x [r, r+1)).
    subject       : subject identifier of every fixation.
    width, height : image size in pixels.
    """
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.subject = np.asarray(self.subject)
        if not len(self.x) == len(self.y) == len(self.subject):
            raise ValueError("x, y and subject must have the same length")
        inside = (self.x >= 0) & (self.x < self.width) & (self.y >= 0) & (self.y < self.height)
        if not inside.all():
            warnings.warn(f"dropping {np.sum(~inside)} fixations that fall outside the image")
            self.x, self.y, self.subject = self.x[inside], self.y[inside], self.subject[inside]
        if len(np.unique(self.subject)) < 2:
            raise ValueError("leave-one-subject-out needs fixations from at least two subjects")


# --------------------------------------------------------------- kernel helpers

def _kernel_mass(cx, cy, sigma_x, sigma_y, width, height):
    """Mass of axis-aligned Gaussian kernels centred at (cx, cy) inside [0, width) x [0, height)."""
    mx = ndtr((width - cx) / sigma_x) - ndtr(-cx / sigma_x)
    my = ndtr((height - cy) / sigma_y) - ndtr(-cy / sigma_y)
    return mx * my


def _logsumexp_rows(a):
    """Row-wise log-sum-exp of a 2-D array. Overwrites `a` (much faster than scipy's on big arrays)."""
    m = a.max(axis=1, keepdims=True)
    m[~np.isfinite(m)] = 0.0
    np.subtract(a, m, out=a)
    np.exp(a, out=a)
    with np.errstate(divide="ignore"):
        return np.log(a.sum(axis=1)) + m[:, 0]


def _log_kde(qx, qy, cx, cy, sigma_x, sigma_y, width, height, max_pairs=10_000_000):
    """Log of the truncated, renormalised Gaussian KDE with centres (cx, cy), at queries (qx, qy).

    The density is per unit area of whatever coordinate system the inputs use.
    """
    log_norm = np.log(2 * np.pi * sigma_x * sigma_y) + np.log(
        _kernel_mass(cx, cy, sigma_x, sigma_y, width, height).sum())
    out = np.empty(len(qx))
    step = max(1, max_pairs // max(len(cx), 1))
    for s in range(0, len(qx), step):
        d = np.subtract.outer(qx[s:s + step], cx)
        d /= sigma_x
        np.square(d, out=d)
        e = np.subtract.outer(qy[s:s + step], cy)
        e /= sigma_y
        np.square(e, out=e)
        d += e
        d *= -0.5
        out[s:s + step] = _logsumexp_rows(d)
    return out - log_norm


def _loo_log_kde(img: ImageFixations, sigma: float) -> np.ndarray:
    """Log KDE (per pixel) at every fixation, built only from the OTHER subjects' fixations."""
    x, y = img.x, img.y
    other = img.subject[:, None] != img.subject[None, :]
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    log_k = np.where(other, -0.5 * d2 / sigma ** 2, -np.inf)
    mass = _kernel_mass(x, y, sigma, sigma, img.width, img.height)
    return (_logsumexp_rows(log_k)
            - np.log(2 * np.pi * sigma ** 2)
            - np.log(other.astype(float) @ mass))


def _fit_weights(log_p, sample_weight=None, n_iter=5000, tol=1e-10):
    """Maximum-likelihood mixture weights for fixed component log-densities (EM).

    log_p : (n_fixations, n_components) natural-log component densities.
    Returns (weights, weighted mean log-likelihood in nats).
    """
    n, k = log_p.shape
    sw = np.full(n, 1.0 / n) if sample_weight is None else sample_weight / sample_weight.sum()
    if k == 1:
        return np.ones(1), float(sw @ log_p[:, 0])
    log_w = np.full(k, -np.log(k))
    prev = -np.inf
    for _ in range(n_iter):
        joint = log_p + log_w
        log_mix = _logsumexp_rows(joint.copy())
        ll = float(sw @ log_mix)
        if ll - prev < tol:
            break
        prev = ll
        w = sw @ np.exp(joint - log_mix[:, None])
        log_w = np.log(np.clip(w, 1e-300, None))
    return np.exp(log_w), ll


# ------------------------------------------------------------------ center bias

@dataclass
class CenterBias:
    """Image-independent prior: KDE over fixations on OTHER images, mixed with a uniform.

    nx, ny : fixation coordinates divided by their own image's width / height (in [0, 1)).
             Must not contain fixations from the image being evaluated.
    sigma  : kernel bandwidth in normalised units.
    eps    : weight of the uniform component.
    """
    nx: np.ndarray
    ny: np.ndarray
    sigma: float = 0.03
    eps: float = 0.01

    def log_prob(self, img: ImageFixations) -> np.ndarray:
        """Natural-log probability per pixel of every fixation of img."""
        log_p = _log_kde(img.x / img.width, img.y / img.height,
                         self.nx, self.ny, self.sigma, self.sigma, 1.0, 1.0)
        if self.eps > 0:
            log_p = np.logaddexp(np.log1p(-self.eps) + log_p, np.log(self.eps))
        return log_p - np.log(img.width * img.height)


def fit_centerbias(nx, ny, image_ids, n_folds=5, sigmas=None, max_queries=2000, seed=0) -> CenterBias:
    """Fit the center-bias bandwidth and uniform weight by K-fold cross-validation over images.

    nx, ny    : normalised fixation coordinates from the other images.
    image_ids : image of every fixation (folds are split by image, not by fixation).
    Scoring uses a random subset of max_queries fixations to keep it fast.
    """
    nx, ny, image_ids = (np.asarray(a) for a in (nx, ny, image_ids))
    rng = np.random.default_rng(seed)
    images = np.unique(image_ids)
    fold_of = dict(zip(images, rng.permutation(len(images)) % n_folds))
    fold = np.array([fold_of[i] for i in image_ids])
    query = rng.permutation(len(nx))[:max_queries]
    sigmas = np.geomspace(0.005, 0.3, 20) if sigmas is None else sigmas

    best_ll, best_sigma, best_eps = -np.inf, None, None
    for s in sigmas:
        parts = []
        for f in range(n_folds):
            q = query[fold[query] == f]
            if len(q):
                train = fold != f
                parts.append(_log_kde(nx[q], ny[q], nx[train], ny[train], s, s, 1.0, 1.0))
        log_kde = np.concatenate(parts)
        # uniform density on the unit square is 1, i.e. log density 0
        w, ll = _fit_weights(np.column_stack([log_kde, np.zeros_like(log_kde)]))
        if ll > best_ll:
            best_ll, best_sigma, best_eps = ll, float(s), float(w[1])
    if best_sigma in (sigmas[0], sigmas[-1]):
        warnings.warn("center-bias bandwidth hit the edge of the search grid")
    return CenterBias(nx, ny, sigma=best_sigma, eps=best_eps)


def _centerbias_log_map(cb: CenterBias, width, height):
    """Center bias on the pixel grid (histogram + blur approximation; used for plotting)."""
    hist = np.zeros((height, width))
    rows = np.minimum((cb.ny * height).astype(int), height - 1)
    cols = np.minimum((cb.nx * width).astype(int), width - 1)
    np.add.at(hist, (rows, cols), 1.0)
    k = gaussian_filter(hist, sigma=(cb.sigma * height, cb.sigma * width), mode="constant")
    k /= k.sum()
    with np.errstate(divide="ignore"):
        return np.log((1 - cb.eps) * k + cb.eps / (width * height))


# ---------------------------------------------------------------- gold standard

@dataclass
class GoldStandard:
    sigma: float    # KDE bandwidth in pixels
    weights: dict   # component name -> mixture weight


def _component_log_probs(img, sigma, components, cb_log_p):
    cols = []
    for c in components:
        if c == "kde":
            cols.append(_loo_log_kde(img, sigma))
        elif c == "centerbias":
            cols.append(cb_log_p)
        elif c == "uniform":
            cols.append(np.full(len(img.x), -np.log(img.width * img.height)))
        else:
            raise ValueError(f"unknown component {c!r}")
    return np.column_stack(cols)


def fit_gold_standard(images, centerbiases=None, components=COMPONENTS,
                      sigma=None, sigma_bounds=None) -> GoldStandard:
    """Fit the KDE bandwidth (pixels) and mixture weights by maximising the
    leave-one-subject-out log-likelihood (averaged per image, then over images).

    images       : one ImageFixations or a list of them (a whole dataset shares the parameters).
    centerbiases : one CenterBias per image, each built WITHOUT that image's fixations.
                   Needed when "centerbias" is a component.
    components   : any subset of ("kde", "centerbias", "uniform") that contains "kde".
    sigma        : fix the bandwidth instead of fitting it (e.g. 1 dva = pixels per degree).
    """
    images = [images] if isinstance(images, ImageFixations) else list(images)
    if centerbiases is not None:
        centerbiases = [centerbiases] if isinstance(centerbiases, CenterBias) else list(centerbiases)
    components = tuple(components)
    if "kde" not in components:
        raise ValueError("the gold standard needs the 'kde' component")
    if "centerbias" in components:
        if centerbiases is None or len(centerbiases) != len(images):
            raise ValueError("pass one CenterBias per image to use the 'centerbias' component")
        cb_log_ps = [cb.log_prob(img) for img, cb in zip(images, centerbiases)]
    else:
        cb_log_ps = [None] * len(images)
    # every image counts equally, every fixation equally within its image
    sw = np.concatenate([np.full(len(img.x), 1.0 / len(img.x)) for img in images]) / len(images)

    def evaluate(s):
        log_p = np.vstack([_component_log_probs(img, s, components, cbl)
                           for img, cbl in zip(images, cb_log_ps)])
        return _fit_weights(log_p, sw)

    if sigma is None:
        lo, hi = sigma_bounds or (1.0, max(max(i.width, i.height) for i in images) / 4)
        grid = np.geomspace(lo, hi, 24)
        lls = [evaluate(s)[1] for s in grid]
        i = int(np.argmax(lls))
        if i in (0, len(grid) - 1):
            warnings.warn("KDE bandwidth hit the edge of the search range; widen sigma_bounds")
        res = minimize_scalar(lambda ls: -evaluate(np.exp(ls))[1], method="bounded",
                              bounds=(np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])))
        sigma = float(np.exp(res.x)) if -res.fun > lls[i] else float(grid[i])
    weights, _ = evaluate(sigma)
    return GoldStandard(sigma=float(sigma), weights=dict(zip(components, weights)))


def gold_standard_log_prob(img: ImageFixations, gs: GoldStandard, centerbias: CenterBias | None = None):
    """Leave-one-subject-out natural-log probability (per pixel) of every fixation."""
    comps = tuple(gs.weights)
    cbl = centerbias.log_prob(img) if "centerbias" in comps else None
    log_p = _component_log_probs(img, gs.sigma, comps, cbl)
    log_w = np.log(np.clip([gs.weights[c] for c in comps], 1e-300, None))
    return logsumexp(log_p + log_w, axis=1)


def explainable_information_gain(img: ImageFixations, gs: GoldStandard, centerbias: CenterBias) -> dict:
    """Explainable IG of the image (bit/fix), plus log-likelihoods relative to uniform."""
    lp_gold = gold_standard_log_prob(img, gs, centerbias)
    lp_cb = centerbias.log_prob(img)
    log_n = np.log(img.width * img.height)
    return {
        "n_fixations": int(len(lp_gold)),
        "ll_gold_bits": float((lp_gold.mean() + log_n) / LN2),
        "ll_centerbias_bits": float((lp_cb.mean() + log_n) / LN2),
        "explainable_ig_bits": float((lp_gold - lp_cb).mean() / LN2),
    }


def model_information_gain(img: ImageFixations, model_log_density, centerbias: CenterBias) -> float:
    """IG (bit/fix) of a model relative to the center bias on this image.

    model_log_density : (height, width) array of natural-log probabilities per pixel
                        (exp of it should sum to 1). IGE = this / explainable IG.
    """
    ld = np.asarray(model_log_density, dtype=float)
    if ld.shape != (img.height, img.width):
        raise ValueError(f"model map has shape {ld.shape}, expected {(img.height, img.width)}")
    total = logsumexp(ld)
    if abs(total) > 1e-3:
        warnings.warn("model density does not sum to 1; renormalising it")
        ld = ld - total
    rows, cols = np.floor(img.y).astype(int), np.floor(img.x).astype(int)
    return float((ld[rows, cols] - centerbias.log_prob(img)).mean() / LN2)


def gold_standard_map(img: ImageFixations, gs: GoldStandard, centerbias: CenterBias | None = None,
                      leave_out=None):
    """Gold-standard natural-log probability per pixel over the whole image (e.g. for plotting).

    Uses all subjects' fixations, or all but subject `leave_out`. The center-bias component is
    approximated on the pixel grid by blurring a histogram.
    """
    H, W = img.height, img.width
    use = np.ones(len(img.x), bool) if leave_out is None else img.subject != leave_out
    qx, qy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    parts = []
    for c, w in gs.weights.items():
        if w <= 0:
            continue
        if c == "kde":
            lp = _log_kde(qx.ravel(), qy.ravel(), img.x[use], img.y[use],
                          gs.sigma, gs.sigma, W, H).reshape(H, W)
        elif c == "centerbias":
            lp = _centerbias_log_map(centerbias, W, H)
        else:
            lp = np.full((H, W), -np.log(W * H))
        parts.append(np.log(w) + lp)
    log_map = logsumexp(np.stack(parts), axis=0)
    return log_map - logsumexp(log_map)


# ------------------------------------------------------------------------- demo

def _demo():
    rng = np.random.default_rng(0)
    W, H = 1024, 768
    ppd = 35.0  # assumed pixels per degree of visual angle, for the "sigma = 1 dva" row

    # Fixations on 200 OTHER images (15 subjects x 8 fixations each), normalised coordinates.
    n = 200 * 15 * 8
    other_ids = np.repeat(np.arange(200), 15 * 8)
    nx, ny = rng.normal(0.5, 0.17, n), rng.normal(0.5, 0.20, n)
    rand = rng.random(n) < 0.1
    nx[rand], ny[rand] = rng.random(rand.sum()), rng.random(rand.sum())
    keep = (nx >= 0) & (nx < 1) & (ny >= 0) & (ny < 1)
    nx, ny, other_ids = nx[keep], ny[keep], other_ids[keep]

    # Target image: three salient blobs + central tendency + random fixations.
    comps = [(np.array([300., 250.]), np.array([30., 30.]), 0.35),
             (np.array([700., 300.]), np.array([45., 45.]), 0.20),
             (np.array([520., 560.]), np.array([25., 25.]), 0.15),
             (np.array([W / 2, H / 2]), np.array([0.17 * W, 0.20 * H]), 0.20)]
    p_uniform = 0.10
    probs = np.array([c[2] for c in comps] + [p_uniform])

    def sample_fixation():
        while True:
            k = rng.choice(len(probs), p=probs)
            p = rng.random(2) * [W, H] if k == len(comps) else rng.normal(comps[k][0], comps[k][1])
            if 0 <= p[0] < W and 0 <= p[1] < H:
                return p

    n_subj, n_fix = 15, 8
    pts = np.array([sample_fixation() for _ in range(n_subj * n_fix)])
    img = ImageFixations(pts[:, 0], pts[:, 1], np.repeat(np.arange(n_subj), n_fix), W, H)

    # The true generating density on the pixel grid, used as an "oracle" model below.
    gx, gy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dens = np.full((H, W), p_uniform / (W * H))
    for mu, sd, w in comps:
        dens += w * np.exp(-0.5 * (((gx - mu[0]) / sd[0]) ** 2 + ((gy - mu[1]) / sd[1]) ** 2)) \
            / (2 * np.pi * sd[0] * sd[1])
    log_true = np.log(dens / dens.sum())

    cb = fit_centerbias(nx, ny, other_ids)
    print(f"Center bias: sigma = {cb.sigma:.3f} (normalised units), uniform weight = {cb.eps:.3f}\n")

    rows = [("KDE, sigma = 1 dva", ("kde",), ppd),
            ("KDE, sigma cross-validated", ("kde",), None),
            ("KDE + uniform", ("kde", "uniform"), None),
            ("KDE + center bias", ("kde", "centerbias"), None),
            ("KDE + uniform + center bias", COMPONENTS, None)]
    print(f"{'Gold standard':30s} {'IG (bit/fix)':>12s} {'sigma (px)':>11s}   weights")
    for name, components, sigma in rows:
        gs = fit_gold_standard(img, cb, components=components, sigma=sigma)
        res = explainable_information_gain(img, gs, cb)
        weights = ", ".join(f"{k}={v:.2f}" for k, v in gs.weights.items())
        print(f"{name:30s} {res['explainable_ig_bits']:12.3f} {gs.sigma:11.1f}   {weights}")

    ig_gold = res["explainable_ig_bits"]
    ig_true = model_information_gain(img, log_true, cb)
    print(f"\nOracle model (the true generating density): IG = {ig_true:.3f} bit/fix, "
          f"IGE = {100 * ig_true / ig_gold:.0f}%")
    log_map = gold_standard_map(img, gs, cb)
    print(f"Gold-standard map: shape {log_map.shape}, total probability {np.exp(log_map).sum():.4f}")


if __name__ == "__main__":
    _demo()

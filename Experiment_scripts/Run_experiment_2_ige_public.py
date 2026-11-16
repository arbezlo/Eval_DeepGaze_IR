#!/usr/bin/env python3
"""Information gain explained (IGE) by DeepGaze on public RGB datasets, in colour or in grayscale.

Same experiment as Run_experiment_2_ige.py, on MIT1003, CAT2000 (training set) and COCO Freeview
(datasets of the MIT/Tübingen saliency benchmark) instead of the paired RGB / IR data. For each
dataset:

  1. the dataset is loaded with pysaliency (downloaded and converted on first use) and put in the
     format of the project: metadata.json (one numeric stem per image) and a fixation table in
     image pixel coordinates. The initial fixation of every scanpath (on the fixation cross) is
     dropped, as for the project's data (--keep-initial-fixation to keep it);
  2. the gold standard of analysis/gold_standard.py is fitted on all its images (KDE bandwidth and
     mixture weights, leave-one-image-out center bias), which gives the explainable information
     gain of every image (Comparator._compute_gold_standard_dataset);
  3. DeepGaze (IIE or MSDB) predicts every image with the gold standard's center bias of the image
     as prior, and its information gain explained is IGE = IG_model / IG_gold, exactly as in
     Comparator.run_deepgaze_ige. With --grayscale, the model gets the grayscale version of each
     image (cv2.COLOR_RGB2GRAY, repeated on the 3 channels) instead of the RGB image.

Outputs, under <output-dir>/<DATASET>/:
    metadata/                         metadata.json (stem -> image file), annotations.json (empty),
                                      fixations.csv (project format)
    gold_standard/                    Saver.save_gold_standard_results, modality = dataset name.
                                      Computed once, reloaded by later runs (--recompute-gold-standard)
    ige[_msdb[_<params>]][_gray]/     Saver.save_deepgaze_ige, modality "RGB" or "GRAY"

Datasets (pysaliency caches the converted datasets under <data-dir>/<name>, this script keeps the
downloaded archives under <downloads-dir> so that they are downloaded once):
    MIT1003        1003 images (longest side 1024 px), 15 subjects, 3 s free viewing (Judd et al., 2009)
    CAT2000        training set: 2000 images 1920 x 1080, 18 subjects per image, 5 s (Borji & Itti, 2015)
    COCO_FREEVIEW  COCO Freeview: 4317 images 1680 x 1050 with public fixations (train + validation
                   splits, see --coco-split), 10 subjects, free viewing (Chen et al., 2021; Yang et al., 2023)

The first conversion of MIT1003 runs its original MATLAB code, so pysaliency needs `matlab` or
`octave` on the PATH that first time only (e.g. `apt install octave`). The first conversion of
CAT2000 needs MATLAB or the mat-io package (`pip install mat-io`): its fixations are stored in a
MATLAB object that Octave cannot read. Afterwards, or when <data-dir>/<name> is copied from another
machine, neither is needed. COCO Freeview needs neither.

Usage:
    python Experiment_scripts/Run_experiment_2_ige_public.py                       # RGB, the 3 datasets
    python Experiment_scripts/Run_experiment_2_ige_public.py --grayscale
    python Experiment_scripts/Run_experiment_2_ige_public.py --datasets MIT1003 CAT2000 --model MSDB --msdb-dataset AUTO
    python Experiment_scripts/Run_experiment_2_ige_public.py --datasets COCO_FREEVIEW --max-images 3   # quick CPU test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from math import radians, tan
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display on the cluster.

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# pysaliency 0.2.x still uses aliases removed from numpy 2.4+ (dataset subsets, hdf5 caches).
for _name, _value in (("in1d", np.isin), ("string_", np.bytes_), ("trapz", getattr(np, "trapezoid", None)),
                      ("asfarray", lambda a, dtype=np.float64: np.asarray(a, dtype=dtype))):
    if not hasattr(np, _name) and _value is not None:
        setattr(np, _name, _value)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from deepgaze_pytorch import MSDBDataset

from analysis.comparator import Comparator
from analysis.gold_standard import COMPONENTS, LN2, CenterBias, _centerbias_log_map, model_information_gain
from analysis.saver import Saver
from saliency.deepgaze import DeepGaze


# Display geometry of each dataset, as the pixels per degree of visual angle (px/deg) of its images.
# It sets the naive model of the gold standard (KDE with sigma = 1 deg) and the input resolution of
# DeepGaze MSDB. The viewing distance and screen width (px) only serve to build an equivalent
# display for Comparator, whose _compute_sigma then gives exactly pixel_per_dva pixels per degree.
# Override per dataset with --pixel-per-dva.
DATASETS = {
    "MIT1003": {
        "cache_dir": "MIT1003",
        # Judd et al. (2009): 19-inch 1280 x 1024 screen viewed from about 2 feet (61 cm), images
        # shown at their native size: about 35 px/deg, the value used for this dataset by DeepGaze MSDB.
        "pixel_per_dva": 35.0,
        "viewing_distance_cm": 61.0,
        "screen_width_px": 1280,
        # pysaliency already discards the initial (central) fixation of every scanpath, as Judd et al.
        "initial_fixation_included": False,
    },
    "CAT2000": {
        "cache_dir": "CAT2000_train_v1.1",
        # Borji & Itti (2015): 42-inch 1920 x 1080 screen (93 cm wide) viewed from 106 cm:
        # 2 * 106 * tan(0.5 deg) * 1920 / 93 = 38 px/deg at the centre of the screen.
        "pixel_per_dva": 38.0,
        "viewing_distance_cm": 106.0,
        "screen_width_px": 1920,
        "initial_fixation_included": True,
    },
    "COCO_FREEVIEW": {
        "cache_dir": "COCO-Freeview",
        # Chen et al. (2021), display of COCO-Search18: 22-inch 1680 x 1050 screen viewed from 47 cm,
        # subtending 54 x 35 deg: 1680 px / 47.9 cm * 2 * 47 * tan(0.5 deg) = 29 px/deg at the centre.
        "pixel_per_dva": 29.0,
        "viewing_distance_cm": 47.0,
        "screen_width_px": 1680,
        # The scanpaths start on the fixation cross (median 22 px from the centre of the screen).
        "initial_fixation_included": True,
    },
}

MODALITY = "visible"

# Octave has no nanmedian / nanvar (MATLAB Statistics Toolbox functions used by the MIT1003
# fixation code in the rare trials without saccade); these stand-ins are written next to the
# scripts pysaliency runs. They only shadow the toolbox functions with the same result on vectors.
_OCTAVE_SHIMS = {
    "nanmedian.m": "function m = nanmedian(x)\n  m = median(x(~isnan(x)));\nend\n",
    "nanvar.m": "function v = nanvar(x)\n  v = var(x(~isnan(x)));\nend\n",
}


# ------------------------------------------------------------------ pysaliency

def _install_download_cache(downloads_dir: Path) -> None:
    """
    Make pysaliency's dataset loaders take their archives from downloads_dir when they are there,
    and keep the ones they download there (pysaliency itself downloads to a temporary directory
    and deletes everything afterwards). The md5 sums are checked as pysaliency does (a warning).
    """
    import pysaliency.utils as pysaliency_utils
    from pysaliency.external_datasets import cat2000, coco_freeview, mit

    original_download_and_check = pysaliency_utils.download_and_check

    def cached_download_and_check(url, target, md5_hash, verify_ssl=True):
        cached = downloads_dir / os.path.basename(url)
        if cached.exists():
            print(f"Using the downloaded {cached}")
            pysaliency_utils.check_file_hash(str(cached), md5_hash)
        else:
            downloads_dir.mkdir(parents=True, exist_ok=True)
            partial = cached.with_name(cached.name + ".part")
            print(f"Downloading {url} to {cached}")
            original_download_and_check(url, str(partial), md5_hash, verify_ssl=verify_ssl)
            partial.replace(cached)
        try:
            os.link(cached, target)  # same file system: no copy
        except OSError:
            shutil.copyfile(cached, target)

    for module in (mit, cat2000, coco_freeview):
        module.download_and_check = cached_download_and_check


def _install_octave_shims() -> None:
    """Write the Octave stand-ins of _OCTAVE_SHIMS where pysaliency runs the MIT1003 MATLAB code."""
    from pysaliency.external_datasets import mit
    from pysaliency.utils import MatlabOptions

    original_run_matlab_cmd = mit.run_matlab_cmd

    def run_matlab_cmd(cmd, cwd=None):
        uses_octave = not any(shutil.which(name) for name in MatlabOptions.matlab_names)
        if cwd is not None and uses_octave:
            for filename, source in _OCTAVE_SHIMS.items():
                path = Path(cwd) / filename
                if not path.exists():
                    path.write_text(source)
        return original_run_matlab_cmd(cmd, cwd=cwd)

    mit.run_matlab_cmd = run_matlab_cmd


def _has_matio() -> bool:
    try:
        import matio  # noqa: F401  (package mat-io)
    except ImportError:
        return False
    return True


def _extract_cat2000_fixations(directory: Path) -> None:
    """
    Python version of pysaliency's load_cat2000.m: split trainSet/allFixData.mat into
    extracted/fix<i>_<j>.mat (fields name, data) for scanpath j of the i-th image.

    allFixData.mat holds a MATLAB containers.Map (image -> 18 records), a MATLAB object that
    Octave and scipy cannot read; the mat-io package (`import matio`) decodes it.
    """
    import glob

    import matio
    from natsort import natsorted
    from scipy.io import savemat

    all_data = matio.load_from_mat(str(directory / "trainSet" / "allFixData.mat"))["allData"]
    keys = list(all_data.keys())
    # pysaliency numbers the images in this order (non-recursive "**", as in pysaliency).
    stimuli_dir = directory / "trainSet" / "Stimuli"
    stimuli = natsorted(os.path.relpath(f, stimuli_dir) for f in glob.glob(str(stimuli_dir / "**" / "*.jpg")))
    if keys != stimuli:
        raise ValueError("CAT2000: the images of allFixData.mat do not match the stimuli of trainSet.zip")

    out_dir = directory / "extracted"
    out_dir.mkdir(exist_ok=True)
    for i, key in enumerate(tqdm(keys, desc="Splitting allFixData.mat"), start=1):
        for j, record in enumerate(np.asarray(all_data[key]).flat, start=1):
            record = np.asarray(record).ravel()[0]
            name = str(np.asarray(record["name"]).ravel()[0])
            data = np.asarray(record["data"], dtype=float)
            data = data.reshape(-1, 2) if data.size else np.zeros((0, 2))
            savemat(str(out_dir / f"fix{i}_{j}.mat"), {"name": name, "data": data})


def _install_cat2000_extractor() -> None:
    """Without MATLAB, run _extract_cat2000_fixations instead of load_cat2000.m (Octave cannot run it)."""
    from pysaliency.external_datasets import cat2000
    from pysaliency.utils import MatlabOptions

    original_run_matlab_cmd = cat2000.run_matlab_cmd

    def run_matlab_cmd(cmd, cwd=None):
        has_matlab = any(shutil.which(name) for name in MatlabOptions.matlab_names)
        if cmd.strip() == "load_cat2000;" and cwd is not None and not has_matlab:
            if not _has_matio():
                raise SystemExit(
                    "CAT2000: without MATLAB, reading allFixData.mat needs the mat-io package: "
                    "`pip install mat-io` (or `uv pip install --python DG/bin/python mat-io`)."
                )
            return _extract_cat2000_fixations(Path(cwd))
        return original_run_matlab_cmd(cmd, cwd=cwd)

    cat2000.run_matlab_cmd = run_matlab_cmd


def check_matlab_available(name: str, data_dir: Path) -> None:
    """Fail before downloading anything when the first conversion of the dataset cannot run."""
    if name not in ("MIT1003", "CAT2000") or (data_dir / DATASETS[name]["cache_dir"]).exists():
        return
    from pysaliency.utils import MatlabOptions

    has_matlab = any(shutil.which(n) for n in MatlabOptions.matlab_names)
    has_octave = any(shutil.which(n) for n in MatlabOptions.octave_names)
    if name == "MIT1003" and not (has_matlab or has_octave):
        raise SystemExit(
            "MIT1003: the first conversion of the dataset by pysaliency runs its original MATLAB code "
            "and needs `matlab` or `octave` on the PATH (e.g. `apt install octave`). Install one of them, "
            f"or copy an already converted {data_dir / DATASETS[name]['cache_dir']} from another machine."
        )
    if name == "CAT2000" and not (has_matlab or _has_matio()):
        raise SystemExit(
            "CAT2000: the first conversion needs MATLAB, or the mat-io package to read allFixData.mat "
            "(`pip install mat-io`; Octave cannot read it). Install it, or copy an already converted "
            f"{data_dir / DATASETS[name]['cache_dir']} from another machine."
        )


def load_pysaliency_dataset(name: str, data_dir: Path, downloads_dir: Path, coco_split: str) -> list[tuple]:
    """
    (split, Stimuli, Fixations) of pysaliency for the dataset, downloading and converting it on
    first use into data_dir. COCO Freeview gives one entry per selected split.
    """
    from pysaliency.external_datasets.cat2000 import get_cat2000_train
    from pysaliency.external_datasets.coco_freeview import get_COCO_Freeview
    from pysaliency.external_datasets.mit import get_mit1003
    from pysaliency.utils import MatlabOptions

    # `octave` may start the GUI build, which can crash in a desktop or snap environment (e.g. a
    # terminal of a snap-packaged editor); the command-line build `octave-cli` does the same work.
    if "octave-cli" not in MatlabOptions.octave_names:
        MatlabOptions.octave_names = ["octave-cli", *MatlabOptions.octave_names]

    _install_download_cache(downloads_dir)
    _install_octave_shims()
    _install_cat2000_extractor()
    check_matlab_available(name, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if name == "MIT1003":
        stimuli, fixations = get_mit1003(location=str(data_dir))
        return [("all", stimuli, fixations)]
    if name == "CAT2000":
        # Version 1.1: repeated fixations removed, MATLAB's 1-based coordinates corrected.
        stimuli, fixations = get_cat2000_train(location=str(data_dir), version="1.1")
        return [("train", stimuli, fixations)]
    if name == "COCO_FREEVIEW":
        stimuli_train, fixations_train, stimuli_val, fixations_val, _ = get_COCO_Freeview(location=str(data_dir))
        parts = {"train": (stimuli_train, fixations_train), "validation": (stimuli_val, fixations_val)}
        splits = ["train", "validation"] if coco_split == "all" else [coco_split]
        return [(split, *parts[split]) for split in splits]
    raise ValueError(f"Unknown dataset {name!r}")


def to_project_format(parts: list[tuple], drop_initial_fixation: bool = False) -> tuple[Path, list[dict], pd.DataFrame]:
    """
    Project format of pysaliency (split, Stimuli, Fixations) parts: the stimulus directory, the
    metadata (one entry per image, stems 000001... in the sorted order of the image file names)
    and the fixations as a DataFrame for Loader.load_dataset.

    pysaliency's fixations are in image pixel coordinates, so the screen and the stimulus are
    given the size of the image (no offset, no scaling in Comparator). Images present in several
    parts (splits) are merged. With drop_initial_fixation, the first fixation of every scanpath
    (the one on the fixation cross) is left out, as processing/cleaner.py does for the project's data.
    """
    all_filenames = [os.path.normpath(filename) for _, stimuli, _ in parts for filename in stimuli.filenames]
    image_dir = Path(os.path.commonpath(all_filenames))
    if image_dir.is_file():
        image_dir = image_dir.parent

    images: dict[str, dict] = {}
    tables = []
    for split, stimuli, fixations in parts:
        names = [os.path.relpath(os.path.normpath(filename), image_dir) for filename in stimuli.filenames]
        sizes = [stimuli.sizes[index] for index in range(len(names))]  # (height, width)
        for index, (name, (height, width)) in enumerate(zip(names, sizes)):
            image = images.setdefault(name, {
                "name": name, "height": int(height), "width": int(width),
                "pysaliency_split": split, "pysaliency_index": int(index),
            })
            if (image["height"], image["width"]) != (int(height), int(width)):
                raise ValueError(f"{name}: different sizes in the parts of the dataset")

        # fixations.lengths: number of previous fixations of the scanpath (0 for its first fixation).
        keep = np.asarray(fixations.lengths) > 0 if drop_initial_fixation else np.ones(len(fixations.x), bool)
        durations = getattr(fixations, "duration", None)  # seconds, when the dataset has them
        tables.append(pd.DataFrame({
            "image_name": np.asarray(names, dtype=object)[np.asarray(fixations.n, dtype=int)[keep]],
            "participant_id": np.asarray(fixations.subjects)[keep].astype(str),
            "x": np.asarray(fixations.x, dtype=float)[keep],
            "y": np.asarray(fixations.y, dtype=float)[keep],
            "t": np.nan_to_num(np.asarray(fixations.t, dtype=float))[keep],
            "duration": (
                np.nan_to_num(np.asarray(durations, dtype=float))[keep] * 1000.0
                if durations is not None else np.zeros(int(keep.sum()))
            ),
        }))

    metadata = []
    for image_id, name in enumerate(sorted(images), start=1):
        image = images[name]
        metadata.append({
            "id": image_id,
            "stem": f"{image_id:06d}",
            "name": name,
            "extension": Path(name).suffix.lstrip(".").lower(),
            "modality": MODALITY,
            "height": image["height"],
            "width": image["width"],
            "pysaliency_split": image["pysaliency_split"],
            "pysaliency_index": image["pysaliency_index"],
        })
    by_name = {entry["name"]: entry for entry in metadata}

    df = pd.concat(tables, ignore_index=True)
    df["image_stem"] = df["image_name"].map(lambda n: by_name[n]["stem"])
    df["image_id"] = df["image_name"].map(lambda n: by_name[n]["id"])
    width = df["image_name"].map(lambda n: by_name[n]["width"])
    height = df["image_name"].map(lambda n: by_name[n]["height"])
    for column in ("image_width", "stimuli_width", "screen_width"):
        df[column] = width
    for column in ("image_height", "stimuli_height", "screen_height"):
        df[column] = height
    df["image_modality"] = MODALITY
    df["eye"] = "unknown"
    df["fixation_id"] = np.arange(len(df))
    df["start"] = df["t"].round().astype(int)
    df["end"] = df["start"]
    df["duration"] = df["duration"].round().astype(int)
    df = df[[
        "image_id", "image_stem", "image_name", "image_modality", "participant_id", "fixation_id",
        "eye", "start", "end", "duration", "x", "y",
        "screen_width", "screen_height", "stimuli_width", "stimuli_height", "image_width", "image_height",
    ]]
    return image_dir, metadata, df


# ------------------------------------------------------------------- pipeline

def display_geometry(pixel_per_dva: float, viewing_distance_cm: float, screen_width_px: int) -> dict:
    """Comparator display arguments giving exactly pixel_per_dva pixels per degree (Comparator._compute_sigma)."""
    pixels_per_cm = pixel_per_dva / (2.0 * viewing_distance_cm * tan(radians(1.0) / 2.0))
    return {
        "viewing_distance_cm": viewing_distance_cm,
        "screen_width_px": screen_width_px,
        "screen_width_cm": screen_width_px / pixels_per_cm,
    }


class LeaveOneImageOutCenterBias:
    """
    CenterBias(nx[image_ids != image_index], ny[image_ids != image_index], sigma, eps) built on
    demand from the arrays shared by all images, instead of one copy of them per image. The
    log-probabilities of the last image evaluated are kept, as the pipeline asks for them several
    times for the same ImageFixations.
    """

    def __init__(self, nx, ny, image_ids, image_index, sigma, eps):
        self._nx, self._ny, self._image_ids, self._image_index = nx, ny, image_ids, image_index
        self.sigma = float(sigma)
        self.eps = float(eps)
        self._cache = None

    @property
    def nx(self):
        return self._nx[self._image_ids != self._image_index]

    @property
    def ny(self):
        return self._ny[self._image_ids != self._image_index]

    def materialise(self) -> CenterBias:
        return CenterBias(self.nx, self.ny, sigma=self.sigma, eps=self.eps)

    def log_prob(self, img) -> np.ndarray:
        if self._cache is None or self._cache[0] is not img:
            self._cache = (img, self.materialise().log_prob(img))
        return self._cache[1].copy()


class PublicDatasetComparator(Comparator):
    """
    Comparator of experiment 2 for one large dataset.

    Comparator keeps, for every image, a copy of the fixations of all the other images for its
    leave-one-image-out center bias, which would take tens of GB here. The center biases are built
    on demand instead (LeaveOneImageOutCenterBias) and, when the dataset has more than
    centerbias_max_fixations fixations, the center bias is estimated from a random subset of that
    many (the same subset for the cross-validated fit and for every image, drawn with
    centerbias_seed, so that a saved gold standard reloads with the same baselines).
    """

    def __init__(self, *args, centerbias_max_fixations: int = 100_000, centerbias_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.centerbias_max_fixations = int(centerbias_max_fixations)
        self.centerbias_seed = centerbias_seed

    def _normalised_fixations(self, images):
        nx, ny, image_ids = Comparator._normalised_fixations(images)
        n_max = self.centerbias_max_fixations
        if n_max > 0 and len(nx) > n_max:
            rng = np.random.default_rng(self.centerbias_seed)
            keep = np.sort(rng.choice(len(nx), size=n_max, replace=False))
            nx, ny, image_ids = nx[keep], ny[keep], image_ids[keep]
        return nx, ny, image_ids

    def _leave_one_image_out_centerbiases(self, images, sigma, eps):
        nx, ny, image_ids = self._normalised_fixations(images)
        return {
            stem: LeaveOneImageOutCenterBias(nx, ny, image_ids, i, sigma, eps)
            for i, stem in enumerate(images)
        }


class LogDensityStore(dict):
    """DeepGaze log densities of a few stems; any other stem is predicted again when asked for."""

    def __init__(self, predict):
        super().__init__()
        self._predict = predict

    def __missing__(self, stem):
        return self._predict(stem)


def prepare_image(image: np.ndarray, grayscale: bool) -> np.ndarray:
    """
    (H, W, 3) uint8 RGB input of DeepGaze. With grayscale, the image is converted with
    cv2.COLOR_RGB2GRAY (Loader.load_image gives RGB) and the gray channel is repeated 3 times:
    DeepGaze._normalize_input would stretch the contrast of a single-channel input (its IR path).
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if grayscale:
        gray = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
        image = np.repeat(gray[..., None], 3, axis=2)
    return np.ascontiguousarray(image)


def draw_example_stems(stems, n_examples: int, seed: int | None) -> list[str]:
    """The stems Saver.save_deepgaze_ige / save_gold_standard_results draw for their figures."""
    stems = sorted(set(stems))
    rng = np.random.default_rng(seed)
    return [str(stem) for stem in rng.choice(stems, size=min(n_examples, len(stems)), replace=False)]


def run_deepgaze_ige_dataset(
    comparator: Comparator,
    data: dict,
    gold: dict,
    image_dir: Path,
    stems: list[str],
    model: str,
    msdb_dataset: int | None,
    grayscale: bool,
    example_stems: list[str],
) -> dict:
    """
    Comparator.run_deepgaze_ige for one dataset (one modality): IGE of DeepGaze on every stem, with
    the gold standard's leave-one-image-out center bias of the image as prior and as baseline.
    Only the log densities of example_stems are kept for the figures of Saver.save_deepgaze_ige.
    """
    deepgaze = DeepGaze(model=model, msdb_dataset=msdb_dataset)

    def predict_log_density(stem: str) -> np.ndarray:
        img = gold["fixations"][stem]
        centerbias = gold["centerbiases"][stem]
        dims = data[stem].dimensions_data
        # Image pixels per degree of visual angle, as displayed (used by DeepGaze MSDB).
        pixel_per_dva = comparator._compute_sigma(1.0, dims.stimuli_width, dims.image_width)
        # The baseline of this image on the pixel grid, as DeepGaze's prior ("gold" of experiment 2).
        prior = _centerbias_log_map(centerbias, img.width, img.height)
        image = prepare_image(comparator.loader.load_image(image_dir / data[stem].image_data.name), grayscale)
        return deepgaze.predict_log_density(image, pixel_per_dva=pixel_per_dva, centerbias=prior)

    per_image, image_paths = {}, {}
    log_densities = LogDensityStore(lambda stem: predict_log_density(stem).astype(np.float32))
    label = f"DeepGaze {model} IGE ({'grayscale' if grayscale else 'RGB'})"
    for stem in tqdm(stems, desc=f"Computing {label}"):
        img = gold["fixations"][stem]
        centerbias = gold["centerbiases"][stem]
        gold_values = gold["images"][stem]

        # Sanity check: the baseline must be the one of the stored gold-standard IG.
        ll_centerbias_bits = float((centerbias.log_prob(img).mean() + np.log(img.width * img.height)) / LN2)
        if not np.isclose(ll_centerbias_bits, gold_values["ll_centerbias_bits"], atol=1e-6):
            raise ValueError(
                f"{stem}: the center-bias baseline differs from the gold standard "
                f"({ll_centerbias_bits:.6f} vs {gold_values['ll_centerbias_bits']:.6f} bit/fix). "
                "Recompute the gold standard (--recompute-gold-standard) with the current data and settings."
            )

        log_density = predict_log_density(stem)
        model_ig = model_information_gain(img, log_density, centerbias)
        explainable_ig = float(gold_values["explainable_ig_bits"])
        naive_ig = float(gold_values["naive_ig_bits"])

        per_image[stem] = {
            "n_participants": int(len(np.unique(img.subject))),
            "n_fixations": int(len(img.x)),
            "ll_centerbias_bits": ll_centerbias_bits,
            "ll_gold_bits": float(gold_values["ll_gold_bits"]),
            "ll_model_bits": ll_centerbias_bits + model_ig,
            "explainable_ig_bits": explainable_ig,
            "naive_ig_bits": naive_ig,
            "model_ig_bits": model_ig,
            # Unstable when the explainable IG of the image is close to 0.
            "ige": model_ig / explainable_ig if explainable_ig != 0 else float("nan"),
            # IGE of the naive model (KDE, sigma = 1 deg, leave-one-subject-out): baseline for "ige".
            "ige_naive": naive_ig / explainable_ig if explainable_ig != 0 else float("nan"),
        }
        if stem in example_stems:
            log_densities[stem] = log_density.astype(np.float32)
        image_paths[stem] = image_dir / data[stem].image_data.name

    explainable_ig_bits = float(np.mean([r["explainable_ig_bits"] for r in per_image.values()]))
    naive_ig_bits = float(np.mean([r["naive_ig_bits"] for r in per_image.values()]))
    model_ig_bits = float(np.mean([r["model_ig_bits"] for r in per_image.values()]))
    return {
        "model": model,
        "msdb_dataset": msdb_dataset,
        "centerbias_source": "gold",
        "centerbias_path": None,
        "grayscale": grayscale,
        "explainable_ig_bits": explainable_ig_bits,
        "naive_ig_bits": naive_ig_bits,
        "model_ig_bits": model_ig_bits,
        "ige": model_ig_bits / explainable_ig_bits,
        "ige_naive": naive_ig_bits / explainable_ig_bits,
        "images": per_image,
        # For the saver's visualisations.
        "log_densities": log_densities,
        "image_paths": image_paths,
        "fixations": {stem: gold["fixations"][stem] for stem in per_image},
        "centerbiases": {stem: gold["centerbiases"][stem] for stem in per_image},
        "gold_standard": gold["gold_standard"],
        "naive_models": {stem: gold["naive_models"][stem] for stem in per_image},
    }


def select_stems(comparator: Comparator, stems: list[str], args) -> list[str]:
    stems = sorted(stems)
    if args.stems:
        wanted = [comparator.index.normalize_stem(stem) for stem in args.stems]
        missing = sorted(set(wanted) - set(stems))
        if missing:
            raise SystemExit(f"Unknown stems: {missing}")
        stems = sorted(set(wanted))
    if args.max_images is not None and args.max_images < len(stems):
        rng = np.random.default_rng(args.seed)
        stems = sorted(str(stem) for stem in rng.choice(stems, size=args.max_images, replace=False))
    return stems


def run_dataset(name: str, args) -> dict:
    spec = DATASETS[name]
    # The stems number the selected images, so a single COCO split gets its own folder.
    label = name if name != "COCO_FREEVIEW" or args.coco_split == "all" else f"{name}_{args.coco_split}"
    output_dir = args.output_dir / label
    print(f"\n=== {label} ===")

    parts = load_pysaliency_dataset(name, args.data_dir, args.downloads_dir, args.coco_split)
    drop_initial_fixation = spec["initial_fixation_included"] and not args.keep_initial_fixation
    image_dir, metadata, fixations_df = to_project_format(parts, drop_initial_fixation)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_json = metadata_dir / "metadata.json"
    annotations_json = metadata_dir / "annotations.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    annotations_json.write_text("{}\n", encoding="utf-8")
    fixations_df.to_csv(metadata_dir / "fixations.csv", index=False)

    pixel_per_dva = args.pixel_per_dva.get(name, spec["pixel_per_dva"])
    comparator = PublicDatasetComparator(
        metadata_json=metadata_json,
        annotations_json=annotations_json,
        sigma_deg=1.0,
        **display_geometry(pixel_per_dva, spec["viewing_distance_cm"], spec["screen_width_px"]),
        centerbias_max_fixations=args.centerbias_max_fixations,
        centerbias_seed=args.seed,
    )
    data = comparator.loader.load_dataset(fixations_df)
    stems = sorted(data)
    print(
        f"{len(stems)} images in {image_dir}, {len(fixations_df)} fixations"
        f"{' (initial fixation of each scanpath dropped)' if drop_initial_fixation else ''}, "
        f"{fixations_df['participant_id'].nunique()} subjects, "
        f"{comparator._compute_sigma(1.0, 1000, 1000):.2f} px/deg"
    )

    saver = Saver(output_dir)

    # Gold standard on every image of the dataset, computed once.
    gold_dir = output_dir / "gold_standard"
    if (gold_dir / name / "images.csv").exists() and not args.recompute_gold_standard:
        print(f"Reloading the gold standard of {gold_dir / name}")
        gold = comparator._load_gold_standard(gold_dir / name, data)
    else:
        gold = comparator._compute_gold_standard_dataset(data, stems, COMPONENTS, None, args.seed, image_dir)
        gold["params"].update({
            "dataset": name,
            "coco_split": args.coco_split if name == "COCO_FREEVIEW" else None,
            "initial_fixation_dropped": drop_initial_fixation,
            "pixel_per_dva": pixel_per_dva,
            "centerbias_max_fixations": comparator.centerbias_max_fixations,
            "n_centerbias_fixations": int(len(gold["centerbias"].nx)),
        })
        saver.save_gold_standard_results({name: gold}, output_dir=gold_dir, n_examples=args.n_examples, seed=args.seed)
    print(
        f"Gold standard: {len(gold['images'])} images, sigma = {gold['gold_standard'].sigma:.1f} px, "
        f"weights = {{{', '.join(f'{k}: {v:.3f}' for k, v in gold['gold_standard'].weights.items())}}}, "
        f"explainable IG = {np.mean([r['explainable_ig_bits'] for r in gold['images'].values()]):.3f} bit/fix"
    )

    # Information gain explained by DeepGaze.
    ige_stems = select_stems(comparator, list(gold["images"]), args)
    msdb_dataset, msdb_name = None, None
    if args.model == "MSDB" and args.msdb_dataset is not None:
        msdb_name = name if args.msdb_dataset == "AUTO" else args.msdb_dataset
        msdb_dataset = getattr(MSDBDataset, msdb_name)
    results = run_deepgaze_ige_dataset(
        comparator, data, gold, image_dir, ige_stems,
        model=args.model, msdb_dataset=msdb_dataset, grayscale=args.grayscale,
        example_stems=draw_example_stems(ige_stems, args.n_examples, args.seed),
    )

    ige_folder = "ige"
    if args.model == "MSDB":
        ige_folder = "ige_msdb" + (f"_{msdb_name.lower()}" if msdb_name else "")
    if args.grayscale:
        ige_folder += "_gray"
    modality = "GRAY" if args.grayscale else "RGB"
    saver.save_deepgaze_ige({modality: results}, output_dir=output_dir / ige_folder,
                            n_examples=args.n_examples, seed=args.seed)
    print(
        f"DeepGaze {args.model} on {modality} ({len(ige_stems)} images): IG = {results['model_ig_bits']:.3f} bit/fix, "
        f"IGE = {results['ige']:.1%} (naive model: {results['ige_naive']:.1%}); saved in {output_dir / ige_folder}"
    )
    return {
        "dataset": label, "modality": modality, "model": args.model, "msdb_dataset": msdb_name,
        "n_images": len(ige_stems), "explainable_ig_bits": results["explainable_ig_bits"],
        "model_ig_bits": results["model_ig_bits"], "ige": results["ige"], "ige_naive": results["ige_naive"],
    }


def parse_pixel_per_dva(values: list[str], parser: argparse.ArgumentParser) -> dict[str, float]:
    overrides = {}
    for value in values:
        try:
            name, number = value.split("=")
            name = name.upper()
            overrides[name] = float(number)
        except ValueError:
            parser.error(f"--pixel-per-dva expects DATASET=VALUE, got {value!r}")
        if name not in DATASETS:
            parser.error(f"--pixel-per-dva: unknown dataset {name!r}, expected one of {list(DATASETS)}")
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold standard and information gain explained by DeepGaze on MIT1003, CAT2000 and COCO Freeview.",
    )
    parser.add_argument("--datasets", nargs="+", type=str.upper, choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to evaluate (default: all three).")
    parser.add_argument("--grayscale", action="store_true",
                        help="Feed DeepGaze the grayscale version of the images (cv2 RGB2GRAY) instead of RGB; "
                             "results go to a folder suffixed with _gray.")
    parser.add_argument("--model", type=str.upper, choices=["IIE", "MSDB"], default="IIE",
                        help="DeepGaze model to evaluate (default: IIE).")
    parser.add_argument("--msdb-dataset", type=str.upper,
                        choices=["AUTO", "MIT1003", "CAT2000", "COCO_FREEVIEW", "DAEMONS", "FIGRIM"], default=None,
                        help="MSDB only: use the parameters learned for this dataset instead of the average over "
                             "its training datasets (default: average). AUTO: the parameters of the evaluated dataset.")
    parser.add_argument("--coco-split", choices=["all", "train", "validation"], default="all",
                        help="COCO Freeview images to use: the train and validation splits (default), or one of them.")
    parser.add_argument("--keep-initial-fixation", action="store_true",
                        help="Keep the first fixation of every scanpath (on the fixation cross) of CAT2000 and "
                             "COCO Freeview. By default it is dropped, as pysaliency already does for MIT1003 and "
                             "processing/cleaner.py for the project's data.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Evaluate DeepGaze on a random subset of this many images (the gold standard is "
                             "still fitted on every image). For quick tests.")
    parser.add_argument("--stems", nargs="+", default=None, help="Evaluate DeepGaze on these stems only.")
    parser.add_argument("--n-examples", type=int, default=5, help="Images shown in the figures (default: 5).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the center-bias fit and subset, of --max-images and of the figures (default: 0).")
    parser.add_argument("--centerbias-max-fixations", type=int, default=100_000,
                        help="Estimate the center bias from at most this many fixations of the dataset "
                             "(default: 100000; 0: all of them, slow on CAT2000 and COCO Freeview).")
    parser.add_argument("--pixel-per-dva", nargs="+", default=[], metavar="DATASET=VALUE",
                        help="Override the pixels per degree of visual angle of a dataset, e.g. MIT1003=35.")
    parser.add_argument("--recompute-gold-standard", action="store_true",
                        help="Fit the gold standard again even if <output-dir>/<DATASET>/gold_standard exists.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "expe2_ige_public",
                        help="Root of the outputs (default: outputs/expe2_ige_public).")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "DATA_OCCULO_2026" / "pysaliency",
                        help="Where pysaliency caches the converted datasets (default: DATA_OCCULO_2026/pysaliency).")
    parser.add_argument("--downloads-dir", type=Path, default=None,
                        help="Where the dataset archives are kept (default: <data-dir>/downloads).")
    parser.add_argument("--tmp-dir", type=Path, default=None,
                        help="Temporary directory for the dataset conversions (default: the system one; "
                             "COCO Freeview needs about 7 GB there).")
    args = parser.parse_args()

    if args.msdb_dataset is not None and args.model != "MSDB":
        parser.error("--msdb-dataset only applies to --model MSDB")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least 1")
    args.pixel_per_dva = parse_pixel_per_dva(args.pixel_per_dva, parser)
    args.data_dir = args.data_dir.resolve()
    args.downloads_dir = (args.downloads_dir or args.data_dir / "downloads").resolve()
    args.output_dir = args.output_dir.resolve()
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(args.tmp_dir.resolve())

    summary = [run_dataset(name, args) for name in args.datasets]

    print("\nSummary")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()

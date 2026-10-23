# Saliency Map Analysis

This directory contains the tools used to compare human fixation distributions between RGB and LWIR, evaluate DeepGaze IIE in both modalities, and perform the associated statistical analyses.

## Structure

```text
analysis/
├── analyzer.py
├── comparator.py
├── metrics.py
├── saver.py
├── stats.py
└── README.md
```

| Module | Purpose |
|---|---|
| `comparator.py` | Computes human–human and DeepGaze–human comparisons for each scene. |
| `metrics.py` | Implements similarity and predictivity metrics. |
| `analyzer.py` | Applies the statistical analyses to scene-level results. |
| `stats.py` | Provides t-tests and multiple-comparison correction. |
| `saver.py` | Exports comparison results and statistical analyses to CSV and JSON. |

## Metrics

The active metrics are registered in `Metrics.registry`.

### Distribution-based metrics

- `cc`: Pearson correlation;
- `sim`: histogram intersection between normalized distributions;
- `kl`: negative Kullback–Leibler divergence, so that higher values indicate better agreement.

### Fixation-based metrics

- `nss`: Normalized Scanpath Saliency;
- `auc_judd`: discrimination between fixated and non-fixated pixels;
- `ig`: mean log-likelihood gain over a uniform baseline, reported in nats per fixation.

### ROI metrics

Computed inside the annotated bounding boxes, per ROI class (`Comparator.run_deepgaze_roi`, run with `Experiment_scripts/Run_experiment_3_roi.py`):

- `mass`: map mass inside the ROI;
- `ratio`: mass inside the ROI divided by the total mass;
- `density`: mass per ROI pixel;
- `normalized_ratio`: `ratio` divided by the fraction of the image covered by the ROI (1 = chance).

ROI-related functions are available in `metrics.py`, but they are not enabled in the current pipeline.

## Comparisons

### Initialization

```python
from analysis.comparator import Comparator

comparator = Comparator(
    metadata_json="metadata/metadata.json",
    annotations_json="metadata/annotations.json",
    sigma_deg=1.0,
    viewing_distance_cm=70.0,
    screen_width_cm=53.1,
    screen_width_px=1360,
    epsilon=1e-12,
)
```

`sigma_deg` defines the Gaussian smoothing standard deviation in degrees of visual angle. It is converted separately for each scene into the original image coordinate system.

The RGB and LWIR DataFrames must be compatible with `Loader.load_dataset`, contain the same `image_stem` values, and provide identical `image_width` and `stimuli_width` values for each paired scene.

### Experiment 1 — Human cross-modal comparison

```python
results_expe1 = comparator.run_expe1(
    infrared_df=infrared_df,
    visible_df=visible_df,
    n_permutations=1000,
    seed=42,
)
```

For each scene:

1. RGB and LWIR fixations are aggregated separately;
2. the two human maps are compared in both directions;
3. observers are pooled and randomly reassigned to two pseudo-groups while preserving the original group sizes;
4. the observed RGB–LWIR agreement is compared with the mean permutation agreement.

The recorded difference is:

```text
delta_perm_inter = score_perm_mean - score_inter
```

A positive value indicates that the observed RGB–LWIR agreement is lower than the agreement obtained after random observer reassignment.

### Experiment 2 — DeepGaze cross-modal comparison

```python
from pathlib import Path

results_expe2 = comparator.run_expe2(
    infrared_df=infrared_df,
    visible_df=visible_df,
    infrared_image_dir=Path("stimuli/infrared"),
    visible_image_dir=Path("stimuli/visible"),
    infrared_centerbias_path=Path("centerbias/infrared.npy"),
    visible_centerbias_path=Path("centerbias/visible.npy"),
)
```

A separate DeepGaze instance is created for each modality using the corresponding center-bias prior. For each paired scene, the prediction is evaluated against the human reference from the same modality.

The recorded difference is:

```text
delta_rgb_ir = score_rgb - score_ir
```

A positive value indicates better performance in RGB, while a negative value indicates better performance in LWIR.

## Statistical Analysis

```python
from analysis.analyzer import Analyzer

analyzer = Analyzer(
    alpha=0.05,
    correction_method="holm",
)
```

### Experiment 1

```python
analysis_expe1 = analyzer.analyze_expe1(results_expe1_df)
```

The analysis is applied to `delta_perm_inter`:

- one-sample t-test against zero;
- one-sided alternative: `greater`;
- Holm correction across the six metrics.

### Experiment 2

```python
analysis_expe2 = analyzer.analyze_expe2(results_expe2_df)
```

The analysis is applied to `delta_rgb_ir`:

- one-sample t-test against zero;
- two-sided alternative;
- Holm correction across the six metrics.

The returned results include the mean, standard deviation, t statistic, degrees of freedom, p-value, confidence interval, and effect size.

## Saving Results

```python
from analysis.saver import Saver

saver = Saver("outputs")

results_expe1_df = saver.save_expe1_results(results_expe1)
analysis_expe1 = analyzer.analyze_expe1(results_expe1_df)
saver.save_expe1_analysis(analysis_expe1)

results_expe2_df = saver.save_expe2_results(results_expe2)
analysis_expe2 = analyzer.analyze_expe2(results_expe2_df)
saver.save_expe2_analysis(analysis_expe2)
```

Generated output structure:

```text
outputs/
├── expe1/
│   ├── comparison/
│   │   ├── similarity/
│   │   ├── predictivity/
│   │   ├── all_results.csv
│   │   └── summary.csv
│   └── statistics/
│       ├── analysis.csv
│       └── analysis.json
└── expe2/
    ├── comparison/
    │   ├── similarity/
    │   ├── predictivity/
    │   ├── all_results.csv
    │   └── summary.csv
    └── statistics/
        ├── analysis.csv
        └── analysis.json
```

## Main Dependencies

- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- Pingouin;
- tqdm;
- local `helpers` and `saliency` modules.

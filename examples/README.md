# Examples

This directory contains executable examples for processing eye-tracking recordings, running the RGB–LWIR analyses, and generating qualitative visualizations.

The files in this directory orchestrate the project modules. The processing, analysis, saliency, and visualization implementations are defined in their respective source directories.

## Structure

```text
examples/
├── processing.ipynb
├── analysis.ipynb
├── visualization.py
└── README.md
```

| File | Purpose |
|---|---|
| `processing.ipynb` | Extracts, profiles, cleans, and organizes the eye-tracking data, then builds the center-bias priors. |
| `analysis.ipynb` | Runs the human cross-modal and DeepGaze IIE analyses and saves the statistical results. |
| `visualization.py` | Generates fixation, heatmap, cross-modal, and DeepGaze–human visualizations. |

## Recommended Order

Run the examples from the project root in the following order:

1. `processing.ipynb`
2. `analysis.ipynb`
3. `visualization.py`, when qualitative figures are required

Launching Jupyter from the project root ensures that local modules and relative paths are resolved correctly.

```bash
jupyter notebook examples/processing.ipynb
jupyter notebook examples/analysis.ipynb
python examples/visualization.py
```

## Data Processing

`processing.ipynb` illustrates the complete eye-tracking preprocessing workflow:

1. parse the EyeLink ASCII files;
2. extract raw fixations, blinks, and trial summaries;
3. evaluate trial and participant quality;
4. remove invalid trials and fixations;
5. create the paired RGB–LWIR fixation tables;
6. create the larger infrared-only fixation table;
7. build the full, RGB, and LWIR center-bias priors.

The notebook expects the source data under:

```text
data/2026/global/
├── asc/
│   ├── visible/
│   └── infrared/
├── images/
├── metadata/
└── dataframes/
```

The main generated datasets are stored under:

```text
data/2026/
├── paired/
│   ├── dataframes/
│   │   ├── fixations_visible.csv
│   │   └── fixations_infrared.csv
│   └── centerbias/
│       ├── full.npy
│       ├── visible.npy
│       └── infrared.npy
└── single/
    ├── fixations.csv
    └── centerbias.npy
```

Path definitions and preprocessing parameters should be reviewed before executing the notebook.

## Statistical Analysis

`analysis.ipynb` runs the two analyses defined in the `analysis` package.

### Human Cross-Modal Analysis

The observed RGB–LWIR agreement is compared with a reference obtained from 1,000 random observer reassignments while preserving the original group sizes.

Default settings:

```text
Gaussian bandwidth: 1 degree of visual angle
Permutations: 1,000
Random seed: 42
```

### DeepGaze IIE Analysis

DeepGaze IIE is evaluated separately on the RGB and LWIR images using the corresponding human fixation maps and modality-specific center-bias priors.

The notebook then applies:

- a one-sided one-sample t-test to the human cross-modal differences;
- a two-sided one-sample t-test to the DeepGaze RGB–LWIR differences;
- Holm correction across the six metrics of each analysis.

Results are written to:

```text
outputs/
├── expe1/
│   ├── comparison/
│   └── statistics/
└── expe2/
    ├── comparison/
    └── statistics/
```

## Visualization

`visualization.py` provides examples for:

- displaying fixation locations;
- generating human fixation heatmaps;
- comparing RGB and LWIR heatmaps;
- comparing DeepGaze predictions with human fixation-density maps.

Configure the paths and display parameters at the beginning of the script. Uncomment the required visualization block before execution.

The current active block generates DeepGaze–human comparisons for the infrared modality. Generated figures are stored under:

```text
outputs/figures/
```

## Main Requirements

The examples require the project dependencies and the following local packages:

- `analysis`;
- `processing`;
- `saliency`;
- `helpers`.

Jupyter and Matplotlib are also required for the notebooks and visual outputs.

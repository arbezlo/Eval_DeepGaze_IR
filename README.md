# InfraRed Gaze Benchmark

IRGazeBenchmark is a modular pipeline for studying human visual attention across RGB and long-wave infrared (LWIR) imagery and for evaluating DeepGaze IIE under spectral domain shift.

The repository covers the complete workflow:

- eye-tracking acquisition with PsychoPy and EyeLink;
- conversion of ASC recordings into structured fixation tables;
- trial-quality assessment and fixation cleaning;
- construction of human fixation-density maps;
- estimation of modality-specific center-bias priors;
- DeepGaze IIE inference on RGB and single-channel LWIR images;
- human cross-modal and DeepGaze–human statistical analyses;
- qualitative visualization of fixation and saliency maps.

## Repository Structure

```text
DeepGazeIR/
├── analysis/
│   ├── analyzer.py
│   ├── comparator.py
│   ├── metrics.py
│   ├── saver.py
│   ├── stats.py
│   └── README.md
├── eyetracking/
│   ├── main.py
│   ├── src/
│   ├── stimuli/
│   ├── tables/
│   ├── results/
│   └── README.md
├── examples/
│   ├── processing.ipynb
│   ├── analysis.ipynb
│   ├── visualization.py
│   └── README.md
├── helpers/
│   ├── indexer.py
│   ├── loader.py
│   ├── models.py
│   ├── utils.py
│   └── visualizer.py
├── processing/
│   ├── parser.py
│   ├── extractor.py
│   ├── profiler.py
│   ├── cleaner.py
│   └── README.md
├── saliency/
│   ├── builder.py
│   ├── centerbias.py
│   ├── deepgaze.py
│   └── README.md
├── data/
├── outputs/
└── README.md
```

## Main Components

| Directory | Purpose |
|---|---|
| `eyetracking/` | Runs the PsychoPy and EyeLink free-viewing experiment. |
| `processing/` | Parses ASC files, extracts events, evaluates trial quality, and cleans fixations. |
| `saliency/` | Builds human fixation maps, center-bias priors, and DeepGaze IIE predictions. |
| `analysis/` | Computes evaluation metrics, statistical tests, and result files. |
| `helpers/` | Provides shared data structures, loading, indexing, saving, and visualization utilities. |
| `examples/` | Contains the notebooks and scripts used to run the complete pipeline. |

Each main directory contains a dedicated README with implementation details.

## Analysis Workflow

### 1. Eye-Tracking Acquisition

The experiment presents RGB or LWIR images under free-viewing conditions and records gaze data with EyeLink.

```bash
cd eyetracking
python main.py
```

The acquisition script:

- assigns each participant to a predefined subset;
- preserves image aspect ratio;
- randomizes image order reproducibly;
- records parser-compatible EyeLink messages;
- stores one EDF file per session.

The EDF files must be converted to ASC before processing. This conversion is performed externally with the SR Research EDF2ASC utility.

### 2. Data Processing

Use the processing notebook from the project root:

```bash
jupyter notebook examples/processing.ipynb
```

The processing pipeline:

1. parses EyeLink ASC events;
2. extracts fixations, blinks, and trial summaries;
3. evaluates trial validity;
4. removes invalid trials and fixations;
5. projects gaze coordinates into stimulus space;
6. creates RGB, LWIR, and combined center-bias priors.

### 3. Statistical Analysis

Run:

```bash
jupyter notebook examples/analysis.ipynb
```

Two experiments are implemented.

#### Human Cross-Modal Analysis

The observed RGB–LWIR fixation agreement is compared with a permutation reference obtained by pooling observers and randomly reassigning them to pseudo-groups while preserving the original group sizes.

For each scene:

```text
delta_perm_inter = score_perm_mean - score_inter
```

A positive value indicates that observed RGB–LWIR agreement is lower than the permutation reference.

#### DeepGaze IIE Cross-Modal Analysis

DeepGaze IIE is evaluated separately on RGB and LWIR images against the human fixation-density map from the same modality.

For each paired scene:

```text
delta_rgb_ir = score_rgb - score_ir
```

A positive value indicates better model performance in RGB. A negative value indicates better performance in LWIR.

Separate center-bias priors can be used for RGB and LWIR inference.

### 4. Visualization

Run:

```bash
python examples/visualization.py
```

The visualization script can generate:

- fixation overlays;
- human fixation heatmaps;
- paired RGB–LWIR comparisons;
- DeepGaze–human comparisons.

## Metrics

The active metrics are divided into two groups.

### Distribution-Based Metrics

- Pearson correlation (`cc`);
- Similarity / histogram intersection (`sim`);
- Negative Kullback–Leibler divergence (`kl`).

The KL score is negated so that higher values indicate better agreement.

### Fixation-Based Metrics

- Normalized Scanpath Saliency (`nss`);
- AUC-Judd (`auc_judd`);
- Information Gain (`ig`).

Information Gain is computed relative to a uniform baseline and reported in nats per fixation.

## DeepGaze IIE and LWIR Inputs

The `DeepGaze` wrapper loads the pretrained DeepGaze IIE model in evaluation mode.

Accepted input shapes are:

```text
(H, W)
(H, W, 1)
(H, W, 3)
```

Single-channel LWIR images are min-max normalized and replicated across three channels before inference. Predictions are returned as normalized two-dimensional fixation-density maps with the same spatial resolution as the input image.

The model automatically uses CUDA when available and otherwise runs on CPU.

## Data Organization

A typical data layout is:

```text
data/
├── global/
│   ├── asc/
│   │   ├── visible/
│   │   └── infrared/
│   ├── images/
│   │   ├── visible/
│   │   └── infrared/
│   ├── metadata/
│   └── dataframes/
├── paired/
│   ├── dataframes/
│   │   ├── fixations_visible.csv
│   │   └── fixations_infrared.csv
│   ├── images/
│   │   ├── visible/
│   │   └── infrared/
│   └── centerbias/
│       ├── full.npy
│       ├── visible.npy
│       └── infrared.npy
└── single/
    ├── images/
    ├── fixations.csv
    └── centerbias.npy
```

The exact paths used by the notebooks should be checked before execution.

## Outputs

The analysis results are organized as follows:

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
├── expe2/
│   ├── comparison/
│   │   ├── similarity/
│   │   ├── predictivity/
│   │   ├── all_results.csv
│   │   └── summary.csv
│   └── statistics/
│       ├── analysis.csv
│       └── analysis.json
└── figures/
```

## Installation

Create and activate a Python environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Main dependencies:

```text
numpy
pandas
scipy
scikit-learn
pingouin
tqdm
opencv-python
torch
deepgaze-pytorch
matplotlib
jupyter
pillow
psychopy
```

EyeLink acquisition additionally requires:

- PyLink;
- the EyeLink Developers Kit;
- the SR Research PsychoPy graphics integration.

Package names and installation procedures for EyeLink components depend on the local acquisition environment.

The EyeLink acquisition components and `EyeLinkCoreGraphicsPsychoPy.py` are subject to the SR Research licensing terms. Their redistribution may be restricted. Check the corresponding source headers and license conditions before distributing the repository.

DeepGaze IIE is provided through the `deepgaze_pytorch` package.

## Recommended Execution Order

From the project root:

```text
1. Convert EDF files to ASC
2. Run examples/processing.ipynb
3. Run examples/analysis.ipynb
4. Run examples/visualization.py when figures are required
```

## Reproducibility

The main analysis defaults are:

```text
Gaussian smoothing: 1 degree of visual angle
Permutation count: 1,000
Random seed: 42
Significance level: 0.05
Multiple-comparison correction: Holm
```

These parameters are configurable in the notebooks and source modules.


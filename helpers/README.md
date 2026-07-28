# Shared Helper Utilities

This directory contains the shared data structures and utility classes used by the acquisition, processing, saliency, analysis, and visualization pipelines.

## Structure

```text
helpers/
├── indexer.py
├── loader.py
├── models.py
├── utils.py
├── visualizer.py
└── README.md
```

| Module | Purpose |
|---|---|
| `models.py` | Defines the dataclasses and type aliases exchanged across the project. |
| `loader.py` | Loads metadata, annotations, fixation tables, NumPy arrays, and images. |
| `indexer.py` | Normalizes identifiers and provides consistent access to metadata and annotations. |
| `utils.py` | Provides small shared utility functions. |
| `visualizer.py` | Generates interactive fixation, heatmap, modality, and DeepGaze–human visualizations. |

## Data Models

`models.py` defines the common structures used throughout the repository.

Main dataclasses:

| Class | Description |
|---|---|
| `DimensionData` | Screen, displayed stimulus, and original image dimensions. |
| `ImageData` | Image identifier, stem, filename, modality, dimensions, and time-of-day label. |
| `FixationData` | Eye, timing, duration, and gaze coordinates for one fixation. |
| `BlinkData` | Eye, timing, and duration for one blink. |
| `TrialState` | Mutable parser state used while reading one EyeLink trial. |
| `RawTrial` | Parsed trial containing stimulus metadata, fixations, and blinks. |
| `TrialData` | Scene-level structure grouping image data, dimensions, and participant fixations. |
| `RoiBox` | One rectangular region of interest. |
| `RoiData` | Collection of regions of interest for one image and modality. |

Main type aliases:

```python
Dataset
Metadata
Annotations
Results
```

For example, a `Dataset` is organized as:

```text
image_stem
└── TrialData
    ├── image_data
    ├── dimensions_data
    └── participants_data
        └── participant_id
            └── list[FixationData]
```

## Loading Data

`Loader` centralizes the loading of project data.

### Initialization

```python
from helpers.loader import Loader

loader = Loader("data/metadata/metadata.json")
```

Initialization loads the metadata and creates an internal `Indexer`.

### Supported Operations

```python
metadata = loader.metadata

array = loader.load_npy("centerbias/visible.npy")
data = loader.load_json("metadata/annotations.json")
df = loader.load_dataframe("data/fixations.csv")
image = loader.load_image("data/images/visible/000001.png")
annotations = loader.load_annotations("metadata/annotations.json")
dataset = loader.load_dataset(df)
```

`load_image()` preserves single-channel images and converts OpenCV BGR/BGRA images to RGB/RGBA.

`load_dataset()` converts a flat fixation DataFrame into the typed scene-level `Dataset` structure used by the saliency and analysis modules.

The input DataFrame must contain at least:

```text
image_stem
participant_id
eye
start
end
duration
x
y
screen_width
screen_height
stimuli_width
stimuli_height
image_width
image_height
fixation_id or id
```

## Metadata and Annotation Indexing

`Indexer` provides consistent access to image metadata and annotations.

### Initialization

```python
from helpers.indexer import Indexer

indexer = Indexer(
    metadata=loader.metadata,
    annotations=annotations,
)
```

### Main Operations

```python
image_id = indexer.get_id("000001", "visible")
image_name = indexer.get_name(image_id)
width, height = indexer.get_size(image_id)
modality = indexer.get_modality(image_id)
visibility = indexer.get_visibility(image_id)
metadata = indexer.get_metadata(image_id)
annotation = indexer.get_annotation("infrared", "000001")
```

The indexer validates:

- modalities: `visible` or `infrared`;
- positive numerical image identifiers;
- non-empty numerical image stems;
- duplicate metadata identifiers;
- duplicate annotations for the same scene and modality;
- consistency between annotation identifiers and metadata.

Image stems are normalized before indexing so that equivalent numerical representations can be matched consistently.

## Visualization

`Visualizer` provides interactive tools for inspecting fixation and saliency results.

### Initialization

```python
from helpers.visualizer import Visualizer

visualizer = Visualizer(
    visible_dir="data/images/visible",
    infrared_dir="data/images/infrared",
    alpha=0.6,
    show_fixation_ids=True,
    keep_only_inside_image=True,
)
```

### Available Visualizations

#### Fixation Locations

```python
visualizer.show_fixations(
    fixations_df="data/fixations_visible.csv",
    participant_id="001",
    image_id="1",
    modality="visible",
)
```

#### Human Heatmaps

```python
visualizer.show_heatmaps(
    fixations_df="data/fixations_visible.csv",
    modality="visible",
    sigma=35.0,
)
```

#### RGB–LWIR Comparison

```python
visualizer.compare_modalities_heatmaps(
    visible_fixations_df="data/fixations_visible.csv",
    infrared_fixations_df="data/fixations_infrared.csv",
)
```

#### DeepGaze–Human Comparison

```python
visualizer.compare_deepgaze_human_heatmaps(
    fixations_df="data/fixations_infrared.csv",
    deepgaze_predictor=deepgaze_model,
    modality="infrared",
)
```

The visualizer projects screen coordinates into image space before plotting. Fixations outside the displayed image can be removed through `keep_only_inside_image`.

Interactive controls:

```text
Right / Down / n  : next image
Left / Up / p     : previous image
s                 : save current figure
q                 : close
```

## Saving DataFrames

`save_dataframe()` writes a DataFrame as CSV and creates the output directory when needed.

```python
from helpers.utils import save_dataframe

save_dataframe(
    dataframe=df,
    output_dir="outputs/clean",
    filename="fixations",
)
```

This produces:

```text
outputs/clean/fixations.csv
```

## Main Dependencies

- NumPy;
- pandas;
- OpenCV;
- Matplotlib;
- SciPy;
- local `saliency` modules for DeepGaze visualizations.

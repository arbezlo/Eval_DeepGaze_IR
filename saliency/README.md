# Saliency Map Construction

This directory contains the tools used to build human fixation maps, estimate center-bias priors, and generate DeepGaze IIE predictions.

## Structure

```text
saliency/
├── builder.py
├── centerbias.py
├── deepgaze.py
└── README.md
```

| Module | Purpose |
|---|---|
| `builder.py` | Builds individual and aggregated fixation maps and converts them into normalized saliency distributions. |
| `centerbias.py` | Estimates center-bias priors from fixation data. |
| `deepgaze.py` | Wraps the pretrained DeepGaze IIE model for RGB and LWIR inference. |

## Human Fixation Maps

`Builder` projects gaze coordinates from screen space to the original image space.

```python
from saliency.builder import Builder
from helpers.loader import Loader

builder = Builder()
loader = Loader("metadata.json")

df = loader.load_dataframe("clean_fixations.csv")
dataset = loader.load_dataset(df)

fixation_map = builder.build_fixation_map(
    data=dataset,
    stem="123",
    participant_id=None,
)
```

When `participant_id=None`, fixations from all participants are aggregated. Fixations outside the displayed stimulus are ignored.

The stimulus is assumed to be centered on the screen. Coordinates are converted as follows:

```text
offset_x = (screen_width - stimuli_width) / 2
offset_y = (screen_height - stimuli_height) / 2

x_stimulus = x_screen - offset_x
y_stimulus = y_screen - offset_y
```

The stimulus coordinates are then scaled to the original image dimensions.

### Participant-Level Maps

```python
participant_maps = builder.build_participant_maps(
    data=dataset,
    stem="000001",
)
```

The returned dictionary associates each participant identifier with a fixation-count map.

### Aggregated Map

```python
aggregate_map = builder.aggregate_fixation_maps(
    list(participant_maps.values())
)
```

All maps must have the same shape.

### Human Saliency Distribution

```python
saliency_map = builder.build_saliency_map(
    fixation_map=aggregate_map,
    sigma=20.0,
)
```

A Gaussian filter is applied to the fixation-count map and the result is normalized to sum to one.

`sigma` is expressed in pixels in the original image space. In the analysis pipeline, it is computed from a visual-angle value before calling this method.

## Center-Bias Estimation

`build_centerbias()` estimates a global fixation prior from one or more fixation sources.

Accepted inputs include:

- pandas DataFrames;
- CSV or TSV files;
- Parquet files;
- Pickle files;
- Excel files.

Example:

```python
from saliency.centerbias import build_centerbias

centerbias = build_centerbias(
    fixation_sources="data/fixations_visible.csv",
    output_path="centerbias/visible.npy",
    template_shape=(512, 512),
    sigma=24.0,
)
```

The function:

1. loads and combines the fixation sources;
2. projects screen coordinates into image coordinates;
3. normalizes positions to the range `[0, 1)`;
4. accumulates fixations in a two-dimensional histogram;
5. applies Gaussian smoothing;
6. normalizes the result to sum to one;
7. saves the prior as a NumPy array.

### Optional Weighting

```python
centerbias = build_centerbias(
    fixation_sources=fixations_df,
    output_path="centerbias/weighted.npy",
    weight_col="weight",
)
```

When no weight column is provided, every fixation contributes equally.

### Leave-One-Scene-Out

```python
centerbias = build_centerbias(
    fixation_sources=fixations_df,
    output_path="centerbias/scene_123.npy",
    leave_out_stem="123",
)
```

This excludes all fixations associated with the selected scene before estimating the prior.

## DeepGaze IIE

`DeepGaze` loads the pretrained `DeepGazeIIE` model and returns a normalized fixation-density prediction.

```python
from saliency.deepgaze import DeepGaze

model = DeepGaze(
    centerbias_path="centerbias/visible.npy",
)

prediction = model.predict(image)
```

The model automatically uses CUDA when available and otherwise runs on CPU.

### Input Images

Accepted input shapes are:

```text
(H, W)
(H, W, 1)
(H, W, 3)
```

Single-channel images are min-max normalized to `[0, 255]` and replicated across three channels before inference. Three-channel images are clipped to `[0, 255]`.

The output:

- is a two-dimensional NumPy array;
- has the same spatial dimensions as the input image;
- contains non-negative values;
- is normalized to sum to one.

### Center-Bias Options

A center-bias prior can be loaded from a `.npy` file:

```python
model = DeepGaze(
    centerbias_path="centerbias/infrared.npy",
)
```

When no file is provided, the wrapper creates either:

```python
DeepGaze(centerbias_type="uniform")
DeepGaze(centerbias_type="gaussian")
```

The prior is resized to the input image dimensions and cached by image shape.

## Main Dependencies

- NumPy;
- pandas;
- SciPy;
- OpenCV;
- PyTorch;
- `deepgaze_pytorch`;
- local `helpers` modules.

# Eye-Tracking Data Processing

This directory contains the pipeline used to convert EyeLink recordings exported as ASCII files into structured and cleaned fixation tables.

EDF-to-ASC conversion is not performed here. EyeLink EDF files must first be converted with the SR Research EDF2ASC utility.

## Structure

```text
processing/
├── __init__.py
├── parser.py
├── extractor.py
├── profiler.py
├── cleaner.py
└── README.md
```

| Module | Purpose |
|---|---|
| `parser.py` | Parses EyeLink ASC events and creates trial objects. |
| `extractor.py` | Converts parsed trials into fixation, blink, and trial-summary DataFrames. |
| `profiler.py` | Evaluates trial quality and identifies invalid trials. |
| `cleaner.py` | Removes invalid data and adds stimulus-relative coordinates. |

The pipeline also depends on the local `helpers` package for metadata access, shared data models, indexing, loading, and saving utilities.

## Pipeline

```text
ASC files -> Parser -> RawTrial objects -> Extractor -> Raw fixations, blinks, and trial summaries -> Profiler -> Trial-quality and participant summaries -> Cleaner -> Clean fixation table and image-level summary
```

## Expected ASC Content

The parser expects EyeLink messages compatible with the acquisition script:

```text
MSG <timestamp> DISPLAY_COORDS 0 0 <x_max> <y_max>
MSG <timestamp> Affichage_Image
MSG <timestamp> !V TRIAL_VAR name <image_name>
MSG <timestamp> !V TRIAL_VAR stimulus_width <width>
MSG <timestamp> !V TRIAL_VAR stimulus_height <height>
EFIX <eye> <start> <end> <duration> <x> <y> <pupil>
EBLINK <eye> <start> <end> <duration>
MSG <timestamp> time_out
```

Supported modalities are:

```text
visible
infrared
```

Stimulus metadata must allow the project `Indexer` to recover the image identifier, canonical filename, and original image dimensions.

## Basic Usage

```python
from pathlib import Path

from helpers.loader import Loader
from processing.parser import Parser
from processing.extractor import Extractor
from processing.profiler import Profiler
from processing.cleaner import Cleaner


modality = "visible"
root_dir = Path("data/")

metadata_path = root_dir / "metadata" / "metadata.json"
asc_dir = root_dir / "asc" / modality
output_dir = root_dir / "dataframes" / modality

loader = Loader(metadata_path)
parser = Parser(loader.metadata)
extractor = Extractor(parser)

raw_fixations_df, raw_blinks_df, raw_summary_df = extractor.run(
    asc_dir=asc_dir,
    modality=modality,
    verbose=True,
    save=True,
    output_dir=output_dir / "raw",
)

profiler = Profiler(
    min_fix_duration=80,
    max_blinks_valid=4,
    center_tolerance_deg=2.0,
    n_min_fixation_per_trial=3,
)

trial_quality_df, participant_summary_df, invalid_trials_df = profiler.run(
    summary_df=raw_summary_df,
    fixations_df=raw_fixations_df,
    blinks_df=raw_blinks_df,
    verbose=True,
    save=True,
    output_dir=output_dir / "quality",
)

cleaner = Cleaner(
    min_fix_duration=80,
    keep_only_inside_screen=True,
    remove_initial_fixation=True,
    initial_fixation_id=0,
)

clean_fixations_df, clean_summary_df = cleaner.run(
    raw_fixations_df=raw_fixations_df,
    trial_summary_df=trial_quality_df,
    verbose=True,
    save=True,
    output_dir=output_dir / "clean",
)
```

## Parsing and Extraction

`Parser.parse_file()` identifies trial boundaries, reads screen and stimulus dimensions, and extracts fixations and blinks.

`Extractor.run()` processes every `.asc` file in a directory and returns:

```python
fixations_df, blinks_df, summary_df
```

Each trial receives a key with the following structure:

```text
<image_stem>_<participant_id>_<modality>_<trial_index>
```

The extracted tables contain screen coordinates, stimulus dimensions, original image dimensions, and timestamps relative to stimulus onset.

## Trial Quality

`Profiler` evaluates every trial using the following default criteria:

- fixation duration of at least 80 ms;
- no more than four blinks;
- first fixation within 2 degrees of the screen center;
- at least three valid fixations;
- fixation coordinates inside the screen.

A trial is marked as invalid when at least one configured criterion is not satisfied.

The profiler returns:

```python
trial_quality_df
participant_summary_df
invalid_trials_df
```

The screen geometry used for angular conversion is configurable through:

```text
viewing_distance_cm
screen_width_cm
screen_height_cm
```

## Cleaning

`Cleaner` applies the following operations:

1. removes trials marked as invalid;
2. optionally removes the initial fixation;
3. removes fixations shorter than the configured duration;
4. optionally removes coordinates outside the screen;
5. computes coordinates relative to the displayed stimulus;
6. records whether each fixation falls inside the stimulus.

The stimulus is assumed to be centered on the screen:

```text
stimulus_left = (screen_width - stimuli_width) / 2
stimulus_top  = (screen_height - stimuli_height) / 2

x_stimulus = x - stimulus_left
y_stimulus = y - stimulus_top
```

Fixations outside the stimulus are retained by default and marked with:

```text
is_inside_stimulus = False
```

## Saved Outputs

When `save=True`, the pipeline writes the following logical outputs:

```text
raw/
├── fixations
├── blinks
└── summary

quality/
├── trial_quality
├── invalid_trials
└── participant_summary

clean/
├── fixations
└── summary
```

The final filename convention is defined by `helpers.utils.save_dataframe`.

## Main Dependencies

- pandas;
- tqdm;
- local `helpers` modules.

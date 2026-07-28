# BIDS Formating

This folder contains the minimal tools required to write the DeepGazeIR eye-tracking dataset in a BIDS-like structure from EyeLink ASC files containing event data only.

Parsing, extraction, quality control, cleaning, and saliency-map generation are not reimplemented here. These operations remain handled by the existing `processing` and `saliency` modules.

## Structure

```text
bids/
├── __init__.py
├── events.py
└── writer.py
```

### `events.py`

This file contains only the functions used to convert the DataFrames produced by the processing pipeline into event tables:

- `stimulus_events()`: converts the trial summary into stimulus-presentation events;
- `fixation_events()`: converts raw or cleaned fixations;
- `blink_events()`: converts blink events detected by EyeLink;
- `quality_events()`: associates trial-quality results with stimulus-presentation timings.

Times are converted from milliseconds to seconds. They are expressed relative to the first stimulus of each ASC recording.

### `writer.py`

This file contains the `BIDSWriter` class, which is responsible for creating directories and writing TSV, JSON, ASC, and NumPy files.

Its main methods are:

- `prepare()`: initializes the dataset, writes the general metadata, and copies the stimuli;
- `write_recording()`: writes the events and derivatives for one ASC recording;
- `write_participants()`: creates `participants.tsv` and `participants.json`;
- `write_group_tables()`: writes the quality-control and cleaning summaries;
- `write_saliency_maps()`: saves fixation maps and saliency maps;
- `subject_label()`: converts participant identifiers into labels such as `sub-001`.

## Reused Existing Modules

The complete dataset construction is launched from `main.py`, which directly uses:

```python
from helpers.loader import Loader
from processing.parser import Parser
from processing.extractor import Extractor
from processing.profiler import Profiler
from processing.cleaner import Cleaner
from saliency.builder import Builder as SaliencyBuilder
```

The `bids` folder is therefore limited to formatting and writing the results. It does not duplicate functions that already exist in the project.

## Input Data

The pipeline expects EyeLink ASC files containing events such as:

```text
EFIX
ESACC
EBLINK
MSG
```

The current ASC files were exported without continuous gaze samples. Therefore, the package does not generate the following files:

```text
*_physio.tsv.gz
*_physio.json
*_physioevents.tsv.gz
*_physioevents.json
```

The original ASC files are preserved in:

```text
sourcedata/eyelink/
```

## Generated Structure

```text
bids_dataset/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── task-<task>_events.json
├── README
├── stimuli/
│   ├── visible/
│   └── infrared/
├── sourcedata/
│   └── eyelink/
│       ├── visible/
│       └── infrared/
├── sub-XXX/
│   └── beh/
│       └── sub-XXX_task-<task>_acq-<modality>_run-XX_events.tsv
└── derivatives/
    └── deepgazeir-processing/
        ├── dataset_description.json
        ├── group_desc-participantquality.tsv
        ├── group_desc-invalidtrials.tsv
        ├── group_desc-cleaningsummary.tsv
        ├── saliency/
        │   ├── visible/
        │   └── infrared/
        └── sub-XXX/
            └── beh/
                ├── *_desc-rawfixations_events.tsv
                ├── *_desc-cleanfixations_events.tsv
                ├── *_desc-blinks_events.tsv
                └── *_desc-trialquality_events.tsv
```

## Raw Events

For each recording, the BIDS `events.tsv` file contains the stimulus presentations.

The main columns are:

- `onset`: start of the presentation in seconds;
- `duration`: presentation duration in seconds;
- `trial_type`: event type;
- `stim_file`: relative path to the displayed image;
- `image_id`: image identifier;
- `image_stem`: image filename without its extension;
- `modality`: `visible` or `infrared`;
- `trial_index`: trial number within the recording.

## Derivatives

Processed data are stored in `derivatives/deepgazeir-processing/`:

- raw fixations extracted from EyeLink events;
- fixations cleaned by `Cleaner`;
- blink events;
- trial-level quality measures produced by `Profiler`;
- participant-level quality summaries;
- invalid trials;
- cleaning summaries;
- fixation maps;
- empirical saliency maps.

Maps are stored as NumPy arrays:

```text
*_desc-fixationmap.npy
*_desc-saliencymap.npy
```

The associated `maps.tsv` file contains, among other information, the image identifier, modality, number of fixations, map dimensions, `sigma` value, and generated filenames.

## Usage

The `bids` folder is not executed directly. Dataset construction is launched from the project root:

```bash
python -m bids.main
```

Paths and processing parameters are defined directly at the beginning of `main.py`. No command-line argument parser is used.

Writer initialization:

```python
from bids import BIDSWriter

writer = BIDSWriter(
    output_dir=BIDS_DIR,
    task="freeview",
    dataset_name="DeepGazeIR Eye-Tracking Dataset",
    authors=["Noé Edouard"],
    overwrite=True,
)

writer.prepare(STIMULI_DIRS)
```

Writing one recording after processing:

```python
writer.write_recording(
    asc_path=asc_path,
    modality=modality,
    run=run,
    summary=summary,
    raw_fixations=raw_fixations,
    blinks=blinks,
    trial_quality=trial_quality,
    clean_fixations=clean_fixations,
)
```

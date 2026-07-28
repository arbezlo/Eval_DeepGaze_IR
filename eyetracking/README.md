# Eye-Tracking Experiment

This directory contains the PsychoPy and EyeLink acquisition script used for the free-viewing RGB and LWIR experiment.

The script assigns each participant to a predefined stimulus subset, presents the corresponding images in a reproducible random order, records gaze data, and stores one EDF file per session.

## Structure

```text
eyetracking/
├── main.py
├── README.md
├── src/
│   ├── EyeLinkCoreGraphicsPsychoPy.py
│   └── fixTarget.bmp
├── stimuli/
│   ├── visible/
│   └── infrared/
├── tables/
│   ├── participants.csv
│   └── metadata.csv
└── results/
    ├── visible/
    └── infrared/
```

## Main Features

- participant-specific subset selection;
- support for RGB and LWIR stimuli;
- deterministic stimulus shuffling for each participant;
- fixed three-second image presentation;
- random central fixation period between 0.8 and 1.2 seconds;
- aspect-ratio-preserving image resizing;
- automatic screen-resolution detection;
- initial calibration and periodic recalibration;
- parser-compatible EyeLink messages;
- automatic EDF transfer to a modality-specific result directory;
- optional dummy and test modes.

## Requirements

- Python;
- PsychoPy;
- pandas;
- Pillow;
- PyLink and the EyeLink Developers Kit;
- `src/EyeLinkCoreGraphicsPsychoPy.py`.

## Configuration

The main parameters are defined at the beginning of `main.py`.

```python
DUMMY_MODE = True
FULL_SCREEN = True
TEST_MODE = False

STIMULUS_MAX_SIZE = (1200, 800)

IMG_TIME = 3.0
FIXATION_MIN_TIME = 0.8
FIXATION_MAX_TIME = 1.2

CENTRAL_RECALIBRATION_EVERY = 25
FULL_RECALIBRATION_EVERY = 50
```

For a real acquisition session:

```python
DUMMY_MODE = False
FULL_SCREEN = True
TEST_MODE = False
```

The monitor geometry must match the experimental setup:

```python
MONITOR_WIDTH_CM = 53.5
MONITOR_DISTANCE_CM = 70.0
```

The screen resolution is detected automatically at runtime.

## Input Tables

### `participants.csv`

This table assigns each participant to a stimulus subset.

Required columns:

| Column | Description |
|---|---|
| `id` | Participant identifier entered at startup. |
| `subset` | Assigned subset, such as `A`, `B`, or `C`. |

Example:

```csv
id,subset
1,A
2,B
3,C
```

### `metadata.csv`

This table describes the stimuli assigned to each subset.

Required columns:

| Column | Description |
|---|---|
| `subset` | Stimulus subset. |
| `modality` | RGB/visible or IR/infrared modality. |
| `image` | Image filename used when no alternative display column is provided. |


Each subset must contain a single modality. Stimulus files must be stored in:

```text
stimuli/visible/
stimuli/infrared/
```

## Running the Experiment

From the `eyetracking` directory:

```bash
python main.py
```

Execution order:

1. enter the participant identifier;
2. load the corresponding subset;
3. determine the subset modality;
4. validate and load the stimulus files;
5. shuffle trials reproducibly;
6. initialize PsychoPy and EyeLink;
7. perform the initial calibration;
8. present all trials;
9. transfer the EDF file to the local results directory.

## Trial Procedure

For each trial:

1. the stimulus is resized without changing its aspect ratio;
2. EyeLink recording starts;
3. a central fixation cross is displayed;
4. the image is shown for three seconds;
5. trial and stimulus metadata are written to the EDF file;
6. recording stops and the trial result is stored.

A central drift-correction block is scheduled every 25 images, and a full recalibration every 50 images.

## EyeLink Messages

The acquisition script records the messages required by the processing pipeline, including:

```text
DISPLAY_COORDS
TRIALID
NUM_IMAGE
Affichage_Image
!V TRIAL_VAR name
!V TRIAL_VAR modality
!V TRIAL_VAR stimulus_width
!V TRIAL_VAR stimulus_height
image_onset
time_out
blank_screen
TRIAL_RESULT
```

These messages allow the processing pipeline to recover trial boundaries, stimulus names, display dimensions, and gaze coordinates.

## Output

Each session is stored under:

```text
results/<modality>/<participant>_<date>_<time>/
└── ID-<participant>-<date>_<time>.EDF
```

For example:

```text
results/infrared/12_2026_07_27_14_30/
└── ID-012-2026_07_27_14_30.EDF
```

When `TEST_MODE=True`, the script uses:

```text
test/tables/
test/stimuli/
results/test/
```

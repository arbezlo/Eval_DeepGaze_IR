from __future__ import annotations

import pandas as pd


def stimulus_events(summary: pd.DataFrame, start_ms: int) -> pd.DataFrame:
    """Convert the trial summary produced by Extractor into BIDS events."""
    events = pd.DataFrame()
    events["onset"] = ((summary["image_onset"] - start_ms) / 1000).round(6)
    events["duration"] = (summary["image_duration"] / 1000).round(6)
    events["trial_type"] = "stimulus"
    events["stim_file"] = summary.apply(
        lambda row: f"stimuli/{row['image_modality']}/{row['image_name']}",
        axis=1,
    )
    events["image_id"] = summary["image_id"].astype(int)
    events["image_stem"] = summary["image_stem"].astype(str)
    events["modality"] = summary["image_modality"].astype(str)
    events["trial_index"] = summary["trial_index"].astype(int) + 1
    return events.sort_values("onset").reset_index(drop=True)


def fixation_events(
    fixations: pd.DataFrame,
    start_ms: int,
    trial_type: str,
) -> pd.DataFrame:
    """Convert raw or cleaned fixations into an event table."""
    if fixations.empty:
        return pd.DataFrame(
            columns=["onset", "duration", "trial_type", "eye", "x", "y"]
        )

    events = pd.DataFrame()
    events["onset"] = ((fixations["start"] - start_ms) / 1000).round(6)
    events["duration"] = (fixations["duration"] / 1000).round(6)
    events["trial_type"] = trial_type

    columns = [
        "trial_key",
        "trial_index",
        "participant_id",
        "image_modality",
        "image_name",
        "image_stem",
        "image_id",
        "fixation_id",
        "eye",
        "x",
        "y",
        "relative_start",
        "relative_end",
        "screen_width",
        "screen_height",
        "stimuli_width",
        "stimuli_height",
        "image_width",
        "image_height",
        "is_inside_screen",
        "x_stimulus",
        "y_stimulus",
        "is_inside_stimulus",
    ]

    for column in columns:
        if column in fixations.columns:
            events[column] = fixations[column].to_numpy()

    return events.sort_values(["onset", "fixation_id"]).reset_index(drop=True)


def blink_events(blinks: pd.DataFrame, start_ms: int) -> pd.DataFrame:
    """Convert EyeLink blink events into an event table."""
    if blinks.empty:
        return pd.DataFrame(
            columns=["onset", "duration", "trial_type", "eye"]
        )

    events = pd.DataFrame()
    events["onset"] = ((blinks["start"] - start_ms) / 1000).round(6)
    events["duration"] = (blinks["duration"] / 1000).round(6)
    events["trial_type"] = "blink"

    for column in [
        "trial_key",
        "trial_index",
        "participant_id",
        "image_modality",
        "image_name",
        "image_stem",
        "image_id",
        "blink_id",
        "eye",
        "relative_start",
        "relative_end",
    ]:
        if column in blinks.columns:
            events[column] = blinks[column].to_numpy()

    return events.sort_values("onset").reset_index(drop=True)


def quality_events(
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    start_ms: int,
) -> pd.DataFrame:
    """Convert Profiler output into events using Extractor trial timings."""
    if quality.empty:
        return pd.DataFrame(
            columns=["onset", "duration", "trial_type", "is_invalid_trial"]
        )

    timings = summary[["trial_key", "image_onset", "image_duration"]]
    events = quality.merge(timings, on="trial_key", how="left")
    events.insert(0, "trial_type", "trial_quality")
    events.insert(0, "duration", (events.pop("image_duration") / 1000).round(6))
    events.insert(0, "onset", ((events.pop("image_onset") - start_ms) / 1000).round(6))
    return events.sort_values("onset").reset_index(drop=True)

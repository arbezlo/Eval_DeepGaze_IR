from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from bids.writer import BIDSWriter
from helpers.loader import Loader
from processing.cleaner import Cleaner
from processing.extractor import Extractor
from processing.parser import Parser
from processing.profiler import Profiler
from saliency.builder import Builder as SaliencyBuilder


# -----------------------------------------------------------------------------
# Paths to adapt to the project
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "2026" / "global"

METADATA_PATH = DATA_DIR / "metadata" / "metadata.json"
PARTICIPANTS_PATH = DATA_DIR / "tables" / "participants.csv"
BIDS_DIR = PROJECT_ROOT / "data" / "2026" / "bids"

ASC_DIRS = {
    "visible": DATA_DIR / "asc" / "visible",
    "infrared": DATA_DIR / "asc" / "infrared",
}

STIMULI_DIRS = {
    "visible": DATA_DIR / "images" / "visible",
    "infrared": DATA_DIR / "images" / "infrared",
}

TASK = "freeview"
OVERWRITE = True
SALIENCY_SIGMA = 20.0

MIN_FIXATION_DURATION = 80
MAX_BLINKS = 4
CENTER_TOLERANCE_DEG = 2.0
MIN_FIXATIONS_PER_TRIAL = 3
VIEWING_DISTANCE_CM = 70.0
SCREEN_WIDTH_CM = 53.5


# -----------------------------------------------------------------------------
# Processing of one ASC recording
# -----------------------------------------------------------------------------


def process_recording(
    asc_path: Path,
    participant_id: str,
    modality: str,
    parser: Parser,
    extractor: Extractor,
    cleaner: Cleaner,
) -> dict[str, pd.DataFrame]:
    trials = parser.parse_file(
        filepath=asc_path,
        participant_id=participant_id,
        modality=modality,
    )
    if not trials:
        raise ValueError(f"No trial found in {asc_path}")

    summary = extractor.extract_summary(trials)
    raw_fixations = extractor.extract_fixations(trials)
    blinks = extractor.extract_blinks(trials)

    screen_width = int(summary["screen_width"].iloc[0])
    screen_height = int(summary["screen_height"].iloc[0])
    screen_height_cm = SCREEN_WIDTH_CM * screen_height / screen_width

    profiler = Profiler(
        min_fix_duration=MIN_FIXATION_DURATION,
        max_blinks_valid=MAX_BLINKS,
        center_tolerance_deg=CENTER_TOLERANCE_DEG,
        n_min_fixation_per_trial=MIN_FIXATIONS_PER_TRIAL,
        viewing_distance_cm=VIEWING_DISTANCE_CM,
        screen_width_cm=SCREEN_WIDTH_CM,
        screen_height_cm=screen_height_cm,
    )
    trial_quality, participant_quality, invalid_trials = profiler.run(
        summary_df=summary,
        fixations_df=raw_fixations,
        blinks_df=blinks,
    )
    clean_fixations, cleaning_summary = cleaner.run(
        raw_fixations_df=raw_fixations,
        trial_summary_df=trial_quality,
    )

    return {
        "summary": summary,
        "raw_fixations": raw_fixations,
        "blinks": blinks,
        "trial_quality": trial_quality,
        "participant_quality": participant_quality,
        "invalid_trials": invalid_trials,
        "clean_fixations": clean_fixations,
        "cleaning_summary": cleaning_summary,
    }


# -----------------------------------------------------------------------------
# Complete construction
# -----------------------------------------------------------------------------


def main() -> None:
    loader = Loader(METADATA_PATH)
    parser = Parser(loader.metadata)
    extractor = Extractor(parser)
    cleaner = Cleaner(
        min_fix_duration=MIN_FIXATION_DURATION,
        keep_only_inside_screen=True,
        remove_initial_fixation=True,
        initial_fixation_id=0,
    )
    saliency_builder = SaliencyBuilder()

    writer = BIDSWriter(
        output_dir=BIDS_DIR,
        task=TASK,
        dataset_name="DeepGazeIR Eye-Tracking Dataset",
        authors=["Noé Edouard", "Jocelyn Chanussot", "Nathalie Guyader", "Ronald Phlypo", "Loïc Arbez"],
        overwrite=OVERWRITE,
    )
    writer.prepare(STIMULI_DIRS)

    participants: set[str] = set()
    run_numbers: defaultdict[tuple[str, str], int] = defaultdict(int)
    clean_by_modality: defaultdict[str, list[pd.DataFrame]] = defaultdict(list)
    participant_quality: list[pd.DataFrame] = []
    invalid_trials: list[pd.DataFrame] = []
    cleaning_summary: list[pd.DataFrame] = []
    n_recordings = 0

    for modality, asc_dir in ASC_DIRS.items():
        if not asc_dir.exists():
            print(f"Skipped missing directory: {asc_dir}")
            continue

        for asc_path in sorted(asc_dir.rglob("*.asc")):
            participant_id = extractor._get_participant_id(asc_path.name)
            run_numbers[(participant_id, modality)] += 1
            run = run_numbers[(participant_id, modality)]

            data = process_recording(
                asc_path=asc_path,
                participant_id=participant_id,
                modality=modality,
                parser=parser,
                extractor=extractor,
                cleaner=cleaner,
            )

            writer.write_recording(
                asc_path=asc_path,
                modality=modality,
                run=run,
                summary=data["summary"],
                raw_fixations=data["raw_fixations"],
                blinks=data["blinks"],
                trial_quality=data["trial_quality"],
                clean_fixations=data["clean_fixations"],
            )

            participants.add(participant_id)
            clean_by_modality[modality].append(data["clean_fixations"])
            participant_quality.append(data["participant_quality"])
            invalid_trials.append(data["invalid_trials"])
            cleaning_summary.append(data["cleaning_summary"])
            n_recordings += 1

    if n_recordings == 0:
        raise RuntimeError("No ASC file was processed.")

    writer.write_participants(participants, PARTICIPANTS_PATH)
    writer.write_group_tables(
        participant_quality=participant_quality,
        invalid_trials=invalid_trials,
        cleaning_summary=cleaning_summary,
    )

    for modality, frames in clean_by_modality.items():
        non_empty = [frame for frame in frames if not frame.empty]
        if not non_empty:
            continue
        dataset = loader.load_dataset(pd.concat(non_empty, ignore_index=True))
        writer.write_saliency_maps(
            modality=modality,
            dataset=dataset,
            saliency_builder=saliency_builder,
            sigma=SALIENCY_SIGMA,
        )

    print(f"BIDS dataset created in: {BIDS_DIR}")
    print(f"ASC recordings processed: {n_recordings}")
    print(f"Participants: {len(participants)}")


if __name__ == "__main__":
    main()

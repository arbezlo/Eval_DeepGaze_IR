from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from bids.events import blink_events, fixation_events, quality_events, stimulus_events


class BIDSWriter:
    """Write the BIDS files without reimplementing the processing pipeline."""

    def __init__(
        self,
        output_dir: str | Path,
        task: str = "freeview",
        dataset_name: str = "DeepGazeIR Eye-Tracking Dataset",
        authors: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task = task
        self.dataset_name = dataset_name
        self.authors = authors or []
        self.overwrite = overwrite
        self.derivative_dir = self.output_dir / "derivatives" / "deepgazeir-processing"

    def prepare(self, stimuli_dirs: dict[str, Path]) -> None:
        if self.overwrite and self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink(missing_ok=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            self.output_dir / "dataset_description.json",
            {
                "Name": self.dataset_name,
                "BIDSVersion": "1.11.1",
                "DatasetType": "raw",
                "Authors": self.authors,
            },
        )
        self._write_text(
            self.output_dir / "README",
            "Event-only EyeLink dataset. The source ASC files contain EyeLink "
            "events but no continuous samples, therefore no physio files are generated.\n",
        )
        self._write_json(
            self.output_dir / f"task-{self.task}_events.json",
            {
                "onset": {
                    "Description": "Stimulus onset relative to the first stimulus of the ASC recording.",
                    "Units": "s",
                },
                "duration": {"Description": "Stimulus duration.", "Units": "s"},
                "trial_type": {"Description": "Type of event."},
                "stim_file": {"Description": "Path to the displayed stimulus."},
                "modality": {"Description": "Visible or infrared image modality."},
            },
        )

        self._write_json(
            self.derivative_dir / "dataset_description.json",
            {
                "Name": "DeepGazeIR eye-tracking processing derivatives",
                "BIDSVersion": "1.11.1",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "DeepGazeIR processing pipeline",
                        "Description": "Parser, Extractor, Profiler, Cleaner and saliency.Builder.",
                    }
                ],
            },
        )

        for modality, source in stimuli_dirs.items():
            source = Path(source)
            if source.exists():
                shutil.copytree(
                    source,
                    self.output_dir / "stimuli" / modality,
                    dirs_exist_ok=True,
                )

    def write_recording(
        self,
        asc_path: Path,
        modality: str,
        run: int,
        summary: pd.DataFrame,
        raw_fixations: pd.DataFrame,
        blinks: pd.DataFrame,
        trial_quality: pd.DataFrame,
        clean_fixations: pd.DataFrame,
    ) -> None:
        participant = str(summary["participant_id"].iloc[0])
        subject = self.subject_label(participant)
        basename = (
            f"{subject}_task-{self.task}_acq-{modality}_run-{run:02d}"
        )
        start_ms = int(summary["image_onset"].min())

        raw_dir = self.output_dir / subject / "beh"
        self._write_tsv(
            raw_dir / f"{basename}_events.tsv",
            stimulus_events(summary, start_ms),
        )

        derivative_dir = self.derivative_dir / subject / "beh"
        tables = {
            "rawfixations": fixation_events(
                raw_fixations, start_ms, "raw_fixation"
            ),
            "cleanfixations": fixation_events(
                clean_fixations, start_ms, "clean_fixation"
            ),
            "blinks": blink_events(blinks, start_ms),
            "trialquality": quality_events(trial_quality, summary, start_ms),
        }

        for description, table in tables.items():
            if "participant_id" in table.columns:
                table["participant_id"] = subject
            stem = f"{basename}_desc-{description}_events"
            self._write_tsv(derivative_dir / f"{stem}.tsv", table)
            self._write_json(
                derivative_dir / f"{stem}.json",
                {
                    "onset": {
                        "Description": "Event onset relative to the first stimulus of the ASC recording.",
                        "Units": "s",
                    },
                    "duration": {"Description": "Event duration.", "Units": "s"},
                    "trial_type": {"Description": "Type of derived event."},
                },
            )

        source_dir = self.output_dir / "sourcedata" / "eyelink" / modality
        source_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asc_path, source_dir / f"{basename}_eyelink.asc")

    def write_participants(
        self,
        participants: set[str],
        source_path: str | Path | None = None,
    ) -> None:
        observed = pd.DataFrame(
            {"participant_id": sorted(self.subject_label(p) for p in participants)}
        )

        if source_path is not None and Path(source_path).exists():
            source = self._read_table(Path(source_path))
            id_column = next(
                (column for column in ["participant_id", "id"] if column in source.columns),
                None,
            )
            if id_column is not None:
                source = source.copy()
                source["participant_id"] = source[id_column].map(self.subject_label)
                source = source.drop(columns=[id_column], errors="ignore")
                source = source.drop_duplicates("participant_id")
                observed = observed.merge(source, on="participant_id", how="left")

        self._write_tsv(self.output_dir / "participants.tsv", observed)
        self._write_json(
            self.output_dir / "participants.json",
            {
                column: {"Description": column.replace("_", " ").capitalize()}
                for column in observed.columns
            },
        )

    def write_group_tables(
        self,
        participant_quality: list[pd.DataFrame],
        invalid_trials: list[pd.DataFrame],
        cleaning_summary: list[pd.DataFrame],
    ) -> None:
        tables = {
            "group_desc-participantquality.tsv": participant_quality,
            "group_desc-invalidtrials.tsv": invalid_trials,
            "group_desc-cleaningsummary.tsv": cleaning_summary,
        }

        for filename, frames in tables.items():
            non_empty = [frame for frame in frames if not frame.empty]
            if not non_empty:
                continue
            table = pd.concat(non_empty, ignore_index=True)
            if "participant_id" in table.columns:
                table["participant_id"] = table["participant_id"].map(self.subject_label)
            self._write_tsv(self.derivative_dir / filename, table)

    def write_saliency_maps(
        self,
        modality: str,
        dataset,
        saliency_builder,
        sigma: float = 20.0,
    ) -> None:
        target = self.derivative_dir / "saliency" / modality
        target.mkdir(parents=True, exist_ok=True)
        rows = []

        for stem in sorted(dataset):
            fixation_map = saliency_builder.build_fixation_map(dataset, stem)
            saliency_map = saliency_builder.build_saliency_map(
                fixation_map,
                sigma=sigma,
            )

            fixation_name = f"{stem}_desc-fixationmap.npy"
            saliency_name = f"{stem}_desc-saliencymap.npy"
            np.save(target / fixation_name, fixation_map)
            np.save(target / saliency_name, saliency_map)

            rows.append(
                {
                    "image_stem": stem,
                    "modality": modality,
                    "n_fixations": int(fixation_map.sum()),
                    "height": int(fixation_map.shape[0]),
                    "width": int(fixation_map.shape[1]),
                    "sigma": float(sigma),
                    "fixation_map": fixation_name,
                    "saliency_map": saliency_name,
                }
            )

        self._write_tsv(target / "maps.tsv", pd.DataFrame(rows))

    @staticmethod
    def subject_label(value: str | int) -> str:
        value = str(value).strip()
        if value.startswith("sub-"):
            value = value[4:]
        if value.isdigit():
            value = f"{int(value):03d}"
        else:
            value = re.sub(r"[^A-Za-z0-9]", "", value)
        if not value:
            raise ValueError("Invalid participant identifier.")
        return f"sub-{value}"

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        table = pd.read_csv(path, sep=None, engine="python")
        table = table.dropna(axis=1, how="all")
        table.columns = [str(column).strip() for column in table.columns]
        return table

    @staticmethod
    def _write_tsv(path: Path, table: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False, na_rep="n/a")

    @staticmethod
    def _write_json(path: Path, content: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(content, file, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

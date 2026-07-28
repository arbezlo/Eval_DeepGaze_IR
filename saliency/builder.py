from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from helpers.models import Dataset, FixationData


class Builder:

    def build_fixation_map(
        self,
        data: Dataset,
        stem: str,
        participant_id: str | None = None,
    ) -> np.ndarray:
        """
        Construit une carte de fixation pour un StemData donné.

        Si participant_id est fourni, la carte contient uniquement les
        fixations de ce participant. Sinon la carte contient les fixations
        agrégées de tous les participants pour ce stem.
        """
        stem_data = data[stem]
        
        image_width  = int(stem_data.dimensions_data.image_width)
        image_height = int(stem_data.dimensions_data.image_height)

        fixation_map = np.zeros((image_height, image_width), dtype=np.uint32)
        fixations: list[FixationData] = []
        
        if participant_id is not None:
            fixations = stem_data.participants_data.get(participant_id, [])
        else:
            for participant_fixations in stem_data.participants_data.values():
                fixations.extend(participant_fixations)

        for fixation in fixations:
            coords = self._get_image_coordinates(
                x_screen=float(fixation.x),
                y_screen=float(fixation.y),
                screen_width=int(stem_data.dimensions_data.screen_width),
                screen_height=int(stem_data.dimensions_data.screen_height),
                stimuli_width=int(stem_data.dimensions_data.stimuli_width),
                stimuli_height=int(stem_data.dimensions_data.stimuli_height),
                image_width=image_width,
                image_height=image_height,
            )

            if coords is None:
                continue

            x_img, y_img = coords
            fixation_map[y_img, x_img] += 1

        return fixation_map
    
    def build_participant_maps(
        self,
        data: Dataset,
        stem: str | int,
    ) -> dict[str, np.ndarray]:
        stem_data = data[stem]

        return {
            participant_id: self.build_fixation_map(
                data=data,
                stem=stem,
                participant_id=participant_id,
            )
            for participant_id in stem_data.participants_data
        }
    

    @staticmethod
    def aggregate_fixation_maps(participant_maps: list[np.ndarray]) -> np.ndarray:
        if not participant_maps:
            raise ValueError("participant_maps is empty")

        base_shape = participant_maps[0].shape
        out = np.zeros(base_shape, dtype=np.uint32)

        for pmap in participant_maps:
            if pmap.shape != base_shape:
                raise ValueError(f"Inconsistent map shape: {pmap.shape} vs {base_shape}")
            out += pmap.astype(np.uint32)

        return out

    @staticmethod
    def build_saliency_map(fixation_map: np.ndarray, sigma: float = 20.0) -> np.ndarray:
        """
        Produit une carte de saillance probabiliste.
        Contrairement à une normalisation par max, la somme vaut 1, ce qui est
        préférable pour comparer des distributions RGB/IR avec KL, CC, SIM, etc.
        """
        saliency = gaussian_filter(fixation_map.astype(np.float32), sigma=float(sigma))
        total = float(saliency.sum())
        if total > 0:
            saliency /= total
        return saliency
    
    def row_selector(
        self,
        fixations_df: pd.DataFrame,
        modality: str,
        stem: str,
        participant_id: str | int | None = None,
    ) -> pd.DataFrame:
        """
        Sélectionne les fixations correspondant à une modalité, une image
        et éventuellement un participant.

        Si participant_id est None, les lignes de tous les participants
        sont conservées.
        """
        modality = str(modality)
        stem = str(stem)

        mask = (
            fixations_df["image_modality"].astype(str).eq(modality)
            & fixations_df["image_stem"].astype(str).eq(stem)
        )

        selected_rows = fixations_df.loc[mask]

        if participant_id is not None:
            participant_id = str(participant_id)

            selected_rows = selected_rows.loc[
                selected_rows["participant_id"]
                .astype(str)
                .eq(participant_id)
            ]
            
        if selected_rows.empty:
            participant_label = (
                "all participants"
                if participant_id is None
                else str(participant_id)
            )

            raise ValueError(
                "No fixation found for "
                f"modality={modality}, "
                f"stem={stem}, "
                f"participant={participant_label}"
            )

        return selected_rows

    @staticmethod
    def _get_image_coordinates(
        x_screen: float,
        y_screen: float,
        screen_width: int,
        screen_height: int,
        stimuli_width: int,
        stimuli_height: int,
        image_width: int,
        image_height: int,
    ) -> Optional[tuple[int, int]]:
        if stimuli_width <= 0 or stimuli_height <= 0:
            return None
        if image_width <= 0 or image_height <= 0:
            return None

        offset_x = (screen_width - stimuli_width) / 2.0
        offset_y = (screen_height - stimuli_height) / 2.0

        x_stim = x_screen - offset_x
        y_stim = y_screen - offset_y

        if not (0 <= x_stim < stimuli_width and 0 <= y_stim < stimuli_height):
            return None

        x_img = int(np.floor(x_stim * image_width / float(stimuli_width)))
        y_img = int(np.floor(y_stim * image_height / float(stimuli_height)))

        x_img = min(max(x_img, 0), image_width - 1)
        y_img = min(max(y_img, 0), image_height - 1)

        return x_img, y_img

    
    @staticmethod
    def _find_image(images_dir: Path, image_name: str, modality: str | None = None) -> Optional[Path]:
        candidates = [images_dir / image_name]

        if modality is not None:
            candidates.append(images_dir / str(modality) / image_name)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        matches = list(images_dir.rglob(image_name))
        if matches:
            return matches[0]

        return None


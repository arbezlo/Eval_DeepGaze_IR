from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

from helpers.indexer import Metadata, Indexer
from helpers.models import Annotations, RoiBox, ImageData, RoiData, FixationData, Dataset, DimensionData, TrialData


class Loader:

    def __init__(
        self,
        metadata_path: str | Path,
    ) -> None:
        self.metadata = self.load_metadata(metadata_path)
        self.index = Indexer(self.metadata)
    
    @staticmethod
    def load_npy(filepath: str | Path) -> np.ndarray:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Missing numpy file: {filepath}")
        return np.load(filepath)
    
    
    @staticmethod
    def load_json(filepath: str | Path) -> Dict:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Missing json file: {filepath}")
        with filepath.open("r", encoding="utf-8") as file:
            return json.load(file)
        
        
    @staticmethod
    def load_dataframe(filepath: str | Path) -> pd.DataFrame:
        return pd.read_csv(filepath)


    @staticmethod  
    def load_image(image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return image


    @staticmethod
    def to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
        """
        Grayscale version of an RGB(A) image from load_image, as an (H, W, 3) uint8 image: converted
        with cv2.COLOR_RGB2GRAY and the gray channel repeated 3 times. Three channels, so that
        DeepGaze sees a colour-less RGB image rather than an IR image (DeepGaze._normalize_input
        stretches the contrast of single-channel inputs). A single-channel image is only repeated.
        """
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[..., 0]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(np.repeat(image[..., None], 3, axis=2))
    
    def load_metadata(self, json_path: str | Path) -> Metadata:
        path = Path(json_path)

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Metadata JSON must be a list of objects")

        metadata: Metadata = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Entry at index {i} must be an object")

            try:
                modality = str(item["modality"]).lower().strip()
                if modality not in ("infrared", "visible"):
                    raise ValueError(f"Invalid modality at index {i}: {modality}")

                stem = str(item["stem"]).strip()
                if not stem:
                    raise ValueError(f"Empty stem at index {i}")
                
                image_data = ImageData(
                    id=int(item["id"]),
                    stem=stem,
                    name=str(item["name"]).strip(),
                    extension=str(item["extension"]).strip().lower(),
                    modality=modality,
                    height=int(item["height"]),
                    width=int(item["width"]),
                    daynight=str(item.get("time_of_day", "unknown"))
                )

            except KeyError as e:
                raise ValueError(f"Missing field {e} at index {i}") from e
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid value at index {i}: {item}") from e

            metadata.append(image_data)

        return metadata

    def load_annotations(self, annotations_path: str | Path) -> Annotations:
        
        annotations_dict: dict[str, dict[str, list]] = self.load_json(annotations_path)
        annotations: Annotations = []
        
        for modality, stems in annotations_dict.items():
            modality = self.index.normalize_modality(modality)
            
            for stem, rois in stems.items():
                stem = self.index.normalize_stem(stem)
                rois_list = []
                for roi in rois:
                    
                    x, y, width, height = roi["bbox"]
                  
                    rois_list.append(RoiBox(
                        class_id=int(roi["category_id"]),
                        x_min=int(round(float(x))),
                        y_min=int(round(float(y))),
                        x_max=int(round((float(x) + float(width)))),
                        y_max=int(round((float(y) + float(height)))),
                    ))
                    
                rois_data = RoiData(
                    image_id=self.index.get_id(stem, modality),
                    image_stem=stem,
                    image_modality=modality,
                    rois=rois_list,
                )  
                
                annotations.append(rois_data)
        
        return annotations
    
    
    def load_dataset(self, df: pd.DataFrame) -> Dataset:
        """
        Convertit un DataFrame d'oculométrie plat en une structure typée Dataset (Dict[str, TrialData]).
        
        Cette méthode regroupe les données par image ('image_stem') et organise les fixations
        par participant sous forme de dataclasses immuables et performantes.
        """
        # 1. Validation des colonnes requises
        required_cols = {
            "image_stem", "participant_id", "eye", "start", "end", "duration", "x", "y",
            "screen_width", "screen_height", "stimuli_width", "stimuli_height", "image_width", "image_height"
        }
        
        # Gestion flexible du nom de l'ID de fixation (fixation_id ou id)
        fix_id_col = "fixation_id" if "fixation_id" in df.columns else "id"
        required_cols.add(fix_id_col)
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in DataFrame to build dataset: {missing}")

        dataset: Dataset = {}

        # 2. Groupement par image pour instancier un TrialData par image_stem
        for raw_stem, group in df.groupby("image_stem"):
            stem = self.index.normalize_stem(str(raw_stem))
            first_row = group.iloc[0]
            
            # Extraction des configurations spatiales constantes du stimulus
            dimensions_data = DimensionData(
                screen_width=int(first_row["screen_width"]),
                screen_height=int(first_row["screen_height"]),
                stimuli_width=int(first_row["stimuli_width"]),
                stimuli_height=int(first_row["stimuli_height"]),
                image_width=int(first_row["image_width"]),
                image_height=int(first_row["image_height"])
            )
            
            # 3. Association avec l'objet ImageData existant dans self.metadata
            modality = str(first_row.get("image_modality", "visible")).lower().strip()
            image_data = None
            
            for meta in self.metadata:
                if meta.stem == stem and meta.modality == modality:
                    image_data = meta
                    break
            
            # Remplacement dynamique si l'image n'était pas déclarée dans le JSON de métadonnées
            if image_data is None:
                img_name = str(first_row.get("image_name", f"{stem}.png"))
                daynight = str(first_row.get("time_of_day", "unknown"))
                ext = Path(img_name).suffix.lower().replace(".", "")
                image_data = ImageData(
                    id=int(first_row.get("image_id", 0)),
                    stem=stem,
                    name=img_name,
                    extension=ext,
                    modality=modality,
                    height=dimensions_data.image_height,
                    width=dimensions_data.image_width,
                    daynight=daynight,
                )

            # 4. Hydratation des listes de fixations par participant (via un dictionnaire intermédiaire)
            participants_data: Dict[str, list[FixationData]] = defaultdict(list)
            
            # itertuples est utilisé ici pour maximiser les performances de bouclage
            for row in group.itertuples(index=False):
                p_id = str(row.participant_id)
                
                fixation = FixationData(
                    id=int(getattr(row, fix_id_col)),
                    eye=str(row.eye),
                    start=int(row.start),
                    end=int(row.end),
                    duration=int(row.duration),
                    x=float(row.x),
                    y=float(row.y)
                )
                participants_data[p_id].append(fixation)

            # 5. Encapsulation finale
            dataset[stem] = TrialData(
                image_data=image_data,
                dimensions_data=dimensions_data,
                participants_data=dict(participants_data)
            )

        return dataset
  
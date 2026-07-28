from __future__ import annotations
from tqdm import tqdm
from pathlib import Path
from math import tan, radians

import numpy as np
import pandas as pd

from analysis.metrics import Metrics
from saliency.builder import Builder
from saliency.deepgaze import DeepGaze
from helpers.indexer import Indexer
from helpers.loader import Loader
from helpers.models import Dataset, Results


class Comparator:
    
    def __init__(self, 
        metadata_json: str | Path, 
        annotations_json: str | Path, 
        sigma_deg: float = 1.0,
        viewing_distance_cm: float = 70.0,
        screen_width_cm: float = 53.1,
        screen_width_px: int = 1360,
        epsilon: float = 1e-12
    ):
        self.sigma_deg = sigma_deg
        self.viewing_distance_cm = viewing_distance_cm
        self.screen_width_cm = screen_width_cm
        self.screen_width_px = screen_width_px
        self.pixels_per_cm = screen_width_px / screen_width_cm
        self.builder = Builder()
        self.metrics = Metrics(epsilon)
        self.loader = Loader(metadata_json)
        self.index = Indexer(
            metadata=self.loader.metadata,
            annotations=self.loader.load_annotations(annotations_json),
        )
        
    def _compute_sigma(self, sigma_deg: float, stimulus_width_px: int, image_width_px: int) -> float:
            """
            Compute gaussian sigma in the original image space from a visual angle degre.
            """
    
            sigma_cm = 2.0 * self.viewing_distance_cm * tan(radians(sigma_deg) / 2.0)
            sigma_screen_px = sigma_cm * self.pixels_per_cm
            display_to_image_scale = image_width_px / stimulus_width_px
            sigma_image_px = sigma_screen_px * display_to_image_scale
    
            return sigma_image_px
    
    
    def _get_stems(self, df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:

        for index, df in enumerate((df1, df2), start=1):
            if "image_stem" not in df.columns:
                raise ValueError(
                    f"Missing column 'image_stem' in df{index}."
                )

        stems1 = {self.index.normalize_stem(value) for value in df1["image_stem"]}
        stems2 = {self.index.normalize_stem(value) for value in df2["image_stem"]}

        if stems1 != stems2:
            only_in_df1 = sorted(stems1 - stems2)
            only_in_df2 = sorted(stems2 - stems1)

            raise ValueError(
                "The DataFrames do not contain the same image stems. "
                f"Only in df1: {only_in_df1}. "
                f"Only in df2: {only_in_df2}."
            )
        stems = sorted(stems1)
        return stems
    

    def _get_dimensions(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict[str, dict[str, int]]:
        g1 = df1.groupby("image_stem")[["image_width", "stimuli_width"]].first()
        g2 = df2.groupby("image_stem")[["image_width", "stimuli_width"]].first()

        if not g1.index.equals(g2.index):
            raise ValueError("Dataframes stems are different.")

        if not g1.equals(g2):
            raise ValueError("Dimensions in df1 and df2 are different.")

        return {
            self.index.normalize_stem(stem): {
                "image_width": int(row["image_width"]),
                "stimuli_width": int(row["stimuli_width"]),
            }
            for stem, row in g1.iterrows()
        }
    
    
    def _compute_human_scores(
        self,
        fixation_map_1: np.ndarray,
        fixation_map_2: np.ndarray,
        sigma: float,
    ) -> dict[str, dict[str, float]]:
        
        saliency_map_1 = self.builder.build_saliency_map(fixation_map_1, sigma)
        saliency_map_2 = self.builder.build_saliency_map(fixation_map_2, sigma)

        scores = {"similarity": {}, "predictivity": {}}

        for metric_name, metric_function in self.metrics.registry["similarity"].items():
            score_12 = metric_function(saliency_map_1, saliency_map_2)
            score_21 = metric_function(saliency_map_2, saliency_map_1)

            scores["similarity"][metric_name] = float(
                np.nanmean([score_12, score_21])
            )

        for metric_name, metric_function in self.metrics.registry["predictivity"].items():
            score_12 = metric_function(fixation_map_1, saliency_map_2)
            score_21 = metric_function(fixation_map_2, saliency_map_1)

            scores["predictivity"][metric_name] = float(
                np.nanmean([score_12, score_21])
            )

        return scores
    
    def _compute_permutation_scores(
        self,
        data_visible: Dataset,
        data_infrared: Dataset,
        stem: str | int,
        sigma: float,
        n_permutations: int = 1000,
        seed: int | None = 42,
    ) -> dict[str, dict[str, np.ndarray]]:
        
        visible_maps = self.builder.build_participant_maps(
            data_visible,
            stem,
        )
        infrared_maps = self.builder.build_participant_maps(
            data_infrared,
            stem,
        )

        all_maps = (
            list(visible_maps.values())
            + list(infrared_maps.values())
        )

        n_visible = len(visible_maps)
        n_infrared = len(infrared_maps)

        if n_visible == 0 or n_infrared == 0:
            raise ValueError(
                f"No participant data available for stem '{stem}'."
            )

        rng = np.random.default_rng(seed)

        permutation_scores = {
            metric_type: {
                metric_name: np.empty(n_permutations, dtype=float)
                for metric_name in self.metrics.registry[metric_type]
            }
            for metric_type in ("similarity", "predictivity")
        }

        for permutation_index in range(n_permutations):
            indices = rng.permutation(len(all_maps))

            visible_indices = indices[:n_visible]
            infrared_indices = indices[
                n_visible:n_visible + n_infrared
            ]

            fixation_map_group_1 = self.builder.aggregate_fixation_maps(
                [all_maps[index] for index in visible_indices]
            )
            fixation_map_group_2 = self.builder.aggregate_fixation_maps(
                [all_maps[index] for index in infrared_indices]
            )

            scores = self._compute_human_scores(
                fixation_map_group_1,
                fixation_map_group_2,
                sigma
            )

            for metric_type, metric_values in scores.items():
                for metric_name, value in metric_values.items():
                    permutation_scores[metric_type][metric_name][
                        permutation_index
                    ] = value

        return permutation_scores
    
    def _permutation_p_value(
        self,
        observed_score: float,
        permutation_scores: np.ndarray,
        metric_name: str,
    ) -> float:
        valid_scores = permutation_scores[
            np.isfinite(permutation_scores)
        ]

        if valid_scores.size == 0 or not np.isfinite(observed_score):
            return float("nan")

        
        extreme_count = np.sum(valid_scores <= observed_score)

        return float(
            (extreme_count + 1) / (valid_scores.size + 1)
        )

    
    
    def run_expe1(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        n_permutations: int = 1000,
        seed: int | None = 42,
    ) -> Results:
        data_infrared = self.loader.load_dataset(infrared_df)
        data_visible = self.loader.load_dataset(visible_df)

        stems = self._get_stems(infrared_df, visible_df)
        dimensions = self._get_dimensions(infrared_df, visible_df)
        results: Results = {
            metric_type: {
                metric_name: {
                    stem: None
                    for stem in stems
                }
                for metric_name in self.metrics.registry[metric_type]
            }
            for metric_type in ("similarity", "predictivity")
        }

        for stem in tqdm(stems, desc="Running human-human analysis"):
            visible_maps = self.builder.build_participant_maps(data_visible, stem)
            infrared_maps = self.builder.build_participant_maps(data_infrared, stem)
            fixation_map_v = self.builder.aggregate_fixation_maps(list(visible_maps.values()))
            fixation_map_i = self.builder.aggregate_fixation_maps(list(infrared_maps.values()))
            
            stimuli_width = dimensions[stem]["stimuli_width"]
            image_width = dimensions[stem]["image_width"]
            sigma = self._compute_sigma(self.sigma_deg, stimuli_width, image_width)
            
            inter_scores = self._compute_human_scores(fixation_map_v, fixation_map_i, sigma)
            permutation_scores = self._compute_permutation_scores(
                data_visible=data_visible,
                data_infrared=data_infrared,
                stem=stem,
                sigma=sigma,
                n_permutations=n_permutations,
                seed=seed,
            )

            for metric_type in ("similarity", "predictivity"):
                for metric_name in self.metrics.registry[metric_type]:
                    
                    inter_score = inter_scores[metric_type][metric_name]
                    perm_scores = permutation_scores[metric_type][metric_name]
                    p_value = self._permutation_p_value(inter_score, perm_scores, metric_name)
                    delta = float(np.nanmean(perm_scores)) - inter_score
                    
                    results[metric_type][metric_name][stem] = {
                        "score_inter": inter_score,
                        "score_perm_mean": float(np.nanmean(perm_scores)),
                        "score_perm_std": float(np.nanstd(perm_scores)),
                        "delta_perm_inter": delta,
                        "p_value": p_value,
                        "n_permutations": n_permutations,
                        "n_visible": len(visible_maps),
                        "n_infrared": len(infrared_maps),
                        "sigma": float(sigma),
                    }

        return results

    
    def run_expe2(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        infrared_image_dir: Path,
        visible_image_dir: Path,
        infrared_centerbias_path: Path,
        visible_centerbias_path: Path,
    ) -> Results:

        deepgaze_v = DeepGaze(centerbias_path=visible_centerbias_path)
        deepgaze_i = DeepGaze(centerbias_path=infrared_centerbias_path)

        modality_v = self.index.normalize_modality("visible")
        modality_i = self.index.normalize_modality("infrared")

        data_i = self.loader.load_dataset(infrared_df)
        data_v = self.loader.load_dataset(visible_df)
        
        stems = self._get_stems(infrared_df, visible_df)
        dimensions = self._get_dimensions(infrared_df, visible_df)


        results: Results = {
            metric_type: {
                metric_name: {
                    stem: None for stem in stems
                }
                for metric_name in metrics
            }
            for metric_type, metrics in self.metrics.registry.items()
            if metric_type in ("similarity", "predictivity")
        }

        for stem in tqdm(stems, desc="Comparing DeepGaze RGB and IR"):

            # Human fixation and saliency maps
            fixation_map_v = self.builder.build_fixation_map(data_v, stem)
            fixation_map_i = self.builder.build_fixation_map(data_i, stem)

            stimuli_width = dimensions[stem]["stimuli_width"]
            image_width = dimensions[stem]["image_width"]
            sigma = self._compute_sigma(self.sigma_deg, stimuli_width, image_width)
            
            saliency_map_v = self.builder.build_saliency_map(fixation_map_v, sigma)
            saliency_map_i = self.builder.build_saliency_map(fixation_map_i, sigma)

            # Image paths
            image_id_v = self.index.get_id(stem, modality_v)
            image_id_i = self.index.get_id(stem, modality_i)

            image_name_v = self.index.get_name(image_id_v)
            image_name_i = self.index.get_name(image_id_i)

            image_v = self.loader.load_image(visible_image_dir/image_name_v)
            image_i = self.loader.load_image(infrared_image_dir/image_name_i)

            # DeepGaze predictions
            deepgaze_map_v = deepgaze_v.predict(image_v)
            deepgaze_map_i = deepgaze_i.predict(image_i)

            for metric_type, metrics in self.metrics.registry.items():
                if metric_type not in ("similarity", "predictivity"):
                    continue

                for metric_name, metric_function in metrics.items():

                    if metric_type == "similarity":
                        score_rgb = metric_function(
                            saliency_map_v,
                            deepgaze_map_v,
                        )
                        score_ir = metric_function(
                            saliency_map_i,
                            deepgaze_map_i,
                        )

                    elif metric_type == "predictivity":
                        score_rgb = metric_function(
                            fixation_map_v,
                            deepgaze_map_v,
                        )
                        score_ir = metric_function(
                            fixation_map_i,
                            deepgaze_map_i,
                        )

                    # Positive delta means better performance on RGB.
                    # Make sur all metrics are correctly oriented.
                    delta = score_rgb - score_ir

                    results[metric_type][metric_name][stem] = {
                        "score_rgb": float(score_rgb),
                        "score_ir": float(score_ir),
                        "delta_rgb_ir": float(delta),
                    }

        return results
    
    
    
    
    
    
    
    # def _compute_intra_scores(
    #         self,
    #         data: Dataset,
    #         stem: str | int,
    #         n_splits: int,
    #         seed: int | None = 42,
    #     ) -> dict[str, dict[str, float]]:
    #         stem_data = data[stem]
    #         participant_ids = list(stem_data.participants_data.keys())
    
    #         participant_maps = {
    #             participant_id: self.builder.build_fixation_map(
    #                 data=data,
    #                 stem=stem,
    #                 participant_id=participant_id,
    #             )
    #             for participant_id in participant_ids
    #         }
    
    #         rng = np.random.default_rng(seed)
    
    #         scores: dict[str, dict[str, list[float]]] = {
    #             metric_type: {
    #                 metric_name: []
    #                 for metric_name in metrics
    #             }
    #             for metric_type, metrics in self.metrics.registry.items()
    #             if metric_type in ("similarity", "predictivity")
    #         }
    
    #         for _ in range(n_splits):
    #             shuffled_ids = rng.permutation(participant_ids)
    #             split_index = len(shuffled_ids) // 2
    
    #             group_1_ids = shuffled_ids[:split_index]
    #             group_2_ids = shuffled_ids[split_index:]
    
    #             fixation_map_1 = self.builder.aggregate_fixation_maps(
    #                 [participant_maps[pid] for pid in group_1_ids]
    #             )
    #             fixation_map_2 = self.builder.aggregate_fixation_maps(
    #                 [participant_maps[pid] for pid in group_2_ids]
    #             )
    
    #             saliency_map_1 = self.builder.build_saliency_map(fixation_map_1, self.sigma)
    #             saliency_map_2 = self.builder.build_saliency_map(fixation_map_2, self.sigma)
    
    #             for metric_name, metric_function in self.metrics.registry["predictivity"].items():
    #                 score_1 = metric_function(fixation_map_1, saliency_map_2)
    #                 score_2 = metric_function(fixation_map_2, saliency_map_1)
    #                 scores["predictivity"][metric_name].append(float(np.nanmean([score_1, score_2])))
    
    #             for metric_name, metric_function in self.metrics.registry["similarity"].items():
    #                 score_1 = metric_function(saliency_map_1, saliency_map_2)
    #                 score_2 = metric_function(saliency_map_2, saliency_map_1)
    #                 scores["similarity"][metric_name].append(float(np.nanmean([score_1, score_2])))
    
    #         return {
    #             metric_type: {
    #                 metric_name: float(np.nanmean(values))
    #                 for metric_name, values in metric_scores.items()
    #             }
    #             for metric_type, metric_scores in scores.items()
    #         }
            
            
    
            
    #     def run_expe1_old(self, infrared_df: pd.DataFrame, visible_df: pd.DataFrame, n_splits: int = 50) -> Results:
    #         data_infrared = self.loader.load_dataset(infrared_df)
    #         data_visible = self.loader.load_dataset(visible_df)
    #         stems = self._get_stems(infrared_df, visible_df) 
    #         modality_v = self.index.normalize_modality("visible")
    #         modality_i = self.index.normalize_modality("infrared")
    #         results: Results = {
    #             metric_type: {
    #                 metric_name: {
    #                     stem: None for stem in stems
    #                     } 
    #                 for metric_name in metric_names
    #             } 
    #             for metric_type, metric_names in self.metrics.registry.items()
    #         }
            
    #         for stem in tqdm(stems, desc="Running modality analysis"):
    #             fixation_map_v = self.builder.build_fixation_map(data_visible, stem)
    #             fixation_map_i = self.builder.build_fixation_map(data_infrared, stem)
    #             saliency_map_v = self.builder.build_saliency_map(fixation_map_v, self.sigma)  
    #             saliency_map_i = self.builder.build_saliency_map(fixation_map_i, self.sigma) 
                
    #             annotations_v = self.index.get_annotation(modality_v, stem)
    #             annotations_i = self.index.get_annotation(modality_i, stem)
                
    #             daynight = self.index.get_visibility(self.index.get_id(stem, "infrared"))
                
    #             intra_v = self.compute_intra_scores(data_visible, stem, n_splits)
    #             intra_i = self.compute_intra_scores(data_infrared, stem, n_splits)
            
    #             for metric_type, metrics in self.metrics.registry.items():
    #                 if metric_type == "roi":
    #                     continue
    #                 for metric_name, metric_function in metrics.items():
    #                     vv = intra_v[metric_type][metric_name] # M_{H-H}^{RGB}
    #                     ii = intra_i[metric_type][metric_name] # M_{H-H}^{LWIR}
                        
    #                     if metric_type == "similarity":
    #                         vi = metric_function(saliency_map_v, saliency_map_i)
    #                         iv = metric_function(saliency_map_i, saliency_map_v)
                            
    #                     if metric_type == "predictivity":
    #                         vi = metric_function(fixation_map_v, saliency_map_i)
    #                         iv = metric_function(fixation_map_i, saliency_map_v)
    
    #                     intra = np.mean([vv, ii]) # M_{H-H}^{intra}
    #                     inter = np.mean([vi, iv]) # M_{H-H}^{inter}
    #                     delta = intra - inter if metric_name not in ["emd", "kl"] else inter - intra # \Delta_{M}^{H-H}
                        
    #                     results[metric_type][metric_name][stem] = {
    #                         "vv": vv,
    #                         "ii": ii,
    #                         "vi": vi,
    #                         "iv": iv,
    #                         "intra": intra,
    #                         "inter": inter,
    #                         "delta": delta,
    #                         "daynight": daynight,
    #                     }
                        
    #         return results
        
    # def run_expe2_old(self, 
    #     infrared_df: pd.DataFrame, 
    #     visible_df: pd.DataFrame, 
    #     infrared_image_dir: Path, 
    #     visible_image_dir: Path, 
    #     infrared_centerbias_path: Path, 
    #     visible_centerbias_path: Path,
    #     n_splits: int,
    # ) -> Results:
        
    #     deepgaze_v = DeepGaze(centerbias_path=visible_centerbias_path)
    #     deepgaze_i = DeepGaze(centerbias_path=infrared_centerbias_path)
        
    #     modality_v = self.index.normalize_modality("visible")
    #     modality_i = self.index.normalize_modality("infrared")
        
    #     data_infrared = self.loader.load_dataset(infrared_df)
    #     data_visible = self.loader.load_dataset(visible_df)
    #     stems = self._get_stems(infrared_df, visible_df) 
        
    #     results: Results = {
    #         metric_type: {
    #             metric_name: {
    #                 stem: None for stem in stems
    #                 } 
    #             for metric_name in metric_names
    #         } 
    #         for metric_type, metric_names in self.metrics.registry.items()
    #     }
        
    #     for stem in tqdm(stems, desc="Running deepgaze analysis"):
    #         fixation_map_v = self.builder.build_fixation_map(data_visible, stem)
    #         fixation_map_i = self.builder.build_fixation_map(data_infrared, stem) 
    #         saliency_map_v = self.builder.build_saliency_map(fixation_map_v, self.sigma) 
    #         saliency_map_i = self.builder.build_saliency_map(fixation_map_i, self.sigma)  
            
    #         annotations_v = self.index.get_annotation(modality_v, stem)
    #         annotations_i = self.index.get_annotation(modality_i, stem)
            
    #         image_id_v = self.index.get_id(stem, modality_v)
    #         image_id_i = self.index.get_id(stem, modality_i)
    #         image_name_v = self.index.get_name(image_id_v)
    #         image_name_i = self.index.get_name(image_id_i)
    #         image_path_v = visible_image_dir / image_name_v
    #         image_path_i = infrared_image_dir / image_name_i
    #         image_v = self.loader.load_image(image_path_v)
    #         image_i = self.loader.load_image(image_path_i)
            
    #         daynight = self.index.get_visibility(image_id_i)
            
    #         saliency_map_dg_v = deepgaze_v.predict(image_v)
    #         saliency_map_dg_i = deepgaze_i.predict(image_i)
            
    #         intra_v = self.compute_intra_scores(data_visible, stem, n_splits)
    #         intra_i = self.compute_intra_scores(data_infrared, stem, n_splits)
                
    #         for metric_type, metrics in self.metrics.registry.items():
    #             if metric_type == "roi":
    #                 continue
    #             for metric_name, metric_function in metrics.items():
                    
    #                 vv = intra_v[metric_type][metric_name] # M_{H-H}^{RGB}
    #                 ii = intra_i[metric_type][metric_name] # M_{H-H}^{LWIR}
                    
    #                 if metric_type == "similarity":
    #                     vd = metric_function(saliency_map_v, saliency_map_dg_v) # M_{DG-H}^{RGB}
    #                     id = metric_function(saliency_map_i, saliency_map_dg_i) # M_{DG-H}^{LWIR}
                        
    #                 if metric_type == "predictivity":
    #                     vd = metric_function(fixation_map_v, saliency_map_dg_v) # M_{DG-H}^{RGB}  
    #                     id = metric_function(fixation_map_i, saliency_map_dg_i) # M_{DG-H}^{LWIR} 

    #                 delta_v = vv - vd if metric_name not in ["emd", "kl"] else vd - vv # \Delta_{M}^{RGB}
    #                 delta_i = ii - id if metric_name not in ["emd", "kl"] else id - ii # \Delta_{M}^{LWIR}
    #                 delta = delta_i - delta_v # \Delta_{M}^{DG-H}
                    
    #                 results[metric_type][metric_name][stem] = {
    #                     "vv": vv,
    #                     "ii": ii,
    #                     "vd": vd,
    #                     "id": id,
    #                     "delta_i": delta_i,
    #                     "delta_v": delta_v,
    #                     "delta": delta,
    #                     "daynight": daynight,
    #                 }
    #     return results


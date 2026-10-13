from __future__ import annotations
from collections.abc import Mapping
from tqdm import tqdm
from pathlib import Path
from math import tan, radians

import numpy as np
import pandas as pd

from analysis.metrics import Metrics
from analysis.gold_standard import (
    COMPONENTS,
    LN2,
    CenterBias,
    GoldStandard,
    ImageFixations,
    explainable_information_gain,
    fit_centerbias,
    fit_gold_standard,
    model_information_gain,
)
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
        self.metadata_json = Path(metadata_json)
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
    
   
    
    # def _permutation_p_value(
    #     self,
    #     observed_score: float,
    #     permutation_scores: np.ndarray,
    #     metric_name: str,
    # ) -> float:
    #     valid_scores = permutation_scores[
    #         np.isfinite(permutation_scores)
    #     ]

    #     if valid_scores.size == 0 or not np.isfinite(observed_score):
    #         return float("nan")

        
    #     extreme_count = np.sum(valid_scores <= observed_score)

    #     return float(
    #         (extreme_count + 1) / (valid_scores.size + 1)
    #     )
    
    
    def run_expe1_old(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        n_permutations: int = 1000,
        seed: int | None = 42,
    ) -> Results:
        data_infrared = self.loader.load_dataset(infrared_df)
        data_visible = self.loader.load_dataset(visible_df)

        def _compute_permutation_scores_old(
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

                visible_indices = indices[:n_visible] # regroups N_visibles maps indifferently IR/RGB
                infrared_indices = indices[n_visible:n_visible + n_infrared] #regroups the rest of the maps indifferently from IR/RGB
                
                #Build a fixation map from the selected fixation maps
                fixation_map_group_1 = self.builder.aggregate_fixation_maps(
                    [all_maps[index] for index in visible_indices]
                )
                fixation_map_group_2 = self.builder.aggregate_fixation_maps(
                    [all_maps[index] for index in infrared_indices]
                )
                #Build a fixation map from the selected fixation maps
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

        stems = self._get_stems(infrared_df, visible_df) #1 stem = 1 scene
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

        for stem in tqdm(stems, desc="Running human-human analysis"): # iter over scenes 
            #Compute the Fixations maps for all observations of RGB and LWIR
            visible_maps = self.builder.build_participant_maps(data_visible, stem)
            infrared_maps = self.builder.build_participant_maps(data_infrared, stem)
            fixation_map_v = self.builder.aggregate_fixation_maps(list(visible_maps.values()))
            fixation_map_i = self.builder.aggregate_fixation_maps(list(infrared_maps.values()))
            
            stimuli_width = dimensions[stem]["stimuli_width"]
            image_width = dimensions[stem]["image_width"]
            sigma = self._compute_sigma(self.sigma_deg, stimuli_width, image_width)
            
            #Compute the Agreement (inter modality score) 
            inter_scores = self._compute_human_scores(fixation_map_v, fixation_map_i, sigma)
            
            #Compute the intra-modality score
            permutation_scores = _compute_permutation_scores_old(
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

    def _compute_permutation_scores(
    self,
    data_visible: Dataset,
    data_infrared: Dataset,
    stem: str | int,
    sigma: float,
    n_permutations: int = 1000,
    seed: int | None = 42,
    ) -> dict[str, dict[str, np.ndarray]]:
        
        def compute_inter_score(vis_g1,vis_g2, ir_g1,ir_g2, sigma):
                inter_1 = self._compute_human_scores(vis_g1, ir_g1, sigma)
                inter_2 = self._compute_human_scores(vis_g2, ir_g2, sigma)
                inter_3 = self._compute_human_scores(vis_g1, ir_g2, sigma)
                inter_4 = self._compute_human_scores(vis_g2, ir_g2, sigma)
                
                inter_score = {
                        metric_type: {
                            metric_name: np.empty(1, dtype=float)
                            for metric_name in self.metrics.registry[metric_type]
                        }
                        for metric_type in ("similarity", "predictivity")
                    }
            
                for metric_type in ("similarity", "predictivity"):
                    for metric_name in self.metrics.registry[metric_type]:
                        i1 = inter_1[metric_type][metric_name]
                        i2 = inter_2[metric_type][metric_name]
                        i3 = inter_3[metric_type][metric_name]
                        i4 = inter_4[metric_type][metric_name]

                        inter_score[metric_type][metric_name] = (1/4)*(i1+i2+i3+i4)
                
                return inter_score
        
        
        visible_maps = self.builder.build_participant_maps(
            data_visible,
            stem,
        )
        infrared_maps = self.builder.build_participant_maps(
            data_infrared,
            stem,
        )

        infrared_maps = list(infrared_maps.values())
        visible_maps = list(visible_maps.values())
        # all_maps = (
        #     list(visible_maps.values())
        #     + list(infrared_maps.values())
        # )

        n_visible = len(visible_maps)
        n_infrared = len(infrared_maps)

        if n_visible == 0 or n_infrared == 0:
            raise ValueError(
                f"No participant data available for stem '{stem}'."
            )

        n = int(min((np.floor(n_visible/2),np.floor(n_infrared/2))))

        rng = np.random.default_rng(seed)

        permutations_scores = {
                modality : {
                    metric_type: {
                    metric_name: np.empty(n_permutations, dtype=float)
                    for metric_name in self.metrics.registry[metric_type]
                    }
                for metric_type in ("similarity", "predictivity")
                }
            for modality in ('VIS','IR')
        }

     

        for permutation_index in range(n_permutations):
            visible_indices = rng.permutation(n_visible)
            infrared_indices = rng.permutation(n_infrared)
            
            visible_indices_1, visible_indices_2 = visible_indices[:n], visible_indices[n:n+n] 
            infrared_indices_1, infrared_indices_2 = infrared_indices[:n], infrared_indices[n:n+n] 
            
            #Sanity check for the following test
            assert(set(visible_indices_1).isdisjoint(set(visible_indices_2)))
            assert(set(infrared_indices_1).isdisjoint(set(infrared_indices_2)))

            #Build fixation maps from the selected participant's observations
            visible_fixation_map_group_1 = self.builder.aggregate_fixation_maps(
                [visible_maps[index] for index in visible_indices_1]
            ) 
            visible_fixation_map_group_2 = self.builder.aggregate_fixation_maps(
                [visible_maps[index] for index in visible_indices_2]
            ) 

            infrared_fixation_map_group_1 = self.builder.aggregate_fixation_maps(
                [infrared_maps[index] for index in infrared_indices_1]
            )

            infrared_fixation_map_group_2 = self.builder.aggregate_fixation_maps(
                [infrared_maps[index] for index in infrared_indices_2]
            )
            
            

            #Compute the score between those maps
            intra_RGB_scores = self._compute_human_scores(
                visible_fixation_map_group_1,
                visible_fixation_map_group_2,
                sigma
            )
            intra_IR_scores = self._compute_human_scores(
                infrared_fixation_map_group_1,
                infrared_fixation_map_group_2,
                sigma
            )

            

            #Compute the inter scores
            inter_score = compute_inter_score(visible_fixation_map_group_1,
                                              visible_fixation_map_group_2,
                                              infrared_fixation_map_group_1,
                                              infrared_fixation_map_group_2,
                                              sigma)

            #compute the intra scores
            visible_intra_score = self._compute_human_scores(visible_fixation_map_group_1, visible_fixation_map_group_2, sigma)
            infrared_intra_score = self._compute_human_scores(infrared_fixation_map_group_1, infrared_fixation_map_group_2, sigma)


            #Logs the score for test computation
            for metric_type in ("similarity", "predictivity"):
                for metric_name in self.metrics.registry[metric_type]:
                    
                    inter_value = inter_score[metric_type][metric_name]
                    visible_intra_value = visible_intra_score[metric_type][metric_name]
                    infrared_intra_value = infrared_intra_score[metric_type][metric_name] 

                    delta_rgb = visible_intra_value - inter_value
                    delta_ir = infrared_intra_value - inter_value


                    permutations_scores['VIS'][metric_type][metric_name][permutation_index] = delta_rgb
                    permutations_scores['IR'][metric_type][metric_name][permutation_index] = delta_ir


        return permutations_scores

    def run_expe1(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        n_permutations: int = 1000,
        seed: int | None = 42,
    ) -> Results:
        data_infrared = self.loader.load_dataset(infrared_df)
        data_visible = self.loader.load_dataset(visible_df)

        stems = self._get_stems(infrared_df, visible_df) #1 stem = 1 scene
        dimensions = self._get_dimensions(infrared_df, visible_df)
        results: Results = {
            modality : {
                metric_type: {
                    metric_name: {
                        stem: None
                        for stem in stems
                    }
                    for metric_name in self.metrics.registry[metric_type]
                }
                for metric_type in ("similarity", "predictivity")
            } 
            for modality in ("VIS","IR")
        }

        for stem in tqdm(stems, desc="Running human-human analysis"): # iter over scenes 
            #Define the number of images to use for computations in the permutations  
            visible_user = len(self.builder.build_participant_maps(data_visible, stem).keys())
            infrared_user = len(self.builder.build_participant_maps(data_infrared, stem).keys())
            
            N = int(min(np.floor(visible_user/2),np.floor(infrared_user/2)))

            #Construct the blurring kernel associated to the current scene
            stimuli_width = dimensions[stem]["stimuli_width"]
            image_width = dimensions[stem]["image_width"]
            sigma = self._compute_sigma(self.sigma_deg, stimuli_width, image_width)
            
            
            #Compute the intra-modality score
            permutation_scores = self._compute_permutation_scores(
                data_visible=data_visible,
                data_infrared=data_infrared,
                stem=stem,
                sigma=sigma,
                n_permutations=n_permutations,
                seed=seed,
            )
            
            for modality in ("VIS", "IR"):
                for metric_type in ("similarity", "predictivity"):
                    for metric_name in self.metrics.registry[metric_type]:
                        
                        perm_scores = permutation_scores[modality][metric_type][metric_name]
                        
                        results[modality][metric_type][metric_name][stem] = {
                            "score_perm_mean": float(np.nanmean(perm_scores)),
                            "score_perm_std": float(np.nanstd(perm_scores)),
                            "n_permutations": n_permutations,
                            "sigma": float(sigma),
                            "N_fix_maps": N,
                
                        }

        return results


    def _get_image_fixations(self, data: Dataset, stem: str) -> ImageFixations | None:
        """
        Collect the fixations of all participants on one stem, in continuous image pixel
        coordinates (same screen -> image transform as Builder, without rounding).
        Returns None when fewer than two participants have a fixation on the image.
        """
        dims = data[stem].dimensions_data
        offset_x = (dims.screen_width - dims.stimuli_width) / 2.0
        offset_y = (dims.screen_height - dims.stimuli_height) / 2.0
        scale_x = dims.image_width / float(dims.stimuli_width)
        scale_y = dims.image_height / float(dims.stimuli_height)

        xs, ys, subjects = [], [], []
        for participant_id, fixations in data[stem].participants_data.items():
            for fixation in fixations:
                x_stim = float(fixation.x) - offset_x
                y_stim = float(fixation.y) - offset_y
                if not (0 <= x_stim < dims.stimuli_width and 0 <= y_stim < dims.stimuli_height):
                    continue
                xs.append(x_stim * scale_x)
                ys.append(y_stim * scale_y)
                subjects.append(participant_id)

        if len(set(subjects)) < 2:
            return None

        return ImageFixations(
            x=np.minimum(xs, np.nextafter(dims.image_width, 0)),
            y=np.minimum(ys, np.nextafter(dims.image_height, 0)),
            subject=np.array(subjects),
            width=int(dims.image_width),
            height=int(dims.image_height),
        )


    @staticmethod
    def _normalised_fixations(
        images: dict[str, ImageFixations],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fixations of every image in [0, 1) coordinates, with the index of their image."""
        stems = list(images)
        nx = np.concatenate([images[s].x / images[s].width for s in stems])
        ny = np.concatenate([images[s].y / images[s].height for s in stems])
        image_ids = np.concatenate([np.full(len(images[s].x), i) for i, s in enumerate(stems)])
        return nx, ny, image_ids


    def _leave_one_image_out_centerbiases(
        self,
        images: dict[str, ImageFixations],
        sigma: float,
        eps: float,
    ) -> dict[str, CenterBias]:
        """Center bias of each image, built only from the fixations on the OTHER images."""
        nx, ny, image_ids = self._normalised_fixations(images)
        return {
            stem: CenterBias(nx[image_ids != i], ny[image_ids != i], sigma=sigma, eps=eps)
            for i, stem in enumerate(images)
        }


    def _load_gold_standard(self, modality_dir: str | Path, data: Dataset) -> dict:
        """
        Reload one modality saved by Saver.save_gold_standard_results (<output>/<modality>/).

        The per-image gold-standard results come from images.csv. The leave-one-image-out center
        biases are rebuilt from the fixations of the saved stems with the fitted parameters of
        params.json, so they are the same baselines as the ones used for the saved IG. The fitted
        gold standard and the naive models are rebuilt from params.json and images.csv.
        """
        modality_dir = Path(modality_dir)
        per_image = pd.read_csv(modality_dir / "images.csv", dtype={"stem": str})
        params = self.loader.load_json(modality_dir / "params.json")

        images = {}
        for stem in per_image["stem"]:
            stem = self.index.normalize_stem(stem)
            image = self._get_image_fixations(data, stem) if stem in data else None
            if image is None:
                raise ValueError(
                    f"Stem '{stem}' of {modality_dir / 'images.csv'} has no usable fixations in the "
                    "DataFrame; recompute the gold standard on the current data."
                )
            images[stem] = image

        rows = {
            self.index.normalize_stem(row["stem"]): row
            for row in per_image.to_dict(orient="records")
        }

        return {
            "gold_standard": GoldStandard(sigma=float(params["sigma_px"]), weights=params["weights"]),
            "naive_models": {
                stem: GoldStandard(sigma=float(row["naive_sigma_px"]), weights={"kde": 1.0})
                for stem, row in rows.items()
            },
            "fixations": images,
            "centerbiases": self._leave_one_image_out_centerbiases(
                images, params["centerbias_sigma"], params["centerbias_eps"]
            ),
            "images": rows,
        }


    def _compute_gold_standard_dataset(
        self,
        data: Dataset,
        stems: list[str],
        components: tuple[str, ...],
        sigma: float | None,
        seed: int | None,
        image_dir: Path,
    ) -> dict:
        images = {}
        for stem in stems:
            image = self._get_image_fixations(data, stem)
            if image is None:
                print(f"Skipping stem '{stem}': fewer than two participants with valid fixations.")
                continue
            images[stem] = image

        if len(images) < 2:
            raise ValueError("The gold standard needs at least two images with valid fixations.")

        kept_stems = list(images)
        nx, ny, image_ids = self._normalised_fixations(images)

        # Center-bias hyperparameters are cross-validated over images once; the center bias of
        # each image is then built only from the fixations on the OTHER images.
        cb_params = fit_centerbias(nx, ny, image_ids, seed=seed)
        centerbiases = list(self._leave_one_image_out_centerbiases(
            images, cb_params.sigma, cb_params.eps
        ).values())

        # One bandwidth and one set of mixture weights for the whole dataset.
        gold_standard = fit_gold_standard(
            [images[s] for s in kept_stems],
            centerbiases,
            components=components,
            sigma=sigma,
        )

        per_image = {}
        naive_models = {}
        for stem, centerbias in tqdm(zip(kept_stems, centerbiases), total=len(kept_stems),
                                     desc="Computing gold standard"):
            # Naive model of the experiment 1 protocol: KDE with sigma = self.sigma_deg of visual
            # angle, built from the other participants (leave-one-subject-out).
            dims = data[stem].dimensions_data
            naive_sigma = self._compute_sigma(self.sigma_deg, dims.stimuli_width, dims.image_width)
            naive_models[stem] = GoldStandard(sigma=float(naive_sigma), weights={"kde": 1.0})
            naive = explainable_information_gain(images[stem], naive_models[stem], centerbias)

            per_image[stem] = {
                **explainable_information_gain(images[stem], gold_standard, centerbias),
                "n_participants": int(len(np.unique(images[stem].subject))),
                "naive_sigma_px": float(naive_sigma),
                "ll_naive_bits": naive["ll_gold_bits"],
                "naive_ig_bits": naive["explainable_ig_bits"],
            }

        return {
            "gold_standard": gold_standard,
            "naive_models": naive_models,
            # Center bias built from every image, for visualisation only.
            "centerbias": CenterBias(nx, ny, sigma=cb_params.sigma, eps=cb_params.eps),
            "centerbiases": dict(zip(kept_stems, centerbiases)),
            "fixations": images,
            "image_names": {stem: data[stem].image_data.name for stem in kept_stems},
            "image_paths": {stem: image_dir / data[stem].image_data.name for stem in kept_stems},
            "params": {
                "sigma_px": gold_standard.sigma,
                "weights": {k: float(v) for k, v in gold_standard.weights.items()},
                "centerbias_sigma": cb_params.sigma,
                "centerbias_eps": cb_params.eps,
            },
            # Dataset-level IG: mean over images of the per-image means (Kümmerer & Bethge, 2023).
            "explainable_ig_bits": float(np.mean([r["explainable_ig_bits"] for r in per_image.values()])),
            "naive_ig_bits": float(np.mean([r["naive_ig_bits"] for r in per_image.values()])),
            "images": per_image,
        }


    def compute_gold_standard(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        components: tuple[str, ...] = COMPONENTS,
        sigma: float | None = None,
        seed: int | None = 0,
        infrared_image_dir: str | Path | None = None,
        visible_image_dir: str | Path | None = None,
    ) -> dict[str, dict]:
        """
        Gold-standard fixation density and explainable information gain on every image of
        both modalities, as defined in analysis/gold_standard.py.

        The KDE bandwidth (pixels) and the mixture weights are fitted once per modality on all
        its images (leave-one-subject-out likelihood). Each image's center bias uses only
        fixations from the other images of the same modality.

        components : subset of ("kde", "centerbias", "uniform"), must contain "kde".
        sigma      : fix the KDE bandwidth in image pixels instead of fitting it.
        infrared_image_dir, visible_image_dir : stimulus directories, used by the saver's
                     visualisations. Default: <data_dir>/images/<modality>, where
                     <data_dir> is the parent of the metadata folder.

        Each image is also scored with the naive model of experiment 1 (KDE with
        sigma = self.sigma_deg of visual angle, leave-one-subject-out).

        Returns {"VIS": ..., "IR": ...}, each with the fitted "gold_standard", the per-image
        "naive_models" and "centerbiases" (reusable with gold_standard.model_information_gain),
        the dataset-wide "centerbias", the "fixations", "image_names" and "image_paths" per stem, the fitted
        "params", the dataset "explainable_ig_bits" and "naive_ig_bits", and per-stem results
        under "images".
        """
        data_infrared = self.loader.load_dataset(infrared_df)
        data_visible = self.loader.load_dataset(visible_df)
        stems = self._get_stems(infrared_df, visible_df)

        images_dir = self.metadata_json.parent.parent / "images"
        image_dirs = {
            "VIS": Path(visible_image_dir) if visible_image_dir is not None else images_dir / "visible",
            "IR": Path(infrared_image_dir) if infrared_image_dir is not None else images_dir / "infrared",
        }

        return {
            modality: self._compute_gold_standard_dataset(
                data, stems, tuple(components), sigma, seed, image_dirs[modality]
            )
            for modality, data in (("VIS", data_visible), ("IR", data_infrared))
        }


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


    def run_deepgaze_ige(
        self,
        infrared_df: pd.DataFrame,
        visible_df: pd.DataFrame,
        infrared_image_dir: Path,
        visible_image_dir: Path,
        infrared_centerbias_path: Path,
        visible_centerbias_path: Path,
        gold_standard: Mapping[str, Mapping] | str | Path,
        model: str = "IIE",
        msdb_dataset: int | None = None,
    ) -> dict[str, dict]:
        """
        Information gain explained (IGE) by DeepGaze on every image of both modalities,
        IGE(Fix_mod, DG(Image_mod)), following Kümmerer & Bethge (2023), Annu. Rev. Vis. Sci. 9:269-91:

            IG_model = mean_i [ log2 p_DG(x_i) - log2 p_centerbias(x_i) ]     (bit / fixation)
            IGE      = IG_model / IG_gold

        IG_gold and the center-bias baseline are the ones of the gold standard, so that both
        information gains are measured against the same baseline. The baseline of each image is
        built only from the fixations on the other images of the same modality.

        DeepGaze gets the stimulus of its modality (IR images are duplicated to 3 channels) and
        the center bias of that modality as prior (infrared_centerbias_path, visible_centerbias_path).

        gold_standard : output of Comparator.compute_gold_standard, or the output directory of
                        Saver.save_gold_standard_results (containing VIS/ and IR/).
        model         : "IIE" (DeepGaze IIE) or "MSDB" (DeepGaze MSDB). MSDB also gets the pixels
                        per degree of visual angle of each image, from the display geometry.
        msdb_dataset  : MSDB only. None for the parameters averaged over its training datasets,
                        or a deepgaze_pytorch.MSDBDataset index (see saliency.deepgaze.DeepGaze).

        Returns {"VIS": ..., "IR": ...}, each with per-stem results under "images" and the dataset
        values: mean "explainable_ig_bits" and "model_ig_bits" over images and their ratio "ige"
        (ratio of the averages, not the mean of the per-image ratios, as in the paper).
        "ige_naive" is the IGE of the naive model of the gold standard (KDE with
        sigma = self.sigma_deg, leave-one-subject-out), a baseline for "ige".
        Also keeps, per stem, the DeepGaze "log_densities", the stimulus "image_paths" and the gold
        standard's "fixations", "centerbiases" and "naive_models", plus its fitted "gold_standard",
        for the visualisations of Saver.save_deepgaze_ige.
        """
        data = {
            "VIS": self.loader.load_dataset(visible_df),
            "IR": self.loader.load_dataset(infrared_df),
        }
        image_dirs = {"VIS": Path(visible_image_dir), "IR": Path(infrared_image_dir)}
        centerbias_paths = {"VIS": Path(visible_centerbias_path), "IR": Path(infrared_centerbias_path)}
        modality_names = {
            "VIS": self.index.normalize_modality("visible"),
            "IR": self.index.normalize_modality("infrared"),
        }

        results = {}
        for modality in ("VIS", "IR"):
            gold = (
                gold_standard[modality]
                if isinstance(gold_standard, Mapping)
                else self._load_gold_standard(Path(gold_standard) / modality, data[modality])
            )
            deepgaze = DeepGaze(
                centerbias_path=centerbias_paths[modality], model=model, msdb_dataset=msdb_dataset
            )

            per_image, log_densities, image_paths = {}, {}, {}
            for stem in tqdm(list(gold["images"]), desc=f"Computing DeepGaze IGE ({modality})"):
                img = gold["fixations"][stem]
                centerbias = gold["centerbiases"][stem]
                gold_values = gold["images"][stem]

                # Sanity check: the baseline must be the one of the stored gold-standard IG.
                ll_centerbias_bits = float(
                    (centerbias.log_prob(img).mean() + np.log(img.width * img.height)) / LN2
                )
                if not np.isclose(ll_centerbias_bits, gold_values["ll_centerbias_bits"], atol=1e-6):
                    raise ValueError(
                        f"{modality} {stem}: the center-bias baseline differs from the gold standard "
                        f"({ll_centerbias_bits:.6f} vs {gold_values['ll_centerbias_bits']:.6f} bit/fix). "
                        "Recompute the gold standard on the current data."
                    )

                image_id = self.index.get_id(stem, modality_names[modality])
                image_path = image_dirs[modality] / self.index.get_name(image_id)
                # Image pixels per degree of visual angle, as displayed during the experiment.
                dims = data[modality][stem].dimensions_data
                pixel_per_dva = self._compute_sigma(1.0, dims.stimuli_width, dims.image_width)
                log_density = deepgaze.predict_log_density(
                    self.loader.load_image(image_path), pixel_per_dva=pixel_per_dva
                )

                model_ig = model_information_gain(img, log_density, centerbias)
                explainable_ig = float(gold_values["explainable_ig_bits"])
                naive_ig = float(gold_values["naive_ig_bits"])

                per_image[stem] = {
                    "n_participants": int(len(np.unique(img.subject))),
                    "n_fixations": int(len(img.x)),
                    "ll_centerbias_bits": ll_centerbias_bits,
                    "ll_gold_bits": float(gold_values["ll_gold_bits"]),
                    "ll_model_bits": ll_centerbias_bits + model_ig,
                    "explainable_ig_bits": explainable_ig,
                    "naive_ig_bits": naive_ig,
                    "model_ig_bits": model_ig,
                    # Unstable when the explainable IG of the image is close to 0.
                    "ige": model_ig / explainable_ig if explainable_ig != 0 else float("nan"),
                    # IGE of the naive model of experiment 1 (KDE, sigma = 1 deg): baseline for "ige".
                    "ige_naive": naive_ig / explainable_ig if explainable_ig != 0 else float("nan"),
                }
                log_densities[stem] = log_density.astype(np.float32)
                image_paths[stem] = image_path

            del deepgaze

            explainable_ig_bits = float(np.mean([r["explainable_ig_bits"] for r in per_image.values()]))
            naive_ig_bits = float(np.mean([r["naive_ig_bits"] for r in per_image.values()]))
            model_ig_bits = float(np.mean([r["model_ig_bits"] for r in per_image.values()]))
            results[modality] = {
                "model": model,
                "msdb_dataset": msdb_dataset,
                "centerbias_path": centerbias_paths[modality],
                "explainable_ig_bits": explainable_ig_bits,
                "naive_ig_bits": naive_ig_bits,
                "model_ig_bits": model_ig_bits,
                "ige": model_ig_bits / explainable_ig_bits,
                "ige_naive": naive_ig_bits / explainable_ig_bits,
                "images": per_image,
                # For the saver's visualisations.
                "log_densities": log_densities,
                "image_paths": image_paths,
                "fixations": {stem: gold["fixations"][stem] for stem in per_image},
                "centerbiases": {stem: gold["centerbiases"][stem] for stem in per_image},
                "gold_standard": gold["gold_standard"],
                "naive_models": {stem: gold["naive_models"][stem] for stem in per_image},
            }

        return results
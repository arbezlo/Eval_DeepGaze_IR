from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.gold_standard import _centerbias_log_map, gold_standard_map
from helpers.loader import Loader
from helpers.models import Results


class Saver:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        
    

    def save_expe1_results_old(
        self,
        results: Results,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe1" / "comparison"
            if output_dir is None
            else Path(output_dir)
        )

        expected_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_inter",
            "score_perm_mean",
            "score_perm_std",
            "delta_perm_inter",
            "p_value",
            "n_permutations",
            "n_visible",
            "n_infrared",
            "sigma",
        ]

        return self._save_results(
            results=results,
            output_dir=output_dir,
            expected_columns=expected_columns,
            summary_measures=[
                "score_inter",
                "score_perm_mean",
                "score_perm_std",
                "delta_perm_inter",
            ],
        )

    def save_expe1_results(
        self,
        results: Results,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe1" / "comparison"
            if output_dir is None
            else Path(output_dir)
        )

        expected_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_perm_mean",
            "score_perm_std",
            "n_permutations",
            "N_fix_maps",
            "sigma",
        ]

        return self._save_results(
            results=results,
            output_dir=output_dir,
            expected_columns=expected_columns,
            summary_measures=[
                "score_perm_mean",
                "score_perm_std",
            ],
        )


    def save_expe2_results(
        self,
        results: Results,
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe2" / "comparison"
            if output_dir is None
            else Path(output_dir)
        )

        expected_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_rgb",
            "score_ir",
            "delta_rgb_ir",
        ]

        return self._save_results(
            results=results,
            output_dir=output_dir,
            expected_columns=expected_columns,
            summary_measures=[
                "score_rgb",
                "score_ir",
                "delta_rgb_ir",
            ],
        )


    def save_gold_standard_results(
        self,
        results: Mapping[str, Mapping[str, Any]],
        output_dir: str | Path | None = None,
        image_dirs: Mapping[str, str | Path] | None = None,
        n_examples: int = 5,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        """
        Save the output of Comparator.compute_gold_standard.

        For each modality:
            <modality>/images.csv        one row per stem (gold standard and naive model)
            <modality>/params.json       fitted parameters and dataset information gains
            <modality>/centerbias.png    center bias of the whole modality
            <modality>/visualisation/    gold standard vs naive model on n_examples random images
        Also writes summary.csv (one row per modality) and all_results.csv (all modalities).

        image_dirs : optional {modality: stimulus directory} (e.g. {"VIS": ..., "IR": ...}) that
                     overrides the "image_paths" stored by Comparator.compute_gold_standard for
                     the stimuli shown in the visualisations.
        The same random stems are used for every modality when they share them.
        """
        output_dir = (
            self.output_dir / "gold_standard"
            if output_dir is None
            else Path(output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_columns = [
            "modality",
            "stem",
            "n_participants",
            "n_fixations",
            "ll_gold_bits",
            "ll_centerbias_bits",
            "explainable_ig_bits",
            "naive_sigma_px",
            "ll_naive_bits",
            "naive_ig_bits",
        ]

        common_stems = sorted(set.intersection(
            *(set(modality_results["images"]) for modality_results in results.values())
        ))
        rng = np.random.default_rng(seed)
        example_stems = list(rng.choice(common_stems, size=min(n_examples, len(common_stems)), replace=False))

        summary_rows: list[dict[str, Any]] = []
        all_dataframes: list[pd.DataFrame] = []

        for modality, modality_results in results.items():
            modality_dir = output_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)

            rows = [
                {"modality": modality, "stem": str(stem), **values}
                for stem, values in modality_results["images"].items()
            ]
            df = self._ensure_columns(pd.DataFrame(rows), expected_columns)
            df.to_csv(modality_dir / "images.csv", index=False)
            all_dataframes.append(df)

            params = {
                **modality_results["params"],
                "explainable_ig_bits": modality_results["explainable_ig_bits"],
                "naive_ig_bits": modality_results["naive_ig_bits"],
                "n_images": len(df),
            }
            self._save_analysis_json(params, modality_dir / "params.json")

            summary_rows.append({
                "modality": modality,
                "n_images": len(df),
                "explainable_ig_bits": modality_results["explainable_ig_bits"],
                "explainable_ig_bits_std": float(df["explainable_ig_bits"].std()),
                "naive_ig_bits": modality_results["naive_ig_bits"],
                "naive_ig_bits_std": float(df["naive_ig_bits"].std()),
                "ll_gold_bits": self._safe_mean(df["ll_gold_bits"]),
                "ll_naive_bits": self._safe_mean(df["ll_naive_bits"]),
                "ll_centerbias_bits": self._safe_mean(df["ll_centerbias_bits"]),
                "sigma_px": params["sigma_px"],
                "naive_sigma_px": self._safe_mean(df["naive_sigma_px"]),
                **{f"weight_{name}": weight for name, weight in params["weights"].items()},
                "centerbias_sigma": params["centerbias_sigma"],
                "centerbias_eps": params["centerbias_eps"],
            })

            self._plot_centerbias(modality, modality_results, modality_dir / "centerbias.png")

            image_dir = None if image_dirs is None else image_dirs.get(modality)
            stems = example_stems or list(rng.choice(
                list(modality_results["images"]),
                size=min(n_examples, len(modality_results["images"])),
                replace=False,
            ))
            for stem in stems:
                self._plot_gold_standard_example(
                    modality,
                    modality_results,
                    stem,
                    modality_results.get("image_paths", {}).get(stem) if image_dir is None
                    else Path(image_dir) / modality_results["image_names"][stem],
                    modality_dir / "visualisation" / f"{stem}.png",
                )

        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

        all_results = (
            pd.concat(all_dataframes, ignore_index=True)
            if all_dataframes
            else pd.DataFrame(columns=expected_columns)
        )
        all_results.to_csv(output_dir / "all_results.csv", index=False)

        return all_results


    def save_deepgaze_ige(
        self,
        results: Mapping[str, Mapping[str, Any]],
        output_dir: str | Path | None = None,
        n_examples: int = 5,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        """
        Save the output of Comparator.run_deepgaze_ige.

            <modality>/images.csv       one row per stem (log-likelihoods, information gains, IGE)
            <modality>/visualisation/   gold standard, naive model and DeepGaze on n_examples random images
            summary.csv                 one row per modality; "ige" is the dataset IGE (ratio of the
                                        mean IGs), "ige_mean_per_image" the mean of the per-image ratios;
                                        "ige_naive*" are the same for the naive model (KDE, sigma = 1 deg),
                                        the baseline for the IGE of DeepGaze
            all_results.csv             stems of both modalities, RGB vs IR, in the run_expe2 format
                                        (metric_type, metric_name, score_rgb, score_ir, delta_rgb_ir),
                                        so it can be passed to Analyzer.analyze_expe2

        The example stems are drawn as in save_gold_standard_results, so with the same seed and
        stems the figures show the same images. Returns the all_results table.
        """
        output_dir = (
            self.output_dir / "expe2" / "ige"
            if output_dir is None
            else Path(output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_columns = [
            "modality",
            "stem",
            "n_participants",
            "n_fixations",
            "ll_centerbias_bits",
            "ll_gold_bits",
            "ll_model_bits",
            "explainable_ig_bits",
            "naive_ig_bits",
            "model_ig_bits",
            "ige",
            "ige_naive",
        ]
        comparison_columns = [
            "stem",
            "metric_type",
            "metric_name",
            "score_rgb",
            "score_ir",
            "delta_rgb_ir",
        ]

        common_stems = sorted(set.intersection(
            *(set(modality_results["images"]) for modality_results in results.values())
        ))
        rng = np.random.default_rng(seed)
        example_stems = list(rng.choice(common_stems, size=min(n_examples, len(common_stems)), replace=False))

        summary_rows: list[dict[str, Any]] = []
        dataframes: dict[str, pd.DataFrame] = {}

        for modality, modality_results in results.items():
            modality_dir = output_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)

            rows = [
                {"modality": modality, "stem": str(stem), **values}
                for stem, values in modality_results["images"].items()
            ]
            df = self._ensure_columns(pd.DataFrame(rows), image_columns)
            df.to_csv(modality_dir / "images.csv", index=False)
            dataframes[modality] = df.set_index("stem")

            summary_rows.append({
                "modality": modality,
                "n_images": len(df),
                "explainable_ig_bits": modality_results["explainable_ig_bits"],
                "model_ig_bits": modality_results["model_ig_bits"],
                "model_ig_bits_std": float(df["model_ig_bits"].std()),
                "ige": modality_results["ige"],
                "ige_mean_per_image": self._safe_mean(df["ige"]),
                "ige_median_per_image": float(df["ige"].median()),
                "naive_ig_bits": modality_results["naive_ig_bits"],
                "ige_naive": modality_results["ige_naive"],
                "ige_naive_mean_per_image": self._safe_mean(df["ige_naive"]),
                "ige_naive_median_per_image": float(df["ige_naive"].median()),
                "ll_centerbias_bits": self._safe_mean(df["ll_centerbias_bits"]),
                "ll_gold_bits": self._safe_mean(df["ll_gold_bits"]),
                "ll_model_bits": self._safe_mean(df["ll_model_bits"]),
                "model": modality_results["model"],
                "msdb_dataset": modality_results["msdb_dataset"],
                "centerbias_path": str(modality_results["centerbias_path"]),
            })

            stems = example_stems or list(rng.choice(
                list(modality_results["images"]),
                size=min(n_examples, len(modality_results["images"])),
                replace=False,
            ))
            for stem in stems:
                self._plot_deepgaze_ige_example(
                    modality,
                    modality_results,
                    stem,
                    modality_dir / "visualisation" / f"{stem}.png",
                )

        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

        # RGB vs IR on the stems kept in both modalities. Positive delta: better on RGB.
        comparison_rows: list[dict[str, Any]] = []
        if {"VIS", "IR"} <= set(dataframes):
            visible, infrared = dataframes["VIS"], dataframes["IR"]
            for metric_name in ("ige", "ige_naive", "model_ig_bits"):
                for stem in sorted(set(visible.index) & set(infrared.index)):
                    score_rgb = float(visible.at[stem, metric_name])
                    score_ir = float(infrared.at[stem, metric_name])
                    comparison_rows.append({
                        "stem": stem,
                        "metric_type": "information_gain",
                        "metric_name": metric_name,
                        "score_rgb": score_rgb,
                        "score_ir": score_ir,
                        "delta_rgb_ir": score_rgb - score_ir,
                    })

        all_results = self._ensure_columns(pd.DataFrame(comparison_rows), comparison_columns)
        all_results.to_csv(output_dir / "all_results.csv", index=False)

        return all_results


    @staticmethod
    def _plot_centerbias(
        modality: str,
        modality_results: Mapping[str, Any],
        output_path: Path,
    ) -> None:
        # Displayed at the size of the most common image of the modality.
        sizes = pd.Series([(img.width, img.height) for img in modality_results["fixations"].values()])
        width, height = sizes.mode()[0]
        centerbias = modality_results["centerbias"]
        density = np.exp(_centerbias_log_map(centerbias, width, height))

        fig, ax = plt.subplots(figsize=(7, 7 * height / width + 0.6))
        im = ax.imshow(density, cmap="magma", extent=(0, width, height, 0))
        ax.contour(np.arange(width) + 0.5, np.arange(height) + 0.5, density,
                   levels=5, colors="white", linewidths=0.6, alpha=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="probability per pixel")
        ax.set_title(
            f"{modality} center bias ({len(sizes)} images, {len(centerbias.nx)} fixations)\n"
            f"sigma = {centerbias.sigma:.3f} (image fraction), uniform weight = {centerbias.eps:.3f}"
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


    @classmethod
    def _plot_gold_standard_example(
        cls,
        modality: str,
        modality_results: Mapping[str, Any],
        stem: str,
        image_path: Path | None,
        output_path: Path,
        scale: float = 2.0,
        dpi: int = 250,
    ) -> None:
        """
        2 x 2 figure: stimulus with fixations | gold standard
                      naive model             | gold standard - naive
        """
        img = modality_results["fixations"][stem]
        centerbias = modality_results["centerbiases"][stem]
        values = modality_results["images"][stem]

        log_gold = gold_standard_map(img, modality_results["gold_standard"], centerbias)
        log_naive = gold_standard_map(img, modality_results["naive_models"][stem])
        gold, naive = np.exp(log_gold), np.exp(log_naive)

        cls._plot_density_grid(
            img,
            image_path,
            f"{modality} {stem}: {len(img.x)} fixations, {values['n_participants']} participants",
            [
                (f"Gold standard (sigma = {modality_results['gold_standard'].sigma:.1f} px)\n"
                 f"IG = {values['explainable_ig_bits']:.3f} bit/fix", gold, "density"),
                (f"Naive KDE (sigma = 1 deg = {values['naive_sigma_px']:.1f} px)\n"
                 f"IG = {values['naive_ig_bits']:.3f} bit/fix", naive, "density"),
                ("Gold standard - naive\n(red: more mass in the gold standard)", gold - naive, "difference"),
            ],
            output_path,
            scale=scale,
            dpi=dpi,
        )


    @classmethod
    def _plot_deepgaze_ige_example(
        cls,
        modality: str,
        modality_results: Mapping[str, Any],
        stem: str,
        output_path: Path,
        scale: float = 2.0,
        dpi: int = 250,
    ) -> None:
        """
        2 x 2 figure: stimulus with fixations | gold standard
                      naive model             | DeepGaze
        """
        img = modality_results["fixations"][stem]
        centerbias = modality_results["centerbiases"][stem]
        values = modality_results["images"][stem]

        gold = np.exp(gold_standard_map(img, modality_results["gold_standard"], centerbias))
        naive = np.exp(gold_standard_map(img, modality_results["naive_models"][stem]))
        deepgaze = np.exp(modality_results["log_densities"][stem].astype(np.float64))
        deepgaze /= deepgaze.sum()

        cls._plot_density_grid(
            img,
            modality_results["image_paths"].get(stem),
            f"{modality} {stem}: {len(img.x)} fixations, {values['n_participants']} participants",
            [
                (f"Gold standard (sigma = {modality_results['gold_standard'].sigma:.1f} px)\n"
                 f"IG = {values['explainable_ig_bits']:.3f} bit/fix", gold, "density"),
                (f"Naive KDE (sigma = {modality_results['naive_models'][stem].sigma:.1f} px)\n"
                 f"IG = {values['naive_ig_bits']:.3f} bit/fix, IGE = {values['ige_naive']:.1%}",
                 naive, "density"),
                (f"DeepGaze {modality_results['model']}\n"
                 f"IG = {values['model_ig_bits']:.3f} bit/fix, IGE = {values['ige']:.1%}",
                 deepgaze, "density"),
            ],
            output_path,
            show_fixations=(True, True, False),
            scale=scale,
            dpi=dpi,
        )


    @staticmethod
    def _plot_density_grid(
        img,
        image_path: Path | None,
        stimulus_title: str,
        panels: Sequence[tuple[str, np.ndarray, str]],
        output_path: Path,
        show_fixations: Sequence[bool] = (True, True, True),
        scale: float = 2.0,
        dpi: int = 250,
    ) -> None:
        """
        2 x 2 figure: the stimulus with the fixations of img, then the three (title, data, kind)
        panels in reading order. kind "density" panels share one colour scale, so they are directly
        comparable; a "difference" panel gets its own symmetric scale. Densities are overlaid on the
        stimulus, with the fixations on the panels where show_fixations is True. Each panel is drawn
        at `scale` output pixels per image pixel.
        """
        width, height = img.width, img.height
        extent = (0, width, height, 0)

        if image_path is not None and Path(image_path).exists():
            stimulus = Loader.load_image(image_path)
        else:
            print(f"Stimulus not found ({image_path}); using a grey background.")
            stimulus = np.full((height, width), 128, dtype=np.uint8)
        # Grey version under the density overlays, so their colours stay readable.
        stimulus_grey = stimulus[..., :3].mean(axis=2) if stimulus.ndim == 3 else stimulus

        # Exact panel size in inches, plus margins for titles and colour bars.
        panel_w, panel_h = width * scale / dpi, height * scale / dpi
        left, gap, title, cbar = 0.1, 0.25, 0.75, 1.1
        fig_w = left + 2 * (panel_w + cbar) + gap + 0.2
        fig_h = 2 * (panel_h + title) + 0.1
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        def panel(row, col):
            x0 = left + col * (panel_w + cbar + gap)
            y0 = fig_h - (row + 1) * (panel_h + title)
            ax = fig.add_axes((x0 / fig_w, y0 / fig_h, panel_w / fig_w, panel_h / fig_h))
            cax = fig.add_axes(((x0 + panel_w + 0.1) / fig_w, y0 / fig_h, 0.15 / fig_w, panel_h / fig_h))
            if row == col == 0:
                ax.imshow(stimulus, cmap="gray", extent=extent)
            else:
                ax.imshow(stimulus_grey, cmap="gray", extent=extent, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax, cax

        def overlay(ax, data, cmap, vmin, vmax):
            """Colour-mapped data whose opacity grows with its magnitude."""
            norm = plt.Normalize(vmin, vmax)
            rgba = plt.get_cmap(cmap)(norm(data))
            magnitude = np.abs(data) / max(abs(vmin), abs(vmax))
            rgba[..., 3] = 0.85 * np.sqrt(np.clip(magnitude, 0, 1))
            ax.imshow(rgba, extent=extent)
            return plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        def colorbar(mappable, cax):
            # Always "x 1e-5"-style ticks, so the tick labels stay short.
            cb = fig.colorbar(mappable, cax=cax, label="probability per pixel")
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()

        def fixations(ax, size=14):
            ax.scatter(img.x, img.y, s=size, c="yellow", edgecolors="black", linewidths=0.5)

        ax, cax = panel(0, 0)
        fixations(ax, size=22)
        ax.set_title(stimulus_title)
        cax.axis("off")

        vmax = max(data.max() for _, data, kind in panels if kind == "density")
        for (row, col), (label, data, kind), show in zip(((0, 1), (1, 0), (1, 1)), panels, show_fixations):
            ax, cax = panel(row, col)
            if kind == "density":
                mappable = overlay(ax, data, "inferno", 0, vmax)
            else:
                limit = float(np.abs(data).max())
                mappable = overlay(ax, data, "RdBu_r", -limit, limit)
            if show:
                fixations(ax)
            ax.set_title(label)
            colorbar(mappable, cax)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)


    def save_expe1_analysis_old(
        self,
        analysis: Mapping[str, Any],
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe1" / "statistics"
            if output_dir is None
            else Path(output_dir)
        )

        return self._save_analysis(
            analysis=analysis,
            output_dir=output_dir,
        )

    def save_expe1_analysis(
    self,
    analysis: Mapping[str, Any],
    output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "statistics"
            if output_dir is None
            else Path(output_dir)
        )

        return self._save_analysis(
            analysis=analysis,
            output_dir=output_dir,
        )


    def save_expe2_analysis(
        self,
        analysis: Mapping[str, Any],
        output_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        output_dir = (
            self.output_dir / "expe2" / "statistics"
            if output_dir is None
            else Path(output_dir)
        )

        return self._save_analysis(
            analysis=analysis,
            output_dir=output_dir,
        )

    def _save_analysis_json(
        self,
        analysis: Mapping[str, Any],
        output_path: Path,
    ) -> None:
        if not isinstance(analysis, Mapping):
            raise TypeError(
                "analysis must be a mapping, "
                f"got {type(analysis).__name__}."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                self._to_json_compatible(analysis),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )


    def _to_csv_value(self, value: Any) -> Any:
        """
        Convertit une valeur de résultat en cellule compatible avec un CSV.

        Les objets comme les intervalles de confiance sont stockés sous
        forme de JSON dans une seule cellule.
        """
        if self._is_scalar_value(value):
            return self._to_python_scalar(value)

        return json.dumps(
            self._to_json_compatible(value),
            ensure_ascii=False,
        )


    @staticmethod
    def _is_scalar_value(value: Any) -> bool:
        return value is None or isinstance(
            value,
            (str, bytes, bool, int, float, np.generic, pd.Timestamp),
        )

    @classmethod
    def _to_json_compatible(cls, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return [
                {str(key): cls._to_json_compatible(item) for key, item in row.items()}
                for row in value.to_dict(orient="records")
            ]
        if isinstance(value, pd.Series):
            return {
                str(key): cls._to_json_compatible(item)
                for key, item in value.to_dict().items()
            }
        if isinstance(value, Mapping):
            return {
                str(key): cls._to_json_compatible(item)
                for key, item in value.items()
            }
        if isinstance(value, np.ndarray):
            return cls._to_json_compatible(value.tolist())
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return [cls._to_json_compatible(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return cls._to_json_compatible(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value)

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, float) and not np.isfinite(value):
            return np.nan
        return value

    def _save_results(
        self,
        results: Results,
        output_dir: Path,
        expected_columns: list[str],
        summary_measures: list[str],
    ) -> pd.DataFrame:

        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        all_dataframes: list[pd.DataFrame] = []

        for metric_type, metrics in results.items():
            metric_type_dir = output_dir / metric_type
            metric_type_dir.mkdir(parents=True, exist_ok=True)

            for metric_name, stem_results in metrics.items():
                rows: list[dict[str, Any]] = []

                for stem, values in stem_results.items():
                    if values is None:
                        continue

                    row = {
                        "stem": str(stem),
                        "metric_type": metric_type,
                        "metric_name": metric_name,
                    }

                    row.update(values)
                    rows.append(row)

                df = pd.DataFrame(rows)

                if df.empty:
                    df = pd.DataFrame(columns=expected_columns)
                else:
                    df = self._ensure_columns(df, expected_columns)
                    all_dataframes.append(df)

                # Un CSV par métrique
                metric_path = metric_type_dir / f"{metric_name}.csv"
                df.to_csv(metric_path, index=False)

                # Résumé moyen de la métrique
                summary_row = {
                    "metric_type": metric_type,
                    "metric_name": metric_name,
                }

                for measure in summary_measures:
                    summary_row[measure] = (
                        self._safe_mean(df[measure])
                        if measure in df.columns
                        else np.nan
                    )

                summary_rows.append(summary_row)

        # Résumé par métrique
        summary_columns = [
            "metric_type",
            "metric_name",
            *summary_measures,
        ]

        summary = pd.DataFrame(summary_rows)
        summary = self._ensure_columns(summary, summary_columns)
        summary.to_csv(output_dir / "summary.csv", index=False)

        # Toutes les observations individuelles
        if all_dataframes:
            all_results = pd.concat(
                all_dataframes,
                ignore_index=True,
            )
            all_results = self._ensure_columns(
                all_results,
                expected_columns,
            )
        else:
            all_results = pd.DataFrame(columns=expected_columns)

        all_results.to_csv(
            output_dir / "all_results.csv",
            index=False,
        )

        return all_results

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan

        return df[columns]

    @staticmethod
    def _safe_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce")
        if values.dropna().empty:
            return float("nan")
        return float(values.mean())
    
    
    def _save_analysis(
        self,
        analysis: Mapping[str, Any],
        output_dir: Path,
    ) -> pd.DataFrame:
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_analysis_json(
            analysis=analysis,
            output_path=output_dir / "analysis.json",
        )

        rows: list[dict[str, Any]] = []

        for metric_key, result in analysis.items():
            if not isinstance(result, Mapping):
                continue

            row = {
                "metric_key": str(metric_key),
                "metric_type": result.get("metric_type"),
                "metric_name": result.get("metric_name"),
                "test_name": result.get(
                    "test",
                    result.get("test_name"),
                ),
            }

            for key, value in result.items():
                if key in {
                    "metric_type",
                    "metric_name",
                    "test",
                    "test_name",
                }:
                    continue

                row[str(key)] = self._to_csv_value(value)

            rows.append(row)

        table = pd.DataFrame(rows)

        first_columns = [
            "metric_key",
            "metric_type",
            "metric_name",
            "test_name",
        ]

        if table.empty:
            table = pd.DataFrame(columns=first_columns)
        else:
            remaining_columns = [
                column
                for column in table.columns
                if column not in first_columns
            ]

            table = table[first_columns + remaining_columns]

        table.to_csv(
            output_dir / "analysis.csv",
            index=False,
        )

        return table

    # ------------------------------------------------------------------
    # ROI ANALYSIS (Comparator.run_deepgaze_roi)
    # ------------------------------------------------------------------

    # Fixed colour per class position (sorted class ids); the 9th and later classes are white
    # dashed boxes. Every box also carries its class name, so colour is never the only cue.
    ROI_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
    ROI_SOURCE_COLORS = {"deepgaze": "#2a78d6", "human": "#eb6834", "centerbias": "#1baf7a"}
    ROI_METRICS = ("mass", "ratio", "density", "normalized_ratio")

    def save_deepgaze_roi(
        self,
        results: Mapping[str, Mapping[str, Any]],
        output_dir: str | Path | None = None,
        visualise: bool = True,
        n_visualisations: int | None = None,
        seed: int | None = 0,
    ) -> pd.DataFrame:
        """
        Save the output of Comparator.run_deepgaze_roi.

            <modality>/images.csv   one row per stem, class and source (deepgaze, human, fixations,
                                    centerbias) with the ROI metrics
            per_class.csv           one row per modality, source and class: number of images
                                    containing the class, mean / std / sem / median of each metric
            all_results.csv         RGB vs IR on the stems where the class is annotated in both
                                    modalities, in the run_expe2 format (metric_type =
                                    "roi_<source>_<class>", metric_name = metric), so it can be
                                    passed to Analyzer.analyze_expe2
            per_class.png           ratio and normalized ratio per class, modality and source
            visualisation/<stem>.png  stimulus, DeepGaze and human maps with the ROIs, VIS and IR
                                    (every stem, or n_visualisations random stems; needs the maps,
                                    i.e. keep_maps=True in run_deepgaze_roi)

        Returns the per_class table.
        """
        output_dir = (
            self.output_dir / "expe3" / "roi"
            if output_dir is None
            else Path(output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_columns = [
            "modality", "stem", "class_id", "class_name", "source", *self.ROI_METRICS,
            "n_rois", "roi_area_px", "roi_area_fraction", "n_fixations", "n_participants",
        ]
        comparison_columns = ["stem", "metric_type", "metric_name", "score_rgb", "score_ir", "delta_rgb_ir"]

        frames = []
        for modality, modality_results in results.items():
            df = self._ensure_columns(pd.DataFrame(modality_results["rows"]), image_columns)
            df["class_id"] = df["class_id"].astype("Int64")
            modality_dir = output_dir / modality
            modality_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(modality_dir / "images.csv", index=False)
            frames.append(df)
        all_rows = pd.concat(frames, ignore_index=True)

        # Per-class summary. Classes keep the order of their ids, "all" last.
        class_order = self._roi_class_order(results)
        summary_rows = []
        for (modality, source, class_name), group in all_rows.groupby(
            ["modality", "source", "class_name"], sort=False
        ):
            row = {
                "modality": modality,
                "source": source,
                "class_id": group["class_id"].iloc[0],
                "class_name": class_name,
                "n_images": int(group["stem"].nunique()),
                "n_rois": int(group["n_rois"].sum()),
                "roi_area_fraction_mean": self._safe_mean(group["roi_area_fraction"]),
            }
            for metric in self.ROI_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
                row[f"{metric}_std"] = float(values.std()) if len(values) > 1 else float("nan")
                row[f"{metric}_sem"] = float(values.sem()) if len(values) > 1 else float("nan")
                row[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
            summary_rows.append(row)
        per_class = pd.DataFrame(summary_rows)
        if not per_class.empty:
            per_class["_order"] = per_class["class_name"].map(class_order.index)
            per_class = (
                per_class.sort_values(["modality", "source", "_order"], kind="stable")
                .drop(columns="_order")
                .reset_index(drop=True)
            )
        per_class.to_csv(output_dir / "per_class.csv", index=False)

        # RGB vs IR, stems where the class is annotated in both modalities. Positive: higher on RGB.
        comparison_rows = []
        if {"VIS", "IR"} <= set(all_rows["modality"]):
            keys = ["stem", "class_name", "source"]
            paired = all_rows[all_rows["modality"] == "VIS"].merge(
                all_rows[all_rows["modality"] == "IR"], on=keys, suffixes=("_rgb", "_ir")
            )
            for _, row in paired.iterrows():
                for metric in self.ROI_METRICS:
                    score_rgb, score_ir = float(row[f"{metric}_rgb"]), float(row[f"{metric}_ir"])
                    comparison_rows.append({
                        "stem": row["stem"],
                        "metric_type": f"roi_{row['source']}_{row['class_name']}",
                        "metric_name": metric,
                        "score_rgb": score_rgb,
                        "score_ir": score_ir,
                        "delta_rgb_ir": score_rgb - score_ir,
                    })
        all_results = self._ensure_columns(pd.DataFrame(comparison_rows), comparison_columns)
        all_results.to_csv(output_dir / "all_results.csv", index=False)

        with (output_dir / "settings.json").open("w", encoding="utf-8") as file:
            first = next(iter(results.values()))
            json.dump(self._to_json_compatible({
                "model": first["model"],
                "msdb_dataset": first["msdb_dataset"],
                "sigma_deg": first["sigma_deg"],
                "class_names": first["class_names"],
                "centerbias_paths": {m: str(r["centerbias_path"]) for m, r in results.items()},
                "n_images": {m: len({row["stem"] for row in r["rows"]}) for m, r in results.items()},
            }), file, indent=2)

        if not per_class.empty:
            self._plot_roi_per_class(per_class, class_order, output_dir / "per_class.png")

        if visualise:
            stems = sorted(set.intersection(*(set(r["maps"]) for r in results.values())))
            if not stems:
                print("No maps kept (keep_maps=False): skipping the ROI visualisations.")
            if n_visualisations is not None and n_visualisations < len(stems):
                rng = np.random.default_rng(seed)
                stems = sorted(rng.choice(stems, size=n_visualisations, replace=False))
            for stem in tqdm(stems, desc="Saving ROI visualisations"):
                self._plot_deepgaze_roi_example(
                    results, stem, all_rows, output_dir / "visualisation" / f"{stem}.png"
                )

        return per_class


    @staticmethod
    def _roi_class_order(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
        names = {}
        for modality_results in results.values():
            names.update(modality_results["class_names"])
        return [names[c] for c in sorted(names)] + ["all"]


    @classmethod
    def _roi_style(cls, class_index: int) -> tuple[str, str]:
        """(colour, linestyle) of the class at this position among the sorted class ids."""
        if class_index < len(cls.ROI_COLORS):
            return cls.ROI_COLORS[class_index], "-"
        return "#ffffff", "--"


    @classmethod
    def _plot_roi_per_class(cls, per_class: pd.DataFrame, class_order: list[str], output_path: Path) -> None:
        """
        2 x 2 grid: rows = ratio, normalized ratio; columns = modality. One group of bars per
        class (mean over the images containing it, error bar = sem), one bar per source.
        """
        sources = [s for s in cls.ROI_SOURCE_COLORS if s in set(per_class["source"])]
        modalities = [m for m in ("VIS", "IR") if m in set(per_class["modality"])]
        classes = [c for c in class_order if c in set(per_class["class_name"])]
        metrics = (("ratio", "Mass in the ROI / total mass"),
                   ("normalized_ratio", "Normalized ratio (1 = chance)"))

        fig, axes = plt.subplots(
            len(metrics), len(modalities),
            figsize=(max(6.0, 0.9 * len(classes)) * len(modalities), 4.2 * len(metrics)),
            sharey="row", squeeze=False, layout="constrained",
        )
        x = np.arange(len(classes))
        width = 0.8 / len(sources)
        for row, (metric, ylabel) in enumerate(metrics):
            for col, modality in enumerate(modalities):
                ax = axes[row, col]
                table = per_class[per_class["modality"] == modality].set_index(["source", "class_name"])
                for k, source in enumerate(sources):
                    means = [table[f"{metric}_mean"].get((source, c), np.nan) for c in classes]
                    sems = [table[f"{metric}_sem"].get((source, c), np.nan) for c in classes]
                    ax.bar(x + (k - (len(sources) - 1) / 2) * width, means, width * 0.92,
                           yerr=sems, color=cls.ROI_SOURCE_COLORS[source], label=source,
                           error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "0.3"})
                if metric == "normalized_ratio":
                    ax.axhline(1.0, color="0.4", linewidth=1, linestyle="--", zorder=0)
                n_images = [
                    int(table["n_images"].get((sources[0], c), 0)) if sources else 0 for c in classes
                ]
                ax.set_xticks(x, [f"{c}\n(n={n})" for c, n in zip(classes, n_images)],
                              rotation=45 if len(classes) > 6 else 0, ha="right" if len(classes) > 6 else "center",
                              fontsize=8)
                ax.grid(axis="y", color="0.9", linewidth=0.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(modality)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=9, ncol=len(sources), loc="outside lower center")
        fig.suptitle("ROI metrics per class (mean over the images containing the class, ± sem)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


    @classmethod
    def _plot_deepgaze_roi_example(
        cls,
        results: Mapping[str, Mapping[str, Any]],
        stem: str,
        all_rows: pd.DataFrame,
        output_path: Path,
        dpi: int = 150,
    ) -> None:
        """
        One row per modality: stimulus with the ROIs and the fixations | DeepGaze | human saliency,
        both maps with the ROIs. The two maps of a row share one colour scale.
        """
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle

        modalities = [m for m in ("VIS", "IR") if m in results and stem in results[m]["maps"]]
        class_ids = sorted({c for r in results.values() for c in r["class_names"]})
        class_names = {c: n for r in results.values() for c, n in r["class_names"].items()}

        first = results[modalities[0]]["maps"][stem]["deepgaze"]
        height, width = first.shape
        panel_w = 4.6
        fig, axes = plt.subplots(
            len(modalities), 3, figsize=(3 * panel_w + 1.0, len(modalities) * panel_w * height / width + 1.6),
            squeeze=False, layout="constrained",
        )

        def draw_rois(ax, rois, labels: bool):
            for roi in rois:
                color, linestyle = cls._roi_style(class_ids.index(roi.class_id))
                ax.add_patch(Rectangle((roi.x_min, roi.y_min), roi.width, roi.height, fill=False,
                                       edgecolor=color, linestyle=linestyle, linewidth=1.2))
                if labels:
                    ax.text(roi.x_min, roi.y_min - 1, class_names[roi.class_id], color=color, fontsize=5,
                            va="bottom", ha="left",
                            bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5, "linewidth": 0})

        def roi_summary(modality, source):
            rows = all_rows[(all_rows["modality"] == modality) & (all_rows["stem"] == stem)
                            & (all_rows["source"] == source) & (all_rows["class_name"] == "all")]
            if rows.empty:
                return "no ROI"
            row = rows.iloc[0]
            return f"{row['ratio']:.1%} of the mass in ROIs ({row['roi_area_fraction']:.1%} of the area)"

        for r, modality in enumerate(modalities):
            res = results[modality]
            maps = res["maps"][stem]
            rois = res["rois"][stem]
            image_path = res["image_paths"].get(stem)
            if image_path is not None and Path(image_path).exists():
                stimulus = Loader.load_image(image_path)
            else:
                stimulus = np.full(maps["deepgaze"].shape, 128, dtype=np.uint8)
            grey = stimulus[..., :3].mean(axis=2) if stimulus.ndim == 3 else stimulus
            vmax = float(max(maps["deepgaze"].max(), maps["human"].max()))
            norm = plt.Normalize(0, vmax)

            ax = axes[r, 0]
            ax.imshow(stimulus, cmap="gray")
            draw_rois(ax, rois, labels=True)
            img = res["fixations"].get(stem)
            if img is not None:
                ax.scatter(img.x, img.y, s=8, c="yellow", edgecolors="black", linewidths=0.4)
            n_fix = 0 if img is None else len(img.x)
            ax.set_title(f"{modality} {stem}: {len(rois)} ROIs, {n_fix} fixations", fontsize=9)

            for c, (source, label) in enumerate((("deepgaze", f"DeepGaze {res['model']}"),
                                                 ("human", f"Human saliency (sigma = {res['sigma_deg']:g} deg)")),
                                                start=1):
                ax = axes[r, c]
                ax.imshow(grey, cmap="gray", alpha=0.6)
                rgba = plt.get_cmap("inferno")(norm(maps[source]))
                rgba[..., 3] = 0.85 * np.sqrt(np.clip(maps[source] / vmax, 0, 1))
                ax.imshow(rgba)
                draw_rois(ax, rois, labels=False)
                ax.set_title(f"{label}\n{roi_summary(modality, source)}", fontsize=9)
            # One colour bar for the row, so the three panels keep the same size.
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="inferno"), ax=axes[r, :],
                              fraction=0.02, pad=0.01, label="probability per pixel")
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        present = sorted({roi.class_id for m in modalities for roi in results[m]["rois"][stem]})
        handles = [
            Line2D([], [], color=cls._roi_style(class_ids.index(c))[0],
                   linestyle=cls._roi_style(class_ids.index(c))[1], label=class_names[c])
            for c in present
        ]
        if handles:
            legend = fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 9),
                                frameon=True, fontsize=8, title="ROI class")
            # Dark frame, so the white boxes of the 9th+ classes stay visible.
            legend.get_frame().set_facecolor("0.25")
            plt.setp(legend.get_texts(), color="white")
            legend.get_title().set_color("white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

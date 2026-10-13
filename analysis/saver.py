from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                (f"DeepGaze IIE\n"
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


    def save_expe1_analysis(
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
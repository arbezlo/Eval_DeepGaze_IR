from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
from saliency.deepgaze import DeepGaze


class Visualizer:
    REQUIRED_COLUMNS = {
        "participant_id", "image_id", "image_name",
        "x", "y",
        "screen_width", "screen_height",
        "stimuli_width", "stimuli_height",
        "image_width", "image_height",
    }

    def __init__(
        self,
        visible_dir: str | Path,
        infrared_dir: str | Path,
        alpha: float = 0.6,
        show_fixation_ids: bool = True,
        keep_only_inside_image: bool = True,
    ) -> None:
        self.infrared_image_dir = Path(infrared_dir)
        self.visible_image_dir = Path(visible_dir)
        self.alpha = alpha
        self.show_fixation_ids = show_fixation_ids
        self.keep_only_inside_image = keep_only_inside_image

        if not self.infrared_image_dir.is_dir():
            raise FileNotFoundError(
                f"Infrared images directory not found: "
                f"{self.infrared_image_dir}"
            )

        if not self.visible_image_dir.is_dir():
            raise FileNotFoundError(
                f"Visible images directory not found: "
                f"{self.visible_image_dir}"
            )

    def show_fixations(
        self,
        fixations_df: pd.DataFrame | str | Path,
        participant_id: Optional[str] = None,
        image_id: Optional[int | str] = None,
        save_dir: Optional[str | Path] = None,
        point_size: float = 40.0,
        modality: Optional[str] = None,
    ) -> None:
        images_dir = self._get_image_dir(modality)
        df = self._prepare_df(
            fixations_df,
            participant_id=participant_id,
            image_id=image_id,
        )
        if df.empty:
            print("[VISUALIZER] No fixation to display.")
            return

        groups = self._groups(df, ["participant_id", "image_id", "image_name"])

        def draw_panel(ax, key, group):
            participant, img_id, image_name = key
            image, _ = self._load_image(image_name, images_dir=images_dir)

            if image is None:
                ax.set_title(f"Image not found: {image_name}")
                ax.axis("off")
                return

            h, w = image.shape[:2]
            group = self._rescale_to_loaded_image(group, w, h)

            self._imshow(ax, image)
            ax.scatter(
                group["x_image"], group["y_image"],
                c="red", s=point_size, alpha=self.alpha,
            )

            if self.show_fixation_ids and "fixation_id" in group.columns:
                for _, row in group.iterrows():
                    if pd.notna(row["fixation_id"]):
                        ax.text(
                            row["x_image"],
                            row["y_image"],
                            str(row["fixation_id"]),
                            fontsize=8,
                        )

            #ax.set_title(f"Participant {participant} | Image {img_id} | {image_name} | n={len(group)}")
            self._format_axis(ax, w, h)

        self._browse_single(
            groups=groups,
            draw_func=draw_panel,
            save_dir=save_dir,
            filename_func=lambda key: (
                f"fixations_participant_{key[0]}_image_{key[1]}_{Path(str(key[2])).stem}.png"
            ),
        )

    def show_heatmaps(
        self,
        fixations_df: pd.DataFrame | str | Path,
        image_id: Optional[int | str] = None,
        save_dir: Optional[str | Path] = None,
        sigma: float = 35.0,
        heatmap_alpha: float = 0.45,
        cmap: str = "jet",
        use_duration_weights: bool = False,
        modality: Optional[str] = None,
        samples=None,
    ) -> None:
        images_dir = self._get_image_dir(modality)
        df = self._prepare_df(fixations_df, image_id=image_id)
        if df.empty:
            print("[VISUALIZER] No fixation to display.")
            return
    
        print(np.unique(df["participant_id"].loc[df['image_id']==image_id].values))
        if samples != None : 

            df = df.loc[df["participant_id"].isin(samples)]
            groups = self._groups(df, ["image_id", "image_name"])
        else : 
            groups = self._groups(df, ["image_id", "image_name"])
       
        print(groups) 
        def draw_panel(ax, key, group):
            img_id, image_name = key
            image, _ = self._load_image(image_name, images_dir=images_dir)

            if image is None:
                ax.set_title(f"Image not found: {image_name}")
                ax.axis("off")
                return

            h, w = image.shape[:2]
            group = self._rescale_to_loaded_image(group, w, h)

            self._imshow(ax, image)
            self._imshow_heatmap(ax, group, w, h, sigma, heatmap_alpha, cmap, use_duration_weights)

            # ax.set_title(
            #     f"Aggregated heatmap | Image {img_id} | {image_name} | "
            #     f"fixations={len(group)} | participants={group['participant_id'].nunique()}"
            # )
            self._format_axis(ax, w, h)

        self._browse_single(
            groups=groups,
            draw_func=draw_panel,
            save_dir=save_dir,
            filename_func=lambda key: f"heatmap_aggregated_image_{key[0]}_{Path(str(key[1])).stem}.png",
        )

    
    def compare_modalities_heatmaps(
        self,
        visible_fixations_df: pd.DataFrame | str | Path,
        infrared_fixations_df: pd.DataFrame | str | Path,
        save_dir: Optional[str | Path] = None,
        sigma: float = 35.0,
        heatmap_alpha: float = 0.45,
        cmap: str = "jet",
        use_duration_weights: bool = False,
        visible_title: str = "VISIBLE",
        infrared_title: str = "INFRARED",
    ) -> None:
        """
        Compare les heatmaps agrégées visible / infrared pour les images
        dont le stem est commun aux deux fichiers de fixations.

        Navigation : n/right/down = suivant, p/left/up = précédent, s = sauvegarder, q = fermer.
        """
        visible_images_dir = self.visible_image_dir
        infrared_images_dir = self.infrared_image_dir

        if not visible_images_dir.exists():
            raise FileNotFoundError(f"Visible images directory not found: {visible_images_dir}")
        if not infrared_images_dir.exists():
            raise FileNotFoundError(f"Infrared images directory not found: {infrared_images_dir}")

        visible_df = self._prepare_df(visible_fixations_df)
        infrared_df = self._prepare_df(infrared_fixations_df)

        if visible_df.empty or infrared_df.empty:
            print("[VISUALIZER] One modality has no fixation to display.")
            return

        visible_groups = {stem: group.copy() for stem, group in visible_df.groupby("image_stem", sort=True)}
        infrared_groups = {stem: group.copy() for stem, group in infrared_df.groupby("image_stem", sort=True)}

        stems = sorted(set(visible_groups) & set(infrared_groups))
        if not stems:
            print("[VISUALIZER] No common image stem found between modalities.")
            return

        def draw_pair(stem):
            panels = []
            for title, group, images_dir in [
                (visible_title, visible_groups[stem], visible_images_dir),
                (infrared_title, infrared_groups[stem], infrared_images_dir),
            ]:
                image_name = str(group["image_name"].iloc[0])
                image_path = self._find_image(image_name, images_dir=images_dir)
                if image_path is None:
                    panels.append((None, f"{title} | image not found: {image_name}"))
                    continue

                image = mpimg.imread(image_path)
                h, w = image.shape[:2]
                group = self._rescale_to_loaded_image(group, w, h)
                panels.append((
                    group,
                    f"{title} | {image_name}\nfixations={len(group)} | participants={group['participant_id'].nunique()}",
                    image, w, h,
                ))

            return panels, f"Common stem: {stem}"

        self._browse_compare_independent(
            keys=stems,
            resolve_func=draw_pair,
            heatmap_kwargs=dict(sigma=sigma, alpha=heatmap_alpha, cmap=cmap, use_duration_weights=use_duration_weights),
            save_dir=save_dir,
            filename_func=lambda stem: f"compare_visible_infrared_heatmaps_{stem}.png",
        )

    def compare_deepgaze_human_heatmaps(
        self,
        fixations_df: pd.DataFrame | str | Path,
        deepgaze_predictor: DeepGaze,
        modality: str,
        image_extensions: tuple[str, ...] = (".png", ".jpg"),
        save_dir: Optional[str | Path] = None,
        channel_mode: str = "auto",
        sigma: float = 35.0,
        heatmap_alpha: float = 0.45,
        cmap: str = "jet",
        use_duration_weights: bool = False,
        normalize_for_display: bool = True,
    ) -> None:
        """
        Compare, pour chaque image du dossier, la prédiction DeepGaze
        avec la carte de saillance humaine générée à partir des fixations.
        """

        df_all = self._prepare_df(fixations_df)
        if df_all.empty:
            print("[VISUALIZER] No human fixation found in the provided dataset.")
            return
        
        images_dir = self._get_image_dir(modality)
        image_paths = self._list_images(images_dir, image_extensions)

        pairs = []
        for image_path in image_paths:
            group_img = df_all[df_all["image_name"] == image_path.name]
            if group_img.empty:
                print(f"[VISUALIZER] Missing human fixations for image: {image_path.name} (Skipped)")
                continue
            pairs.append((image_path, group_img))

        if not pairs:
            print("[VISUALIZER] No intersection between images folder and fixations data.")
            return

        save_dir = Path(save_dir) if save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        idx = [0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        def draw():
            image_path, group_img = pairs[idx[0]]
            image = self._read_image_path(image_path)

            for ax in axes:
                ax.clear()

            if image is None:
                fig.suptitle(f"Image not readable: {image_path}")
                fig.canvas.draw_idle()
                return

            h, w = image.shape[:2]

            prediction = deepgaze_predictor.predict(image)
            deepgaze_map = self._prepare_saliency_for_display(
                prediction, width=w, height=h, normalize=normalize_for_display
            )

            group_rescaled = self._rescale_to_loaded_image(group_img, w, h)
            reference_map = self._heatmap(group_rescaled, w, h, sigma, use_duration_weights)
            if normalize_for_display:
                reference_map = self._prepare_saliency_for_display(reference_map, width=w, height=h, normalize=True)

            self._imshow(axes[0], image)
            axes[0].imshow(deepgaze_map, cmap=cmap, alpha=heatmap_alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            axes[0].set_title("DeepGaze prediction")
            self._format_axis(axes[0], w, h)

            self._imshow(axes[1], image)
            axes[1].imshow(reference_map, cmap=cmap, alpha=heatmap_alpha, interpolation="bilinear", vmin=0.0, vmax=1.0)
            axes[1].set_title(f"Human Saliency")
            self._format_axis(axes[1], w, h)

            fig.suptitle(f"[{idx[0] + 1}/{len(pairs)}] {image_path.name}", fontsize=12)
            plt.tight_layout()
            fig.canvas.draw_idle()

        def save():
            if save_dir is None:
                print("[VISUALIZER] No save_dir specified.")
                return
            image_path, _ = pairs[idx[0]]
            output_path = save_dir / f"compare_deepgaze_human_{image_path.stem}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[VISUALIZER] Saved comparison to: {output_path}")

        self._connect_keys(fig, idx, len(pairs), draw, save)
        draw()
        plt.show()


    def _read_df(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        df = pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()
        missing = self.REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        return df

    def _prepare_df(
        self,
        data: pd.DataFrame | str | Path,
        participant_id: Optional[str] = None,
        image_id: Optional[int | str] = None,
    ) -> pd.DataFrame:
        df = self._read_df(data).copy()

        if participant_id is not None:
            df = df[df["participant_id"].astype(str) == str(participant_id)]
            
        if image_id is not None:
            df = df[df["image_id"].astype(str) == str(image_id)]
            
        if df.empty:
            return df

        numeric_columns = [
            "x", "y",
            "screen_width", "screen_height",
            "stimuli_width", "stimuli_height",
            "image_width", "image_height",
        ]
        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        offset_x = (df["screen_width"] - df["stimuli_width"]) / 2
        offset_y = (df["screen_height"] - df["stimuli_height"]) / 2

        x_stim = df["x"] - offset_x
        y_stim = df["y"] - offset_y

        valid = (
            df[numeric_columns].notna().all(axis=1)
            & (df["screen_width"] > 0)
            & (df["screen_height"] > 0)
            & (df["stimuli_width"] > 0)
            & (df["stimuli_height"] > 0)
            & (df["image_width"] > 0)
            & (df["image_height"] > 0)
        )

        if self.keep_only_inside_image:
            valid &= (
                (x_stim >= 0)
                & (x_stim < df["stimuli_width"])
                & (y_stim >= 0)
                & (y_stim < df["stimuli_height"])
            )

        df = df.loc[valid].copy()
        x_stim = x_stim.loc[valid]
        y_stim = y_stim.loc[valid]

        if df.empty:
            return df

        df["x_image"] = np.floor(
            x_stim * df["image_width"] / df["stimuli_width"]
        ).astype(int)

        df["y_image"] = np.floor(
            y_stim * df["image_height"] / df["stimuli_height"]
        ).astype(int)

        return df

    @staticmethod
    def _groups(df: pd.DataFrame, cols: list[str]):
        return list(df.groupby(cols, sort=True))

    def _get_image_dir(
        self,
        modality: Optional[str],
    ) -> Optional[Path]:
        if modality is None:
            return None

        modality = modality.lower()

        if modality == "visible":
            return self.visible_image_dir
        if modality == "infrared":
            return self.infrared_image_dir

        raise ValueError(
            "Unknown modality. Expected 'infrared' or 'visible'."
        )

    def _find_image(
        self,
        image_name: str,
        images_dir: Optional[str | Path] = None,
    ) -> Optional[Path]:
        if pd.isna(image_name):
            return None

        image_name = str(image_name)
        filename = Path(image_name).name

        if images_dir is None:
            directories = [
                self.visible_image_dir,
                self.infrared_image_dir,
            ]
        else:
            directories = [Path(images_dir)]

        for directory in directories:
            direct_path = directory / image_name
            if direct_path.is_file():
                return direct_path

            matches = list(directory.rglob(filename))
            if matches:
                return matches[0]

        return None

    def _load_image(
        self,
        image_name: str,
        images_dir: Optional[str | Path] = None,
    ) -> tuple[Optional[np.ndarray], Optional[Path]]:
        path = self._find_image(image_name, images_dir=images_dir)

        if path is None:
            return None, None

        try:
            return mpimg.imread(path), path
        except (OSError, ValueError):
            return None, path

    @staticmethod
    def _read_image_path(image_path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            return None

        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return image

    @staticmethod
    def _list_images(
        images_dir: str | Path,
        extensions: tuple[str, ...],
    ) -> list[Path]:
        images_dir = Path(images_dir)
        extensions = tuple(ext.lower() for ext in extensions)

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}"
            )

        return sorted(
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _imshow(ax, image) -> None:
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)

    @staticmethod
    def _format_axis(ax, width: int, height: int) -> None:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xlabel("x image")
        ax.set_ylabel("y image")

    @staticmethod
    def _rescale_to_loaded_image(
        group: pd.DataFrame,
        loaded_width: int,
        loaded_height: int,
    ) -> pd.DataFrame:
        group = group.copy()

        if group.empty:
            return group

        ref_w = float(group["image_width"].iloc[0])
        ref_h = float(group["image_height"].iloc[0])

        if ref_w <= 0 or ref_h <= 0:
            raise ValueError("Image dimensions must be strictly positive.")

        group["x_image"] = (
            group["x_image"] * loaded_width / ref_w
        )
        group["y_image"] = (
            group["y_image"] * loaded_height / ref_h
        )

        return group

    def _heatmap(self, group: pd.DataFrame, width: int, height: int, sigma: float, use_duration_weights: bool) -> np.ndarray:
        heatmap = np.zeros((height, width), dtype=np.float32)

        x = np.rint(group["x_image"].to_numpy()).astype(int)
        y = np.rint(group["y_image"].to_numpy()).astype(int)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[valid], y[valid]

        if len(x) == 0:
            return heatmap

        if use_duration_weights and "duration" in group.columns:
            weights = group["duration"].to_numpy(dtype=np.float32)[valid]
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = weights / weights.mean() if weights.mean() > 0 else np.ones_like(weights)
        else:
            weights = np.ones(len(x), dtype=np.float32)

        np.add.at(heatmap, (y, x), weights)
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def _imshow_heatmap(self, ax, group, width, height, sigma, alpha, cmap, use_duration_weights) -> None:
        ax.imshow(
            self._heatmap(group, width, height, sigma, use_duration_weights),
            cmap=cmap, alpha=alpha, interpolation="bilinear", vmin=0.0, vmax=1.0,
        )

    @staticmethod
    def _prepare_saliency_for_display(saliency_map: np.ndarray, width: int, height: int, normalize: bool = True) -> np.ndarray:
        saliency_map = np.asarray(saliency_map, dtype=np.float32)

        if saliency_map.ndim == 3:
            if saliency_map.shape[-1] in (1, 3, 4):
                saliency_map = saliency_map[..., 0]
            elif saliency_map.shape[0] in (1, 3, 4):
                saliency_map = saliency_map[0]
            else:
                raise ValueError(f"Unsupported saliency map shape: {saliency_map.shape}")

        if saliency_map.ndim != 2:
            raise ValueError(f"Expected a 2D saliency map, got: {saliency_map.shape}")

        if saliency_map.shape != (height, width):
            saliency_map = cv2.resize(saliency_map, (width, height), interpolation=cv2.INTER_LINEAR)

        saliency_map = np.nan_to_num(
            saliency_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if normalize:
            saliency_map = saliency_map - saliency_map.min()
            max_value = saliency_map.max()
            if max_value > 0:
                saliency_map = saliency_map / max_value

        return saliency_map.astype(np.float32)

    # ------------------------------------------------------------------
    # Navigation (shared by all interactive browsers)
    # ------------------------------------------------------------------

    @staticmethod
    def _connect_keys(fig, idx: list[int], n: int, draw: Callable, save: Callable) -> None:
        def on_key(event):
            if event.key in {"right", "down", "n"}:
                idx[0] = (idx[0] + 1) % n
                draw()
            elif event.key in {"left", "up", "p"}:
                idx[0] = (idx[0] - 1) % n
                draw()
            elif event.key == "s":
                save()
            elif event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

    def _browse_single(self, groups: list, draw_func: Callable, save_dir: Optional[str | Path], filename_func: Callable) -> None:
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        idx = [0]
        fig, ax = plt.subplots(figsize=(10, 7))

        def draw():
            ax.clear()
            key, group = groups[idx[0]]
            draw_func(ax, key, group)
            #fig.suptitle(f"[{idx[0] + 1}/{len(groups)}]", fontsize=12)
            plt.tight_layout()
            fig.canvas.draw_idle()

        def save():
            if save_dir is None:
                print("[VISUALIZER] No save_dir specified.")
                return
            key, _ = groups[idx[0]]
            output = save_dir / filename_func(key)
            ax.set_axis_off()
            fig.savefig(output, dpi=600, bbox_inches="tight",pad_inches=0)
            print(f"[VISUALIZER] Saved image to: {output}")

        self._connect_keys(fig, idx, len(groups), draw, save)
        draw()
        plt.show()
        save()

    def _browse_compare(
        self,
        keys: list,
        n_panels: int,
        resolve_func: Callable,
        heatmap_kwargs: dict,
        save_dir: Optional[str | Path],
        filename_func: Callable,
    ) -> None:
        """
        Shared background image, N heatmap panels drawn side by side
        (used by compare_raw_clean_heatmaps).
        """
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        idx = [0]
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(8 * n_panels, 7),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes)

        def draw():
            key = keys[idx[0]]
            result, suptitle = resolve_func(key)

            for ax in axes:
                ax.clear()

            if result is None:
                fig.suptitle(f"[{idx[0] + 1}/{len(keys)}] {suptitle}")
                fig.canvas.draw_idle()
                return

            image, w, h, panels = result
            for ax, (group, title) in zip(axes, panels):
                self._imshow(ax, image)
                if group is not None and not group.empty:
                    self._imshow_heatmap(ax, group, w, h, **heatmap_kwargs)
                ax.set_title(title)
                self._format_axis(ax, w, h)

            fig.suptitle(f"[{idx[0] + 1}/{len(keys)}] {suptitle}", fontsize=12)
            plt.tight_layout()
            fig.canvas.draw_idle()

        def save():
            if save_dir is None:
                print("[VISUALIZER] No save_dir specified.")
                return
            output = save_dir / filename_func(keys[idx[0]])
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"[VISUALIZER] Saved comparison to: {output}")

        self._connect_keys(fig, idx, len(keys), draw, save)
        draw()
        plt.show()

    def _browse_compare_independent(
        self,
        keys: list,
        resolve_func: Callable,
        heatmap_kwargs: dict,
        save_dir: Optional[str | Path],
        filename_func: Callable,
    ) -> None:
        """
        Each panel has its own background image (used by
        compare_modalities_heatmaps, where visible/infrared images differ).
        """
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        idx = [0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False, sharey=False)

        def draw():
            key = keys[idx[0]]
            panels, suptitle = resolve_func(key)

            for ax in axes:
                ax.clear()

            for ax, panel in zip(axes, panels):
                if panel[0] is None:
                    ax.set_title(panel[1])
                    ax.axis("off")
                    continue
                group, title, image, w, h = panel
                self._imshow(ax, image)
                self._imshow_heatmap(ax, group, w, h, **heatmap_kwargs)
                ax.set_title(title)
                self._format_axis(ax, w, h)

            fig.suptitle(f"[{idx[0] + 1}/{len(keys)}] {suptitle}", fontsize=12)
            plt.tight_layout()
            fig.canvas.draw_idle()

        def save():
            if save_dir is None:
                print("[VISUALIZER] No save_dir specified.")
                return
            output = save_dir / filename_func(keys[idx[0]])
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"[VISUALIZER] Saved modality comparison to: {output}")

        self._connect_keys(fig, idx, len(keys), draw, save)
        draw()
        plt.show()


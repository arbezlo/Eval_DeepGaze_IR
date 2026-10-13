from __future__ import annotations

from pathlib import Path
from typing import Optional
from scipy.special import logsumexp
from deepgaze_pytorch import DeepGazeIIE, DeepGazeMSDB

import cv2
import numpy as np
import torch


MODELS = {"IIE": DeepGazeIIE, "MSDB": DeepGazeMSDB}


class DeepGaze:


    def __init__(
        self,
        device: Optional[str] = None,
        centerbias_path: Optional[str | Path] = None,
        centerbias_type: Optional[str] = "uniform",
        centerbias_size: Optional[tuple] = (1024, 1024),
        model: str = "IIE",
        msdb_dataset: Optional[int] = None,
    ) -> None:
        """
        model        : "IIE" (DeepGaze IIE) or "MSDB" (DeepGaze MSDB, Kümmerer et al., ICCV 2025).
        msdb_dataset : MSDB only. None uses the parameters averaged over its training datasets
                       (the setting for a new dataset), or a deepgaze_pytorch.MSDBDataset index to
                       use the parameters learned for one of them.
        """
        if model not in MODELS:
            raise ValueError(f"Unknown DeepGaze model {model!r}, expected one of {list(MODELS)}")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.centerbias = self._load_centerbias(centerbias_path, centerbias_type, centerbias_size)
        self._centerbias_cache: dict[tuple[int, int], torch.Tensor] = {}

        self.model_name = model
        self.msdb_dataset = msdb_dataset
        self.model = MODELS[model](pretrained=True).to(self.device)
        self.model.eval()
    
    def _get_centerbias(
        self,
        height: int,
        width: int,
    ) -> torch.Tensor:
        key = (height, width)

        if key not in self._centerbias_cache:
            centerbias = self._resize_centerbias(height, width)

            centerbias_tensor = (
                torch.from_numpy(centerbias)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
            )

            self._centerbias_cache[key] = centerbias_tensor

        return self._centerbias_cache[key]
            
    def _load_centerbias(self, centerbias_path: Path = None, centerbias_type: str = "uniform", centerbias_size: tuple = (1024, 1024)) -> np.ndarray:
        if centerbias_path is not None:
            centerbias = np.load(centerbias_path).astype(np.float64)
            if np.all(centerbias >= 0):
                # Density map (e.g. saliency.centerbias.build_centerbias): DeepGaze expects a log-density.
                centerbias = np.log(centerbias / centerbias.sum() + 1e-12)

        else:
            width, height = centerbias_size[0], centerbias_size[1]
            if centerbias_type == "uniform":
                centerbias = self._make_uniform_centerbias(height, width)
            else:
                centerbias = self._make_gaussian_centerbias(height, width)
            
        centerbias -= logsumexp(centerbias)
        return centerbias.astype(np.float32)
    
    def _resize_centerbias(self, height: int, width: int) -> np.ndarray:

        centerbias = cv2.resize(self.centerbias, (width, height), interpolation=cv2.INTER_LINEAR)
        if np.all(centerbias >= 0):
            centerbias = np.log(centerbias / (centerbias.sum() + 1e-12) + 1e-12)
        centerbias -= logsumexp(centerbias)
        
        return centerbias.astype(np.float32)

    @staticmethod
    def _normalize_input(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)

        # IR image: (H, W) or (H, W, 1)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            infrared = image if image.ndim == 2 else image[..., 0]
            infrared = infrared.astype(np.float32)

            if not np.all(np.isfinite(infrared)):
                raise ValueError("Infrared image contains NaN or infinite values")

            value_min = infrared.min()
            value_max = infrared.max()

            if value_max > value_min:
                infrared = (
                    (infrared - value_min)
                    / (value_max - value_min)
                    * 255.0
                )
            else:
                infrared = np.zeros_like(infrared)
            # [0, 1] -> [0, 255]
            infrared = np.clip(infrared, 0, 255).astype(np.uint8)

            # (H,W) -> (H, W, 3)
            image = np.repeat(infrared[..., None], 3, axis=2)

        # RGB image: (H, W, 3)
        elif image.ndim == 3 and image.shape[2] == 3:
            if not np.all(np.isfinite(image)):
                raise ValueError("RGB image contains NaN or infinite values")

            image = np.clip(image, 0, 255).astype(np.uint8)

        else:
            raise ValueError(
                f"Expected shape (H, W), (H, W, 1) or (H, W, 3), "
                f"got {image.shape}"
            )

        return np.ascontiguousarray(image)
            
            
    @staticmethod
    def _make_gaussian_centerbias(height: int = 1024, width: int = 1024, sigma_ratio: float = 0.25):

        sigma_x = width * sigma_ratio
        sigma_y = height * sigma_ratio
        
        x = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
        y = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
        xx, yy = np.meshgrid(x, y)
        
        gaussian = np.exp(-((xx**2 / (2.0 * sigma_x**2)) + (yy**2 / (2.0 * sigma_y**2))))
        gaussian /= (gaussian.sum() + 1e-12)
        
        centerbias = np.log(gaussian + 1e-12)
        centerbias -= logsumexp(centerbias)
        
        return centerbias.astype(np.float32)

    @staticmethod
    def _make_uniform_centerbias(height: int = 1024, width: int = 1024):
        centerbias = np.zeros((height, width), dtype=np.float64)
        return centerbias.astype(np.float32)


    def predict(self, image: np.ndarray, pixel_per_dva: Optional[float] = None) -> np.ndarray:
        return np.exp(self.predict_log_density(image, pixel_per_dva))


    def predict_log_density(self, image: np.ndarray, pixel_per_dva: Optional[float] = None) -> np.ndarray:
        """
        Natural-log probability per pixel, normalised over the image (used for information gains).

        pixel_per_dva : pixels of `image` per degree of visual angle, as it was displayed. Required
                        by DeepGaze MSDB, which rescales the image to fixed visual resolutions;
                        ignored by DeepGaze IIE.
        """
        if self.model_name == "MSDB" and pixel_per_dva is None:
            raise ValueError("DeepGaze MSDB needs the pixels per degree of visual angle of the image.")

        image = self._normalize_input(image)
        height, width = image.shape[:2]
        
        image_tensor = (
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        )

        centerbias_tensor = self._get_centerbias(height, width)

        with torch.inference_mode():
            if self.model_name == "MSDB":
                log_density = self.model(
                    image_tensor, centerbias_tensor, pixel_per_dva=pixel_per_dva, dataset=self.msdb_dataset
                )
                # (B, H, W) -> (B, 1, H, W), the output shape of DeepGaze IIE.
                log_density = log_density.unsqueeze(1)
            else:
                log_density = self.model(image_tensor, centerbias_tensor)


            if log_density.shape[-2:] != (height, width):
                raise RuntimeError(
                    f"Unexpected output shape {log_density.shape[-2:]}, "
                    f"expected {(height, width)}"
                )
            # Normalization
            log_density = log_density - torch.logsumexp(
                log_density,
                dim=(-2, -1),
                keepdim=True,
            )

        log_density = (
            log_density[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        return log_density
    
    
    @staticmethod
    def _to_logdensity(
        density: np.ndarray,
        epsilon: float = 1e-12,
    ) -> np.ndarray:
        """Convertit une densité positive en log-densité normalisée."""
        density = np.asarray(density, dtype=np.float64)

        if density.ndim != 2:
            raise ValueError(
                f"Expected a 2D density, got shape {density.shape}."
            )

        if not np.all(np.isfinite(density)):
            raise ValueError("Density contains NaN or infinite values.")

        if np.any(density < 0):
            raise ValueError("Density contains negative values.")

        total = density.sum()

        if total <= epsilon:
            raise ValueError("Density has a null or invalid total mass.")

        density = density / total

        log_density = np.log(density + epsilon)
        log_density -= logsumexp(log_density)

        return log_density.astype(np.float32)


    @staticmethod
    def _to_density(log_density: np.ndarray) -> np.ndarray:
        """Convertit une log-densité en densité positive normalisée."""
        log_density = np.asarray(log_density, dtype=np.float64)

        if log_density.ndim != 2:
            raise ValueError(
                f"Expected a 2D log-density, got shape {log_density.shape}."
            )

        if not np.all(np.isfinite(log_density)):
            raise ValueError("Log-density contains NaN or infinite values.")

        normalized_log_density = (
            log_density - logsumexp(log_density)
        )

        density = np.exp(normalized_log_density)
        density /= density.sum()

        return density.astype(np.float32)

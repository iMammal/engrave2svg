from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi

from .skeleton import skeletonize_binary


@dataclass(frozen=True)
class RidgeParams:
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    beta: float = 0.5
    gamma: float = 15.0
    threshold: float = 0.05

    def to_dict(self) -> dict[str, object]:
        return {
            "ridge_sigmas": ",".join(_format_float(value) for value in self.sigmas),
            "ridge_beta": self.beta,
            "ridge_gamma": self.gamma,
            "ridge_threshold": self.threshold,
        }


@dataclass(frozen=True)
class RidgeExtractionResult:
    response: np.ndarray
    response_uint8: np.ndarray
    mask: np.ndarray
    skeleton: np.ndarray


def parse_sigmas(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(value, tuple):
        sigmas = value
    elif isinstance(value, list):
        sigmas = tuple(float(item) for item in value)
    else:
        sigmas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not sigmas:
        raise ValueError("At least one ridge sigma is required.")
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError("Ridge sigmas must be positive.")
    return tuple(float(sigma) for sigma in sigmas)


def extract_ridges(
    normalized_gray: np.ndarray,
    params: RidgeParams,
    min_component_size: int = 0,
) -> RidgeExtractionResult:
    image = normalized_gray.astype(np.float32) / 255.0
    response = frangi(
        image,
        sigmas=params.sigmas,
        beta=params.beta,
        gamma=params.gamma,
        black_ridges=False,
    )
    response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    response = np.clip(response, 0.0, None)
    response_normalized = _normalize_response(response)
    response_uint8 = np.clip(response_normalized * 255.0, 0, 255).astype(np.uint8)
    intensity_support = image >= 0.05
    mask = ((response_normalized >= params.threshold) & intensity_support).astype(np.uint8) * 255
    if min_component_size > 0:
        mask = _remove_small_components(mask, min_component_size)
    skeleton = skeletonize_binary(mask)
    return RidgeExtractionResult(
        response=response,
        response_uint8=response_uint8,
        mask=mask,
        skeleton=skeleton,
    )


def save_ridge_debug(
    result: RidgeExtractionResult | None,
    normalized_gray: np.ndarray,
    debug_dir: str | Path,
) -> None:
    if result is None:
        return
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / "06b_ridge_response.png"), result.response_uint8)
    cv2.imwrite(str(path / "06c_ridge_threshold_mask.png"), result.mask)
    cv2.imwrite(str(path / "06d_ridge_skeleton.png"), result.skeleton)

    overlay = cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)
    overlay[result.skeleton > 0] = (0, 0, 255)
    cv2.imwrite(str(path / "06e_ridge_overlay.png"), overlay)


def _response_to_uint8(response: np.ndarray) -> np.ndarray:
    return np.clip(_normalize_response(response) * 255.0, 0, 255).astype(np.uint8)


def _normalize_response(response: np.ndarray) -> np.ndarray:
    max_value = float(response.max()) if response.size else 0.0
    if max_value <= 0:
        return np.zeros(response.shape, dtype=np.float32)
    return np.clip(response / max_value, 0.0, 1.0)


def _remove_small_components(binary: np.ndarray, min_component_size: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    filtered = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_size:
            filtered[labels == label] = 255
    return filtered


def _format_float(value: float) -> str:
    text = f"{float(value):g}"
    return text

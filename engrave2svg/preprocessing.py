from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


ThresholdMode = Literal["otsu", "global", "adaptive"]
CropMode = Literal["auto", "none"]


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse(cls, value: str) -> "CropBox":
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("Crop boxes must be formatted as x,y,width,height.")
        x, y, width, height = parts
        if width <= 0 or height <= 0:
            raise ValueError("Crop width and height must be positive.")
        return cls(x=x, y=y, width=width, height=height)


@dataclass(frozen=True)
class PreprocessParams:
    crop: str = "auto"
    threshold_mode: ThresholdMode = "otsu"
    threshold_value: int = 128
    adaptive_block_size: int = 35
    adaptive_c: int = -5
    morph_kernel_size: int = 3
    min_component_size: int = 12
    denoise_kernel_size: int = 3
    clahe_clip_limit: float = 2.0
    auto_crop_padding: int = 8


@dataclass(frozen=True)
class PreprocessResult:
    original_bgr: np.ndarray
    cropped_bgr: np.ndarray
    crop_box: CropBox
    grayscale: np.ndarray
    denoised: np.ndarray
    normalized: np.ndarray
    thresholded: np.ndarray
    cleaned: np.ndarray


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def crop_image(image: np.ndarray, crop: str, padding: int = 8) -> tuple[np.ndarray, CropBox]:
    height, width = image.shape[:2]
    if crop == "none":
        box = CropBox(0, 0, width, height)
    elif crop == "auto":
        box = _auto_crop_lower_panel(image, padding=padding)
    else:
        box = CropBox.parse(crop)

    x0 = max(0, box.x)
    y0 = max(0, box.y)
    x1 = min(width, box.x + box.width)
    y1 = min(height, box.y + box.height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Crop is outside image bounds: {box}")
    bounded = CropBox(x0, y0, x1 - x0, y1 - y0)
    return image[y0:y1, x0:x1].copy(), bounded


def preprocess_image(image: np.ndarray, params: PreprocessParams) -> PreprocessResult:
    cropped, crop_box = crop_image(
        image, crop=params.crop, padding=params.auto_crop_padding
    )
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    denoised = _denoise(gray, params.denoise_kernel_size)
    normalized = _normalize_contrast(denoised, params.clahe_clip_limit)
    thresholded = _threshold(
        normalized,
        mode=params.threshold_mode,
        value=params.threshold_value,
        block_size=params.adaptive_block_size,
        c=params.adaptive_c,
    )
    cleaned = _cleanup_binary(
        thresholded,
        kernel_size=params.morph_kernel_size,
        min_component_size=params.min_component_size,
    )
    return PreprocessResult(
        original_bgr=image,
        cropped_bgr=cropped,
        crop_box=crop_box,
        grayscale=gray,
        denoised=denoised,
        normalized=normalized,
        thresholded=thresholded,
        cleaned=cleaned,
    )


def save_preprocess_debug(result: PreprocessResult, debug_dir: str | Path) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / "01_cropped.png"), result.cropped_bgr)
    cv2.imwrite(str(path / "02_grayscale.png"), result.grayscale)
    cv2.imwrite(str(path / "03_denoised.png"), result.denoised)
    cv2.imwrite(str(path / "04_normalized.png"), result.normalized)
    cv2.imwrite(str(path / "05_thresholded.png"), result.thresholded)
    cv2.imwrite(str(path / "06_cleaned.png"), result.cleaned)


def _auto_crop_lower_panel(image: np.ndarray, padding: int) -> CropBox:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    lower_start = height // 2
    lower = gray[lower_start:, :]

    blurred = cv2.GaussianBlur(lower, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If Otsu floods the panel because of a pale background, keep only bright ink.
    if np.mean(mask > 0) > 0.45:
        mask = cv2.inRange(blurred, 180, 255)

    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return CropBox(0, lower_start, width, height - lower_start)

    x0 = max(0, int(cols.min()) - padding)
    x1 = min(width, int(cols.max()) + padding + 1)
    y0 = max(0, lower_start + int(rows.min()) - padding)
    y1 = min(height, lower_start + int(rows.max()) + padding + 1)
    return CropBox(x0, y0, x1 - x0, y1 - y0)


def _denoise(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return gray.copy()
    size = _odd_at_least(kernel_size, 3)
    return cv2.medianBlur(gray, size)


def _normalize_contrast(gray: np.ndarray, clip_limit: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _threshold(
    gray: np.ndarray,
    mode: ThresholdMode,
    value: int,
    block_size: int,
    c: int,
) -> np.ndarray:
    if mode == "global":
        _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    elif mode == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            _odd_at_least(block_size, 3),
            c,
        )
    elif mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    return binary


def _cleanup_binary(
    binary: np.ndarray, kernel_size: int, min_component_size: int
) -> np.ndarray:
    cleaned = binary.copy()
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd_at_least(kernel_size, 3),) * 2
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    if min_component_size > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        filtered = np.zeros_like(cleaned)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_size:
                filtered[labels == label] = 255
        cleaned = filtered
    return cleaned


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 else value + 1

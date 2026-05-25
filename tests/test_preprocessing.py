import cv2
import numpy as np

from engrave2svg.preprocessing import PreprocessParams, crop_image, preprocess_image
from engrave2svg.skeleton import skeletonize_binary


def test_auto_crop_selects_lower_bright_tracing():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.line(image, (20, 90), (140, 95), (220, 220, 220), 2)

    cropped, box = crop_image(image, "auto", padding=4)

    assert box.y >= 56
    assert cropped.shape[0] < image.shape[0]
    assert cropped.shape[1] < image.shape[1]


def test_preprocess_and_skeletonize_synthetic_line():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.line(image, (10, 60), (90, 60), (230, 230, 230), 3)
    params = PreprocessParams(
        crop="none",
        threshold_mode="global",
        threshold_value=100,
        morph_kernel_size=1,
        min_component_size=1,
    )

    result = preprocess_image(image, params)
    skeleton = skeletonize_binary(result.cleaned)

    assert np.count_nonzero(result.cleaned) > 0
    assert np.count_nonzero(skeleton) < np.count_nonzero(result.cleaned)

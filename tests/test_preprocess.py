import cv2
import numpy as np
import pytest

from ast_disk_ocr.preprocess import detect_baseline, preprocess_disk_text


def test_blank_disk_returns_reviewable_preprocessed_image():
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    prepared, baseline = preprocess_disk_text(image, 60, 60, 45)
    assert prepared.ndim == 2
    assert prepared.dtype == np.uint8
    assert not baseline.found


def test_horizontal_print_line_is_detected():
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (65, 150), (175, 150), (0, 0, 0), 8)
    baseline = detect_baseline(image, 100)
    assert baseline.found
    assert abs(baseline.angle_degrees) <= 5


def test_non_positive_radius_is_rejected():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="radius"):
        preprocess_disk_text(image, 10, 10, 0)


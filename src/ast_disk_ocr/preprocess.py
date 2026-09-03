"""Orientation and enhancement for an already localized AST disk."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TextBaseline:
    angle_degrees: float
    midpoint_y: float
    length: float
    found: bool


def sharpen_text(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    blurred = cv2.GaussianBlur(luminance, (0, 0), 1.4)
    lab[:, :, 0] = cv2.addWeighted(luminance, 1.85, blurred, -0.85, 0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _dark_text_mask(image: np.ndarray, disk_radius: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = np.zeros_like(binary)
    cv2.circle(
        mask,
        (binary.shape[1] // 2, binary.shape[0] // 2),
        round(disk_radius * 0.78),
        255,
        -1,
    )
    return cv2.bitwise_and(binary, mask)


def prepare_ocr_input(image: np.ndarray, disk_radius: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    outside = np.full_like(binary, 255)
    cv2.circle(
        outside,
        (binary.shape[1] // 2, binary.shape[0] // 2),
        round(disk_radius * 0.88),
        0,
        -1,
    )
    return cv2.bitwise_or(binary, outside)


def detect_baseline(image: np.ndarray, disk_radius: float) -> TextBaseline:
    binary = _dark_text_mask(image, disk_radius)
    minimum_length = max(14, round(disk_radius * 0.22))
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, round(disk_radius * 0.12)),
        minLineLength=minimum_length,
        maxLineGap=max(5, round(disk_radius * 0.07)),
    )
    center_y = image.shape[0] / 2.0
    if lines is None:
        return TextBaseline(0.0, center_y, 0.0, False)

    center_x = image.shape[1] / 2.0
    candidates: list[tuple[float, float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        midpoint_y = (float(y1) + float(y2)) / 2.0
        midpoint_x = (float(x1) + float(x2)) / 2.0
        candidates.append(
            (
                length,
                angle,
                midpoint_x,
                midpoint_y,
                math.hypot(midpoint_x - center_x, midpoint_y - center_y),
            )
        )
    if not candidates:
        return TextBaseline(0.0, center_y, 0.0, False)

    def angle_distance(first: float, second: float) -> float:
        difference = abs(first - second) % 180.0
        return min(difference, 180.0 - difference)

    cluster_distance = max(14.0, disk_radius * 0.20)
    foreground_y, foreground_x = np.nonzero(binary)
    best: tuple[float, float, float, float] | None = None
    for length, angle, midpoint_x, midpoint_y, distance_from_center in candidates:
        support = sum(
            other_length
            for other_length, other_angle, other_x, other_y, _ in candidates
            if angle_distance(angle, other_angle) <= 8.0
            and math.hypot(midpoint_x - other_x, midpoint_y - other_y) <= cluster_distance
        )
        radians = math.radians(angle)
        signed_distance = (
            (foreground_x - midpoint_x) * -math.sin(radians)
            + (foreground_y - midpoint_y) * math.cos(radians)
        )
        longitudinal = (
            (foreground_x - midpoint_x) * math.cos(radians)
            + (foreground_y - midpoint_y) * math.sin(radians)
        )
        outside_line = signed_distance[
            np.abs(signed_distance) > max(4.0, disk_radius * 0.03)
        ]
        if outside_line.size:
            one_sidedness = max(
                float(np.mean(outside_line > 0)), float(np.mean(outside_line < 0))
            )
            separation = abs(float(np.median(outside_line)))
        else:
            one_sidedness = 0.5
            separation = 0.0
        nearby = (np.abs(longitudinal) <= length / 2.0 + 15.0) & (
            np.abs(signed_distance) <= 24.0
        )
        line_core = (np.abs(longitudinal) <= length / 2.0 + 5.0) & (
            np.abs(signed_distance) <= 7.0
        )
        branch_density = float(np.count_nonzero(nearby & ~line_core)) / max(1.0, length)
        score = (
            support
            + 0.12 * length
            + 0.08 * distance_from_center
            + 2.5 * separation
            + 120.0 * one_sidedness
            - 90.0 * branch_density
        )
        if best is None or score > best[0]:
            best = (score, angle, midpoint_y, length)
    assert best is not None
    _, angle, midpoint_y, length = best
    return TextBaseline(angle, midpoint_y, length, True)


def _rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _rotate_y(image: np.ndarray, angle_degrees: float, x: float, y: float) -> float:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    return float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])


def preprocess_disk_text(
    image: np.ndarray, cx: float, cy: float, radius: float
) -> tuple[np.ndarray, TextBaseline]:
    """Crop, enlarge, orient, and binarize one localized disk."""
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if radius <= 0:
        raise ValueError("radius must be positive")
    half_size = max(20, round(radius * 1.75))
    left = max(0, round(cx) - half_size)
    top = max(0, round(cy) - half_size)
    right = min(image.shape[1], round(cx) + half_size + 1)
    bottom = min(image.shape[0], round(cy) + half_size + 1)
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("disk center is outside the image")
    scale = 6.0
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    sharpened = sharpen_text(enlarged)
    baseline = detect_baseline(sharpened, radius * scale)
    if not baseline.found:
        return prepare_ocr_input(sharpened, radius * scale), baseline
    rotated = _rotate(sharpened, baseline.angle_degrees)
    rotated_y = _rotate_y(
        rotated, baseline.angle_degrees, rotated.shape[1] / 2.0, baseline.midpoint_y
    )
    if rotated_y < rotated.shape[0] / 2.0:
        rotated = _rotate(rotated, 180.0)
        rotated_y = rotated.shape[0] - rotated_y
    return prepare_ocr_input(rotated, radius * scale), TextBaseline(
        0.0, rotated_y, baseline.length, True
    )


"""Catalog-constrained, review-first OCR for printed AST disk labels."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DiskOCRResult:
    raw_code: str
    raw_dose: str
    code: str | None
    dose: str | None
    label: str | None
    raw_confidence: float
    confidence: float
    edit_distance: int | None
    accepted: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_catalog(path: Path) -> dict[str, list[str]]:
    """Load `{code: [content, ...]}` while ignoring `_meta` style keys."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OCR catalog must be a JSON object")
    catalog = {
        str(code).upper(): [str(dose).upper() for dose in doses]
        for code, doses in data.items()
        if not str(code).startswith("_") and isinstance(doses, list)
    }
    if not catalog:
        raise ValueError("OCR catalog has no code/content entries")
    return catalog


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_char in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_char in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + int(first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def select_catalog_candidate(
    raw_code: str,
    raw_dose: str,
    code_score: float,
    dose_score: float,
    catalog: dict[str, list[str]],
    minimum_confidence: float = 0.55,
) -> DiskOCRResult:
    """Return a unique near candidate or an explicit review result."""
    raw_code = _normalize_token(raw_code)
    raw_dose = _normalize_token(raw_dose)
    raw_confidence = math.sqrt(max(0.0, code_score) * max(0.0, dose_score))

    candidates: list[tuple[int, int, int, str, str]] = []
    for code, doses in catalog.items():
        for dose in doses:
            code_distance = _edit_distance(raw_code, code)
            dose_distance = _edit_distance(raw_dose, dose)
            candidates.append(
                (code_distance + dose_distance, code_distance, dose_distance, code, dose)
            )
    candidates.sort()
    if not candidates:
        return DiskOCRResult(
            raw_code, raw_dose, None, None, None, raw_confidence, 0.0, None, False,
            "catalog_empty",
        )

    best = candidates[0]
    unique = len(candidates) == 1 or candidates[1][0] > best[0]
    allowed = unique and best[1] <= 1 and best[2] <= 1
    confidence = raw_confidence * max(0.0, 1.0 - 0.15 * best[0])
    accepted = allowed and confidence >= minimum_confidence
    code = best[3] if allowed else None
    dose = best[4] if allowed else None
    return DiskOCRResult(
        raw_code=raw_code,
        raw_dose=raw_dose,
        code=code,
        dose=dose,
        label=f"{code} {dose}" if code and dose else None,
        raw_confidence=raw_confidence,
        confidence=confidence,
        edit_distance=best[0],
        accepted=accepted,
        status="accepted" if accepted else "review",
    )


def _find_text_rows(image: np.ndarray) -> list[np.ndarray]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (gray < 128).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        ink, 1, np.pi / 180, threshold=20, minLineLength=35, maxLineGap=12
    )
    baseline_y = gray.shape[0] * 0.78
    if lines is not None:
        horizontal: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = float(np.hypot(x2 - x1, y2 - y1))
            midpoint_y = (float(y1) + float(y2)) / 2.0
            if abs(angle) <= 7.0 and midpoint_y > gray.shape[0] * 0.45:
                horizontal.append((length, midpoint_y))
        if horizontal:
            baseline_y = max(horizontal)[1]

    without_line = ink.copy()
    line_y = round(baseline_y)
    without_line[max(0, line_y - 12) : min(gray.shape[0], line_y + 13), :] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(without_line, 8)
    components: list[tuple[int, int, int, int, int, float]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 35 or height < 8 or y + height >= baseline_y + 5:
            continue
        components.append((x, y, width, height, area, float(centroids[index][1])))
    if len(components) < 2:
        return []

    centers = np.array([component[-1] for component in components], dtype=np.float32)
    order = np.argsort(centers)
    best_split: tuple[float, set[int], set[int]] | None = None
    for split in range(1, len(order)):
        first = centers[order[:split]]
        second = centers[order[split:]]
        cost = float(
            np.sum((first - np.mean(first)) ** 2)
            + np.sum((second - np.mean(second)) ** 2)
        )
        candidate = (
            cost,
            {int(value) for value in order[:split]},
            {int(value) for value in order[split:]},
        )
        if best_split is None or candidate[0] < best_split[0]:
            best_split = candidate
    if best_split is None:
        return []

    rows: list[np.ndarray] = []
    for indices in best_split[1:]:
        selected = [components[index] for index in indices]
        left = min(item[0] for item in selected)
        top = min(item[1] for item in selected)
        right = max(item[0] + item[2] for item in selected)
        bottom = max(item[1] + item[3] for item in selected)
        padding = 12
        rows.append(
            gray[
                max(0, top - padding) : min(gray.shape[0], bottom + padding),
                max(0, left - padding) : min(gray.shape[1], right + padding),
            ]
        )
    return rows


def _recognize_line(engine: object, image: np.ndarray) -> tuple[str, float]:
    result = engine(image, use_det=False, use_cls=False, use_rec=True)  # type: ignore[operator]
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if not texts or not scores:
        return "", 0.0
    return _normalize_token(str(texts[0])), float(scores[0])


def _recognize_once(
    engine: object,
    prepared_image: np.ndarray,
    catalog: dict[str, list[str]],
    minimum_confidence: float,
) -> DiskOCRResult:
    rows = _find_text_rows(prepared_image)
    if len(rows) != 2:
        return DiskOCRResult("", "", None, None, None, 0.0, 0.0, None, False, "layout_unclear")
    raw_code, code_score = _recognize_line(engine, rows[0])
    raw_dose, dose_score = _recognize_line(engine, rows[1])
    return select_catalog_candidate(
        raw_code, raw_dose, code_score, dose_score, catalog, minimum_confidence
    )


def recognize_disk_text(
    engine: object,
    prepared_image: np.ndarray,
    catalog: dict[str, list[str]],
    minimum_confidence: float = 0.55,
) -> DiskOCRResult:
    """Recognize one disk, preserving review when rotations conflict."""
    primary = _recognize_once(engine, prepared_image, catalog, minimum_confidence)
    if primary.accepted:
        return primary

    accepted: list[DiskOCRResult] = []
    for rotation in (
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ):
        candidate = _recognize_once(
            engine, cv2.rotate(prepared_image, rotation), catalog, minimum_confidence
        )
        if candidate.accepted:
            accepted.append(candidate)
    if len({candidate.label for candidate in accepted}) == 1:
        return max(accepted, key=lambda candidate: candidate.confidence)
    return primary

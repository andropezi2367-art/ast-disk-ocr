"""Conservative OCR helpers for printed AST disks."""

from .core import DiskOCRResult, load_catalog, recognize_disk_text, select_catalog_candidate
from .preprocess import TextBaseline, preprocess_disk_text

__all__ = [
    "DiskOCRResult",
    "TextBaseline",
    "load_catalog",
    "preprocess_disk_text",
    "recognize_disk_text",
    "select_catalog_candidate",
]

__version__ = "0.1.0"


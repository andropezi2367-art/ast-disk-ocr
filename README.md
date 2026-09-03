# AST Disk OCR

Conservative OCR preprocessing and catalog-constrained recognition for printed
antimicrobial susceptibility testing (AST) disks.

The project accepts an already located disk crop (or a center/radius inside an
image), normalizes its orientation, reads the printed drug code and disk content,
and returns either a unique catalog-supported candidate or an explicit review state.

## Safety boundary

This library is research software. It does **not** identify organisms, measure
inhibition zones, select breakpoint standards, produce S/I/R interpretations, or
authorize clinical reporting. A successful OCR result is evidence that still needs
the laboratory's specimen context, method, disk content, current verified rule,
quality controls, and qualified human review.

Unclear layout, low OCR confidence, catalog ambiguity, or conflicting rotations
returns `accepted=false` and `status="review"`/`"layout_unclear"`. Applications must
not silently convert those states into a drug identity.

## What is included

- disk crop enhancement and printed-baseline orientation;
- two-row code/content extraction;
- conservative catalog matching with explicit review states;
- 90/180/270-degree recovery without forcing conflicting candidates;
- a small CLI and synthetic/unit tests.

No trained weights, clinical images, patient data, breakpoint rules, or proprietary
drug catalogs are included. Bring an OCR engine compatible with the callable API and
your own reviewed code/content catalog.

## Install

```bash
python -m pip install -e ".[rapidocr]"
```

For development:

```bash
python -m pip install -e ".[rapidocr,dev]"
pytest -q
ruff check .
```

## CLI

For a centered disk crop:

```bash
ast-disk-ocr path/to/disk-crop.png \
  --catalog examples/catalog.example.json \
  --minimum-confidence 0.55
```

For a disk within a larger image:

```bash
ast-disk-ocr path/to/image.png \
  --catalog my-reviewed-catalog.json \
  --center-x 420 --center-y 315 --radius 31
```

The command prints JSON to stdout. Example:

```json
{
  "raw_code": "CIP",
  "raw_dose": "5",
  "code": "CIP",
  "dose": "5",
  "label": "CIP 5",
  "raw_confidence": 0.91,
  "confidence": 0.91,
  "edit_distance": 0,
  "accepted": true,
  "status": "accepted"
}
```

Catalog format:

```json
{
  "_meta": {"purpose": "OCR candidates only; not breakpoint rules"},
  "CIP": ["5"],
  "AMP": ["10"]
}
```

Metadata keys beginning with `_` are ignored. The catalog is a recognition
constraint, not evidence that a drug/content combination is appropriate for a
specimen or breakpoint system.

## Python API

```python
from pathlib import Path

import cv2
from rapidocr import RapidOCR

from ast_disk_ocr import load_catalog, preprocess_disk_text, recognize_disk_text

image = cv2.imread("disk-crop.png")
catalog = load_catalog(Path("examples/catalog.example.json"))
prepared, orientation = preprocess_disk_text(
    image,
    cx=image.shape[1] / 2,
    cy=image.shape[0] / 2,
    radius=min(image.shape[:2]) * 0.45,
)
result = recognize_disk_text(RapidOCR(), prepared, catalog)
print(result.to_dict())
```

## Provenance and limitations

The initial implementation was extracted and generalized from an AST workbench.
Only generic OCR/preprocessing logic is published here. The original model package
was not included because its redistribution rights and independent external
performance were not established. Current tests establish software behavior, not
clinical accuracy.

See [SECURITY.md](SECURITY.md) for responsible disclosure and [THIRD_PARTY.md](THIRD_PARTY.md)
for dependency notes.

## License

Apache License 2.0. See [LICENSE](LICENSE).


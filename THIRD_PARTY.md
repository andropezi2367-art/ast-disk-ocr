# Third-party components

This repository contains original glue, preprocessing, and conservative matching
logic. It depends on third-party packages but does not vendor their source or models.

- NumPy — numerical arrays; see the installed package for its license.
- OpenCV — image processing; Apache-2.0 project.
- RapidOCR — optional OCR runtime; check the selected release and model licenses.
- ONNX Runtime — optional inference runtime; MIT project.

No OCR model is bundled. Users are responsible for verifying the license and intended
use of any model they supply.


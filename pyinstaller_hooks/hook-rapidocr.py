"""Freeze only RapidOCR's ONNX Runtime path and bundled Chinese models."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, get_package_paths


_, package_dir = get_package_paths("rapidocr")
package_dir = Path(package_dir)

datas = [
    (str(package_dir / "config.yaml"), "rapidocr"),
    (str(package_dir / "default_models.yaml"), "rapidocr"),
    (
        str(package_dir / "models" / "PP-OCRv6_det_small.onnx"),
        "rapidocr/models",
    ),
    (
        str(package_dir / "models" / "PP-OCRv6_rec_small.onnx"),
        "rapidocr/models",
    ),
    (
        str(package_dir / "models" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        "rapidocr/models",
    ),
]

# RapidOCR imports its public class lazily, and selects its inference engine at
# runtime. Make the intended ONNX path explicit so a frozen build is complete.
hiddenimports = [
    "rapidocr.main",
    "rapidocr.inference_engine.onnxruntime",
]
binaries = collect_dynamic_libs("onnxruntime")

# These are supported by RapidOCR but are not used by this project. In
# particular, do not let its optional PyTorch path duplicate the YOLO graph.
excludedimports = [
    "rapidocr.inference_engine.mnn",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.pytorch",
    "rapidocr.inference_engine.tensorrt",
]

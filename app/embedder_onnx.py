from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageEnhance


class OnnxImageEmbedder:
    """ONNXRuntime-backed image embedder with CLIP-style preprocessing."""
    # Multi-view defaults tuned for partial-panel robustness with modest CPU overhead.
    CENTER_CROP_RATIO = 0.85
    CORNER_CROP_RATIO = 0.75
    CONTRAST_FACTOR = 1.15

    def __init__(self, onnx_path: str, input_size: int = 224, intra_threads: int = 4):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - runtime dependency check
            raise RuntimeError("onnxruntime is required to use OnnxImageEmbedder") from exc

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(intra_threads))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_size = int(input_size)

        self._mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self._std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(denom, eps)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        scale = self._input_size / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)

        rw, rh = resized.size
        left = (rw - self._input_size) // 2
        top = (rh - self._input_size) // 2
        cropped = resized.crop((left, top, left + self._input_size, top + self._input_size))

        arr = np.asarray(cropped, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr.astype(np.float32)

    def embed_pil(self, image: Image.Image) -> np.ndarray:
        inp = self._preprocess(image)
        output = self._session.run([self._output_name], {self._input_name: inp})[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 3:
            output = output.mean(axis=1)
        return self._l2_normalize(output)[0]

    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.embed_pil(image)

    def multi_views(
        self,
        image_bytes: bytes,
        include_corners: bool = True,
        include_contrast: bool = True,
    ) -> list[np.ndarray]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            width, height = image.size
            side = min(width, height)

            views: list[Image.Image] = [image.copy()]

            center_side = max(1, int(side * self.CENTER_CROP_RATIO))
            cx = (width - center_side) // 2
            cy = (height - center_side) // 2
            views.append(image.crop((cx, cy, cx + center_side, cy + center_side)))

            if include_corners:
                corner_side = max(1, int(side * self.CORNER_CROP_RATIO))
                views.extend(
                    [
                        image.crop((0, 0, corner_side, corner_side)),
                        image.crop((width - corner_side, 0, width, corner_side)),
                        image.crop((0, height - corner_side, corner_side, height)),
                        image.crop((width - corner_side, height - corner_side, width, height)),
                    ]
                )

            if include_contrast:
                views.append(ImageEnhance.Contrast(image).enhance(self.CONTRAST_FACTOR))

            return [self.embed_pil(v) for v in views]

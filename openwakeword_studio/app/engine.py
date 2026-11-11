"""Streaming wake word detection built on openWakeWord."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import openwakeword
from openwakeword.model import Model

from .registry import ModelInfo, base_name
from .settings import Settings

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms, the native step of openWakeWord
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


@dataclass
class FrameResult:
    scores: dict[str, float]
    level: float
    vad: float | None = None
    detections: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        data: dict = {
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "level": round(self.level, 4),
        }
        if self.vad is not None:
            data["vad"] = round(self.vad, 4)
        if self.detections:
            data["detections"] = self.detections
        return data


class Detector:
    """Runs a set of wake word models over one 16 kHz mono 16-bit audio stream.

    Thresholds, trigger levels and the refractory period are read from the
    settings on every frame, so changes made in the UI apply immediately.
    Not thread-safe: call `process` from one thread at a time.
    """

    def __init__(self, models: list[ModelInfo], settings: Settings) -> None:
        self.settings = settings
        self.model_infos = [m for m in models if m.usable]
        self._pending = b""
        self._frame_index = 0
        self._activations: dict[str, int] = {}
        self._last_detection: dict[str, float] = {}
        self._engines: list[Model] = []
        self._label_parent: dict[str, str] = {}
        self._signature: tuple | None = None
        self._build()

    # ------------------------------------------------------------------
    def _current_signature(self) -> tuple:
        return (
            self.settings.get("vad_threshold") > 0,
            bool(self.settings.get("noise_suppression")),
        )

    def _build(self) -> None:
        use_vad, use_ns = self._current_signature()
        self._engines = []
        self._label_parent = {}
        for framework in ("tflite", "onnx"):
            group = [m for m in self.model_infos if m.framework == framework]
            if not group:
                continue
            kwargs = {}
            if use_ns:
                try:
                    import speexdsp_ns  # noqa: F401

                    kwargs["enable_speex_noise_suppression"] = True
                except ImportError:
                    _LOGGER.debug("Speex noise suppression is not available")
            engine = Model(
                wakeword_models=[m.path for m in group],
                inference_framework=framework,
                vad_threshold=self.settings.get("vad_threshold") if use_vad else 0,
                **kwargs,
            )
            for name, n_out in engine.model_outputs.items():
                if n_out > 1:
                    # Multi-class model: class 0 is "nothing", the rest are labels.
                    known = openwakeword.model_class_mappings.get(base_name(name))
                    mapping = dict(known) if known else {
                        str(i): f"{name}:{i}" for i in range(1, n_out)
                    }
                    engine.class_mapping[name] = mapping
                    for label in mapping.values():
                        self._label_parent[label] = name
                else:
                    self._label_parent[name] = name
            self._engines.append(engine)
        self._signature = (use_vad, use_ns)

    @property
    def labels(self) -> dict[str, str]:
        """Mapping of score label -> model id."""
        return dict(self._label_parent)

    def reset(self) -> None:
        self._pending = b""
        self._frame_index = 0
        self._activations.clear()
        self._last_detection.clear()
        for engine in self._engines:
            engine.reset()
            if hasattr(engine, "vad"):
                engine.vad.reset_states()
                engine.vad.prediction_buffer.clear()

    # ------------------------------------------------------------------
    def process(self, pcm: bytes, flush: bool = False) -> list[FrameResult]:
        """Feed raw PCM and return one result per completed 80 ms frame."""
        self._pending += pcm
        if flush and len(self._pending) % FRAME_BYTES:
            self._pending += b"\0" * (FRAME_BYTES - len(self._pending) % FRAME_BYTES)
        results = []
        while len(self._pending) >= FRAME_BYTES:
            frame = np.frombuffer(self._pending[:FRAME_BYTES], dtype=np.int16)
            self._pending = self._pending[FRAME_BYTES:]
            results.append(self._process_frame(frame))
        return results

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        if self._signature != self._current_signature():
            self._build()

        level = float(np.sqrt(np.mean(np.square(frame.astype(np.float32) / 32768.0))))
        scores: dict[str, float] = {}
        vad: float | None = None
        for engine in self._engines:
            if hasattr(engine, "vad"):
                engine.vad_threshold = self.settings.get("vad_threshold")
            for label, score in engine.predict(frame).items():
                scores[label] = float(score)
            if hasattr(engine, "vad") and engine.vad.prediction_buffer:
                vad = float(engine.vad.prediction_buffer[-1])

        self._frame_index += 1
        now = self._frame_index * FRAME_SECONDS
        refractory = self.settings.get("refractory_seconds")
        detections = []
        for label, score in scores.items():
            parent = self._label_parent.get(label, label)
            if score < self.settings.threshold_for(parent):
                self._activations[label] = max(0, self._activations.get(label, 0) - 1)
                continue
            count = self._activations.get(label, 0) + 1
            self._activations[label] = count
            if count < self.settings.trigger_level_for(parent):
                continue
            last = self._last_detection.get(parent)
            if last is None or now - last >= refractory:
                self._last_detection[parent] = now
                self._activations[label] = 0
                detections.append({"label": label, "model": parent, "score": round(score, 4)})
        return FrameResult(scores=scores, level=level, vad=vad, detections=detections)

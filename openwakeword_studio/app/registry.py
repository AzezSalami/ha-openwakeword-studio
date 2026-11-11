"""Discovery of built-in and custom wake word models."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import openwakeword

_LOGGER = logging.getLogger(__name__)

MODEL_EXTENSIONS = (".tflite", ".onnx")
_VERSION_RE = re.compile(r"_v\d+(\.\d+)*$")

BUILTIN_PHRASES = {
    "alexa": "Alexa",
    "hey_mycroft": "Hey Mycroft",
    "hey_jarvis": "Hey Jarvis",
    "hey_rhasspy": "Hey Rhasspy",
    "timer": "Set a timer",
    "weather": "What's the weather",
}


def tflite_available() -> bool:
    try:
        import ai_edge_litert.interpreter  # noqa: F401

        return True
    except ImportError:
        return False


def speex_available() -> bool:
    try:
        import speexdsp_ns  # noqa: F401

        return True
    except ImportError:
        return False


def phrase_from_id(model_id: str) -> str:
    base = _VERSION_RE.sub("", model_id)
    words = re.split(r"[_\-\s]+", base)
    return " ".join(w.capitalize() for w in words if w) or model_id


def base_name(model_id: str) -> str:
    return _VERSION_RE.sub("", model_id)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    phrase: str
    path: str
    framework: str  # "tflite" | "onnx"
    source: str  # "builtin" | "custom"
    size: int
    usable: bool = True

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "phrase": self.phrase,
            "framework": self.framework,
            "source": self.source,
            "size": self.size,
            "usable": self.usable,
            "file": os.path.basename(self.path),
        }


class Registry:
    """Keeps the list of available wake word models."""

    def __init__(self, custom_dir: Path) -> None:
        self.custom_dir = custom_dir
        self.has_tflite = tflite_available()
        self.has_speex = speex_available()
        self.models: dict[str, ModelInfo] = {}
        self.generation = 0
        self.scan()

    def scan(self) -> None:
        found: dict[str, ModelInfo] = {}

        for name, meta in openwakeword.MODELS.items():
            path = meta["model_path"]
            if not self.has_tflite or not os.path.exists(path):
                path = path.replace(".tflite", ".onnx")
            if not os.path.exists(path):
                _LOGGER.warning("Built-in model %s is missing (%s)", name, path)
                continue
            model_id = Path(path).stem
            found[model_id] = ModelInfo(
                id=model_id,
                phrase=BUILTIN_PHRASES.get(name, phrase_from_id(model_id)),
                path=path,
                framework=Path(path).suffix[1:],
                source="builtin",
                size=os.path.getsize(path),
            )

        custom: dict[str, ModelInfo] = {}
        if self.custom_dir.is_dir():
            for path in sorted(self.custom_dir.rglob("*")):
                if (
                    path.suffix.lower() not in MODEL_EXTENSIONS
                    or path.name.startswith(".")
                    or not path.is_file()
                ):
                    continue
                framework = path.suffix.lower()[1:]
                usable = framework == "onnx" or self.has_tflite
                existing = custom.get(path.stem)
                # When both formats exist prefer a usable one, then tflite.
                if existing and (existing.usable and (existing.framework == "tflite" or not usable)):
                    continue
                custom[path.stem] = ModelInfo(
                    id=path.stem,
                    phrase=phrase_from_id(path.stem),
                    path=str(path),
                    framework=framework,
                    source="custom",
                    size=path.stat().st_size,
                    usable=usable,
                )
        found.update(custom)  # a custom model overrides a built-in with the same id

        self.models = found
        self.generation += 1

    def get(self, model_id: str) -> ModelInfo | None:
        return self.models.get(model_id)

    def sorted(self) -> list[ModelInfo]:
        return sorted(self.models.values(), key=lambda m: (m.source != "custom", m.phrase.lower()))

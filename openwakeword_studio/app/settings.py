"""Persistent, UI-editable settings."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "threshold": 0.5,
    "trigger_level": 1,
    "refractory_seconds": 2.0,
    "vad_threshold": 0.0,
    "noise_suppression": False,
}

# key -> (type, min, max)
_LIMITS = {
    "threshold": (float, 0.01, 0.99),
    "trigger_level": (int, 1, 20),
    "refractory_seconds": (float, 0.0, 30.0),
    "vad_threshold": (float, 0.0, 0.95),
}


def _clamp(key: str, value: Any) -> Any:
    kind, low, high = _LIMITS[key]
    try:
        number = kind(float(value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"'{key}' must be a number") from err
    if number != number:  # NaN
        raise ValueError(f"'{key}' must be a number")
    return max(low, min(high, number))


class Settings:
    """Thread-safe settings store backed by a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._global: dict[str, Any] = dict(DEFAULTS)
        self._models: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as err:
            _LOGGER.warning("Could not read %s (%s); using defaults", self._path, err)
            return
        try:
            self.update_global({k: v for k, v in raw.items() if k in DEFAULTS}, save=False)
            for model_id, cfg in (raw.get("models") or {}).items():
                self.update_model(str(model_id), cfg, save=False)
        except (ValueError, AttributeError, TypeError) as err:
            _LOGGER.warning("Ignoring invalid settings in %s: %s", self._path, err)

    def _save(self) -> None:
        data = {**self._global, "models": self._models}
        tmp = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self._path)
        except OSError as err:
            _LOGGER.error("Could not save settings to %s: %s", self._path, err)

    # -- global ---------------------------------------------------------
    def update_global(self, changes: dict[str, Any], save: bool = True) -> None:
        cleaned: dict[str, Any] = {}
        for key, value in changes.items():
            if key == "noise_suppression":
                cleaned[key] = bool(value)
            elif key in _LIMITS:
                cleaned[key] = _clamp(key, value)
            else:
                raise ValueError(f"Unknown setting '{key}'")
        with self._lock:
            self._global.update(cleaned)
            if save:
                self._save()

    def get(self, key: str) -> Any:
        return self._global[key]

    # -- per model ------------------------------------------------------
    def update_model(self, model_id: str, changes: dict[str, Any], save: bool = True) -> None:
        cleaned: dict[str, Any] = {}
        for key, value in changes.items():
            if key == "enabled":
                cleaned[key] = bool(value)
            elif key in ("threshold", "trigger_level"):
                cleaned[key] = None if value is None else _clamp(key, value)
            else:
                raise ValueError(f"Unknown model setting '{key}'")
        with self._lock:
            self._models.setdefault(model_id, {}).update(cleaned)
            if save:
                self._save()

    def forget_model(self, model_id: str) -> None:
        with self._lock:
            if self._models.pop(model_id, None) is not None:
                self._save()

    def model(self, model_id: str) -> dict[str, Any]:
        cfg = self._models.get(model_id, {})
        return {
            "enabled": cfg.get("enabled", True),
            "threshold": cfg.get("threshold"),
            "trigger_level": cfg.get("trigger_level"),
        }

    def is_enabled(self, model_id: str) -> bool:
        return self._models.get(model_id, {}).get("enabled", True)

    def threshold_for(self, model_id: str) -> float:
        value = self._models.get(model_id, {}).get("threshold")
        return self._global["threshold"] if value is None else value

    def trigger_level_for(self, model_id: str) -> int:
        value = self._models.get(model_id, {}).get("trigger_level")
        return self._global["trigger_level"] if value is None else value

    def as_dict(self) -> dict[str, Any]:
        return dict(self._global)

"""Shared application state and live event fan-out to the web UI."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import openwakeword

from .registry import Registry
from .settings import Settings

try:
    from importlib.metadata import version as _pkg_version

    OWW_VERSION = _pkg_version("openwakeword")
except Exception:  # pragma: no cover
    OWW_VERSION = getattr(openwakeword, "__version__", "unknown")


class Hub:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        version: str = "dev",
        wyoming_port: int = 10400,
        hostname: str = "",
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.version = version
        self.wyoming_port = wyoming_port
        self.hostname = hostname
        self.started = time.time()
        self.clients: dict[int, dict[str, Any]] = {}
        self.detections: deque[dict[str, Any]] = deque(maxlen=200)
        self.total_detections = 0
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="oww")
        self._subscribers: set[asyncio.Queue] = set()
        self._ids = itertools.count(1)
        self.trainer = None  # set by attach_trainer()

    # -- training -------------------------------------------------------
    def attach_trainer(self, data_dir, loop: asyncio.AbstractEventLoop) -> None:
        from .trainer import Trainer

        def on_update() -> None:
            loop.call_soon_threadsafe(self._publish_training)

        def on_model(model_id: str, meta: dict) -> None:
            loop.call_soon_threadsafe(self._model_trained, model_id, meta)

        self.trainer = Trainer(data_dir, self.registry.custom_dir, on_update, on_model)

    def _publish_training(self) -> None:
        if self._subscribers and self.trainer:
            self.publish({"type": "training", "status": self.trainer.status.as_dict()})

    def _model_trained(self, model_id: str, meta: dict) -> None:
        self.registry.scan()
        recommended = meta["metrics"]["recommended_threshold"]
        same = abs(recommended - self.settings.get("threshold")) < 1e-6
        self.settings.update_model(
            model_id, {"enabled": True, "threshold": None if same else recommended}
        )
        self.publish_state()

    # -- pub/sub --------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    def publish(self, message: dict[str, Any]) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # slow browser; live data is disposable

    def publish_state(self) -> None:
        if self._subscribers:
            self.publish({"type": "state", "state": self.state()})

    # -- clients --------------------------------------------------------
    def next_id(self) -> int:
        return next(self._ids)

    def add_client(self, client_id: int, peer: str) -> dict[str, Any]:
        client = {
            "id": client_id,
            "peer": peer,
            "connected": time.time(),
            "models": [],
            "requested": None,
            "warning": None,
        }
        self.clients[client_id] = client
        self.publish_state()
        return client

    def remove_client(self, client_id: int) -> None:
        if self.clients.pop(client_id, None) is not None:
            self.publish_state()

    # -- detections -----------------------------------------------------
    def record_detection(self, detection: dict[str, Any], source: str, kind: str) -> None:
        info = self.registry.get(detection["model"])
        entry = {
            "time": time.time(),
            "model": detection["model"],
            "label": detection["label"],
            "phrase": info.phrase if info else detection["model"],
            "score": detection["score"],
            "source": source,
            "kind": kind,  # "wyoming" | "test"
        }
        self.detections.appendleft(entry)
        self.total_detections += 1
        self.publish({"type": "detection", "detection": entry, "total": self.total_detections})

    # -- snapshots ------------------------------------------------------
    def models(self) -> list[dict[str, Any]]:
        return [
            {**info.as_dict(), **self.settings.model(info.id)}
            for info in self.registry.sorted()
        ]

    def enabled_models(self):
        return [
            m for m in self.registry.sorted() if m.usable and self.settings.is_enabled(m.id)
        ]

    def state(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "openwakeword_version": OWW_VERSION,
            "started": self.started,
            "now": time.time(),
            "hostname": self.hostname,
            "wyoming_port": self.wyoming_port,
            "custom_dir": str(self.registry.custom_dir),
            "capabilities": {
                "tflite": self.registry.has_tflite,
                "noise_suppression": self.registry.has_speex,
            },
            "settings": self.settings.as_dict(),
            "models": self.models(),
            "clients": list(self.clients.values()),
            "detections": list(self.detections)[:50],
            "total_detections": self.total_detections,
            "training": self.trainer.status.as_dict() if self.trainer else None,
            "can_train": self.trainer is not None,
        }

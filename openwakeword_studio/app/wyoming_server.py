"""Wyoming protocol server so Home Assistant can use the wake words."""

from __future__ import annotations

import asyncio
import logging

from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from .engine import Detector
from .hub import OWW_VERSION, Hub

_LOGGER = logging.getLogger(__name__)

_ATTRIBUTION = Attribution(name="dscripka", url="https://github.com/dscripka/openWakeWord")
_SCORE_EVERY_FRAMES = 3


def build_info(hub: Hub) -> Info:
    return Info(
        wake=[
            WakeProgram(
                name="openwakeword",
                description="openWakeWord Studio",
                attribution=_ATTRIBUTION,
                installed=True,
                version=OWW_VERSION,
                models=[
                    WakeModel(
                        name=model.id,
                        description=model.phrase,
                        phrase=model.phrase,
                        attribution=_ATTRIBUTION,
                        installed=True,
                        languages=["en"] if model.source == "builtin" else [],
                        version=None,
                    )
                    for model in hub.enabled_models()
                ],
            )
        ]
    )


class WakeHandler(AsyncEventHandler):
    """One connected Wyoming client (Home Assistant or a satellite)."""

    def __init__(self, hub: Hub, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hub = hub
        self.client_id = hub.next_id()
        peer = self.writer.get_extra_info("peername")
        self.peer = peer[0] if peer else "unknown"
        self.converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.requested: list[str] | None = None
        self.detector: Detector | None = None
        self.detector_key: tuple | None = None
        self.client: dict | None = None
        self.detected = False
        self._frames = 0
        self._peak: dict[str, float] = {}

    def _ensure_client(self) -> dict:
        if self.client is None:
            self.client = self.hub.add_client(self.client_id, self.peer)
            _LOGGER.info("Client %s connected from %s", self.client_id, self.peer)
        return self.client

    async def _ensure_detector(self) -> Detector:
        hub = self.hub
        enabled = hub.enabled_models()
        if self.requested:
            models = [m for m in enabled if m.id in self.requested]
        else:
            models = enabled
        key = (hub.registry.generation, tuple(m.id for m in models))
        if self.detector is None or key != self.detector_key:
            loop = asyncio.get_running_loop()
            self.detector = await loop.run_in_executor(
                hub.executor, Detector, models, hub.settings
            )
            self.detector_key = key
            client = self._ensure_client()
            client["models"] = [m.id for m in models]
            client["requested"] = self.requested
            missing = [n for n in (self.requested or []) if n not in client["models"]]
            if missing:
                client["warning"] = (
                    "Asked for wake word(s) that are disabled or missing: " + ", ".join(missing)
                )
                _LOGGER.warning("Client %s: %s", self.client_id, client["warning"])
            elif not models:
                client["warning"] = "No wake words are enabled"
            else:
                client["warning"] = None
            hub.publish_state()
        return self.detector

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(build_info(self.hub).event())
            return True

        if Detect.is_type(event.type):
            self.requested = Detect.from_event(event).names or None
            self._ensure_client()
            await self._ensure_detector()
        elif AudioStart.is_type(event.type):
            self.detected = False
            if self.detector is not None:
                self.detector.reset()
        elif AudioChunk.is_type(event.type):
            chunk = self.converter.convert(AudioChunk.from_event(event))
            detector = await self._ensure_detector()
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(self.hub.executor, detector.process, chunk.audio)
            for result in results:
                for detection in result.detections:
                    self.detected = True
                    _LOGGER.info(
                        "Client %s: detected %s (score %.2f)",
                        self.client_id, detection["label"], detection["score"],
                    )
                    await self.write_event(
                        Detection(name=detection["model"], timestamp=chunk.timestamp).event()
                    )
                    self.hub.record_detection(detection, source=self.peer, kind="wyoming")
                self._publish_scores(result)
        elif AudioStop.is_type(event.type):
            if not self.detected:
                await self.write_event(NotDetected().event())
        return True

    def _publish_scores(self, result) -> None:
        """Send throttled per-model peak scores to the UI's live monitor."""
        labels = self.detector.labels if self.detector else {}
        for label, score in result.scores.items():
            model = labels.get(label, label)
            if score > self._peak.get(model, 0.0):
                self._peak[model] = score
        self._frames += 1
        if self._frames % _SCORE_EVERY_FRAMES:
            return
        if self.hub.has_subscribers:
            self.hub.publish(
                {
                    "type": "scores",
                    "client": self.client_id,
                    "scores": {k: round(v, 3) for k, v in self._peak.items()},
                    "level": round(result.level, 4),
                }
            )
        self._peak = {}

    async def disconnect(self) -> None:
        if self.client is not None:
            _LOGGER.info("Client %s disconnected", self.client_id)
        self.hub.remove_client(self.client_id)

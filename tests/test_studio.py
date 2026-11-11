"""End-to-end tests. Need openwakeword (with downloaded models), wyoming, aiohttp,
pytest and pytest-asyncio. Set OWW_TEST_WAV to a 16 kHz mono recording of "alexa"
(e.g. tests/data/alexa_test.wav from the openWakeWord repository)."""

import asyncio
import os
import wave
from functools import partial
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncTcpServer
from wyoming.wake import Detect, Detection, NotDetected

from app.engine import Detector
from app.hub import Hub
from app.registry import Registry
from app.settings import Settings
from app.web import create_app
from app.wyoming_server import WakeHandler

WAV = os.environ.get("OWW_TEST_WAV")
needs_wav = pytest.mark.skipif(not WAV, reason="OWW_TEST_WAV not set")
ALEXA = "alexa_v0.1"


def read_pcm() -> bytes:
    with wave.open(WAV, "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1
        return b"\0" * 32000 + wav.readframes(wav.getnframes()) + b"\0" * 32000


@pytest.fixture
def hub(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    return Hub(Settings(tmp_path / "settings.json"), Registry(custom), wyoming_port=0)


def test_settings_validate_and_persist(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    settings.update_global({"threshold": 5, "trigger_level": "3"})
    settings.update_model("foo", {"enabled": False, "threshold": 0.7})
    with pytest.raises(ValueError):
        settings.update_global({"bogus": 1})
    with pytest.raises(ValueError):
        settings.update_global({"threshold": "abc"})
    again = Settings(path)
    assert again.get("threshold") == 0.99
    assert again.get("trigger_level") == 3
    assert not again.is_enabled("foo")
    assert again.threshold_for("foo") == 0.7
    assert again.threshold_for("other") == 0.99


def test_registry_finds_builtins_and_custom(hub):
    ids = set(hub.registry.models)
    assert {ALEXA, "hey_jarvis_v0.1", "timer_v0.1"} <= ids
    src = Path(hub.registry.get(ALEXA).path)
    (hub.registry.custom_dir / f"ok_computer_v2{src.suffix}").write_bytes(src.read_bytes())
    hub.registry.scan()
    custom = hub.registry.get("ok_computer_v2")
    assert custom.source == "custom" and custom.phrase == "Ok Computer"


@needs_wav
def test_detector_detects_once_and_respects_threshold(hub):
    models = [hub.registry.get(ALEXA), hub.registry.get("hey_jarvis_v0.1")]
    detector = Detector(models, hub.settings)
    results = detector.process(read_pcm(), flush=True)
    detections = [d for r in results for d in r.detections]
    assert [d["model"] for d in detections] == [ALEXA]
    assert max(r.scores[ALEXA] for r in results) > 0.5
    assert max(r.scores["hey_jarvis_v0.1"] for r in results) < 0.5

    hub.settings.update_model(ALEXA, {"trigger_level": 20})
    detector.reset()
    results = detector.process(read_pcm(), flush=True)
    assert not [d for r in results for d in r.detections]


@needs_wav
def test_detector_with_vad(hub):
    hub.settings.update_global({"vad_threshold": 0.3})
    detector = Detector([hub.registry.get(ALEXA)], hub.settings)
    results = detector.process(read_pcm(), flush=True)
    assert any(r.vad is not None and r.vad > 0.3 for r in results)
    assert [d["model"] for r in results for d in r.detections] == [ALEXA]


def test_multiclass_labels(hub):
    detector = Detector([hub.registry.get("timer_v0.1")], hub.settings)
    assert set(detector.labels.values()) == {"timer_v0.1"}
    assert "1_minute_timer" in detector.labels
    results = detector.process(b"\0" * 32000)
    assert len(results) == 12 and not results[0].detections


@pytest.fixture
async def wyoming_port(hub):
    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(partial(WakeHandler, hub))
    yield server._server.sockets[0].getsockname()[1]
    await server.stop()


async def test_wyoming_describe_lists_enabled_models(hub, wyoming_port):
    hub.settings.update_model("weather_v0.1", {"enabled": False})
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await asyncio.wait_for(client.read_event(), 5))
    names = [m.name for m in info.wake[0].models]
    assert ALEXA in names and "weather_v0.1" not in names
    assert next(m for m in info.wake[0].models if m.name == ALEXA).phrase == "Alexa"
    assert not hub.clients  # a probe is not a listening client


async def _stream(client, pcm, rate=16000):
    await client.write_event(AudioStart(rate=rate, width=2, channels=1).event())
    for i in range(0, len(pcm), 2048):
        await client.write_event(
            AudioChunk(rate=rate, width=2, channels=1, audio=pcm[i:i + 2048]).event()
        )
    await client.write_event(AudioStop().event())


@needs_wav
async def test_wyoming_detection_and_live_events(hub, wyoming_port):
    queue = hub.subscribe()
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Detect(names=[ALEXA]).event())
        await _stream(client, read_pcm())
        event = await asyncio.wait_for(client.read_event(), 20)
        assert Detection.is_type(event.type)
        assert Detection.from_event(event).name == ALEXA
        assert len(hub.clients) == 1
        assert next(iter(hub.clients.values()))["models"] == [ALEXA]
    await asyncio.sleep(0.2)
    assert not hub.clients
    assert hub.detections[0]["phrase"] == "Alexa" and hub.detections[0]["kind"] == "wyoming"
    kinds = set()
    while not queue.empty():
        kinds.add(queue.get_nowait()["type"])
    assert {"state", "scores", "detection"} <= kinds


@needs_wav
async def test_wyoming_not_detected_for_other_or_disabled_model(hub, wyoming_port):
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Detect(names=["hey_jarvis_v0.1"]).event())
        await _stream(client, read_pcm())
        assert NotDetected.is_type((await asyncio.wait_for(client.read_event(), 20)).type)

    hub.settings.update_model(ALEXA, {"enabled": False})
    async with AsyncTcpClient("127.0.0.1", wyoming_port) as client:
        await client.write_event(Detect(names=[ALEXA]).event())
        await _stream(client, read_pcm())
        assert NotDetected.is_type((await asyncio.wait_for(client.read_event(), 20)).type)
        assert "disabled or missing" in next(iter(hub.clients.values()))["warning"]


@pytest.fixture
async def api(hub):
    runner = web.AppRunner(create_app(hub))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with aiohttp.ClientSession(base_url=f"http://127.0.0.1:{port}") as session:
        yield session
    await runner.cleanup()


async def test_api_state_settings_and_models(hub, api):
    async with api.get("/") as resp:
        assert resp.status == 200 and "openWakeWord" in await resp.text()
    async with api.get("/api/state") as resp:
        state = await resp.json()
    assert len(state["models"]) == 6 and state["settings"]["threshold"] == 0.5

    async with api.put("/api/settings", json={"threshold": 0.7}) as resp:
        assert (await resp.json())["settings"]["threshold"] == 0.7
    async with api.put("/api/settings", json={"nope": 1}) as resp:
        assert resp.status == 400
    async with api.put(f"/api/models/{ALEXA}", json={"enabled": False, "threshold": 0.3}) as resp:
        model = next(m for m in (await resp.json())["models"] if m["id"] == ALEXA)
        assert model["enabled"] is False and model["threshold"] == 0.3
    async with api.put("/api/models/unknown", json={"enabled": False}) as resp:
        assert resp.status == 404
    async with api.delete(f"/api/models/{ALEXA}") as resp:
        assert resp.status == 400


async def test_api_upload_and_delete(hub, api):
    src = Path(hub.registry.get("hey_jarvis_v0.1").path)

    form = aiohttp.FormData()
    form.add_field("file", b"this is not a model", filename="broken.onnx")
    form.add_field("file", b"nope", filename="notes.txt")
    async with api.post("/api/models", data=form) as resp:
        body = await resp.json()
        assert resp.status == 400 and len(body["errors"]) == 2
    assert not [p for p in hub.registry.custom_dir.iterdir()]

    form = aiohttp.FormData()
    form.add_field("file", src.read_bytes(), filename=f"../../hey computer_v1{src.suffix}")
    async with api.post("/api/models", data=form) as resp:
        body = await resp.json()
        assert resp.status == 200 and body["added"] == ["hey_computer_v1"], body
    assert (hub.registry.custom_dir / f"hey_computer_v1{src.suffix}").exists()
    model = next(m for m in body["state"]["models"] if m["id"] == "hey_computer_v1")
    assert model["source"] == "custom" and model["phrase"] == "Hey Computer"

    async with api.delete("/api/models/hey_computer_v1") as resp:
        assert resp.status == 200
    assert not [p for p in hub.registry.custom_dir.iterdir()]


@needs_wav
async def test_ws_test_stream(hub, api):
    pcm = read_pcm()
    async with api.ws_connect("/ws/test") as ws:
        await ws.send_json({"type": "start", "models": [ALEXA], "source": "file"})
        started = await ws.receive_json(timeout=10)
        assert started["type"] == "started" and started["labels"] == {ALEXA: ALEXA}
        for i in range(0, len(pcm), 16000):
            await ws.send_bytes(pcm[i:i + 16000])
        await ws.send_json({"type": "end"})
        frames = []
        while True:
            msg = await ws.receive_json(timeout=20)
            if msg["type"] == "done":
                break
            frames.extend(msg["frames"])
    assert len(frames) == -(-len(pcm) // 2560)
    assert sum(len(f.get("detections", [])) for f in frames) == 1
    assert hub.detections[0]["kind"] == "test" and hub.detections[0]["source"] == "File test"

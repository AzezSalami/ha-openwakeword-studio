"""HTTP + WebSocket API and static web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from urllib.parse import unquote

from aiohttp import WSMsgType, web

from .engine import Detector
from .hub import Hub
from .registry import MODEL_EXTENSIONS, ModelInfo, phrase_from_id

_LOGGER = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_MODEL_BYTES = 50 * 1024 * 1024
INGRESS_ADDRESSES = {"172.30.32.2", "127.0.0.1", "::1"}
HUB = web.AppKey("hub", Hub)


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


@web.middleware
async def ingress_only(request: web.Request, handler):
    if request.remote not in INGRESS_ADDRESSES:
        return web.Response(status=403, text="Open this page from the Home Assistant sidebar.")
    return await handler(request)


@web.middleware
async def no_cache(request: web.Request, handler):
    response = await handler(request)
    if not request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def index(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    html = (STATIC_DIR / "index.html").read_text().replace("__VERSION__", hub.version)
    return web.Response(text=html, content_type="text/html")


async def get_state(request: web.Request) -> web.Response:
    return web.json_response(request.app[HUB].state())


async def put_settings(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    try:
        changes = await request.json()
        if not isinstance(changes, dict):
            raise ValueError("Expected an object")
        if changes.get("noise_suppression") and not hub.registry.has_speex:
            raise ValueError("Noise suppression is not available on this system")
        hub.settings.update_global(changes)
    except (ValueError, json.JSONDecodeError) as err:
        return _error(400, str(err))
    hub.publish_state()
    return web.json_response(hub.state())


async def put_model(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    model_id = request.match_info["model_id"]
    if hub.registry.get(model_id) is None:
        return _error(404, "Unknown wake word")
    try:
        changes = await request.json()
        if not isinstance(changes, dict):
            raise ValueError("Expected an object")
        hub.settings.update_model(model_id, changes)
    except (ValueError, json.JSONDecodeError) as err:
        return _error(400, str(err))
    hub.publish_state()
    return web.json_response(hub.state())


def _try_load(info: ModelInfo, hub: Hub) -> None:
    detector = Detector([info], hub.settings)
    detector.process(b"\0" * 2560)


async def upload_models(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    custom_dir = hub.registry.custom_dir
    try:
        custom_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        return _error(500, f"Cannot create {custom_dir}: {err}")

    if not request.content_type.startswith("multipart/"):
        return _error(400, "Expected a file upload")

    added: list[str] = []
    errors: list[str] = []
    reader = await request.multipart()
    loop = asyncio.get_running_loop()
    async for part in reader:
        if not part.filename:
            continue
        name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", Path(unquote(part.filename)).name).lstrip(".")
        suffix = Path(name).suffix.lower()
        if suffix not in MODEL_EXTENSIONS or not Path(name).stem:
            errors.append(f"{part.filename}: only .tflite and .onnx files are supported")
            continue
        if suffix == ".tflite" and not hub.registry.has_tflite:
            errors.append(f"{part.filename}: .tflite models are not supported on this system")
            continue
        target = custom_dir / (Path(name).stem + suffix)
        # Hidden temp file that keeps the extension (openWakeWord loads by extension).
        temp = custom_dir / f".upload-{target.name}"
        size = 0
        try:
            with temp.open("wb") as handle:
                while chunk := await part.read_chunk(65536):
                    size += len(chunk)
                    if size > MAX_MODEL_BYTES:
                        break
                    handle.write(chunk)
            if size > MAX_MODEL_BYTES:
                errors.append(f"{part.filename}: file is too large")
                continue
            candidate = ModelInfo(
                id=target.stem,
                phrase=phrase_from_id(target.stem),
                path=str(temp),
                framework=suffix[1:],
                source="custom",
                size=size,
            )
            try:
                await loop.run_in_executor(hub.executor, _try_load, candidate, hub)
            except Exception as err:  # openWakeWord raises many kinds
                _LOGGER.warning("Rejected upload %s: %s", part.filename, err)
                errors.append(f"{part.filename}: not a valid openWakeWord model")
                continue
            temp.replace(target)
            target.with_suffix(".json").unlink(missing_ok=True)  # metadata of a replaced model
            added.append(target.stem)
            _LOGGER.info("Added custom wake word %s", target.name)
        except OSError as err:
            errors.append(f"{part.filename}: {err}")
        finally:
            temp.unlink(missing_ok=True)

    hub.registry.scan()
    hub.publish_state()
    status = 200 if added or not errors else 400
    return web.json_response({"added": added, "errors": errors, "state": hub.state()}, status=status)


async def delete_model(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    info = hub.registry.get(request.match_info["model_id"])
    if info is None:
        return _error(404, "Unknown wake word")
    if info.source != "custom":
        return _error(400, "Built-in wake words cannot be deleted; disable it instead")
    path = Path(info.path)
    try:
        for suffix in (*MODEL_EXTENSIONS, ".json"):
            path.with_suffix(suffix).unlink(missing_ok=True)
    except OSError as err:
        return _error(500, f"Could not delete: {err}")
    hub.settings.forget_model(info.id)
    hub.registry.scan()
    hub.publish_state()
    return web.json_response(hub.state())


# -- training --------------------------------------------------------------
def _trainer(request: web.Request):
    trainer = request.app[HUB].trainer
    if trainer is None:
        raise web.HTTPServiceUnavailable(text="Training is not available in this installation")
    return trainer


def _phrase(request: web.Request) -> str:
    phrase = " ".join(request.query.get("phrase", "").split())
    if len(phrase) < 3 or len(phrase) > 40:
        raise ValueError("Enter the wake word first")
    return phrase


async def training_languages(request: web.Request) -> web.Response:
    from . import trainer as tr

    trainer = _trainer(request)
    loop = asyncio.get_running_loop()
    try:
        index = await loop.run_in_executor(None, tr.load_voice_index, trainer.assets)
        return web.json_response({"languages": tr.languages(index), "online": True})
    except Exception as err:  # offline: training would fail too, but let the UI say so
        _LOGGER.warning("Could not load the voice list: %s", err)
        return web.json_response({"languages": tr.FALLBACK_LANGUAGES, "online": False})


async def training_start(request: web.Request) -> web.Response:
    from .trainer import TrainingConfig

    trainer = _trainer(request)
    try:
        data = await request.json()
        trainer.start(TrainingConfig(
            phrase=str(data.get("phrase", "")),
            language=str(data.get("language") or "en_US"),
            quality=str(data.get("quality") or "standard"),
        ))
    except (ValueError, json.JSONDecodeError, AttributeError) as err:
        return _error(400, str(err))
    return web.json_response(trainer.status.as_dict())


async def training_cancel(request: web.Request) -> web.Response:
    trainer = _trainer(request)
    trainer.cancel()
    return web.json_response(trainer.status.as_dict())


async def samples_list(request: web.Request) -> web.Response:
    trainer = _trainer(request)
    try:
        return web.json_response({"samples": trainer.list_samples(_phrase(request))})
    except ValueError:
        return web.json_response({"samples": []})


async def samples_add(request: web.Request) -> web.Response:
    """Body: raw 16 kHz mono 16-bit PCM of one recording of the wake word."""
    trainer = _trainer(request)
    try:
        phrase = _phrase(request)
        pcm = await request.read()
        if len(pcm) > 16000 * 2 * 15:
            raise ValueError("The recording is too long - say only the wake word")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, trainer.add_sample, phrase, pcm)
    except ValueError as err:
        return _error(400, str(err))
    return web.json_response({"samples": trainer.list_samples(phrase)})


async def samples_get(request: web.Request) -> web.StreamResponse:
    trainer = _trainer(request)
    name = request.match_info["name"]
    try:
        phrase = _phrase(request)
    except ValueError as err:
        return _error(400, str(err))
    path = trainer.sample_dir(phrase) / name
    if not re.fullmatch(r"\d+\.wav", name) or not path.is_file():
        return _error(404, "Unknown recording")
    return web.FileResponse(path)


async def samples_delete(request: web.Request) -> web.Response:
    trainer = _trainer(request)
    try:
        phrase = _phrase(request)
        trainer.delete_sample(phrase, request.match_info["name"])
    except ValueError as err:
        return _error(400, str(err))
    return web.json_response({"samples": trainer.list_samples(phrase)})


async def rescan(request: web.Request) -> web.Response:
    hub = request.app[HUB]
    hub.registry.scan()
    hub.publish_state()
    return web.json_response(hub.state())


async def ws_events(request: web.Request) -> web.WebSocketResponse:
    hub = request.app[HUB]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    queue = hub.subscribe()

    async def sender() -> None:
        await ws.send_json({"type": "state", "state": hub.state()})
        while True:
            await ws.send_json(await queue.get())

    task = asyncio.create_task(sender())
    try:
        async for _ in ws:
            pass
    finally:
        task.cancel()
        hub.unsubscribe(queue)
    return ws


async def ws_test(request: web.Request) -> web.WebSocketResponse:
    """Browser streams 16 kHz mono 16-bit PCM; we answer with per-frame scores."""
    hub = request.app[HUB]
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    detector: Detector | None = None
    source = "Browser test"

    async def send_results(results) -> None:
        if not results:
            return
        for result in results:
            for detection in result.detections:
                hub.record_detection(detection, source=source, kind="test")
        await ws.send_json({"type": "frames", "frames": [r.as_dict() for r in results]})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "start":
                    wanted = data.get("models")
                    models = [
                        m for m in hub.registry.sorted()
                        if m.usable and (wanted is None or m.id in wanted)
                    ]
                    source = "File test" if data.get("source") == "file" else "Microphone test"
                    try:
                        detector = await loop.run_in_executor(
                            hub.executor, Detector, models, hub.settings
                        )
                    except Exception as err:
                        _LOGGER.exception("Could not start test")
                        await ws.send_json({"type": "error", "error": f"Could not load models: {err}"})
                        detector = None
                        continue
                    await ws.send_json({"type": "started", "labels": detector.labels})
                elif kind == "end" and detector is not None:
                    await send_results(
                        await loop.run_in_executor(hub.executor, detector.process, b"", True)
                    )
                    await ws.send_json({"type": "done"})
                    detector = None
            elif msg.type == WSMsgType.BINARY and detector is not None:
                await send_results(
                    await loop.run_in_executor(hub.executor, detector.process, msg.data)
                )
    except ConnectionResetError:
        pass
    return ws


def create_app(hub: Hub, restrict_to_ingress: bool = False) -> web.Application:
    middlewares = [no_cache]
    if restrict_to_ingress:
        middlewares.insert(0, ingress_only)
    app = web.Application(middlewares=middlewares, client_max_size=MAX_MODEL_BYTES * 4)
    app[HUB] = hub
    app.router.add_get("/", index)
    app.router.add_get("/api/state", get_state)
    app.router.add_put("/api/settings", put_settings)
    app.router.add_post("/api/models", upload_models)
    app.router.add_post("/api/rescan", rescan)
    app.router.add_put("/api/models/{model_id}", put_model)
    app.router.add_delete("/api/models/{model_id}", delete_model)
    app.router.add_get("/api/training/languages", training_languages)
    app.router.add_post("/api/training/start", training_start)
    app.router.add_post("/api/training/cancel", training_cancel)
    app.router.add_get("/api/training/samples", samples_list)
    app.router.add_post("/api/training/samples", samples_add)
    app.router.add_get("/api/training/samples/{name}", samples_get)
    app.router.add_delete("/api/training/samples/{name}", samples_delete)
    app.router.add_get("/ws/events", ws_events)
    app.router.add_get("/ws/test", ws_test)
    app.router.add_static("/static/", STATIC_DIR)
    return app

"""Entry point: starts the Wyoming server and the web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
from functools import partial
from pathlib import Path

import aiohttp
from aiohttp import web
from wyoming.server import AsyncTcpServer

from .hub import OWW_VERSION, Hub
from .registry import Registry
from .settings import Settings
from .web import create_app
from .wyoming_server import WakeHandler

_LOGGER = logging.getLogger("studio")

SUPERVISOR = "http://supervisor"


def load_options(data_dir: Path) -> dict:
    try:
        return json.loads((data_dir / "options.json").read_text())
    except (OSError, ValueError):
        return {}


async def supervisor_setup(port: int) -> str:
    """Look up our hostname and announce the Wyoming service to Home Assistant."""
    token = os.environ.get("SUPERVISOR_TOKEN")
    hostname = socket.gethostname()
    if not token:
        return hostname
    headers = {"Authorization": f"Bearer {token}"}
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(f"{SUPERVISOR}/addons/self/info") as resp:
                if resp.status == 200:
                    hostname = (await resp.json())["data"].get("hostname") or hostname
            payload = {"service": "wyoming", "config": {"uri": f"tcp://{hostname}:{port}"}}
            async with session.post(f"{SUPERVISOR}/discovery", json=payload) as resp:
                if resp.status == 200:
                    _LOGGER.info("Announced wake word service to Home Assistant")
                else:
                    _LOGGER.warning("Discovery failed (%s): %s", resp.status, await resp.text())
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as err:
        _LOGGER.warning("Could not talk to the Supervisor: %s", err)
    return hostname


async def main() -> None:
    data_dir = Path(os.environ.get("STUDIO_DATA_DIR", "/data"))
    options = load_options(data_dir)
    logging.basicConfig(
        level=logging.DEBUG if options.get("debug_logging") else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    custom_dir = Path(
        os.environ.get("STUDIO_CUSTOM_DIR")
        or options.get("custom_model_dir")
        or "/share/openwakeword"
    )
    wyoming_port = int(os.environ.get("STUDIO_WYOMING_PORT", "10400"))
    web_port = int(os.environ.get("STUDIO_WEB_PORT", "8099"))
    version = os.environ.get("STUDIO_VERSION", "dev")
    under_supervisor = bool(os.environ.get("SUPERVISOR_TOKEN"))

    _LOGGER.info("openWakeWord Studio %s (openWakeWord %s)", version, OWW_VERSION)
    try:
        custom_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        _LOGGER.warning("Cannot create custom model folder %s: %s", custom_dir, err)

    settings = Settings(data_dir / "settings.json")
    registry = Registry(custom_dir)
    if not registry.has_speex and settings.get("noise_suppression"):
        settings.update_global({"noise_suppression": False})
    hub = Hub(settings, registry, version=version, wyoming_port=wyoming_port)
    _LOGGER.info(
        "Found %d wake word models (%d custom in %s)",
        len(registry.models),
        sum(1 for m in registry.models.values() if m.source == "custom"),
        custom_dir,
    )

    try:
        hub.attach_trainer(data_dir, asyncio.get_running_loop())
    except ImportError as err:
        _LOGGER.warning("Wake word training is unavailable: %s", err)

    wyoming = AsyncTcpServer("0.0.0.0", wyoming_port)
    await wyoming.start(partial(WakeHandler, hub))
    _LOGGER.info("Wyoming server listening on port %d", wyoming_port)

    runner = web.AppRunner(create_app(hub, restrict_to_ingress=under_supervisor), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", web_port).start()
    _LOGGER.info("Web UI listening on port %d", web_port)

    hub.hostname = await supervisor_setup(wyoming_port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()

    _LOGGER.info("Shutting down")
    await wyoming.stop()
    await runner.cleanup()
    hub.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    asyncio.run(main())

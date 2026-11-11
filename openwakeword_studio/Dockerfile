# Debian-based Python image: openWakeWord's dependencies (onnxruntime, LiteRT,
# speexdsp-ns) only ship glibc wheels, so the usual Alpine add-on base won't do.
# The image is multi-arch (amd64 + aarch64); BUILD_FROM is intentionally unused.
FROM python:3.12-slim-bookworm

# Pinned openWakeWord commit (https://github.com/dscripka/openWakeWord)
ARG OPENWAKEWORD_REF=368c03716d1e92591906a84949bc477f3a834455

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/studio

COPY requirements.txt ./
RUN pip install -r requirements.txt \
        "openwakeword @ https://github.com/dscripka/openWakeWord/archive/${OPENWAKEWORD_REF}.tar.gz" \
    # Bake the feature, VAD and pre-trained wake word models into the image
    && python -c "from openwakeword.utils import download_models; download_models()"

COPY app ./app

ARG BUILD_VERSION=dev
ENV STUDIO_VERSION=${BUILD_VERSION}

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="addon" \
    io.hass.arch="amd64|aarch64"

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8099/api/state', timeout=5)" || exit 1

CMD ["python", "-m", "app"]

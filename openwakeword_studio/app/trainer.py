"""Train a custom wake word model inside the add-on.

The approach follows openWakeWord: a small classifier on top of the frozen
speech-embedding features, trained on synthetic speech. To stay light enough
for Home Assistant hardware it uses Piper (ONNX) for speech synthesis and a
NumPy-trained network, instead of the PyTorch pipeline of the upstream project:

1. positives: the phrase spoken by many Piper voices/speakers (+ the user's own
   recordings), augmented with noise, reverb, level and timing changes
2. negatives: ~11 hours of pre-computed features of speech, noise and music
   (the openWakeWord validation set), similar-sounding phrases, cut-off phrases
3. training with hard-negative mining, evaluation on held-out data
4. export as an ONNX model that openWakeWord loads like any other custom model
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_LOGGER = logging.getLogger(__name__)

SR = 16000
N_FRAMES = 16
LONG = 3 * SR  # training clips are 3 s = 28 embedding frames = 13 sliding model inputs
N_WINDOWS = 13


def window_end(j: int) -> int:
    """Sample position where the j-th 16-frame window of a clip ends."""
    return 1280 * j + 31600

EMB = 96

HF = "https://huggingface.co"
VOICES_INDEX_URL = f"{HF}/rhasspy/piper-voices/resolve/main/voices.json"
VOICE_BASE_URL = f"{HF}/rhasspy/piper-voices/resolve/main/"
NEGATIVE_FEATURES_URL = (
    f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
)
NEGATIVE_FEATURES_BYTES = 184836608

QUALITY = {
    # name: (synthetic positives, description)
    "quick": 800,
    "standard": 2500,
    "thorough": 6000,
}
MAX_VOICES = 3
DEFAULT_LANGUAGE = "en_US"
UNSUPPORTED_FAMILIES = {"zh", "ja", "th"}  # need extra phonemizer packages in Piper
FALLBACK_LANGUAGES = [{"code": "en_US", "name": "English (United States)", "voices": 3}]

COMMON_WORDS = (
    "the and you that was for are with his they this have from one had word but not what all "
    "were when your can said there use each which she how their will other about out many then "
    "them these some her would make like him into time has look two more write see number way "
    "could people my than first water been call who oil now find long down day did get come made "
    "may part over new sound take only little work know place year live me back give most very "
    "after thing our just name good sentence man think say great where help through much before "
    "line right too mean old any same tell boy follow came want show also around form three small "
    "set put end does another well large must big even such because turn here why ask went men "
    "read need land different home us move try kind hand picture again change off play spell air "
    "away animal house point page letter mother answer found study still learn should america world "
    "okay hello please thanks today tomorrow music light kitchen weather open close start stop"
).split()


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def slugify(phrase: str) -> str:
    text = unicodedata.normalize("NFKD", phrase).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        import hashlib

        slug = "wake_word_" + hashlib.sha1(phrase.encode()).hexdigest()[:8]
    return slug[:48]


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def write_wav_16k(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm)


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == SR:
        return audio.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(SR, rate)
    return resample_poly(audio, SR // g, rate // g).astype(np.float32)


def trim_silence(audio: np.ndarray, rel: float = 0.04, pad: int = 480) -> np.ndarray:
    """Cut leading/trailing silence using 10 ms energy frames."""
    if audio.size < 320:
        return audio
    frame = 160
    n = audio.size // frame
    energy = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    peak = float(energy.max())
    if peak < 1e-4:
        return audio[:0]
    active = np.nonzero(energy > max(peak * rel, 2e-3))[0]
    start = max(0, active[0] * frame - pad)
    end = min(audio.size, (active[-1] + 1) * frame + pad)
    return audio[start:end]


def colored_noise(rng: np.random.Generator, n: int, exponent: float) -> np.ndarray:
    """exponent 0 = white, 1 = pink, 2 = brown."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2), n)
    return (noise / (np.sqrt(np.mean(noise**2)) + 1e-9)).astype(np.float32)


def add_reverb(rng: np.random.Generator, speech: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    rt60 = rng.uniform(0.15, 0.7)
    n = int(rt60 * SR)
    tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    tail *= rng.uniform(0.15, 0.6) / (np.sqrt(np.sum(tail**2)) + 1e-9)
    tail[0] = 1.0
    wet = fftconvolve(speech, tail)[: speech.size + int(0.12 * SR)]
    return wet.astype(np.float32)


def make_clip(
    rng: np.random.Generator,
    speech: np.ndarray,
    babble: list[np.ndarray] | None = None,
    end_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Put speech into a 3 s clip with room and background noise.

    Returns the int16 clip and the sample positions where the speech starts and ends.
    """
    if speech.size and rng.random() < 0.4:
        speech = add_reverb(rng, speech)
    clip = np.zeros(LONG, dtype=np.float32)
    level = 0.05
    start = end = 0
    if speech.size:
        speech = speech * (rng.uniform(0.08, 0.9) / (np.abs(speech).max() + 1e-9))
        low, high = end_range or (min(speech.size, LONG), LONG)
        end = int(rng.integers(low, high + 1))
        speech = speech[-end:]
        start = end - speech.size
        clip[start:end] = speech
        level = float(np.sqrt(np.mean(speech**2))) + 1e-6

    kind = rng.random()
    if kind < 0.12:
        noise = colored_noise(rng, LONG, 0) * 1e-3 / level  # near silence
        snr = 0.0
    else:
        if kind < 0.35 and babble:
            other = babble[int(rng.integers(len(babble)))]
            if other.size > 1600:
                reps = int(np.ceil(LONG / other.size)) + 1
                offset = int(rng.integers(other.size))
                noise = np.tile(other, reps)[offset:offset + LONG]
                noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-9)
            else:
                noise = colored_noise(rng, LONG, 1)
            snr = rng.uniform(6, 25)
        else:
            noise = colored_noise(rng, LONG, float(rng.choice([0, 1, 1, 2])))
            snr = rng.uniform(3, 30)
        noise = noise * level / (10 ** (snr / 20))
    clip = clip + noise.astype(np.float32)
    return np.clip(clip * 32767.0, -32768, 32767).astype(np.int16), start, end


# --------------------------------------------------------------------------
# text generation for negatives
# --------------------------------------------------------------------------
_VOWELS = "aeiouy"


def _perturb_word(rng: np.random.Generator, word: str) -> str:
    letters = list(word)
    idx = [i for i, c in enumerate(letters) if c.isalpha()]
    if len(idx) < 2:
        return str(rng.choice(COMMON_WORDS))
    n_change = max(2, int(round(len(idx) * rng.uniform(0.3, 0.6))))
    for i in rng.choice(idx, size=min(n_change, len(idx)), replace=False):
        c = letters[i].lower()
        if c in _VOWELS:
            pool = [v for v in "aeiou" if v != c]
        elif c.isascii():
            pool = [k for k in "bdfgklmnprstvz" if k != c]
        else:
            continue
        letters[i] = str(rng.choice(pool))
    return "".join(letters)


_ONSETS = "b d f g h j k l m n p r s t v w z ch sh st tr pl br kr".split()
_NUCLEI = "a e i o u ai ee oo ou er".split()
_CODAS = ["", "", "", "n", "r", "s", "t", "m", "l", "k"]


def _pseudo_word(rng: np.random.Generator) -> str:
    """A pronounceable made-up word, for phonetic variety in any language."""
    return "".join(
        str(rng.choice(_ONSETS)) + str(rng.choice(_NUCLEI)) + str(rng.choice(_CODAS))
        for _ in range(int(rng.integers(1, 4)))
    )


def negative_texts(rng: np.random.Generator, phrase: str, n: int) -> list[str]:
    """Phrases that sound similar to, or share parts with, the wake phrase - plus random speech."""
    words = phrase.split()
    def word() -> str:
        if rng.random() < 0.5:
            return str(rng.choice(COMMON_WORDS))
        return _pseudo_word(rng)

    texts: list[str] = []
    while len(texts) < n:
        r = rng.random()
        if r < 0.15 and len(words) > 1:
            # some of the words, possibly followed or preceded by something else
            k = int(rng.integers(1, len(words)))
            start = int(rng.integers(0, len(words) - k + 1))
            part = words[start:start + k]
            extra = rng.random()
            if extra < 0.35:
                part = part + [word()]
            elif extra < 0.5:
                part = [word()] + part
            text = " ".join(part)
        elif r < 0.3:
            # one of the words inside an ordinary sentence ("... in the kitchen today")
            before = [word() for _ in range(int(rng.integers(1, 4)))]
            after = [word() for _ in range(int(rng.integers(0, 3)))]
            text = " ".join(before + [str(rng.choice(words))] + after)
        elif r < 0.47:
            # same beginning, different ending ("hey com" + "e here")
            cut = int(round(len(phrase) * rng.uniform(0.35, 0.75)))
            tail = " ".join(word() for _ in range(int(rng.integers(1, 3))))
            text = phrase[:cut] + (tail if rng.random() < 0.5 else " " + tail)
        elif r < 0.6:
            # different beginning, same ending
            cut = int(round(len(phrase) * rng.uniform(0.3, 0.65)))
            head = " ".join(word() for _ in range(int(rng.integers(1, 3))))
            text = (head if rng.random() < 0.5 else head + " ") + phrase[cut:]
        elif r < 0.8:
            # one word swapped for another word or a similar-sounding non-word
            changed = list(words)
            i = int(rng.integers(len(words)))
            if len(words) > 1 and rng.random() < 0.5:
                changed[i] = word()
            else:
                changed[i] = _perturb_word(rng, words[i])
                if len(words) > 1 and rng.random() < 0.3:
                    j = int(rng.integers(len(words)))
                    changed[j] = _perturb_word(rng, words[j])
            text = " ".join(changed)
        else:
            text = " ".join(word() for _ in range(int(rng.integers(1, 6))))
        text = " ".join(text.split())
        if text and text.lower() != phrase.lower():
            texts.append(text)
    return texts


# --------------------------------------------------------------------------
# assets: Piper voices and negative features
# --------------------------------------------------------------------------
def download(url: str, target: Path, progress: Callable[[float], None], cancel: threading.Event,
             expected: int | None = None) -> None:
    import requests

    if target.exists() and (expected is None or target.stat().st_size == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or expected or 0)
        done = 0
        with temp.open("wb") as handle:
            for chunk in resp.iter_content(1 << 18):
                if cancel.is_set():
                    raise Cancelled()
                handle.write(chunk)
                done += len(chunk)
                if total:
                    progress(min(1.0, done / total))
    if expected is not None and temp.stat().st_size != expected:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {target.name} was incomplete")
    os.replace(temp, target)


def load_voice_index(assets: Path, refresh: bool = False) -> dict:
    path = assets / "voices.json"
    if refresh or not path.exists():
        import requests

        resp = requests.get(VOICES_INDEX_URL, timeout=30)
        resp.raise_for_status()
        assets.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return json.loads(path.read_text())


def voices_for(index: dict, language: str) -> list[dict]:
    """Up to MAX_VOICES voices for a language, preferring many speakers and small models."""
    rank = {"medium": 0, "low": 1, "x_low": 2, "high": 3}
    voices = [v for v in index.values() if v["language"]["code"] == language]
    voices.sort(key=lambda v: (-(v["num_speakers"] > 1), rank.get(v["quality"], 9), -v["num_speakers"], v["key"]))
    chosen, names = [], set()
    for voice in voices:  # one quality per voice name
        if voice["name"] in names:
            continue
        names.add(voice["name"])
        chosen.append(voice)
        if len(chosen) == MAX_VOICES:
            break
    return chosen


def languages(index: dict) -> list[dict]:
    seen: dict[str, dict] = {}
    for voice in index.values():
        lang = voice["language"]
        if lang["family"] in UNSUPPORTED_FAMILIES:
            continue
        entry = seen.setdefault(
            lang["code"],
            {"code": lang["code"], "name": f"{lang['name_english']} ({lang['country_english']})", "voices": 0},
        )
        entry["voices"] += 1
    result = sorted(seen.values(), key=lambda e: e["name"])
    for entry in result:
        entry["voices"] = min(entry["voices"], MAX_VOICES)
    return result


# --------------------------------------------------------------------------
# the network (NumPy)
# --------------------------------------------------------------------------
class Net:
    """1536 -> 96 -> 48 -> 1 MLP with input standardisation."""

    def __init__(self, rng: np.random.Generator, mean: np.ndarray, std: np.ndarray, hidden=(96, 48)) -> None:
        self.mean = mean.astype(np.float32)
        self.inv_std = (1.0 / np.maximum(std, 1e-3)).astype(np.float32)
        sizes = [N_FRAMES * EMB, *hidden, 1]
        self.W = [
            (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.b[-1][:] = -2.0  # start out predicting "no wake word"
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = (x.reshape(len(x), -1) - self.mean) * self.inv_std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                np.maximum(h, 0, out=h)
        return 1.0 / (1.0 + np.exp(-np.clip(h[:, 0], -30, 30)))

    def train_step(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, lr: float, decay: float = 1e-4) -> float:
        acts = [(x.reshape(len(x), -1) - self.mean) * self.inv_std]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = acts[-1] @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        z = np.clip(acts[-1][:, 0], -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(np.mean(w * -(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))))
        delta = ((p - y) * w / len(x))[:, None].astype(np.float32)
        grads_W, grads_b = [], []
        for i in reversed(range(len(self.W))):
            grads_W.insert(0, acts[i].T @ delta + decay * self.W[i])
            grads_b.insert(0, delta.sum(axis=0))
            if i:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        self._t += 1
        params = self.W + self.b
        for k, grad in enumerate(grads_W + grads_b):
            self._m[k] = 0.9 * self._m[k] + 0.1 * grad
            self._v[k] = 0.999 * self._v[k] + 0.001 * grad * grad
            m_hat = self._m[k] / (1 - 0.9**self._t)
            v_hat = self._v[k] / (1 - 0.999**self._t)
            params[k] -= (lr * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)
        return loss

    def export_onnx(self, path: Path) -> None:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        init = [
            numpy_helper.from_array(np.array([1, N_FRAMES * EMB], dtype=np.int64), "shape"),
            numpy_helper.from_array(self.mean.reshape(1, -1), "mean"),
            numpy_helper.from_array(self.inv_std.reshape(1, -1), "inv_std"),
        ]
        nodes = [
            helper.make_node("Reshape", ["input", "shape"], ["flat"]),
            helper.make_node("Sub", ["flat", "mean"], ["centered"]),
            helper.make_node("Mul", ["centered", "inv_std"], ["h0"]),
        ]
        last = "h0"
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            init += [numpy_helper.from_array(W, f"W{i}"), numpy_helper.from_array(b, f"b{i}")]
            nodes.append(helper.make_node("Gemm", [last, f"W{i}", f"b{i}"], [f"z{i}"]))
            if i < len(self.W) - 1:
                nodes.append(helper.make_node("Relu", [f"z{i}"], [f"a{i}"]))
                last = f"a{i}"
            else:
                nodes.append(helper.make_node("Sigmoid", [f"z{i}"], ["output"]))
        graph = helper.make_graph(
            nodes,
            "wakeword",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, N_FRAMES, EMB])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
            initializer=init,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                                  producer_name="openwakeword-studio")
        model.ir_version = 8
        onnx.checker.check_model(model)
        temp = path.with_name("." + path.name + ".tmp")
        onnx.save(model, str(temp))
        os.replace(temp, path)


def gather(frames: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Gather (len(starts), 16, 96) windows from a (N, 96) frame sequence."""
    return frames[starts[:, None] + np.arange(N_FRAMES)[None, :]]


def score_stream(net: Net, frames: np.ndarray, start: int, stop: int, chunk: int = 8192) -> np.ndarray:
    """Score every 16-frame window in frames[start:stop] (simulates streaming)."""
    scores = np.empty(max(0, stop - start - N_FRAMES + 1), dtype=np.float32)
    for a in range(0, len(scores), chunk):
        n = min(chunk, len(scores) - a)
        block = np.asarray(frames[start + a:start + a + n + N_FRAMES - 1])
        view = np.lib.stride_tricks.sliding_window_view(block, (N_FRAMES, EMB))[:, 0]
        scores[a:a + n] = net.predict(view)
    return scores


def count_events(scores: np.ndarray, threshold: float, refractory: int = 25) -> int:
    hits = np.nonzero(scores >= threshold)[0]
    events, last = 0, -refractory
    for h in hits:
        if h - last >= refractory:
            events += 1
            last = h
    return events


# --------------------------------------------------------------------------
# the job
# --------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    phrase: str
    language: str = DEFAULT_LANGUAGE
    quality: str = "standard"
    seed: int = 0


@dataclass
class TrainingStatus:
    state: str = "idle"  # idle | running | done | error | cancelled
    phrase: str = ""
    model_id: str = ""
    stage: str = ""
    progress: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    log: list[str] = field(default_factory=list)
    result: dict | None = None

    def as_dict(self) -> dict:
        return {**self.__dict__, "log": self.log[-30:]}


# (stage name, share of the progress bar)
STAGES = [
    ("Downloading voices and data", 0.08),
    ("Generating speech", 0.27),
    ("Preparing audio", 0.40),
    ("Training", 0.20),
    ("Testing", 0.05),
]


class Trainer:
    """Runs one training job at a time in a background thread."""

    def __init__(self, data_dir: Path, custom_dir: Path, on_update: Callable[[], None],
                 on_model: Callable[[str, dict], None]) -> None:
        self.assets = data_dir / "training" / "assets"
        self.samples_dir = data_dir / "training" / "samples"
        self.custom_dir = custom_dir
        self.status = TrainingStatus()
        self._on_update = on_update
        self._on_model = on_model
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_push = 0.0

    # -- user recordings ------------------------------------------------
    def sample_dir(self, phrase: str) -> Path:
        return self.samples_dir / slugify(phrase)

    def list_samples(self, phrase: str) -> list[dict]:
        folder = self.sample_dir(phrase)
        if not folder.is_dir():
            return []
        return [
            {"name": p.name, "seconds": round((p.stat().st_size - 44) / 2 / SR, 2)}
            for p in sorted(folder.glob("*.wav"))
        ]

    def add_sample(self, phrase: str, pcm: bytes) -> dict:
        audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        speech = trim_silence(audio)
        if speech.size < 0.2 * SR:
            raise ValueError("No speech found in the recording - please try again, a bit louder")
        if speech.size > 3 * SR:
            raise ValueError("The recording is too long - say only the wake word")
        folder = self.sample_dir(phrase)
        folder.mkdir(parents=True, exist_ok=True)
        if len(list(folder.glob("*.wav"))) >= 100:
            raise ValueError("That's plenty - 100 recordings is the maximum")
        name = f"{int(time.time() * 1000)}.wav"
        write_wav_16k(folder / name, (np.clip(speech, -1, 1) * 32767).astype(np.int16).tobytes())
        return {"name": name, "seconds": round(speech.size / SR, 2)}

    def delete_sample(self, phrase: str, name: str) -> None:
        if not re.fullmatch(r"\d+\.wav", name):
            raise ValueError("Unknown recording")
        (self.sample_dir(phrase) / name).unlink(missing_ok=True)

    # -- job control ----------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, config: TrainingConfig) -> None:
        if self.running:
            raise ValueError("A wake word is already being trained")
        phrase = " ".join(config.phrase.split())
        if len(phrase) < 3:
            raise ValueError("Please enter a wake word of at least 3 characters")
        if len(phrase) > 40 or len(phrase.split()) > 5:
            raise ValueError("Please use a shorter wake word (at most 5 words / 40 characters)")
        if config.quality not in QUALITY:
            raise ValueError("Unknown quality setting")
        config.phrase = phrase
        self._cancel.clear()
        self.status = TrainingStatus(
            state="running", phrase=phrase, model_id=slugify(phrase), started=time.time(),
            stage=STAGES[0][0],
        )
        self._thread = threading.Thread(target=self._run, args=(config,), name="trainer", daemon=True)
        self._thread.start()
        self._on_update()

    def cancel(self) -> None:
        if self.running:
            self._cancel.set()

    # -- progress -------------------------------------------------------
    def _log(self, message: str) -> None:
        _LOGGER.info("[training] %s", message)
        self.status.log.append(message)
        self._push(force=True)

    def _stage(self, index: int, fraction: float = 0.0) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.status.stage = STAGES[index][0]
        self.status.progress = sum(s for _, s in STAGES[:index]) + STAGES[index][1] * min(1.0, fraction)
        self._push(force=fraction == 0.0)

    def _push(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self._last_push > 0.5:
            self._last_push = now
            self._on_update()

    # -- pipeline -------------------------------------------------------
    def _run(self, config: TrainingConfig) -> None:
        status = self.status
        try:
            status.result = self._train(config)
            status.state = "done"
            status.progress = 1.0
            status.stage = "Finished"
        except Cancelled:
            status.state = "cancelled"
            status.stage = "Cancelled"
            self._log("Training was cancelled")
        except Exception as err:  # report anything to the UI
            _LOGGER.exception("Training failed")
            status.state = "error"
            status.error = _friendly_error(err)
        status.finished = time.time()
        self._on_update()

    def _train(self, config: TrainingConfig) -> dict:
        from openwakeword.utils import AudioFeatures
        from piper import PiperVoice, SynthesisConfig

        rng = np.random.default_rng(config.seed or int(time.time()))
        phrase = config.phrase
        n_pos = QUALITY[config.quality]
        cpus = max(1, (os.cpu_count() or 2) - 1)

        # 1. assets ------------------------------------------------------
        self._stage(0)
        index = load_voice_index(self.assets)
        chosen = voices_for(index, config.language)
        if not chosen:
            raise RuntimeError(f"No voices are available for language {config.language}")
        files: list[tuple[str, Path, int]] = []
        for voice in chosen:
            for rel, meta in voice["files"].items():
                if rel.endswith((".onnx", ".onnx.json")):
                    files.append((VOICE_BASE_URL + rel, self.assets / "voices" / Path(rel).name, meta["size_bytes"]))
        features_path = self.assets / "negative_features.npy"
        files.append((NEGATIVE_FEATURES_URL, features_path, NEGATIVE_FEATURES_BYTES))
        total_bytes = sum(size for _, _, size in files)
        done_bytes = 0
        for url, target, size in files:
            if not (target.exists() and target.stat().st_size == size):
                self._log(f"Downloading {target.name} ({size / 1e6:.0f} MB)")
            base = done_bytes
            download(url, target, lambda f: self._stage(0, (base + f * size) / total_bytes), self._cancel, size)
            done_bytes += size

        voices = []
        for voice in chosen:
            onnx_name = next(Path(r).name for r in voice["files"] if r.endswith(".onnx"))
            voices.append((PiperVoice.load(str(self.assets / "voices" / onnx_name)), max(1, voice["num_speakers"])))
        n_speakers = sum(n for _, n in voices)
        self._log(f"Using {len(voices)} voice model(s) with {n_speakers} different speakers")

        def speak(text: str) -> np.ndarray:
            # voices with more speakers are used more often
            weights = np.array([np.sqrt(n) for _, n in voices])
            voice, speakers = voices[int(rng.choice(len(voices), p=weights / weights.sum()))]
            cfg = SynthesisConfig(
                speaker_id=int(rng.integers(speakers)) if speakers > 1 else None,
                length_scale=float(rng.uniform(0.8, 1.35)),
                noise_scale=float(rng.uniform(0.35, 0.9)),
                noise_w_scale=float(rng.uniform(0.5, 1.0)),
            )
            parts = [
                np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                for chunk in voice.synthesize(text, syn_config=cfg)
            ]
            if not parts:
                return np.zeros(0, dtype=np.float32)
            return trim_silence(to_16k(np.concatenate(parts), voice.config.sample_rate))

        # 2. speech ------------------------------------------------------
        self._stage(1)
        n_neg = int(n_pos * 1.5)
        variants = [phrase, phrase + ".", phrase + "!", phrase + "?", phrase.replace(" ", ", ", 1)]
        neg_texts = negative_texts(rng, phrase, n_neg)
        pos_speech: list[np.ndarray] = []
        neg_speech: list[np.ndarray] = []
        total = n_pos + n_neg
        for i in range(total):
            if i < n_pos:
                audio = speak(variants[i % len(variants)])
                if audio.size > 0.15 * SR:
                    pos_speech.append(audio)
            else:
                audio = speak(neg_texts[i - n_pos])
                if audio.size > 0.1 * SR:
                    neg_speech.append(audio)
            if i % 20 == 0:
                self._stage(1, i / total)
        if len(pos_speech) < n_pos * 0.5:
            raise RuntimeError("The speech voices could not pronounce this wake word")
        durations = np.array([a.size for a in pos_speech]) / SR
        self._log(f"Generated {len(pos_speech)} examples of “{phrase}” and {len(neg_speech)} of similar phrases")
        if np.median(durations) > 1.7:
            self._log("Note: this wake word is long; shorter phrases (under ~1.5 s) work better")
        del voices

        user = [read_wav_16k(p) for p in sorted(self.sample_dir(phrase).glob("*.wav"))]
        user = [u for u in user if u.size > 0.15 * SR]
        if user:
            self._log(f"Adding your {len(user)} own recording(s)")

        # 3. audio -> clips -> features -----------------------------------
        self._stage(2)
        order = rng.permutation(len(pos_speech))
        n_val = max(20, len(pos_speech) // 8)
        val_ids, train_ids = order[:n_val], order[n_val:]
        user_val = user[::5] if len(user) >= 10 else []
        user_train = [u for k, u in enumerate(user) if not (len(user) >= 10 and k % 5 == 0)]

        def speed(audio: np.ndarray) -> np.ndarray:
            factor = rng.uniform(0.9, 1.1)
            n = max(1, int(audio.size / factor))
            return np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(np.float32)

        # Every clip is embedded as a whole and cut into the sliding windows the
        # detector will see while streaming: windows that end just after the
        # complete phrase are positives; windows that only contain part of the
        # phrase, and every window of other speech, are negatives.
        pos_range = (32500, 44500)
        plan: list[tuple[str, Callable[[], tuple[np.ndarray, int, int]]]] = []
        for i in train_ids:
            for _ in range(2):  # two different augmentations of every synthetic positive
                plan.append(("pos", lambda i=i: make_clip(rng, pos_speech[i], neg_speech, pos_range)))
        if user_train:
            copies = int(np.clip(len(train_ids) * 2 * 0.5 / len(user_train), 20, 150))
            for u in user_train:
                for _ in range(copies):
                    plan.append(("pos", lambda u=u: make_clip(rng, speed(u), neg_speech, pos_range)))
        for i in val_ids:
            plan.append(("pos_val", lambda i=i: make_clip(rng, pos_speech[i], neg_speech, pos_range)))
        for u in user_val:
            for _ in range(5):
                plan.append(("user_val", lambda u=u: make_clip(rng, speed(u), neg_speech, pos_range)))
        n_neg_val = max(20, len(neg_speech) // 8)
        for k, audio in enumerate(neg_speech):
            plan.append(("neg_val" if k < n_neg_val else "neg", lambda a=audio: make_clip(rng, a, neg_speech)))
        for _ in range(max(30, n_pos // 20)):
            plan.append(("neg", lambda: make_clip(rng, np.zeros(0, dtype=np.float32), neg_speech)))

        extractor = AudioFeatures(inference_framework="onnx", ncpu=cpus)
        feats: dict[str, list[np.ndarray]] = {}
        keep_neg = 6 if n_pos <= 3000 else 4  # windows kept per negative clip (memory)
        batch = 64
        for a in range(0, len(plan), batch):
            chunk = plan[a:a + batch]
            made = [make() for _, make in chunk]
            embedded = extractor.embed_clips(np.stack([m[0] for m in made]), batch_size=batch, ncpu=cpus)
            for (label, _), (_, start, end), emb in zip(chunk, made, embedded):
                wins = np.lib.stride_tricks.sliding_window_view(emb, (N_FRAMES, EMB))[:, 0]
                if label in ("pos", "pos_val", "user_val"):
                    first = int(np.ceil((end - 480 - window_end(0)) / 1280))
                    hits = wins[first:first + 3]
                    if label == "pos":
                        feats.setdefault("pos", []).extend(hits)
                        partial = [j for j in range(first) if window_end(j) < start + 0.6 * (end - start)]
                        for j in rng.permutation(partial)[:2]:
                            feats.setdefault("neg", []).append(wins[j])
                    else:
                        feats.setdefault(label, []).append(hits)
                elif label == "neg":
                    feats.setdefault("neg", []).extend(wins[rng.permutation(N_WINDOWS)[:keep_neg]])
                else:
                    feats.setdefault(label, []).append(wins)
            self._stage(2, (a + batch) / len(plan))
        del extractor, plan
        X = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
        del feats
        pos, neg = X["pos"], X["neg"]
        self._log(f"Prepared {len(pos)} positive and {len(neg)} hard negative training examples")

        # 4. training ----------------------------------------------------
        self._stage(3)
        general = np.load(features_path)  # (frames, 96), ~185 MB
        split = int(len(general) * 0.85)
        feat_mean = np.tile(general[:split].mean(axis=0), N_FRAMES)
        feat_std = np.tile(general[:split].std(axis=0), N_FRAMES)
        net = Net(rng, feat_mean, feat_std)
        jitter = (0.1 * feat_std).reshape(1, N_FRAMES, EMB).astype(np.float32)
        mined = np.empty((0, N_FRAMES, EMB), dtype=np.float32)
        rounds = [(2500, 2e-3), (1500, 1e-3), (1500, 5e-4)]
        total_steps = sum(s for s, _ in rounds)
        step_done = 0
        B = 96  # per class group
        for r, (steps, lr) in enumerate(rounds):
            for step in range(steps):
                gen = gather(general, rng.integers(0, split - N_FRAMES, size=B))
                adv = neg[rng.integers(0, len(neg), size=B)]
                p = pos[rng.integers(0, len(pos), size=B)]
                parts = [p, gen, adv]
                if len(mined):
                    parts.append(mined[rng.integers(0, len(mined), size=B // 2)])
                x = np.concatenate(parts)
                x = x + rng.standard_normal(x.shape, dtype=np.float32) * jitter
                y = np.zeros(len(x), dtype=np.float32)
                y[:B] = 1.0
                w = np.full(len(x), 1.0 + 0.5 * r, dtype=np.float32)
                w[:B] = 1.0
                net.train_step(x, y, w, lr * (0.5 * (1 + np.cos(np.pi * step / steps)) * 0.9 + 0.1))
                step_done += 1
                if step % 50 == 0:
                    self._stage(3, step_done / total_steps)
            if r < len(rounds) - 1:
                # hard-negative mining: everything in the general data that still scores high
                scores = score_stream(net, general, 0, split)
                top = np.nonzero(scores > 0.2)[0]
                if len(top) > 20000:
                    top = top[np.argsort(scores[top])[-20000:]]
                if len(top):
                    mined = np.concatenate([mined, gather(general, np.sort(top))])[-40000:]
                self._log(f"Round {r + 1}: found {len(top)} confusing moments in background audio to learn from")

        # 5. evaluation --------------------------------------------------
        self._stage(4)
        held = score_stream(net, general, split, len(general))
        hours = len(held) * 0.08 / 3600

        def clip_scores(x: np.ndarray) -> np.ndarray:
            """Best score over the windows of each clip, as in streaming."""
            return net.predict(x.reshape(-1, N_FRAMES, EMB)).reshape(len(x), -1).max(axis=1)

        pos_scores = clip_scores(X["pos_val"])
        similar_scores = clip_scores(X["neg_val"])
        user_scores = clip_scores(X["user_val"]) if "user_val" in X else None
        table = []
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            table.append({
                "threshold": thr,
                "recall": float(np.mean(pos_scores >= thr)),
                "false_per_hour": count_events(held, thr) / hours,
                "similar_rejected": float(np.mean(similar_scores < thr)),
            })
        # lowest threshold with at most 0.5 false activations per hour; else the strictest useful one
        ok = [t for t in table if t["false_per_hour"] <= 0.5]
        best = ok[0] if ok else min(table, key=lambda t: (t["false_per_hour"], -t["recall"]))
        metrics = {
            "recommended_threshold": best["threshold"],
            "recall": best["recall"],
            "false_per_hour": best["false_per_hour"],
            "similar_rejected": best["similar_rejected"],
            "user_recall": None if user_scores is None else float(np.mean(user_scores >= best["threshold"])),
            "held_out_hours": round(hours, 2),
            "positives": int(len(pos)),
            "user_recordings": len(user),
            "table": table,
        }

        # 6. export ------------------------------------------------------
        model_id = slugify(phrase)
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        net.export_onnx(self.custom_dir / f"{model_id}.onnx")
        (self.custom_dir / f"{model_id}.tflite").unlink(missing_ok=True)  # replaced by the new model
        meta = {
            "phrase": phrase,
            "language": config.language,
            "quality": config.quality,
            "trained": time.time(),
            "metrics": metrics,
        }
        (self.custom_dir / f"{model_id}.json").write_text(json.dumps(meta, indent=2))
        self._log(
            f"Done: recognises {metrics['recall']:.0%} of test voices with about "
            f"{metrics['false_per_hour']:.1f} false activations per hour at threshold {best['threshold']}"
        )
        self._on_model(model_id, meta)
        return {"model_id": model_id, **meta}


def _friendly_error(err: Exception) -> str:
    text = str(err) or err.__class__.__name__
    name = err.__class__.__name__
    if "ConnectionError" in name or "Timeout" in name or "HTTPError" in name or "resolve" in text.lower():
        return ("Could not download the speech voices or training data. "
                "The add-on needs internet access (huggingface.co) for training. " + text[:200])
    if isinstance(err, MemoryError):
        return "Not enough memory to train on this system. Try the 'quick' quality."
    return text[:400]

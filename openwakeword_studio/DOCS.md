# openWakeWord Studio

## Getting started

1. Install and start the add-on, then open **Wake Words** in the sidebar
   (or *Open Web UI* on the add-on page).
2. Go to **Settings → Devices & services**. Home Assistant discovers the add-on
   as a **Wyoming Protocol** integration — press **Add**.
   If it isn't discovered, add *Wyoming Protocol* manually with the host name
   shown on the add-on's **Setup guide** tab and port `10400`.
3. Go to **Settings → Voice assistants**, edit your assistant and choose
   **openwakeword** as the wake word engine, then pick a wake word.

Do not run this add-on together with the official *openWakeWord* add-on on the
same port: both use `10400` by default. Change the port of one of them under
**Configuration → Network** if you want to keep both.

## The web UI

| Tab | What it is for |
| --- | --- |
| **Overview** | Devices that are streaming audio right now with their live scores, and the most recent detections. |
| **Wake words** | Enable/disable wake words, give a wake word its own sensitivity, upload or delete custom models. |
| **Test** | Speak into your browser's microphone or load a recording (WAV, MP3, …) and watch the score of every selected wake word against the threshold. |
| **Settings** | Default threshold, trigger level, pause after detection, voice activity filter and noise suppression. |
| **Setup guide** | The steps above, with your add-on's host name filled in. |

All settings made in the UI are stored in the add-on's data folder and are
included in add-on backups.

### Microphone testing

Browsers only allow microphone access on secure pages. The microphone test
therefore works when you open Home Assistant via `https://` (for example
through Home Assistant Cloud) — on a plain `http://` address use the audio file
test instead. Audio from the test is processed inside the add-on and is not
stored.

## Custom wake words

Upload `.tflite` or `.onnx` openWakeWord models on the **Wake words** tab. They
are saved to `/share/openwakeword` (configurable). You can also copy files into
that folder yourself and press **Rescan folder**.

The wake word's name is taken from the file name: `hey_computer_v2.tflite`
becomes "Hey Computer".

After you add, remove, enable or disable wake words, reload the Wyoming
integration (**Settings → Devices & services → Wyoming Protocol → ⋮ → Reload**)
so Home Assistant refreshes its list.

To train your own model see the
[openWakeWord documentation](https://github.com/dscripka/openWakeWord#training-new-models).

## Tuning

- **Threshold** (default 0.5): the score a wake word must reach. Lower = more
  sensitive, more false activations.
- **Trigger level** (default 1): number of 80 ms audio frames that must be above
  the threshold before a detection counts.
- **Pause after detection** (default 2 s): time before the same wake word can
  trigger again.
- **Voice activity filter** (default off): only accept detections while speech
  is present. Try 0.3–0.5 if music or TV noise causes false activations.
- **Noise suppression** (default off): Speex noise suppression for steady
  background noise.

A good routine: on the **Test** tab say the wake word a few times and note the
*peak* score, then talk normally for a while and note the peak again. Set the
threshold between the two.

## Add-on configuration

| Option | Default | Description |
| --- | --- | --- |
| `custom_model_dir` | `/share/openwakeword` | Folder for custom models (inside `/share`). |
| `debug_logging` | `false` | Verbose add-on log. |

## Supported systems

`amd64` and `aarch64` (64-bit Raspberry Pi 4/5, most mini PCs, Home Assistant
Green/Yellow). 32-bit systems are not supported because openWakeWord's
inference libraries are not available for them.

## Credits

Wake word engine and pre-trained models:
[openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
(code Apache-2.0; the pre-trained models are licensed CC BY-NC-SA 4.0).

# Architecture

Voice Studio is a single-user local web app. The browser talks to a FastAPI process, and that process calls one Python engine for speech and media.

```text
Browser  http://127.0.0.1:7860
  static/index.html    layout and styles
  static/app.js        UI state and fetch() calls
        │  /api/*
        ▼
  server.py            validation, uploads, project payload
        │
        ▼
  engine.py            Chatterbox TTS, project files, FFmpeg, captions
        │
        ├── PyTorch on MPS or CPU
        ├── data/projects/<id>/project.json + WAV files
        └── data/pronunciations.json
```

There is no database, no accounts, and no background job queue. One process-wide lock in `server.py` (`BUSY`) covers generation, regeneration, export, and video render. Some smaller edits, such as pause changes, do not take that lock. Avoid editing a project while a generation is running.

## Where to change things

| You want to… | Edit |
|---|---|
| Labels, layout, buttons, browser state | `voice_studio/static/index.html`, `voice_studio/static/app.js` |
| A route, request body, or what the browser receives | `voice_studio/server.py` |
| Chunking, synthesis, pauses, export, captions, video | `voice_studio/engine.py` |
| Where projects and the voice file are stored | `voice_studio/paths.py` |

`engine.py` still imports Gradio. The old Gradio screen is gone. Gradio remains because several engine functions raise `gr.Error` and the generator accepts a Gradio progress object. The web API catches those errors and returns HTTP 400. Replacing that with a normal exception is safe later; do it in one pass and re-test generation.

Presets are defined twice today: `PRESETS` in `engine.py` and a separate map in `static/app.js`. Change both until they share one source.

## HTTP API

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Serves the UI |
| `/api/start` | GET | Recent projects, last project, dictionary, memory |
| `/api/project?id=` | GET | Opens a project |
| `/api/generate` | POST | Creates a new project from a script |
| `/api/project/{segment}/regenerate` | POST | Regenerates one segment |
| `/api/project/{segment}/pause` | POST | Changes the pause after one segment |
| `/api/project/rename` | POST | Renames the project |
| `/api/project/snapshot` | POST | Saves a checkpoint |
| `/api/project/versions` | GET | Lists checkpoints |
| `/api/project/restore` | POST | Restores a checkpoint |
| `/api/project/export` | POST | Mastered WAV and MP3 |
| `/api/project/captions` | POST | SRT and VTT |
| `/api/project/waveform` | GET | Waveform PNG |
| `/api/project/trim` | POST | Trims one segment and rebuilds the mix |
| `/api/project/video` | POST | Renders a narrated MP4 |
| `/api/upload?kind=voice\|video` | POST | Stores an upload under `data/uploads` |
| `/api/media` | GET | Plays a file from the open project folder |
| `/api/settings` | GET | Dictionary and memory |
| `/api/settings/dictionary` | POST | Saves pronunciations |
| `/api/model/unload` | POST | Drops the loaded model |
| `/api/model/cleanup` | POST | Garbage-collects and clears the MPS cache |

Request fields are the Pydantic models at the top of `server.py`.

`/api/media` only serves a file that sits directly inside the requested project folder. Uploads are a separate directory and are passed back to later routes as local paths.

## Project files

Each project is a folder under `data/projects/`:

- `project.json` is the manifest the engine reads and writes.
- `segment_*.wav` and trimmed revisions are the takes.
- `voice_reference.wav` is the reference copied into the project.
- `combined_raw.wav` is the joined narration.
- `versions/<timestamp>/snapshot.json` plus copied WAVs are checkpoints.

The manifest stores **absolute paths**. `server.py` builds a smaller object for the browser (`id`, segment audio URLs, durations). Do not save that browser object back as `project.json`.

If you move `data/projects`, rewrite the absolute paths inside every `project.json` and `snapshot.json`, and update `data/preferences.json` (`last_project`). The app will not find audio if those strings still point at the old folder.

A manifest looks like this:

```json
{
  "folder": "/Users/you/voice-cloning/data/projects/PROJECT_ID",
  "name": "Product launch demo",
  "script": "Welcome. [pause:500] Here's the feature.",
  "reference": "/Users/you/voice-cloning/data/projects/PROJECT_ID/voice_reference.wav",
  "device": "MPS (Apple GPU)",
  "exaggeration": 0.5,
  "cfg": 0.5,
  "seed": 42,
  "raw": "/Users/you/voice-cloning/data/projects/PROJECT_ID/combined_raw.wav",
  "version": 1,
  "segments": [
    {
      "text": "Welcome.",
      "pause_ms": 500,
      "file": "/Users/you/voice-cloning/data/projects/PROJECT_ID/segment_001.wav"
    }
  ]
}
```

Saves are not transactional. A crash while a manifest or combined WAV is being written can leave a project half-updated. Copy `data/projects` before a risky change.

## Speech model

The engine loads Resemble AI's original English Chatterbox model:

```python
from chatterbox.tts import ChatterboxTTS
```

This is zero-shot cloning from a short reference. The app does not fine-tune a model. Keep the Perth watermarker path; do not remove it to work around an import error.

Useful controls, stored on the project:

| Field | Effect |
|---|---|
| `exaggeration` | More animated delivery. About 0.45–0.55 for demos. |
| `cfg` | How strongly the reference conditions the voice. Higher is not always closer. |
| `seed` | Comparison knob. The same seed is not a cross-machine guarantee. |
| `device` | `MPS (Apple GPU)` or `CPU`. |
| chunk size | Max characters per synthesized chunk, default about 300. |
| pause | Silence after automatic chunks. An explicit `[pause:500]` can add more silence on top. |

Audio is saved at the model's sample rate, commonly 24 kHz. A Voice Memos export can be converted with:

```bash
ffmpeg -i input.m4a -ar 24000 -ac 1 data/voices/my_voice.wav
```

The pronunciation file is case-insensitive text replacement, not a phoneme dictionary. Changing it does not regenerate audio you already have.

## Memory

Apple Silicon shares memory between CPU and GPU. PyTorch's MPS driver figure is not the same number as Activity Monitor's memory pressure, and the two should not be added together.

On this Mac, one generation reported a very large MPS driver allocation that dropped after **Clear unused memory**. That spike is not fully explained. Do not treat it as a measured leak, and do not set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` to hide it.

Practical use: leave the model loaded while you regenerate several segments, then unload it when you are done. `cleanup()` does not free the live model.

## Known limits

- Editing the master script does not update existing audio. Generate a new project, or regenerate one segment from Review.
- A segment can contain more than one sentence.
- Waveform trim is numeric, not a drag handle.
- Video cues are a list of start times. The render uses raw segment audio, not the mastered export.
- Captions follow the raw timeline. Changing export speed does not retime them.
- Generation cannot be cancelled mid-chunk.

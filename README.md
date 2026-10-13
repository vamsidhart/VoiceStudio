# Voice Studio

Local app for turning a script into narration in your own voice, then trimming segments, exporting audio and captions, and laying that narration over a screen recording.

It runs on your Mac. Generation uses the installed [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) model. Nothing is sent to a hosted speech API.

## Run

```bash
cd ~/voice-cloning
./scripts/run.sh
```

Open http://127.0.0.1:7860. Stop the server with Ctrl+C.

The script activates `.venv` and starts one local worker. Do not bind this app to anything other than `127.0.0.1`. There is no login.

## What you do in the app

1. Paste a script. `[pause:500]` inserts 500 ms of silence.
2. Generate a project. The app splits the script into chunks and writes one WAV per chunk.
3. In Review, edit or regenerate a single chunk. Other chunks stay as they are, and the previous take is snapshotted first.
4. Export mastered WAV and MP3, or SRT/VTT captions.
5. Optionally import a screen recording and set a start time for each chunk, then export an MP4.

The default voice file is `data/voices/my_voice.wav`. Pronunciations live in `data/pronunciations.json` (see `examples/pronunciations.example.json`).

## Layout

```text
voice_studio/          application
  server.py            HTTP API
  engine.py            speech, projects, audio, captions, video
  paths.py             where local data is stored
  static/              browser UI
data/                  your files only; not committed
  projects/            project folders and audio
  uploads/             uploaded voice references and videos
  voices/my_voice.wav  default reference recording
  pronunciations.json
  preferences.json
docs/                  how the app is built and how to change it
```

## Keep the working environment

`.venv` already has the Chatterbox, PyTorch, and Perth packages this Mac was tested with. Install only the web dependencies if they are missing:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Do not upgrade `torch`, `chatterbox-tts`, or `resemble-perth` as a routine update. On this machine, Chatterbox's watermarker also depended on `setuptools==81.0.0`. Capture `pip freeze` before you change packages.

FFmpeg and ffprobe must be on `PATH` (`brew install ffmpeg`). Word-level captions optionally need `faster-whisper`.

## Before you put this on GitHub

`data/` and `.venv/` are gitignored on purpose. They contain your voice, scripts, and generated audio. The repository should contain the application, docs, and the example dictionary only.

Details for future changes: [docs/architecture.md](docs/architecture.md) and [docs/maintaining.md](docs/maintaining.md).

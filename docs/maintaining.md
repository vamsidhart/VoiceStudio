# Maintaining Voice Studio

Use this when you change the app after a real session, not as a feature wishlist.

## Safe change loop

1. Copy `data/projects` if the change touches manifests, segment files, or restore.
2. Change the smallest layer that owns the behavior. See the table in [architecture.md](architecture.md).
3. Run the checks below that match the change.
4. Leave `torch`, `chatterbox-tts`, and `resemble-perth` alone unless the task is specifically a dependency fix. If you do touch them, save `pip freeze` first.

The app is local and single-user. A broken generation should leave the previous WAV and snapshot in place. If a change can overwrite audio, confirm a snapshot is written before the replacement.

## Checks

Import and project load, without generating speech:

```bash
source .venv/bin/activate
python -c "from voice_studio.server import safe_state, engine; ps=engine.list_projects(); print(len(ps), 'projects'); print(safe_state(ps[0][1])['name'] if ps else 'none')"
```

After a UI or API change, start `./scripts/run.sh` and walk the path you touched:

- Home reopens the last project.
- A short script with `[pause:500]` generates two segments and both WAVs play.
- Regenerating one segment leaves the other WAV unchanged and adds a snapshot.
- Export produces a WAV and an MP3.
- Caption export writes SRT and VTT.
- Unload and cleanup return the app to a usable state.

Use a real MPS generation only when you changed synthesis, devices, or memory behavior. Mock or skip the model for wording, layout, and route-shape edits.

## Dependency notes from this machine

- Python 3.11, Apple Silicon, FFmpeg on `PATH`.
- `perth.PerthImplicitWatermarker` once failed with `TypeError: 'NoneType' object is not callable` even with `resemble-perth==1.0.1`. Installing `setuptools==81.0.0` in `.venv` fixed it here. Do not assume that pin is required on every machine, and do not remove it from this environment without a generation test.
- First model download uses Hugging Face. After the weights are cached, generation can run offline. An unauthenticated Hub warning is a rate limit, not a quality setting.

## What not to commit

- `data/` (voice recordings, scripts, project audio, uploads, last-project preference)
- `.venv/`
- Model caches outside this repo

`examples/pronunciations.example.json` is the committed sample. Your real dictionary stays in `data/pronunciations.json`.

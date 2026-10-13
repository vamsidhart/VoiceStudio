"""Speech, project, and media engine for the local Voice Studio app."""
import gc
import csv
import math
import copy
import datetime
import hashlib
import tempfile
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import gradio as gr
import numpy as np
import torch
import torchaudio
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from chatterbox.tts import ChatterboxTTS

from .paths import DICTIONARY, DEFAULT_VOICE, PROJECTS, ensure_data_dirs

ensure_data_dirs()
V3_PROJECTS = PROJECTS
LOCK = threading.RLock()
MODEL = None
MODEL_DEVICE = None
PRESETS = {
    'Product demo': (0.45, 0.55, 1.0, 300),
    'Founder update': (0.55, 0.50, 1.0, 350),
    'Launch announcement': (0.70, 0.40, 1.03, 400),
    'Educational explainer': (0.40, 0.60, 0.95, 300),
    'Custom': (0.50, 0.50, 1.0, 300),
}
_original_load = torch.load

def _mac_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device(MODEL_DEVICE or 'cpu')
    return _original_load(*args, **kwargs)
torch.load = _mac_load


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except Exception: pass


def memory_status():
    try:
        import psutil
        s = f'App RAM: {psutil.Process().memory_info().rss / 1024**3:.2f} GB'
    except ImportError: s = 'App RAM: install psutil for usage'
    if torch.backends.mps.is_available():
        try:
            s += f' | MPS tensors: {torch.mps.current_allocated_memory()/1024**3:.2f} GB'
            s += f' | MPS driver: {torch.mps.driver_allocated_memory()/1024**3:.2f} GB'
        except Exception: pass
    return s + f' | Model: {MODEL_DEVICE or "unloaded"}'


def load_model(choice):
    global MODEL, MODEL_DEVICE
    with LOCK:
        dev = 'mps' if choice == 'MPS (Apple GPU)' else 'cpu'
        if dev == 'mps' and not torch.backends.mps.is_available(): raise gr.Error('MPS unavailable; select CPU.')
        if MODEL is not None and MODEL_DEVICE == dev: return 'Model already loaded. ' + memory_status()
        MODEL = None
        MODEL_DEVICE = None
        cleanup()
        MODEL_DEVICE = dev
        try: MODEL = ChatterboxTTS.from_pretrained(device=dev)
        except Exception:
            MODEL = None
            MODEL_DEVICE = None
            cleanup()
            raise
        return 'Model loaded. ' + memory_status()


def unload_model():
    global MODEL, MODEL_DEVICE
    with LOCK:
        MODEL = None
        MODEL_DEVICE = None
        cleanup()
        return 'Model unloaded. ' + memory_status()


def clear_cache():
    with LOCK:
        cleanup()
        return 'Cache cleared. ' + memory_status()


def read_dictionary():
    if not DICTIONARY.exists(): return {}
    try:
        data = json.loads(DICTIONARY.read_text(encoding='utf-8'))
        return {str(k): str(v) for k,v in data.items() if str(k).strip() and str(v).strip()}
    except (OSError, ValueError, TypeError): return {}


def save_dictionary(text):
    try:
        data = json.loads(text or '{}')
        if not isinstance(data, dict): raise ValueError('Expected a JSON object')
        cleaned = {str(k).strip():str(v).strip() for k,v in data.items() if str(k).strip() and str(v).strip()}
        DICTIONARY.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding='utf-8')
        return f'Saved {len(cleaned)} pronunciation replacement(s) to {DICTIONARY}'
    except Exception as e: raise gr.Error(f'Invalid dictionary JSON: {e}')


def apply_dictionary(text, dictionary):
    for original in sorted(dictionary, key=len, reverse=True):
        pattern = re.compile(r'(?<!\w)' + re.escape(original) + r'(?!\w)', re.IGNORECASE)
        text = pattern.sub(lambda _m: dictionary[original], text)
    return text


def split_text(text, max_chars):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    pieces = []
    for sentence in sentences:
        words = sentence.split()
        part = ''
        for word in words:
            if len(word) > max_chars: raise gr.Error(f'One word exceeds {max_chars} characters.')
            candidate = (part + ' ' + word).strip()
            if len(candidate) > max_chars and part:
                pieces.append(part)
                part = word
            else: part = candidate
        if part: pieces.append(part)
    chunks, current = [], ''
    for piece in pieces:
        candidate = (current + ' ' + piece).strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = piece
        else: current = candidate
    if current: chunks.append(current)
    return chunks


def parse_script(script, max_chars, default_pause, dictionary):
    """Turn explicit [pause:500] tokens into silence after preceding speech."""
    tokens = re.split(r'(\[pause:\s*\d+\s*\])', script, flags=re.IGNORECASE)
    segments = []
    pending_leading_pause = 0
    for token in tokens:
        match = re.fullmatch(r'\[pause:\s*(\d+)\s*\]', token.strip(), re.I)
        if match:
            duration = int(match.group(1))
            if duration > 10000: raise gr.Error('Pause markers are limited to 10,000 ms.')
            if segments: segments[-1]['pause_ms'] += duration
            else: pending_leading_pause += duration
            continue
        chunks = split_text(token, int(max_chars)) if token.strip() else []
        for chunk in chunks:
            segments.append({'text': apply_dictionary(chunk, dictionary), 'pause_ms': int(default_pause), 'file': None})
    if not segments: raise gr.Error('Enter text to narrate.')
    segments[-1]['pause_ms'] = 0 if segments[-1]['pause_ms'] == int(default_pause) else segments[-1]['pause_ms'] - int(default_pause)
    # Leading silence is stored as a separate metadata value.
    return segments, pending_leading_pause


def new_project():
    folder = V3_PROJECTS / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    return folder


def reference_path(uploaded):
    ref = uploaded or (str(DEFAULT_VOICE) if DEFAULT_VOICE.exists() else None)
    if not ref or not Path(ref).is_file(): raise gr.Error('Upload a reference recording or add data/voices/my_voice.wav.')
    return str(ref)


def synthesize(text, ref, seed, exaggeration, cfg, dest):
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available(): torch.mps.manual_seed(int(seed))
    with torch.inference_mode():
        audio = MODEL.generate(text, audio_prompt_path=ref,
                               exaggeration=float(exaggeration), cfg_weight=float(cfg))
    torchaudio.save(str(dest), audio.detach().cpu(), MODEL.sr)
    del audio


def append_wav(outpath, segments, lead_pause=0):
    """Stream PCM samples to disk; avoid concatenating long audio in RAM."""
    import soundfile as sf
    with sf.SoundFile(segments[0]['file']) as first:
        sr, channels = first.samplerate, first.channels
    with sf.SoundFile(str(outpath), mode='w', samplerate=sr, channels=channels, subtype='PCM_16') as dst:
        if lead_pause: dst.write(np.zeros((round(sr*lead_pause/1000), channels), dtype=np.float32))
        for seg in segments:
            with sf.SoundFile(seg['file']) as source:
                if source.samplerate != sr or source.channels != channels:
                    raise gr.Error('Inconsistent segment sample rate or channel count.')
                while True:
                    frames = source.read(65536, dtype='float32', always_2d=True)
                    if len(frames) == 0: break
                    dst.write(frames)
            if seg['pause_ms']:
                remaining = round(sr*seg['pause_ms']/1000)
                while remaining:
                    n = min(remaining, 65536)
                    dst.write(np.zeros((n, channels), dtype=np.float32))
                    remaining -= n


def save_manifest(state):
    if not state: return
    folder = Path(state['folder'])
    (folder / 'project.json').write_text(json.dumps(state, indent=2), encoding='utf-8')


def segment_choices(state):
    return [(f'{i+1:02d} — {s["text"][:72]}', str(i)) for i,s in enumerate(state['segments'])]


def generate(script, uploaded, device, exaggeration, cfg, seed, max_chars, default_pause,
             clean, progress=gr.Progress()):
    if not script.strip(): raise gr.Error('Enter a narration script.')
    with LOCK:
        ref = reference_path(uploaded)
        segments, leading = parse_script(script, int(max_chars), int(default_pause), read_dictionary())
        folder = new_project()
        stored_ref = folder / ('voice_reference' + Path(ref).suffix.lower())
        shutil.copy2(ref, stored_ref)
        ref = str(stored_ref)
        progress(0, desc='Loading model')
        load_model(device)
        for i, seg in enumerate(segments):
            progress(i/len(segments), desc=f'Generating {i+1}/{len(segments)}')
            path = folder / f'segment_{i+1:03d}.wav'
            synthesize(seg['text'], ref, int(seed)+i, exaggeration, cfg, path)
            seg['file'] = str(path)
            if clean: cleanup()
        combined = folder / 'combined_raw.wav'
        append_wav(combined, segments, leading)
        state = {'folder':str(folder), 'segments':segments, 'leading_pause_ms':leading,
                 'reference':ref, 'device':device, 'exaggeration':float(exaggeration),
                 'cfg':float(cfg), 'seed':int(seed), 'raw':str(combined), 'script':script, 'version':1, 'created':time.time()}
        save_manifest(state)
        choices = segment_choices(state)
        progress(1, desc='Finished')
        return (state, str(combined), str(combined),
                gr.update(choices=choices, value='0'), segments[0]['text'],
                f'Generated {len(segments)} segments. {memory_status()}\nProject: {folder}')


def choose_segment(index, state):
    if not state or index is None: return '', None
    seg = state['segments'][int(index)]
    return seg['text'], seg['file']


def regenerate(index, revised_text, state, device, exaggeration, cfg, seed, clean):
    if not state or index is None: raise gr.Error('Generate a script first.')
    if not revised_text.strip(): raise gr.Error('Segment text cannot be empty.')
    with LOCK:
        i = int(index)
        load_model(device)
        seg = state['segments'][i]
        new_file = Path(state['folder']) / f'segment_{i+1:03d}_revision_{uuid.uuid4().hex[:6]}.wav'
        # Apply current dictionary to the edited text, without changing other segments.
        spoken = apply_dictionary(revised_text.strip(), read_dictionary())
        synthesize(spoken, state['reference'], int(seed)+i, exaggeration, cfg, new_file)
        snapshot_version(state, 'before_segment_revision')
        seg['text'] = spoken
        seg['file'] = str(new_file)
        combined = Path(state['folder']) / 'combined_raw.wav'
        append_wav(combined, state['segments'], state['leading_pause_ms'])
        state['version'] = int(state.get('version',1)) + 1
        save_manifest(state)
        if clean: cleanup()
        return (state, str(combined), str(combined), str(new_file),
                gr.update(choices=segment_choices(state), value=str(i)),
                f'Regenerated segment {i+1}. {memory_status()}')


def export_audio(state, speed, normalize, compress, mp3_bitrate):
    if not state: raise gr.Error('Generate audio before exporting.')
    if not shutil.which('ffmpeg'): raise gr.Error('FFmpeg not found. Run: brew install ffmpeg')
    with LOCK:
        folder = Path(state['folder'])
        raw = folder / 'combined_raw.wav'
        # Always reconstruct from saved individual segments, preserving precise pauses.
        append_wav(raw, state['segments'], state['leading_pause_ms'])
        filters = []
        if float(speed) != 1.0: filters.append(f'atempo={float(speed):.3f}')
        if compress: filters.append('acompressor=threshold=-20dB:ratio=2:attack=10:release=120')
        if normalize: filters.append('loudnorm=I=-16:TP=-1.5:LRA=11')
        wav = folder / 'final_mastered.wav'
        mp3 = folder / 'final_mastered.mp3'
        base = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(raw)]
        if filters: base += ['-af', ','.join(filters)]
        try:
            subprocess.run(base + ['-c:a','pcm_s16le',str(wav)], check=True, capture_output=True, text=True)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(wav),
                            '-c:a','libmp3lame','-b:a',str(mp3_bitrate),str(mp3)],
                           check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise gr.Error('FFmpeg export failed: ' + (e.stderr or str(e))[-800:])
        return str(wav), str(wav), str(mp3), f'Exported mastered WAV + MP3. {memory_status()}'



# -------------------- v3 production tools --------------------

def safe_project(folder):
    root = V3_PROJECTS.resolve()
    path = Path(folder).resolve()
    if path == root or root not in path.parents:
        raise gr.Error('Project path is outside the v3 projects folder.')
    return path


def snapshot_version(state, reason='manual'):
    if not state: raise gr.Error('Load or generate a project first.')
    folder = safe_project(state['folder'])
    versions = folder / 'versions'
    versions.mkdir(exist_ok=True)
    version_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest = versions / version_id
    dest.mkdir()
    for i,seg in enumerate(state['segments']):
        src = Path(seg['file'])
        if src.is_file(): shutil.copy2(src, dest / f'segment_{i+1:03d}.wav')
    raw = Path(state.get('raw',''))
    if raw.is_file(): shutil.copy2(raw, dest / 'combined_raw.wav')
    snapshot = copy.deepcopy(state)
    for i,seg in enumerate(snapshot['segments']):
        seg['file'] = str(dest / f'segment_{i+1:03d}.wav')
    snapshot['raw'] = str(dest / 'combined_raw.wav')
    snapshot['snapshot_reason'] = reason
    (dest / 'snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    return f'Version saved: {version_id}'


def list_projects():
    result=[]
    for folder in sorted(V3_PROJECTS.iterdir(),reverse=True):
        if folder.is_dir() and (folder/'project.json').is_file():
            result.append((folder.name, str(folder)))
    return result


def list_versions(state):
    if not state: return gr.update(choices=[],value=None)
    folder=safe_project(state['folder'])
    vdir=folder/'versions'
    options=[(p.name,str(p/'snapshot.json')) for p in sorted(vdir.glob('*/snapshot.json'),reverse=True)] if vdir.exists() else []
    return gr.update(choices=options,value=None)


def load_project(project):
    if not project: raise gr.Error('Select a project.')
    folder=safe_project(project)
    manifest=folder/'project.json'
    if not manifest.exists(): raise gr.Error('Project manifest missing.')
    state=json.loads(manifest.read_text(encoding='utf-8'))
    validate_project_state(state)
    return project_ui(state, 'Project loaded')


def validate_project_state(state):
    folder=safe_project(state['folder'])
    for seg in state['segments']:
        f=Path(seg['file']).resolve()
        if not f.is_file() or (folder not in f.parents):
            raise gr.Error('A project audio segment is missing or outside its project folder.')


def project_ui(state, message):
    validate_project_state(state)
    raw=Path(state.get('raw',''))
    if not raw.exists():
        raw=safe_project(state['folder'])/'combined_raw.wav'
        append_wav(raw,state['segments'],state.get('leading_pause_ms',0))
        state['raw']=str(raw)
    choices=segment_choices(state)
    return (state,state.get('script',''),str(raw),str(raw),
            gr.update(choices=choices,value='0'),state['segments'][0]['text'],
            state['segments'][0]['file'],f'{message}. {memory_status()}')


def restore_version(snapshot_path):
    if not snapshot_path: raise gr.Error('Select a saved version.')
    snapshot_file=Path(snapshot_path).resolve()
    # Only allow snapshots within the v3 project tree.
    safe_project(snapshot_file.parent)
    if snapshot_file.name!='snapshot.json': raise gr.Error('Invalid version selection.')
    state=json.loads(snapshot_file.read_text(encoding='utf-8'))
    validate_project_state(state)
    folder=safe_project(state['folder'])
    current_manifest=folder/'project.json'
    if current_manifest.exists():
        current=json.loads(current_manifest.read_text(encoding='utf-8'))
        snapshot_version(current,'before_restore')
    for i,seg in enumerate(state['segments']):
        new_path=folder/f'restored_{i+1:03d}_{uuid.uuid4().hex[:6]}.wav'
        shutil.copy2(seg['file'],new_path)
        seg['file']=str(new_path)
    state['raw']=str(folder/'combined_raw.wav')
    append_wav(state['raw'],state['segments'],state.get('leading_pause_ms',0))
    state['version']=int(state.get('version',1))+1
    save_manifest(state)
    return project_ui(state,'Version restored')


def save_current_version(state):
    return snapshot_version(state)


def timeline_entries(state):
    if not state: return []
    lead=int(state.get('leading_pause_ms',0))/1000
    entries=[]
    t=lead
    for i,seg in enumerate(state['segments']):
        with sf.SoundFile(seg['file']) as audio:
            duration=len(audio)/audio.samplerate
        entries.append({'i':i,'start':t,'end':t+duration,'duration':duration,'text':seg['text'], 'pause_ms':seg.get('pause_ms',0)})
        t+=duration+seg.get('pause_ms',0)/1000
    return entries


def duration_report(state,target):
    if not state: return 'Generate or load audio first.'
    entries=timeline_entries(state)
    actual=entries[-1]['end'] if entries else 0
    target=float(target or 0)
    if target<=0: return f'Current duration: {actual:.1f} seconds. Set a target to get a speed recommendation.'
    ratio=actual/target
    if 0.75<=ratio<=1.25:
        advice=f'Suggested export speed: {ratio:.2f}× (set the playback speed control).'
    else:
        advice='The required speed is outside the safe 0.75–1.25× range. Shorten or lengthen your script instead.'
    return f'Actual: {actual:.1f}s | Target: {target:.1f}s | Difference: {actual-target:+.1f}s. {advice}'


def waveform_image(state, width=1100, height=260):
    if not state: return None
    entries=timeline_entries(state)
    if not entries: return None
    total=max(entries[-1]['end'],0.01)
    canvas=Image.new('RGB',(width,height),'#111827')
    d=ImageDraw.Draw(canvas)
    for j,ent in enumerate(entries):
        x0=int(width*ent['start']/total)
        x1=max(x0+1,int(width*ent['end']/total))
        if j%2==0: d.rectangle((x0,0,x1,height),fill='#182a40')
        d.line((x0,0,x0,height),fill='#38bdf8',width=2)
        d.text((x0+5,6),f'{j+1}',fill='#f8fafc')
        with sf.SoundFile(state['segments'][j]['file']) as f:
            # streaming waveform envelope, bounded memory even for long recordings
            bins=max(1,x1-x0)
            step=max(1,math.ceil(len(f)/bins))
            for k in range(bins):
                data=f.read(step,dtype='float32',always_2d=True)
                if not len(data): break
                amp=float(np.max(np.abs(data)))
                amp=min(1.0,amp)
                x=x0+k
                y=int(amp*(height/2-22))
                d.line((x,height//2-y,x,height//2+y),fill='#2dd4bf')
    d.line((0,height//2,width,height//2),fill='#475569')
    return canvas


def waveform_click(state,evt:gr.SelectData):
    if not state: raise gr.Error('Generate or load a project first.')
    try: x=int(evt.index[0])
    except (TypeError,IndexError,ValueError): raise gr.Error('Click inside the waveform.')
    entries=timeline_entries(state)
    total=max(entries[-1]['end'],0.01)
    clicked=x/1100*total
    idx=min(range(len(entries)),key=lambda i: 0 if entries[i]['start']<=clicked<=entries[i]['end'] else min(abs(entries[i]['start']-clicked),abs(entries[i]['end']-clicked)))
    seg=state['segments'][idx]
    return gr.update(value=str(idx)),seg['text'],seg['file'],f'Selected segment {idx+1} at {clicked:.2f}s.'


def trim_selected(state,index,trim_start,trim_end,pause_after):
    if not state or index is None: raise gr.Error('Choose a segment.')
    with LOCK:
        i=int(index)
        seg=state['segments'][i]
        with sf.SoundFile(seg['file']) as src:
            sr=src.samplerate
            n=len(src)
            begin=int(float(trim_start)*sr/1000)
            end=int(float(trim_end)*sr/1000)
            if begin+end>=n-100: raise gr.Error('Trim amount exceeds the segment duration.')
        snapshot_version(state,'before_trim')
        dest=safe_project(state['folder'])/f'trimmed_{i+1:03d}_{uuid.uuid4().hex[:6]}.wav'
        with sf.SoundFile(seg['file']) as src, sf.SoundFile(str(dest),'w',samplerate=src.samplerate,channels=src.channels,subtype='PCM_16') as out:
            src.seek(begin)
            remaining=n-begin-end
            while remaining>0:
                frames=src.read(min(65536,remaining),dtype='float32',always_2d=True)
                if not len(frames): break
                out.write(frames)
                remaining-=len(frames)
        seg['file']=str(dest)
        seg['pause_ms']=int(pause_after)
        state['version']=int(state.get('version',1))+1
        append_wav(state['raw'],state['segments'],state.get('leading_pause_ms',0))
        save_manifest(state)
        return state,str(state['raw']),str(state['raw']),str(dest),waveform_image(state),duration_report(state,0),f'Trimmed segment {i+1}; original retained in version history.'


def seconds_to_stamp(sec, sep=','):
    ms=round(max(0,sec)*1000)
    h,ms=divmod(ms,3600000)
    m,ms=divmod(ms,60000)
    s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}{sep}{ms:03}'


def export_captions(state, mode):
    if not state: raise gr.Error('Generate or load audio first.')
    entries=timeline_entries(state)
    folder=safe_project(state['folder'])
    rows=[]
    if mode=='Whisper word alignment (optional)':
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise gr.Error('Install optional dependency: python -m pip install faster-whisper')
        # CPU int8 avoids large concurrent MPS allocations with Chatterbox.
        whisper=WhisperModel('base.en',device='cpu',compute_type='int8')
        for ent in entries:
            seg_audio=state['segments'][ent['i']]['file']
            transcription,_=whisper.transcribe(seg_audio,language='en',word_timestamps=True,beam_size=1)
            for segment in transcription:
                if segment.words:
                    for word in segment.words:
                        rows.append((ent['start']+word.start,ent['start']+word.end,word.word.strip()))
                else:
                    rows.append((ent['start']+segment.start,ent['start']+segment.end,segment.text.strip()))
        del whisper
        cleanup()
    else:
        rows=[(ent['start'],ent['end'],ent['text']) for ent in entries]
    if not rows: raise gr.Error('No captions generated.')
    srt=folder/'captions.srt'
    vtt=folder/'captions.vtt'
    srt.write_text('\n\n'.join(f'{i}\n{seconds_to_stamp(a)} --> {seconds_to_stamp(b)}\n{t}' for i,(a,b,t) in enumerate(rows,1))+'\n',encoding='utf-8')
    vtt.write_text('WEBVTT\n\n'+'\n\n'.join(f'{seconds_to_stamp(a,".")} --> {seconds_to_stamp(b,".")}\n{t}' for a,b,t in rows)+'\n',encoding='utf-8')
    note='Whisper timestamps are estimates; review proper nouns.' if mode.startswith('Whisper') else 'Captions follow generated segment timing (not word-level).'
    return str(srt),str(vtt),f'Exported {len(rows)} captions. {note}'


def probe_video(video):
    if not video: raise gr.Error('Upload a demo video first.')
    if not shutil.which('ffprobe'): raise gr.Error('ffprobe missing; install FFmpeg.')
    cmd=['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video)]
    try:
        data=json.loads(subprocess.run(cmd,check=True,capture_output=True,text=True).stdout)
        duration=float(data['format']['duration'])
    except (KeyError,ValueError,subprocess.CalledProcessError) as e:
        raise gr.Error(f'Could not read video duration: {e}')
    return duration


def default_cues(state,video):
    if not state: raise gr.Error('Generate or load narration first.')
    duration=probe_video(video)
    entries=timeline_entries(state)
    # Template only: user should place cues to match the on-screen actions.
    lines=['segment,start_seconds']
    lines.extend(f'{e["i"]+1},{e["start"]:.2f}' for e in entries)
    return '\n'.join(lines),f'Video duration: {duration:.2f}s. Adjust each cue start to match your screen recording.'


def render_demo(state,video,cues_text,keep_original):
    if not state or not video: raise gr.Error('Generate narration and upload a demo video first.')
    if not shutil.which('ffmpeg'): raise gr.Error('FFmpeg not found.')
    duration=probe_video(video)
    lines=list(csv.DictReader((cues_text or '').splitlines()))
    if len(lines)!=len(state['segments']):
        raise gr.Error('Provide exactly one cue row per segment.')
    cues={}
    for row in lines:
        try:
            i=int(row['segment'])-1
            start=float(row['start_seconds'])
        except (ValueError,KeyError,TypeError): raise gr.Error('Cue CSV must have segment,start_seconds columns with valid numbers.')
        if i<0 or i>=len(state['segments']) or i in cues or not math.isfinite(start) or start<0 or start>duration:
            raise gr.Error('Cue numbers must be unique and times within video duration.')
        cues[i]=start
    if len(cues)!=len(state['segments']): raise gr.Error('Cue list is incomplete.')
    folder=safe_project(state['folder'])
    output=folder/f'demo_narrated_{uuid.uuid4().hex[:6]}.mp4'
    # Input 0: original video. Remaining inputs: segment WAVs. Use bounded
    # ffmpeg filter graph instead of loading all audio into Python memory.
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video)]
    for seg in state['segments']: cmd+=['-i',str(seg['file'])]
    filters=[]
    for i in range(len(state['segments'])):
        delay=round(cues[i]*1000)
        filters.append(f'[{i+1}:a]aresample=48000,adelay={delay}:all=1[a{i}]')
    labels=''.join(f'[a{i}]' for i in range(len(state['segments'])))
    filters.append(f'{labels}amix=inputs={len(state["segments"])}:duration=longest:normalize=0[narr]')
    if keep_original:
        # Optional original soundtrack; requires an audio stream in the video.
        filters.append('[0:a]volume=0.25[orig]')
        filters.append('[narr][orig]amix=inputs=2:duration=longest:normalize=0[final]')
    else:
        filters.append('[narr]anull[final]')
    cmd+=['-filter_complex',';'.join(filters),'-map','0:v:0','-map','[final]',
          '-c:v','copy','-c:a','aac','-b:a','192k','-t',f'{duration:.3f}',
          '-movflags','+faststart',str(output)]
    try:
        subprocess.run(cmd,check=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as e:
        raise gr.Error('Video render failed. If preserving original audio, ensure the video has an audio track. '+(e.stderr or '')[-900:])
    return str(output),str(output),f'Exported demo video: {output.name} ({duration:.1f}s).'


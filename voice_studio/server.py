"""Local-only Voice Studio API."""
import json, os, shutil, threading, time, uuid, copy
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import engine
from .paths import PREFERENCES, UPLOADS, ensure_data_dirs

ensure_data_dirs()
META=PREFERENCES
APP=FastAPI(title='Voice Studio',docs_url=None,redoc_url=None)
APP.mount('/ui',StaticFiles(directory=Path(__file__).parent/'static',html=True),name='ui')
BUSY=threading.Lock()
TASKS={}

def fail(e):
    raise HTTPException(400,str(e))
def safe_state(pid):
    folder=engine.safe_project(pid)
    manifest=folder/'project.json'
    if not manifest.is_file(): raise HTTPException(404,'Project not found')
    state=json.loads(manifest.read_text())
    engine.validate_project_state(state)
    return state

def save_state(s):
    engine.save_manifest(s)
    META.write_text(json.dumps({'last_project':s['folder']}))

def payload(s):
    entries=engine.timeline_entries(s)
    return {'id':s['folder'],'name':s.get('name') or Path(s['folder']).name,'script':s.get('script',''),
            'segments':[{'index':i,'text':seg['text'],'pause_ms':seg.get('pause_ms',0),
                         'start':round(entries[i]['start'],2),'duration':round(entries[i]['duration'],2),
                         'audio':f'/api/media?project={s["folder"]}&file={Path(seg["file"]).name}'} for i,seg in enumerate(s['segments'])],
            'audio':f'/api/media?project={s["folder"]}&file={Path(s["raw"]).name}',
            'version':s.get('version',1),'duration':round(entries[-1]['end']+s['segments'][-1].get('pause_ms',0)/1000,2) if entries else 0,
            'device':s.get('device','MPS (Apple GPU)'), 'cfg':s.get('cfg',0.5),'expression':s.get('exaggeration',0.5),
            'seed':s.get('seed',42),'reference':Path(s.get('reference','')).name}

class Generate(BaseModel):
    script:str
    reference:str|None=None
    device:str='MPS (Apple GPU)'
    expression:float=0.5
    cfg:float=0.5
    seed:int=42
    chunk:int=300
    pause:int=250
    cleanup:bool=False
    name:str='Untitled voiceover'
class Edit(BaseModel):
    text:str
    expression:float=0.5
    cfg:float=0.5
    seed:int=42
    device:str='MPS (Apple GPU)'
class Pause(BaseModel):
    pause_ms:int
class Export(BaseModel):
    speed:float=1.0
    normalize:bool=True
    compress:bool=False
    bitrate:str='192k'
class Timing(BaseModel):
    target:float
class VideoRender(BaseModel):
    video:str
    cues:list[float]
    keep_original:bool=False

@APP.get('/')
def index(): return FileResponse(Path(__file__).parent/'static'/'index.html')
@APP.get('/api/start')
def start():
    projects=engine.list_projects()
    recent=None
    if META.exists():
        try: recent=json.loads(META.read_text()).get('last_project')
        except Exception: pass
    paths=[v for _,v in projects]
    if recent not in paths: recent=paths[0] if paths else None
    return {'projects':[{'name':n,'id':v} for n,v in projects], 'project':payload(safe_state(recent)) if recent else None,
            'dictionary':engine.read_dictionary(),'memory':engine.memory_status(), 'default_voice':str(engine.DEFAULT_VOICE) if engine.DEFAULT_VOICE.exists() else None}
@APP.get('/api/project')
def project(id:str):
    s=safe_state(id);save_state(s);return payload(s)
@APP.post('/api/generate')
def generate(req:Generate):
    if not BUSY.acquire(blocking=False): raise HTTPException(409,'Another generation or export is running')
    try:
        s=engine.generate(req.script,req.reference,req.device,req.expression,req.cfg,req.seed,req.chunk,req.pause,req.cleanup)[0]
        s['name']=req.name[:100]
        save_state(s)
        return payload(s)
    except Exception as e: fail(e)
    finally: BUSY.release()
@APP.post('/api/project/{segment}/regenerate')
def regen(segment:int,req:Edit,project:str):
    if not BUSY.acquire(blocking=False): raise HTTPException(409,'Another operation is running')
    try:
        s=safe_state(project)
        if segment<0 or segment>=len(s['segments']): raise ValueError('Invalid segment')
        engine.regenerate(str(segment),req.text,s,req.device,req.expression,req.cfg,req.seed,False)
        save_state(s);return payload(s)
    except Exception as e: fail(e)
    finally: BUSY.release()
@APP.post('/api/project/{segment}/pause')
def pause(segment:int,req:Pause,project:str):
    try:
        s=safe_state(project)
        if not 0<=segment<len(s['segments']) or not 0<=req.pause_ms<=10000: raise ValueError('Invalid pause')
        engine.snapshot_version(s,'before_pause_edit')
        s['segments'][segment]['pause_ms']=req.pause_ms
        engine.append_wav(s['raw'],s['segments'],s.get('leading_pause_ms',0))
        s['version']=s.get('version',1)+1
        save_state(s);return payload(s)
    except Exception as e: fail(e)
@APP.post('/api/project/rename')
def rename(project:str,name:str):
    s=safe_state(project);s['name']=name[:100];save_state(s);return payload(s)
@APP.post('/api/project/snapshot')
def snapshot(project:str):
    try: return {'message':engine.snapshot_version(safe_state(project))}
    except Exception as e: fail(e)
@APP.get('/api/project/versions')
def versions(project:str):
    s=safe_state(project);p=Path(s['folder'])/'versions'
    return [{'name':x.parent.name,'id':str(x)} for x in sorted(p.glob('*/snapshot.json'),reverse=True)] if p.exists() else []
@APP.post('/api/project/restore')
def restore(project:str,snapshot:str):
    try:
        s=safe_state(project);p=Path(snapshot).resolve()
        if p.name!='snapshot.json' or Path(s['folder']).resolve() not in p.parents: raise ValueError('Invalid snapshot')
        restored=engine.restore_version(str(p))[0]
        save_state(restored);return payload(restored)
    except Exception as e: fail(e)
@APP.post('/api/project/export')
def export(project:str,req:Export):
    if not BUSY.acquire(blocking=False): raise HTTPException(409,'Another operation is running')
    try:
        if not .75<=req.speed<=1.25: raise ValueError('Speed must be 0.75–1.25')
        s=safe_state(project);wav,_,mp3,_=engine.export_audio(s,req.speed,req.normalize,req.compress,req.bitrate)
        return {'wav':f'/api/media?project={project}&file={Path(wav).name}','mp3':f'/api/media?project={project}&file={Path(mp3).name}', 'caption_warning':req.speed!=1}
    except Exception as e: fail(e)
    finally: BUSY.release()
@APP.post('/api/project/captions')
def captions(project:str,mode:str='segments'):
    try:
        s=safe_state(project)
        a,b,note=engine.export_captions(s,'Whisper word alignment (optional)' if mode=='words' else 'Segment timestamps')
        return {'srt':f'/api/media?project={project}&file={Path(a).name}','vtt':f'/api/media?project={project}&file={Path(b).name}','message':note}
    except Exception as e: fail(e)
@APP.get('/api/project/waveform')
def waveform(project:str):
    try:
        s=safe_state(project);image=engine.waveform_image(s)
        import io
        from fastapi.responses import Response
        buf=io.BytesIO();image.save(buf,format='PNG')
        return Response(buf.getvalue(),media_type='image/png',headers={'Cache-Control':'no-store'})
    except Exception as e: fail(e)
@APP.post('/api/project/trim')
def trim(project:str,segment:int,start_ms:int=0,end_ms:int=0,pause_ms:int=250):
    try:
        s=safe_state(project);engine.trim_selected(s,str(segment),start_ms,end_ms,pause_ms)
        save_state(s);return payload(s)
    except Exception as e: fail(e)
@APP.get('/api/media')
def media(project:str,file:str):
    s=safe_state(project);folder=Path(s['folder']).resolve();p=(folder/file).resolve()
    if p.parent!=folder or not p.is_file(): raise HTTPException(404,'File not found')
    return FileResponse(p,filename=p.name)
@APP.post('/api/upload')
async def upload(file:UploadFile=File(...),kind:str='voice'):
    if kind not in ('voice','video'): raise HTTPException(400,'Invalid upload kind')
    ext=Path(file.filename or '').suffix.lower()
    if ext not in (('.wav','.m4a','.mp3','.flac') if kind=='voice' else ('.mp4','.mov','.m4v')): raise HTTPException(400,'Unsupported file type')
    folder=UPLOADS
    dest=folder/(uuid.uuid4().hex+ext)
    with dest.open('wb') as out:
        while data:=await file.read(1024*1024): out.write(data)
    return {'path':str(dest),'name':file.filename}
@APP.post('/api/project/video')
def video(project:str,req:VideoRender):
    if not BUSY.acquire(blocking=False): raise HTTPException(409,'Another operation is running')
    try:
        s=safe_state(project)
        if len(req.cues)!=len(s['segments']): raise ValueError('Each segment needs a start time')
        csv='segment,start_seconds\n'+'\n'.join(f'{i+1},{v}' for i,v in enumerate(req.cues))
        output,_,_=engine.render_demo(s,req.video,csv,req.keep_original)
        return {'video':f'/api/media?project={project}&file={Path(output).name}'}
    except Exception as e: fail(e)
    finally: BUSY.release()
@APP.get('/api/settings')
def settings(): return {'dictionary':engine.read_dictionary(),'memory':engine.memory_status()}
@APP.post('/api/settings/dictionary')
def dictionary(entries:dict[str,str]):
    try:
        engine.save_dictionary(json.dumps(entries));return {'ok':True}
    except Exception as e: fail(e)
@APP.post('/api/model/unload')
def unload(): return {'message':engine.unload_model()}
@APP.post('/api/model/cleanup')
def cleanup(): return {'message':engine.clear_cache()}

# app.py — EN->ES como Google Cloud (si hay credenciales) o deep_translator (fallback)
# - Carga automática de .env (Azure Speech y Google Cloud)
# - SIN campos de Azure en la barra lateral
# - Azure TTS: SSML con <sub alias="..."> SOLO en URLs/emails (no dice "punto" al final)
# - Piper TTS: alternativa local (verbaliza SOLO URL/EMAIL/DOMINIO+RUTA)
# - Sincroniza audio↔vídeo con ffmpeg

import os, re, io, glob, shutil, subprocess, pathlib, platform
from pathlib import Path
import numpy as np
import streamlit as st
from pydub import AudioSegment
from urllib.parse import urlparse
from deep_translator import GoogleTranslator

import tempfile, json

# Carga .env si existe (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Lee secrets sin romper si no existen (local)
try:
    _secrets = st.secrets
except Exception:
    _secrets = {}

# Azure Speech → ENV
if not os.getenv("AZURE_SPEECH_KEY") and _secrets.get("AZURE_SPEECH_KEY"):
    os.environ["AZURE_SPEECH_KEY"] = str(_secrets["AZURE_SPEECH_KEY"])
if not os.getenv("AZURE_SPEECH_REGION") and _secrets.get("AZURE_SPEECH_REGION"):
    os.environ["AZURE_SPEECH_REGION"] = str(_secrets["AZURE_SPEECH_REGION"])

# Google Cloud Translate → crea JSON temporal y apunta la ruta
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and _secrets.get("gcp_service_account"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(dict(_secrets["gcp_service_account"])).encode("utf-8"))
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
# --- fin bootstrap ---

# Mapear Secrets 
if "REMOVED_AZURE_KEY" in st.secrets:
    os.environ["REMOVED_AZURE_KEY"] = st.secrets["REMOVED_AZURE_KEY"]
if "REMOVED_AZURE_REGION" in st.secrets:
    os.environ["REMOVED_AZURE_REGION"] = st.secrets["REMOVED_AZURE_REGION"]

# Google Cloud: 
if "gcp_service_account" in st.secrets:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(st.secrets["gcp_service_account"]).encode("utf-8"))
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ff) + os.pathsep + os.environ.get("PATH","")
except Exception:
    pass

# === Cargar .env automáticamente (mismo dir que este archivo; fallback a find_dotenv) ===
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
    if not Path(".env").exists():
        load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

import yt_dlp
import whisper

# ===== Azure TTS (opcional) =====
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_OK = True
except Exception:
    AZURE_OK = False

# ---------- Utils ----------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _ffmpeg_ok(): return shutil.which("ffmpeg") is not None
def _to_float(s: str) -> float:
    s=(s or "").strip().replace(",", ".")
    try: return float(s)
    except: return 0.0
def _ffprobe_text(args):
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        return out.decode(errors="ignore")
    except Exception:
        return ""
def _probe_duration(path: str) -> float:
    if not os.path.exists(path) or os.path.getsize(path)==0: return 0.0
    out=_ffprobe_text(['ffprobe','-v','error','-show_entries','format=duration',
                       '-of','default:nokey=1:noprint_wrappers=1', path])
    v=_to_float(out)
    if v>0: return v
    out=_ffprobe_text(['ffprobe','-v','error','-select_streams','v:0',
                       '-show_entries','stream=duration',
                       '-of','default:nokey=1:noprint_wrappers=1', path])
    v=_to_float(out)
    if v>0: return v
    try:
        seg=AudioSegment.from_file(path)
        return len(seg)/1000.0
    except Exception:
        return 0.0
def ensure_video_ok(video_path:str)->str:
    if _probe_duration(video_path)>0.1: return video_path
    remux=os.path.splitext(video_path)[0]+"_genpts.mp4"
    subprocess.run(['ffmpeg','-y','-fflags','+genpts','-i',video_path,'-c','copy',
                    '-movflags','+faststart', remux],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return remux if _probe_duration(remux)>0.1 else video_path

# ---------- Descarga / entrada ----------
def _first_url(s:str):
    m=re.search(r'https?://\S+', s or '')
    return m.group(0) if m else None
def _looks_local(s:str)->bool:
    if not s: return False
    s=s.strip().strip('"').strip("'")
    return pathlib.Path(s).exists()
def download_video(url:str)->str:
    if not _ffmpeg_ok(): raise RuntimeError("ffmpeg no encontrado.")
    formats=[
        "best[ext=mp4][vcodec*=avc1][acodec*=mp4a]/best[ext=mp4]",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "bv*+ba/b","22/18"
    ]
    ydl_opts={
        "outtmpl":"%(id)s.%(ext)s","merge_output_format":"mp4","noplaylist":True,"quiet":True,
        "retries":25,"fragment_retries":25,"concurrent_fragment_downloads":5,
        "nocheckcertificate":True,"geo_bypass":True,"http_headers":{"User-Agent":UA},
        "postprocessor_args":{"FFmpegVideoRemuxer":["-movflags","faststart"]},
        "postprocessors":[{"key":"FFmpegVideoRemuxer","preferedformat":"mp4"}],
    }
    last=None
    for f in formats:
        try:
            opts=dict(ydl_opts); opts["format"]=f
            with yt_dlp.YoutubeDL(opts) as ydl:
                info=ydl.extract_info(url, download=True)
                fn=ydl.prepare_filename(info)
                if not fn.endswith(".mp4"): fn=os.path.splitext(fn)[0]+".mp4"
                if not os.path.exists(fn):
                    files=glob.glob("*.mp4")
                    fn=max(files,key=os.path.getctime)
                return fn
        except Exception as e:
            last=e
    try:
        from pytube import YouTube
        yt=YouTube(url)
        stream=(yt.streams.filter(progressive=True,file_extension="mp4")
                .order_by("resolution").desc().first()
                or yt.streams.filter(progressive=True,file_extension="mp4",res="360p").first())
        return stream.download(filename=f"{yt.video_id}.mp4")
    except Exception as e2:
        raise RuntimeError(f"Fallo descarga: {last} / {e2}")
def resolve_source(user_in:str)->str:
    s=user_in.strip().strip('"').strip("'")
    if s.lower().startswith(('http://','https://')) or _first_url(s):
        url=s if s.lower().startswith(('http://','https://')) else _first_url(s)
        path=download_video(url)
        return ensure_video_ok(path)
    if _looks_local(s):
        p=pathlib.Path(s)
        if p.suffix.lower()==".mp4": return ensure_video_ok(str(p.resolve()))
        out=p.with_suffix(".mp4")
        subprocess.run(["ffmpeg","-y","-i",str(p),"-c:v","copy","-c:a","aac","-b:a","192k",
                        "-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out.resolve()))
    raise RuntimeError("Proporciona URL de YouTube o ruta local válida.")

# ---------- Audio / ASR ----------
def extract_audio(video:str, out="audio.wav")->str:
    subprocess.run(["ffmpeg","-y","-i",video,"-ac","1","-ar","16000","-vn",
                    "-acodec","pcm_s16le",out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def transcribe_segments(audio:str, model_size='small'):
    model=whisper.load_model(model_size, device='cpu')
    res=model.transcribe(
        audio, language='en', task='transcribe',
        temperature=0.0, best_of=5, beam_size=5,
        condition_on_previous_text=False, fp16=False,
        no_speech_threshold=0.2, logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
    full = res.get('text','')
    return res['segments'], full

# ---------- Detección de direcciones ----------
URL_RE   = re.compile(r'(?i)\bhttps?://[^\s]+')
EMAIL_RE = re.compile(r'(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b')
DOMAIN_WITH_PATH_RE = re.compile(
    r'(?ix)(?<!://)\b(?![\w\.-]+@)'
    r'((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})'
    r'(/[^\s\)\]\}\.,;:!?]+)'
)

# ---------- Traducción (Google Cloud > deep_translator) ----------
def translate_google_cloud(texts):
    from google.cloud import translate_v2 as translate
    client=translate.Client()
    if isinstance(texts,str): texts=[texts]
    outs=[]
    for t in texts:
        r=client.translate(t, source_language='en', target_language='es', format_='text')
        outs.append(r['translatedText'])
    return outs

def translate_fallback(texts):
    gt=GoogleTranslator(source="en", target="es")
    if isinstance(texts,str): texts=[texts]
    return [gt.translate(t) if (t or "").strip() else "" for t in texts]

def translate_en2es_exact_google(texts):
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            return translate_google_cloud(texts)
        except Exception:
            pass
    return translate_fallback(texts)

def build_spanish_text(full_en: str) -> str:
    return translate_en2es_exact_google(full_en)[0]

# ---------- “Forma hablada” para URLs/emails ----------
def pronounce_host_es(host:str)->str:
    host=host.strip().strip('.,;:!?)]}')
    spoken=[]
    if host.lower().startswith('www.'):
        spoken.append('triple doble u punto'); host=host[4:]
    labels=[l for l in host.split('.') if l]
    labs=[]
    for lab in labels:
        lab=lab.replace('-', ' guion ').replace('_',' guion bajo ')
        lab=re.sub(r'\s{2,}',' ',lab).strip()
        labs.append(lab)
    if labs: spoken.append(' punto '.join(labs))
    return ' '.join(spoken).strip()
def pronounce_url_es(s:str)->str:
    raw=s.strip().strip('.,;:!?)]}')
    if not re.match(r'(?i)^https?://', raw):
        raw='http://'+raw
    u=urlparse(raw)
    spoken=pronounce_host_es(u.netloc or '')
    if u.path and u.path!='/':
        for seg in [seg for seg in u.path.split('/') if seg]:
            seg=(seg.replace('-', ' guion ').replace('_',' guion bajo ').replace('%20',' espacio '))
            seg=re.sub(r'\s{2,}',' ',seg).strip()
            if seg: spoken+=' barra '+seg
    return re.sub(r'\s{2,}',' ',spoken).strip()
def pronounce_email_es(email:str)->str:
    e=email.strip().strip('.,;:!?)]}')
    if '@' not in e: return pronounce_url_es(e)
    local,dom=e.split('@',1)
    local=(local.replace('.',' punto ').replace('-',' guion ').replace('_',' guion bajo '))
    local=re.sub(r'\s{2,}',' ',local).strip()
    return f"{local} arroba {pronounce_host_es(dom)}"

# ---------- SSML seguro (Azure) ----------
PAUSE_COMMA_MS = 200
PAUSE_SEMI_MS  = 260
PAUSE_COLON_MS = 220
PAUSE_SENT_MS  = 360

def _ssml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))

def _collect_address_spans(text: str):
    spans = []
    for m in URL_RE.finditer(text):
        spans.append((m.start(), m.end(), "url", text[m.start():m.end()]))
    for m in EMAIL_RE.finditer(text):
        spans.append((m.start(), m.end(), "email", text[m.start():m.end()]))
    for m in DOMAIN_WITH_PATH_RE.finditer(text):
        spans.append((m.start(), m.end(), "domain_path", text[m.start():m.end()]))
    spans.sort(key=lambda x: x[0])
    compact=[]
    last_e=-1
    for s,e,t,v in spans:
        if s>=last_e:
            compact.append((s,e,t,v)); last_e=e
    return compact

def _mark_breaks_with_placeholders(s: str) -> str:
    s = re.sub(r'\s*\n+\s*', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    s = re.sub(r',\s*', ',__BRK_COMMA__', s)
    s = re.sub(r';\s*', ';__BRK_SEMI__', s)
    s = re.sub(r':\s*', ':__BRK_COLON__', s)
    s = re.sub(r'([\.!?])\s*', r'\1__BRK_SENT__ ', s)
    return s
def _restore_break_placeholders(escaped_text: str) -> str:
    escaped_text = escaped_text.replace('__BRK_COMMA__', f'<break time="{PAUSE_COMMA_MS}ms"/> ')
    escaped_text = escaped_text.replace('__BRK_SEMI__',  f'<break time="{PAUSE_SEMI_MS}ms"/> ')
    escaped_text = escaped_text.replace('__BRK_COLON__', f'<break time="{PAUSE_COLON_MS}ms"/> ')
    escaped_text = escaped_text.replace('__BRK_SENT__',  f'<break time="{PAUSE_SENT_MS}ms"/> ')
    return escaped_text
def _ssml_plain_with_breaks(s: str) -> str:
    if not s: return ""
    marked = _mark_breaks_with_placeholders(s)
    escaped = _ssml_escape(marked)
    return _restore_break_placeholders(escaped)

def _build_ssml_with_address_subs(full_text_es: str) -> str:
    if not full_text_es:
        return "<s></s>"
    text = full_text_es
    spans = _collect_address_spans(text)
    if not spans:
        return f"<s>{_ssml_plain_with_breaks(text)}</s>"
    out = []; last=0
    for (s,e,typ,frag) in spans:
        if s>last: out.append(_ssml_plain_with_breaks(text[last:s]))
        alias = pronounce_email_es(frag) if typ=="email" else pronounce_url_es(frag)
        out.append(f'<sub alias="{_ssml_escape(alias)}">{_ssml_escape(frag)}</sub>')
        last=e
    if last<len(text): out.append(_ssml_plain_with_breaks(text[last:]))
    return "<s>"+"".join(out)+"</s>"

# ---------- Piper (texto plano a partir del SSML anterior) ----------
BREAKER = re.compile(r'<break[^>]*time="(\d+)ms"[^>]*/>')
def ssml_to_punct_for_local(ssml: str) -> str:
    if not ssml: return ""
    def repl(m):
        t = int(m.group(1))
        if t >= 330: return '. '
        if t >= 250: return '; '
        if t >= 180: return ', '
        return ' '
    txt = BREAKER.sub(repl, ssml)
    txt = re.sub(r'</?s>', '', txt)
    txt = re.sub(r'\s{2,}', ' ', txt)
    return txt.strip()

def speakify_addresses_text_piper(text: str) -> str:
    def repl_url(m):   return pronounce_url_es(m.group(0))
    def repl_mail(m):  return pronounce_email_es(m.group(0))
    def repl_path(m):  return pronounce_url_es(m.group(0))
    text = URL_RE.sub(repl_url, text)
    text = EMAIL_RE.sub(repl_mail, text)
    text = DOMAIN_WITH_PATH_RE.sub(repl_path, text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def prepare_text_for_piper(es_text: str) -> str:
    spoken = speakify_addresses_text_piper(es_text)
    ssml = _build_ssml_with_address_subs(spoken)
    return ssml_to_punct_for_local(ssml)

# ---------- TTS ----------
IS_WIN = platform.system() == "Windows"
PIPER_BIN_DEFAULT = "./piper/piper.exe" if IS_WIN else "./piper/piper"
PIPER_BIN   = os.getenv("PIPER_BIN",   PIPER_BIN_DEFAULT)
PIPER_VOICE = os.getenv("PIPER_VOICE", "./voices/es_ES-mls_10246-low.onnx")
PIPER_RATE  = os.getenv("PIPER_RATE",  "22050")

def has_piper() -> bool:
    return (shutil.which(PIPER_BIN) is not None) or os.path.exists(PIPER_BIN)

def tts_piper(text: str, outfile="tts_es.wav") -> str:
    if not has_piper() or not os.path.exists(PIPER_VOICE):
        raise RuntimeError("Piper no disponible. Configura PIPER_BIN y PIPER_VOICE.")
    prepared = prepare_text_for_piper(text)
    cmd = [PIPER_BIN, "-m", PIPER_VOICE, "-q", "50", "-s", str(PIPER_RATE), "-f", outfile]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.communicate(input=prepared.encode("utf-8"))
    if proc.returncode != 0 or not os.path.exists(outfile):
        raise RuntimeError("Piper falló generando TTS.")
    return outfile

def get_azure_creds():
    key=os.getenv("REMOVED_AZURE_KEY")
    region=os.getenv("REMOVED_AZURE_REGION")
    return key, region

def tts_azure(text_es: str, voice="es-ES-DarioNeural", outfile="tts_es.wav", rate_pct=-6):
    if not AZURE_OK: raise RuntimeError("azure-cognitiveservices-speech no instalado.")
    key, region = get_azure_creds()
    if not key or not region: raise RuntimeError("Faltan REMOVED_AZURE_KEY / REMOVED_AZURE_REGION")
    body = _build_ssml_with_address_subs(text_es)
    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-ES">
      <voice name="{voice}">
        <mstts:express-as style="newscast-casual">
          <prosody rate="{rate_pct:+d}%">{body}</prosody>
        </mstts:express-as>
      </voice>
    </speak>"""
    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    res = synth.speak_ssml_async(ssml).get()
    if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS falló: {res.reason}")
    raw = bytes(res.audio_data)
    AudioSegment.from_file(io.BytesIO(raw), format="wav").export(outfile, format="wav")
    return outfile

def synthesize_es(text_es: str, voice="es-ES-DarioNeural", outfile="tts_es.wav") -> str:
    key, region = get_azure_creds()
    if AZURE_OK and key and region:
        return tts_azure(text_es, voice=voice, outfile=outfile, rate_pct=-6)
    else:
        return tts_piper(text_es, outfile=outfile)

# ---------- Sincronización ----------
def _atempo_chain(factor:float):
    if factor<=0: factor=1.0
    chain=[]; f=factor
    while f<0.5 or f>2.0:
        step=0.5 if f<0.5 else 2.0
        chain.append(f"atempo={step}"); f=f/step
    chain.append(f"atempo={f}")
    return ",".join(chain)
def fit_audio_to_video(video:str, audio_in:str, audio_out:str)->str:
    v=_probe_duration(video); a=_probe_duration(audio_in)
    if v<=0: video=ensure_video_ok(video); v=_probe_duration(video)
    if v<=0: raise RuntimeError("No se pudo medir duración del vídeo.")
    if a<=0: raise RuntimeError("Audio TTS vacío.")
    if abs(a-v)<0.01:
        shutil.copyfile(audio_in, audio_out); return audio_out
    if a>v:
        factor=a/v
        subprocess.run(['ffmpeg','-y','-i',audio_in,'-filter:a',_atempo_chain(factor),audio_out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        pad=(v-a)+0.05
        tmp=audio_out+".tmp.wav"
        subprocess.run(['ffmpeg','-y','-i',audio_in,'-af',f'apad=pad_dur={pad:.3f}','-t',f'{v:.3f}', tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(tmp, audio_out)
    subprocess.run(['ffmpeg','-y','-i',audio_out,'-t',f'{v:.3f}', audio_out+".fix.wav"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    shutil.move(audio_out+".fix.wav", audio_out)
    return audio_out
def mux_video_audio(video:str, audio:str, out="video_doblado.mp4")->str:
    subprocess.run(['ffmpeg','-y','-i',video,'-i',audio,'-map','0:v:0','-map','1:a:0',
                    '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart', out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- UI ----------
st.title("🎬 Dobador de videos EN→ES")

has_azure = bool(os.getenv("AZURE_SPEECH_KEY"))
azure_region = os.getenv("AZURE_SPEECH_REGION") or "—"

engine = "Google Cloud" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "deep_translator (fallback)"
st.caption(f"Motor de traducción activo: {engine}  |  Azure KEY: {'✔️' if os.getenv('REMOVED_AZURE_KEY') else '❌'}  |  Región: {os.getenv('REMOVED_AZURE_REGION') or '—'}")

source = st.text_input("🔗 URL de YouTube o 📁 ruta local al vídeo")
accion = st.radio("Acción", ["Obtener el texto en inglés","Obtener la traducción a español","Hacer el doblaje del video"], index=2)

colA,colB = st.columns(2)
with colA:
    model = st.selectbox("Modelo Whisper", ["small","base"], index=0)
with colB:
    voice = st.selectbox("Voz TTS (Azure)", ["es-ES-DarioNeural","es-ES-AlvaroNeural","es-ES-TeoNeural",
                                             "es-ES-ArnauNeural","es-ES-ElviraNeural","es-ES-LiaNeural"], index=0)

if st.button("Procesar"):
    if not source:
        st.error("Introduce una URL o ruta.")
    else:
        with st.spinner("Cargando vídeo y transcribiendo..."):
            video = resolve_source(source)
            audio_wav = extract_audio(video)
            segments, full_en = transcribe_segments(audio_wav, model_size=model)

        if accion == "Obtener el texto en inglés":
            st.success("✅ Transcripción (EN) lista.")
            st.download_button("⬇️ Descargar EN (.txt)", (full_en or "").encode("utf-8"),
                               file_name="transcripcion_en.txt", mime="text/plain")

        elif accion == "Obtener la traducción a español":
            with st.spinner("Traduciendo…"):
                full_es = build_spanish_text(full_en)
            st.success("✅ Traducción (ES) lista.")
            c1,c2=st.columns(2)
            with c1:
                st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                                   file_name="transcripcion_en.txt", mime="text/plain")
            with c2:
                st.download_button("⬇️ ES (.txt)", (full_es or "").encode("utf-8"),
                                   file_name="traduccion_es.txt", mime="text/plain")

            if st.button("Doblaje ahora"):
                try:
                    with st.spinner("Sintetizando y ajustando al tiempo del vídeo..."):
                        wav = synthesize_es(full_es, voice=voice, outfile="tts_es.wav")
                        fit = fit_audio_to_video(video, wav, "tts_fit.wav")
                        out = mux_video_audio(video, fit, "video_doblado.mp4")
                    st.success("✅ Doblaje listo")
                    st.video(out)
                    st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                       file_name="video_doblado.mp4")
                except Exception as e:
                    st.error(str(e))

        else:  # Hacer el doblaje del video
            with st.spinner("Traduciendo…"):
                full_es = build_spanish_text(full_en)
            try:
                with st.spinner("Sintetizando y ajustando al tiempo del vídeo..."):
                    wav = synthesize_es(full_es, voice=voice, outfile="tts_es.wav")
                    fit = fit_audio_to_video(video, wav, "tts_fit.wav")
                    out = mux_video_audio(video, fit, "video_doblado.mp4")
                st.success("✅ Doblaje listo")
                st.video(out)
                st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                   file_name="video_doblado.mp4")
            except Exception as e:
                st.error(str(e))

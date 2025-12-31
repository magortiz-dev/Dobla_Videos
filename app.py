# app.py
# EN -> ES (España) con detección automática de credenciales:
# - Lee AZURE_SPEECH_KEY / AZURE_SPEECH_REGION y gcp_service_account desde st.secrets (o .env)
# - Google Cloud Translate si hay credenciales; si no, deep_translator fallback
# - Azure TTS (es-ES) y ajuste exacto de audio al vídeo
# - ffmpeg portable vía imageio-ffmpeg

import os, re, io, glob, shutil, subprocess, pathlib, json, tempfile
import numpy as np
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile
from urllib.parse import urlparse

# ========= 0) Bootstrap de secrets / entorno =========
# (Carga .env si existe y mapea st.secrets -> variables de entorno ANTES de usar nada)
# --- Secrets & ENV bootstrap (colocar al principio de app.py) ---
from collections.abc import Mapping

# Carga .env si existe (opcional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Acceso tolerante a st.secrets (en local puede no existir)
try:
    _secrets = st.secrets
except Exception:
    _secrets = {}

def _to_plain(obj):
    """Convierte AttrDict/Mapping/list/tuple recursivamente a tipos JSON-serializables."""
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    return obj

# Azure Speech -> ENV
if not os.getenv("AZURE_SPEECH_KEY") and _secrets.get("AZURE_SPEECH_KEY"):
    os.environ["AZURE_SPEECH_KEY"] = str(_secrets["AZURE_SPEECH_KEY"])
if not os.getenv("AZURE_SPEECH_REGION") and _secrets.get("AZURE_SPEECH_REGION"):
    os.environ["AZURE_SPEECH_REGION"] = str(_secrets["AZURE_SPEECH_REGION"])

# Google Cloud Translate -> crea JSON temporal desde TOML (sección gcp_service_account)
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and _secrets.get("gcp_service_account"):
    svc_plain = _to_plain(_secrets["gcp_service_account"])
    # Normaliza la private_key por si la pegaron con '\n' en lugar de saltos reales
    pk = svc_plain.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "BEGIN PRIVATE KEY" in pk:
        svc_plain["private_key"] = pk.replace("\\n", "\n")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(svc_plain).encode("utf-8"))
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
# --- fin bootstrap ---

# ========= 0.1) ffmpeg portable (forzar binario y registrar en pydub) =========
FFMPEG_BIN = None
FFPROBE_BIN = None

def _setup_ffmpeg():
    global FFMPEG_BIN, FFPROBE_BIN
    try:
        import imageio_ffmpeg, os
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()  # ruta absoluta al ffmpeg embebido
        os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN
    except Exception:
        pass

    # Registrar en pydub
    try:
        from pydub.utils import which
        from pydub import AudioSegment as _AS
        if not FFMPEG_BIN:
            FFMPEG_BIN = which("ffmpeg")
        if FFMPEG_BIN:
            _AS.converter = FFMPEG_BIN
        # ffprobe (opcional): intenta en PATH o junto a ffmpeg
        FFPROBE_BIN = which("ffprobe")
        if not FFPROBE_BIN and FFMPEG_BIN:
            guess = os.path.join(os.path.dirname(FFMPEG_BIN), "ffprobe")
            if os.path.exists(guess):
                FFPROBE_BIN = guess
        if FFPROBE_BIN:
            _AS.ffprobe = FFPROBE_BIN
    except Exception:
        pass

_setup_ffmpeg()

# ========= 1) Resto de imports que dependen del entorno =========
import yt_dlp
import whisper
from deep_translator import GoogleTranslator

# Azure Speech SDK (opcional)
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_OK = True
except Exception:
    AZURE_OK = False

# ---------- utilidades de sistema / ffprobe ----------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _ffmpeg_ok():
    return bool(FFMPEG_BIN) or shutil.which("ffmpeg") is not None

def _to_float(s: str) -> float:
    s=(s or "").strip().replace(",", ".")
    try: return float(s)
    except: return 0.0

def _ffprobe_text(args):
    # Si no hay ffprobe, devolvemos vacío para que _probe_duration use el fallback con pydub
    if not FFPROBE_BIN:
        return ""
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
def ensure_video_ok(video_path: str) -> str:
    """
    Garantiza que el contenedor final sea MP4 con timestamps correctos.
    Si ya es válido, lo devuelve tal cual.
    """
    try:
        dur = _probe_duration(video_path)
        if dur > 0.1 and video_path.lower().endswith(".mp4"):
            return video_path
    except Exception:
        pass

    remux = os.path.splitext(video_path)[0] + "_genpts.mp4"
    # Copia vídeo sin recodificar + audio AAC (si no lo estaba) y faststart
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", video_path,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", remux],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    # Si por cualquier motivo falla, devolvemos el original
    return remux if _probe_duration(remux) > 0.1 else video_path

# ---------- descarga / entrada ----------
def _first_url(s:str):
    m=re.search(r'https?://\S+', s or '')
    return m.group(0) if m else None
def _looks_local(s:str)->bool:
    if not s: return False
    s=s.strip().strip('"').strip("'")
    return pathlib.Path(s).exists()

# --- Descarga robusta: inspecciona formatos y elige el que exista ---
def download_video(url: str) -> str:
    if not _ffmpeg_ok():
        raise RuntimeError("ffmpeg no encontrado.")

    # Opciones base para yt-dlp
    ydl_base = {
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": UA},
        "allow_multiple_video_streams": False,
        "allow_multiple_audio_streams": False,
        # usar el ffmpeg portable
        **({"ffmpeg_location": os.path.dirname(FFMPEG_BIN)} if FFMPEG_BIN else {}),
        # convertir SIEMPRE a mp4 (aunque baje webm/hls)
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "postprocessor_args": {
            "FFmpegVideoConvertor": ["-movflags", "faststart"]
        },
    }

    # 1) Inspecciona formatos sin descargar
    with yt_dlp.YoutubeDL({**ydl_base, "format": "best"}) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("No se pudo obtener la info del vídeo.")

    fmts = info.get("formats") or []

    # Helpers de selección
    def is_progressive(f):
        return f.get("vcodec") != "none" and f.get("acodec") != "none"
    def is_video_only(f):
        return f.get("vcodec") != "none" and f.get("acodec") == "none"
    def is_audio_only(f):
        return f.get("acodec") != "none" and f.get("vcodec") == "none"

    # 2) Preferimos progresivo MP4 (vídeo+audio)
    prog_mp4 = [f for f in fmts if is_progressive(f) and f.get("ext") == "mp4"]
    if prog_mp4:
        best = max(prog_mp4, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
        fmt = best["format_id"]
        try:
            with yt_dlp.YoutubeDL({**ydl_base, "format": fmt}) as ydl:
                info2 = ydl.extract_info(url, download=True)
                fn = ydl.prepare_filename(info2)
                base, _ = os.path.splitext(fn)
                mp4 = base + ".mp4"
                return ensure_video_ok(mp4 if os.path.exists(mp4) else fn)
        except Exception as e:
            last_err = e
    else:
        last_err = None

    # 3) Si no hay progresivo, combinamos bestvideo + bestaudio
    vids = [f for f in fmts if is_video_only(f)]
    auds = [f for f in fmts if is_audio_only(f)]
    if vids and auds:
        vids_sorted = sorted(vids, key=lambda f: (f.get("ext") == "mp4", f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        auds_sorted = sorted(auds, key=lambda f: (f.get("ext") in ("m4a", "mp4"), f.get("abr") or 0, f.get("tbr") or 0), reverse=True)
        v = vids_sorted[0]; a = auds_sorted[0]
        fmt = f"{v['format_id']}+{a['format_id']}"
        try:
            with yt_dlp.YoutubeDL({**ydl_base, "format": fmt}) as ydl:
                info2 = ydl.extract_info(url, download=True)
                fn = ydl.prepare_filename(info2)
                base, _ = os.path.splitext(fn)
                mp4 = base + ".mp4"
                return ensure_video_ok(mp4 if os.path.exists(mp4) else fn)
        except Exception as e:
            last_err = e

    # 4) Último recurso: 'best' y convertimos a mp4
    try:
        with yt_dlp.YoutubeDL({**ydl_base, "format": "best"}) as ydl:
            info2 = ydl.extract_info(url, download=True)
            fn = ydl.prepare_filename(info2)
            base, _ = os.path.splitext(fn)
            mp4 = base + ".mp4"
            return ensure_video_ok(mp4 if os.path.exists(mp4) else fn)
    except Exception as e:
        raise RuntimeError(f"Fallo descarga (yt-dlp): {e if last_err is None else last_err}")

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
        subprocess.run([FFMPEG_BIN,"-y","-i",str(p),"-c:v","copy","-c:a","aac","-b:a","192k",
                        "-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out.resolve()))
    raise RuntimeError("Proporciona URL de YouTube o ruta local válida.")

# ---------- audio ----------
def extract_audio(video:str, out="audio.wav")->str:
    subprocess.run([FFMPEG_BIN,"-y","-i",video,"-ac","1","-ar","16000","-vn",
                    "-acodec","pcm_s16le",out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- ASR Whisper + normalización ligera de URLs ----------
def normalize_glued_phrases_en(text:str)->str:
    if not text: return text
    text=re.sub(r"you\s*can\s*also\s*sign\s*up\s*for\s*a\s*free\s*trial\s*at",
                "you can also sign up for a free trial at", text, flags=re.IGNORECASE)
    text=re.sub(r"youcanalsosignupforafreetrialat",
                "you can also sign up for a free trial at", text, flags=re.IGNORECASE)
    text=re.sub(r"sign\s*up","sign up", text, flags=re.IGNORECASE)
    return text

URL_WORDS = r"(?:dot|period|\.)"
SLASH_WORDS = r"(?:slash|forward slash|/)"
HOST_LABEL = r"[A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)*"
URL_HOST_PATH_PATTERN = re.compile(
    rf"\b({HOST_LABEL})\s*(?:{URL_WORDS})\s*(com|co|org|net|ai|io|dev|app|es|uk|edu|gov|info|biz)\b\s*(?:{SLASH_WORDS})\s*([A-Za-z0-9\-_/%.]+)",
    re.IGNORECASE
)
URL_HOST_ONLY_PATTERN = re.compile(
    rf"\b({HOST_LABEL})\s*(?:{URL_WORDS})\s*(com|co|org|net|ai|io|dev|app|es|uk|edu|gov|info|biz)\b",
    re.IGNORECASE
)
def _squash(label:str)->str: return re.sub(r"\s+","",label)
def normalize_english_urls(text:str)->str:
    if not text: return text
    def _host_path(m):
        host=_squash(m.group(1)); tld=m.group(2).lower(); path=m.group(3)
        path=re.sub(r"\s*/\s*","/", path)
        return f"{host}.{tld}/{path}"
    def _host_only(m):
        host=_squash(m.group(1)); tld=m.group(2).lower()
        return f"{host}.{tld}"
    text=URL_HOST_PATH_PATTERN.sub(_host_path, text)
    text=URL_HOST_ONLY_PATTERN.sub(_host_only, text)
    text=re.sub(r"\b([A-Za-z0-9-]+)\.\s+(com|co|org|net|ai|io|dev|app|es|uk|edu|gov|info|biz)\b",
                r"\1.\2", text, flags=re.IGNORECASE)
    return text

def transcribe_segments(audio:str, model_size='small'):
    model=whisper.load_model(model_size, device='cpu')
    res=model.transcribe(
        audio, language='en', task='transcribe',
        temperature=0.0, best_of=5, beam_size=5,
        condition_on_previous_text=False, fp16=False,
        no_speech_threshold=0.2, logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
    full = normalize_english_urls(normalize_glued_phrases_en(res.get('text','')))
    segs = res['segments']
    for s in segs:
        t=normalize_english_urls(normalize_glued_phrases_en(s.get('text','')))
        s['text']=t
    return segs, full

# ---------- Protección de direcciones + TECH acronyms ----------
URL_RE   = re.compile(r'(?i)\bhttps?://[^\s]+')
EMAIL_RE = re.compile(r'(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b')
DOMAIN_HOST_RE = re.compile(r'(?ix)(?<!://)\b(?![\w\.-]+@)(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b')
DOMAIN_WITH_PATH_RE = re.compile(
    r'(?ix)(?<!://)\b(?![\w\.-]+@)'
    r'((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})'
    r'(/[^\s\)\]\}\.,;:!?]+)'
)
BASE_TECH_TERMS = {
    "RAG","LLM","GPT","GPU","TPU","API","SDK","SQL","NoSQL","ETL","ELT",
    "BI","KPI","CI/CD","MLOps","DevOps","RBAC","SSO","IAM","SLA","SLO","K8S","ML","AI","NLP"
}
UPPER_TECH = {t.upper() for t in BASE_TECH_TERMS}
PLACE_PREF="__XTOK_"; PLACE_SUFF="__"

def canonicalize_tech_acronyms(text: str) -> str:
    if not text: return text
    def repl(m):
        core = m.group("core"); suf = m.group("suf") or ""; comp = m.group("comp") or ""
        core_up = core.upper()
        if core_up in UPPER_TECH: return core_up + suf + comp
        return m.group(0)
    pat = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<core>[A-Za-z]{2,6})(?P<suf>s|es)?(?P<comp>[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)")
    return pat.sub(repl, text)

def compile_tech_regex(terms: set[str], protect_all_caps=True):
    parts=[]
    if terms:
        term_alts=[]
        for t in sorted(terms, key=len, reverse=True):
            core=re.escape(t)
            term_alts.append(rf"(?<![A-Za-z0-9]){core}(?:s|es)?(?:[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)")
        parts.append("(?:%s)" % "|".join(term_alts))
    if protect_all_caps:
        parts.append(r"(?<![A-Za-z0-9])([A-Z]{2,6})(?:s|es)?(?:[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)")
    if not parts: return None
    return re.compile("(?:" + "|".join(parts) + ")")

def protect_spans_with_types(text: str, tech_re=None):
    if not text: return text, {}, {}
    spans=[]
    def collect(pat, typ):
        for m in pat.finditer(text):
            s,e=m.span()
            if any(s<e2 and e>s2 for s2,e2,_ in spans): continue
            spans.append((s,e,typ))
    for pat,typ in [(URL_RE,"url"),(EMAIL_RE,"email"),(DOMAIN_WITH_PATH_RE,"domain_path"),(DOMAIN_HOST_RE,"domain")]:
        collect(pat,typ)
    if tech_re is not None:
        for m in tech_re.finditer(text):
            s,e=m.span(); frag=text[s:e]
            if frag.upper()!=frag: continue
            if any(s<e2 and e>s2 for s2,e2,_ in spans): continue
            spans.append((s,e,"tech"))
    if not spans: return text, {}, {}
    spans.sort()
    out=[]; last=0; mapping={}; mapping_types={}
    for idx,(s,e,typ) in enumerate(spans):
        out.append(text[last:s])
        tok=f"{PLACE_PREF}{idx}{PLACE_SUFF}"
        mapping[tok]=text[s:e]; mapping_types[tok]=typ
        out.append(tok); last=e
    out.append(text[last:])
    return "".join(out), mapping, mapping_types

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

def to_spoken_spanish_from_raw_address(raw: str) -> str:
    raw = raw.strip()
    if re.match(URL_RE, raw): return pronounce_url_es(raw)
    if re.match(EMAIL_RE, raw): return pronounce_email_es(raw)
    m = DOMAIN_WITH_PATH_RE.match(raw)
    if m: return pronounce_url_es(m.group(1) + m.group(2))
    if re.match(DOMAIN_HOST_RE, raw): return pronounce_host_es(raw)
    return raw

def unprotect_addresses_as_spoken(text: str, mapping: dict, mapping_types: dict) -> str:
    if not mapping: return text
    for token, original in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        typ = mapping_types.get(token, "")
        if typ in {"url","email","domain_path","domain"}:
            text = text.replace(token, to_spoken_spanish_from_raw_address(original))
        else:
            text = text.replace(token, original)  # TECH -> exacto
    return re.sub(r'\s{2,}', ' ', text).strip()

# ---------- Traducción EN->ES ----------
def translate_google_cloud(texts):
    from google.cloud import translate_v2 as translate
    client=translate.Client()
    if isinstance(texts,str): texts=[texts]
    outs=[]
    for t in texts:
        t=t or ""
        if not t.strip(): outs.append(""); continue
        parts=[]; chunk=[]; total=0
        for w in t.split():
            lw=len(w)+1
            if total+lw>4500: parts.append(" ".join(chunk)); chunk=[w]; total=lw
            else: chunk.append(w); total+=lw
        if chunk: parts.append(" ".join(chunk))
        segs=[]
        for p in parts:
            r=client.translate(p, source_language='en', target_language='es', format_='text')
            segs.append(r['translatedText'])
        outs.append(" ".join(segs))
    return outs

def translate_google_fallback(texts):
    gt=GoogleTranslator(source="en", target="es")
    if isinstance(texts,str): texts=[texts]
    outs=[]
    for t in texts:
        t=(t or "").strip()
        if not t: outs.append(""); continue
        parts=[]; chunk=[]; total=0
        for w in t.split():
            lw=len(w)+1
            if total+lw>4500: parts.append(" ".join(chunk)); chunk=[w]; total=lw
            else: chunk.append(w); total+=lw
        if chunk: parts.append(" ".join(chunk))
        outs.append(" ".join(gt.translate(p) for p in parts if p.strip()))
    return outs

def translate_en2es(texts):
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try: return translate_google_cloud(texts)
        except Exception: return translate_google_fallback(texts)
    return translate_google_fallback(texts)

# ---------- Post-edición ES ----------
def clean_spaces(s:str)->str:
    s=re.sub(r'\s*\n+\s*',' ', s); s=re.sub(r'\s{2,}',' ', s); return s.strip()
def spain_register(s:str)->str:
    rules=[
        (r'\bles voy a mostrar\b','os voy a enseñar'),
        (r'\bles mostraré\b','os voy a enseñar'),
        (r'\bimplementar\b','poner en marcha'),
        (r'\bustedes\b','vosotros'),
        (r'\bUstedes\b','Vosotros'),
        (r'\bles (muestro|enseño|explico|presento)\b', r'os \1'),
    ]
    out=s
    for pat,rep in rules: out=re.sub(pat,rep,out,flags=re.IGNORECASE)
    return out
def fix_spanish_intro(es_text:str)->str:
    if not es_text: return es_text
    es_text = re.sub(
        r'(?i)\b(?:aquí\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]+)|([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]+)\s+aquí)\b',
        lambda m: f"soy {m.group(1) or m.group(2)}",
        es_text
    )
    es_text = re.sub(
        r'(?i)\b(hola(?: a todos| a todas| a todos y todas| a [\w\s]+)?),\s*([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]+)\s+aquí\b',
        lambda m: f"{m.group(1)}, soy {m.group(2)}",
        es_text
    )
    es_text = re.sub(
        r'(^|\.\s+|\!\s+|\?\s+)soy\s+([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚñ]+)',
        lambda m: f"{m.group(1)}Soy {m.group(2)}",
        es_text
    )
    return es_text
def post_edit_es(s:str)->str:
    if not s: return s
    s=clean_spaces(s); s=spain_register(s); s=fix_spanish_intro(s); s=clean_spaces(s)
    return s

# ---------- Azure TTS ----------
def get_azure_creds():
    key=os.getenv("AZURE_SPEECH_KEY")
    region=os.getenv("AZURE_SPEECH_REGION")
    return key, region

def tts_azure(text:str, voice="es-ES-DarioNeural", outfile="tts_es.wav", rate_pct=-4):
    if not AZURE_OK: raise RuntimeError("azure-cognitiveservices-speech no instalado.")
    key,region=get_azure_creds()
    if not key or not region: raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")
    text=(text or "").strip()
    if not text:
        wavfile.write(outfile, 24000, np.zeros(int(0.05*24000), dtype=np.int16))
        return outfile
    rate=f"{rate_pct:+d}%"
    ssml=f"""<speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-ES">
<voice name="{voice}">
<mstts:express-as style="newscast-casual">
<prosody rate="{rate}">{text}</prosody>
</mstts:express-as>
</voice>
</speak>"""
    cfg=speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synth=speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    res=synth.speak_ssml_async(ssml).get()
    if res.reason!=speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS falló: {res.reason}")
    audio=bytes(res.audio_data)
    AudioSegment.from_file(io.BytesIO(audio), format="wav").export(outfile, format="wav")
    return outfile

# ---------- Ajuste exacto audio-vídeo ----------
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
        subprocess.run([FFMPEG_BIN,'-y','-i',audio_in,'-filter:a',_atempo_chain(factor),audio_out],
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
    subprocess.run([FFMPEG_BIN,'-y','-i',video,'-i',audio,'-map','0:v:0','-map','1:a:0',
                    '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart', out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- Construcción texto español (sin toggles) ----------
def build_spanish_text(full_en: str) -> str:
    norm_en = canonicalize_tech_acronyms(full_en or "")
    tech_re = compile_tech_regex(set(BASE_TECH_TERMS), protect_all_caps=True)
    prot, mapping, mapping_types = protect_spans_with_types(norm_en, tech_re=tech_re)
    es = translate_en2es(prot)[0]
    es = unprotect_addresses_as_spoken(es, mapping, mapping_types)
    es = post_edit_es(es)
    return es

# ============================ UI ============================
st.set_page_config(page_title="Doblador EN→ES", page_icon="🎬", layout="centered")
st.title("🎬 Doblad**or** de videos EN→ES")  # pequeño guiño

# Estado de motores
translator_active = "Google Cloud" if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") else "deep_translator (fallback)"
has_azure = bool(os.getenv("AZURE_SPEECH_KEY"))
azure_region = os.getenv("AZURE_SPEECH_REGION") or "—"
st.caption(f"Motor de traducción activo: **{translator_active}** | Azure KEY: {'✅' if has_azure else '❌'} | Región: {azure_region}")

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
        try:
            with st.spinner("Cargando vídeo y transcribiendo..."):
                video = resolve_source(source)
                audio_wav = extract_audio(video)
                segments, full_en = transcribe_segments(audio_wav, model_size=model)

            if accion == "Obtener el texto en inglés":
                st.success("✅ Transcripción (EN) lista.")
                st.download_button("⬇️ Descargar EN (.txt)", (full_en or "").encode("utf-8"),
                                   file_name="transcripcion_en.txt", mime="text/plain")

            elif accion == "Obtener la traducción a español":
                with st.spinner("Traduciendo y aplicando reglas..."):
                    full_es = build_spanish_text(full_en)
                st.success("✅ Traducción (ES) lista.")
                c1,c2=st.columns(2)
                with c1:
                    st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                                       file_name="transcripcion_en.txt", mime="text/plain")
                with c2:
                    st.download_button("⬇️ ES (.txt)", (full_es or "").encode("utf-8"),
                                       file_name="traduccion_es.txt", mime="text/plain")

                if AZURE_OK and has_azure:
                    if st.button("Doblaje ahora con Azure TTS"):
                        with st.spinner("Sintetizando y ajustando al tiempo del vídeo..."):
                            wav = tts_azure(full_es, voice=voice, outfile="tts_es.wav", rate_pct=-4)
                            fit = fit_audio_to_video(video, wav, "tts_fit.wav")
                            out = mux_video_audio(video, fit, "video_doblado.mp4")
                        st.success("✅ Doblaje listo")
                        st.video(out)
                        st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                           file_name="video_doblado.mp4")
                else:
                    st.info("Para doblar directamente aquí, configura AZURE_SPEECH_KEY y AZURE_SPEECH_REGION en *Secrets*.")

            else:  # Hacer el doblaje del video
                with st.spinner("Traduciendo y aplicando reglas..."):
                    full_es = build_spanish_text(full_en)
                if not (AZURE_OK and has_azure):
                    st.error("Falta Azure Speech (SDK o claves). Configura *Secrets* y vuelve a intentar.")
                else:
                    with st.spinner("Sintetizando y ajustando al tiempo del vídeo..."):
                        wav = tts_azure(full_es, voice=voice, outfile="tts_es.wav", rate_pct=-4)
                        fit = fit_audio_to_video(video, wav, "tts_fit.wav")
                        out = mux_video_audio(video, fit, "video_doblado.mp4")
                    st.success("✅ Doblaje listo")
                    st.video(out)
                    st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                       file_name="video_doblado.mp4")

        except Exception as e:
            st.error(str(e))

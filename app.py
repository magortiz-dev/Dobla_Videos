# app.py — EN→ES privado (Hugging Face, offline) + TTS Azure es-ES
# Entrada estable (archivo / Drive / Dropbox / OneDrive / HTTP) + YouTube (plan C)
# - ASR Whisper (CPU)
# - Traducción local con M2M100 (MIT) + reglas:
#     * preserva términos técnicos (RAG, LLM, GPT, GPU, ...) y acrónimos 2–6 MAYÚSCULAS
#     * URLs/emails/domains se convierten a forma hablada en español (punto/barra/arroba/guion/guion bajo)
#     * “Nick here” -> “Soy Nick” (si aparece)
# - Doblaje: sincro por clústeres (une segmentos para evitar pausas raras) + ajuste de rate SSML en Azure
#
# Nota: M2M100 requiere 'sentencepiece' y 'transformers'. Es 100% local (no Google).

import os, re, io, shutil, tempfile, subprocess, pathlib, time
from typing import List, Tuple, Dict

import requests
import numpy as np
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile

# ---------- Azure TTS ----------
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_OK = True
except Exception:
    AZURE_OK = False

# ---------- YouTube / descargas ----------
import yt_dlp

# ---------- Whisper ----------
import whisper

# ---------- Hugging Face (M2M100, MIT) ----------
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# =============================================================================
# FFmpeg portable (sin ffprobe)
# =============================================================================
FFMPEG_BIN = None

def _setup_ffmpeg():
    """Preferir ffmpeg del sistema; fallback a imageio-ffmpeg. No dependemos de ffprobe."""
    global FFMPEG_BIN
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
    else:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN
    # Registrar SOLO ffmpeg en pydub
    AudioSegment.converter = FFMPEG_BIN
import warnings
from pydub.utils import which as pydub_which

# Silencia el warning de pydub (ya tenemos FFMPEG_BIN)
warnings.filterwarnings(
    "ignore",
    message="Couldn't find ffmpeg or avconv*",
    category=RuntimeWarning,
)

# Asegura que pydub encuentra ffmpeg en reruns
if FFMPEG_BIN:
    os.environ["FFMPEG_BINARY"] = FFMPEG_BIN
    os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
def _ffmpeg_ok() -> bool:
    return bool(FFMPEG_BIN) or (shutil.which("ffmpeg") is not None)

_setup_ffmpeg()

def _duration_from_ffmpeg_stderr(txt: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt or "")
    if not m:
        return 0.0
    hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return hh * 3600 + mm * 60 + ss

def _probe_duration(path: str) -> float:
    """Mide duración sin ffprobe (parsea 'ffmpeg -i' + fallback pydub)."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return 0.0
        p = subprocess.run([FFMPEG_BIN, "-hide_banner", "-i", path],
                           capture_output=True, text=True)
        dur = _duration_from_ffmpeg_stderr(p.stderr or "")
        if dur > 0:
            return float(dur)
        seg = AudioSegment.from_file(path)
        return len(seg) / 1000.0
    except Exception:
        return 0.0

def ensure_video_ok(video_path: str) -> str:
    """Normaliza a MP4 con timestamps correctos si hace falta."""
    dur = _probe_duration(video_path)
    if dur > 0.1 and video_path.lower().endswith(".mp4"):
        return video_path

    remux = os.path.splitext(video_path)[0] + "_genpts.mp4"
    subprocess.run([FFMPEG_BIN, "-y", "-fflags", "+genpts", "-i", video_path,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", remux],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if _probe_duration(remux) > 0.1:
        return remux

    # Último recurso: recodificar (lento pero robusto)
    rec = os.path.splitext(video_path)[0] + "_reencode.mp4"
    subprocess.run([FFMPEG_BIN, "-y", "-i", video_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", rec],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return rec if _probe_duration(rec) > 0.1 else video_path


# =============================================================================
# UI título (banderas con Twemoji)
# =============================================================================
def render_title_text_first():
    GB = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ec-1f1e7.svg"  # 🇬🇧
    ES = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ea-1f1f8.svg"  # 🇪🇸
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; margin-top:6px; margin-bottom:10px;">
          <span style="font-size:2rem; line-height:1;">🎬</span>
          <span style="font-size:1.9rem; font-weight:700; letter-spacing:0.2px;">
            Doblador Videos
          </span>
          <div style="display:flex; align-items:center; gap:10px; margin-left:8px;">
            <img src="{GB}" style="height:1.6rem; vertical-align:middle;">
            <span style="font-weight:700; font-size:1.25rem;">EN</span>
            <span style="opacity:0.7; font-size:1.25rem;">→</span>
            <img src="{ES}" style="height:1.6rem; vertical-align:middle;">
            <span style="font-weight:700; font-size:1.25rem;">ES</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================================================================
# Entrada: archivo/URLs directas + YouTube plan C
# =============================================================================
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url or "", re.I))

def _is_gdrive(url: str) -> bool:
    return "drive.google.com" in (url or "") or "docs.google.com/uc" in (url or "")

def _is_dropbox(url: str) -> bool:
    return "dropbox.com" in (url or "")

def _is_onedrive(url: str) -> bool:
    return ("1drv.ms" in (url or "")) or ("onedrive.live.com" in (url or "")) or ("sharepoint.com" in (url or ""))

def _download_http(url: str, out_path: str, chunk: int = 1 << 20) -> str:
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)
    return out_path

def _gdrive_file_id(url: str) -> str | None:
    m = re.search(r"/d/([A-Za-z0-9_-]{10,})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", url)
    return m.group(1) if m else None

def download_from_gdrive(url: str, out_path: str) -> str:
    try:
        import gdown
    except Exception:
        raise RuntimeError("Falta gdown. Añade gdown==5.2.0 a requirements.txt")
    fid = _gdrive_file_id(url)
    if fid:
        gdown.download(id=fid, output=out_path, quiet=True)
    else:
        gdown.download(url=url, output=out_path, quiet=True)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Descarga de Google Drive falló (archivo vacío).")
    return out_path

def _dropbox_direct(url: str) -> str:
    if "dl=" in url:
        return re.sub(r"dl=\d", "dl=1", url)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}dl=1"

def _onedrive_direct(url: str) -> str:
    if re.search(r"[?&]download=1", url):
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}download=1"

def download_video_stable(url: str) -> str:
    """Drive/Dropbox/OneDrive/HTTP directo."""
    tmp_dir = tempfile.gettempdir()
    base = f"video_in_{int(time.time())}"
    if _is_gdrive(url):
        path = os.path.join(tmp_dir, base + ".bin")
        download_from_gdrive(url, path)
        return ensure_video_ok(path)
    if _is_dropbox(url):
        path = os.path.join(tmp_dir, base + ".mp4")
        _download_http(_dropbox_direct(url), path)
        return ensure_video_ok(path)
    if _is_onedrive(url):
        path = os.path.join(tmp_dir, base + ".mp4")
        _download_http(_onedrive_direct(url), path)
        return ensure_video_ok(path)
    if re.match(r"^https?://", url or "", re.I):
        ext = ".mp4" if ".mp4" in url.lower() else (".webm" if ".webm" in url.lower() else ".bin")
        path = os.path.join(tmp_dir, base + ext)
        _download_http(url, path)
        return ensure_video_ok(path)
    raise RuntimeError("URL no soportada para descarga directa.")

# Cookies/proxy opcional desde st.secrets (si no hay, no pasa nada)
YTDLP_COOKIEFILE = None
YTDLP_PROXY = None

def _bootstrap_ytdlp_auth():
    global YTDLP_COOKIEFILE, YTDLP_PROXY
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}
    cookies_blob = os.getenv("YTDLP_COOKIES") or (secrets.get("YTDLP_COOKIES") if hasattr(secrets, "get") else None)
    YTDLP_PROXY = os.getenv("YTDLP_PROXY") or (secrets.get("YTDLP_PROXY") if hasattr(secrets, "get") else None)
    if isinstance(cookies_blob, str) and cookies_blob.strip():
        blob = cookies_blob.replace("\r\n", "\n")
        if "Netscape HTTP Cookie File" not in (blob.splitlines() or [""])[0]:
            blob = "# Netscape HTTP Cookie File\n" + blob.lstrip()
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".cookies.txt", mode="w", encoding="utf-8")
        tf.write(blob); tf.close()
        YTDLP_COOKIEFILE = tf.name

_bootstrap_ytdlp_auth()

def download_youtube(url: str) -> str:
    """Plan C: YouTube con yt-dlp. Puede fallar en cloud por 403; mejor subir archivo."""
    ydl_base = {
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "retries": 8,
        "fragment_retries": 8,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": UA},
        "extractor_args": {"youtube": {"player_client": ["web", "android", "ios", "tv"]}},
        "ffmpeg_location": os.path.dirname(FFMPEG_BIN) if FFMPEG_BIN else None,
        "postprocessors": [{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        "postprocessor_args": {"FFmpegVideoConvertor": ["-movflags", "faststart"]},
        "allow_multiple_video_streams": False,
        "allow_multiple_audio_streams": False,
        "format_sort": ["proto:https", "ext:mp4:m4a", "vcodec:h264:avc1", "acodec:aac:mp4a", "res", "tbr"],
        "compat_opts": ["format-sort-force"],
        "sleep_interval_requests": 0.5,
        "throttled_rate": 1024 * 1024,
    }
    if YTDLP_COOKIEFILE:
        ydl_base["cookiefile"] = YTDLP_COOKIEFILE
    if YTDLP_PROXY:
        ydl_base["proxy"] = YTDLP_PROXY

    attempts = ["bv*+ba/b[ext=mp4]/b[ext=mp4]", "bestvideo*+bestaudio*/best", "best"]
    last_err = None
    for fmt in attempts:
        try:
            opts = dict(ydl_base)
            opts["format"] = fmt
            # limpia ffmpeg_location None para evitar warnings
            if opts.get("ffmpeg_location") is None:
                opts.pop("ffmpeg_location", None)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                fn = ydl.prepare_filename(info)
                base, _ = os.path.splitext(fn)
                mp4 = base + ".mp4"
                final = mp4 if os.path.exists(mp4) else fn
                return ensure_video_ok(final)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Fallo descarga YouTube: {last_err}")

def resolve_source(user_in: str, uploaded_path: str | None) -> str:
    if uploaded_path:
        p = pathlib.Path(uploaded_path)
        if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}:
            return ensure_video_ok(str(p.resolve()))
        out = p.with_suffix(".mp4")
        subprocess.run([FFMPEG_BIN, "-y", "-i", str(p),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))

    s = (user_in or "").strip().strip('"').strip("'")
    if not s:
        raise RuntimeError("Proporciona una URL o sube un archivo de vídeo.")
    if re.match(r"^https?://", s, re.I):
        if _is_youtube(s):
            return download_youtube(s)
        return download_video_stable(s)
    if pathlib.Path(s).exists():
        p = pathlib.Path(s)
        if p.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov"}:
            return ensure_video_ok(str(p.resolve()))
        out = p.with_suffix(".mp4")
        subprocess.run([FFMPEG_BIN, "-y", "-i", str(p),
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))
    raise RuntimeError("Entrada no válida. Sube un archivo o pega una URL directa/YouTube.")


# =============================================================================
# ASR: extraer audio y transcribir sin que Whisper llame a ffmpeg
# =============================================================================
def extract_audio(video: str, out: str = "audio.wav") -> str:
    subprocess.run([FFMPEG_BIN, "-y", "-i", video, "-ac", "1", "-ar", "16000", "-vn",
                    "-acodec", "pcm_s16le", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def transcribe_segments(audio_wav_path: str, model_size: str = "small"):
    sr, pcm = wavfile.read(audio_wav_path)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != 16000:
        fixed = audio_wav_path + ".16k.wav"
        subprocess.run([FFMPEG_BIN, "-y", "-i", audio_wav_path, "-ac", "1", "-ar", "16000",
                        "-acodec", "pcm_s16le", fixed],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        sr, pcm = wavfile.read(fixed)
    audio = pcm.astype(np.float32) / 32768.0

    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        temperature=0.0,
        best_of=5,
        beam_size=5,
        condition_on_previous_text=False,
        fp16=False
    )
    return result.get("segments", []), result.get("text", "")


# =============================================================================
# Protección: URLs/emails/TECH TERMS + pronunciación en ES
# =============================================================================
URL_RE   = re.compile(r'(?i)\bhttps?://[^\s]+')
EMAIL_RE = re.compile(r'(?i)\b[\w\.-]+@[\w\.-]+\.\w+\b')
DOMAIN_HOST_RE = re.compile(r'(?ix)(?<!://)\b(?![\w\.-]+@)(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b')
DOMAIN_WITH_PATH_RE = re.compile(
    r'(?ix)(?<!://)\b(?![\w\.-]+@)'
    r'((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})'
    r'(/[^\s\)\]\}\.,;:!?]+)'
)

BASE_TECH_TERMS = {
    "RAG", "LLM", "GPT", "GPU", "TPU", "API", "SDK", "SQL", "NoSQL",
    "ETL", "ELT", "BI", "KPI", "A/B", "CI/CD", "MLOps", "DevOps", "Ops",
    "RBAC", "SSO", "IAM", "SLA", "SLO", "K8S", "ML", "AI", "NLP"
}
UPPER_TECH = {t.upper() for t in BASE_TECH_TERMS}

PLACE_PREF = "__XTOK_"
PLACE_SUFF = "__"

def canonicalize_tech_acronyms(text: str) -> str:
    """rag -> RAG, llm -> LLM, etc., solo si coincide con nuestro set."""
    if not text:
        return text
    pat = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<core>[A-Za-z]{2,6})(?P<suf>s|es)?(?P<comp>[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)")
    def repl(m):
        core = m.group("core")
        suf  = m.group("suf") or ""
        comp = m.group("comp") or ""
        up = core.upper()
        if up in UPPER_TECH:
            return up + suf + comp
        return m.group(0)
    return pat.sub(repl, text)

def compile_tech_regex(terms: set[str], protect_all_caps: bool = True):
    parts = []
    if terms:
        term_alts = []
        for t in sorted(terms, key=len, reverse=True):
            core = re.escape(t)
            term_alts.append(
                rf"(?<![A-Za-z0-9]){core}(?:s|es)?(?:[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)"
            )
        parts.append("(?:%s)" % "|".join(term_alts))
    if protect_all_caps:
        parts.append(r"(?<![A-Za-z0-9])([A-Z]{2,6})(?:s|es)?(?:[-_][A-Za-z0-9]+)?(?=[\s\)\]\}},\.;:!?]|$)")
    if not parts:
        return None
    return re.compile("(?:" + "|".join(parts) + ")")

def protect_spans_with_types(text: str, tech_re=None):
    """Devuelve (texto_protegido, mapping, mapping_types)."""
    if not text:
        return text, {}, {}

    spans = []

    def collect(pat, typ):
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < e2 and e > s2 for s2, e2, _ in spans):
                continue
            spans.append((s, e, typ))

    # Direcciones primero
    for pat, typ in [(URL_RE, "url"), (EMAIL_RE, "email"),
                     (DOMAIN_WITH_PATH_RE, "domain_path"), (DOMAIN_HOST_RE, "domain")]:
        collect(pat, typ)

    # TECH (solo si viene en MAYÚSCULAS ya o está en nuestra lista tras canonicalize)
    if tech_re is not None:
        for m in tech_re.finditer(text):
            s, e = m.span()
            frag = text[s:e]
            # protejo también si es de nuestro set aunque no esté full caps
            if frag.upper() not in UPPER_TECH and frag.upper() != frag:
                continue
            if any(s < e2 and e > s2 for s2, e2, _ in spans):
                continue
            spans.append((s, e, "tech"))

    if not spans:
        return text, {}, {}

    spans.sort()
    out = []
    last = 0
    mapping = {}
    mapping_types = {}

    for idx, (s, e, typ) in enumerate(spans):
        out.append(text[last:s])
        tok = f"{PLACE_PREF}{idx}{PLACE_SUFF}"
        mapping[tok] = text[s:e]
        mapping_types[tok] = typ
        out.append(tok)
        last = e
    out.append(text[last:])
    return "".join(out), mapping, mapping_types

def pronounce_host_es(host: str) -> str:
    host = host.strip().strip('.,;:!?)]}')
    spoken = []
    if host.lower().startswith('www.'):
        spoken.append('triple doble u punto')
        host = host[4:]
    labels = [l for l in host.split('.') if l]
    labs = []
    for lab in labels:
        lab = lab.replace('-', ' guion ').replace('_', ' guion bajo ')
        lab = re.sub(r'\s{2,}', ' ', lab).strip()
        labs.append(lab)
    if labs:
        spoken.append(' punto '.join(labs))
    return ' '.join(spoken).strip()

def pronounce_url_es(raw: str) -> str:
    s = raw.strip().strip('.,;:!?)]}')
    # no añadimos esquema en el hablado; solo lo necesitamos para parsear
    tmp = s
    if not re.match(r'(?i)^https?://', tmp):
        tmp = 'http://' + tmp
    from urllib.parse import urlparse
    u = urlparse(tmp)
    spoken = pronounce_host_es(u.netloc or '')
    if u.path and u.path != '/':
        for seg in [seg for seg in u.path.split('/') if seg]:
            seg = seg.replace('-', ' guion ').replace('_', ' guion bajo ').replace('%20', ' espacio ')
            seg = re.sub(r'\s{2,}', ' ', seg).strip()
            if seg:
                spoken += ' barra ' + seg
    return re.sub(r'\s{2,}', ' ', spoken).strip()

def pronounce_email_es(email: str) -> str:
    e = email.strip().strip('.,;:!?)]}')
    if '@' not in e:
        return pronounce_url_es(e)
    local, dom = e.split('@', 1)
    local = local.replace('.', ' punto ').replace('-', ' guion ').replace('_', ' guion bajo ')
    local = re.sub(r'\s{2,}', ' ', local).strip()
    return f"{local} arroba {pronounce_host_es(dom)}"

def to_spoken_spanish_from_raw_address(raw: str) -> str:
    raw = raw.strip()
    if re.match(URL_RE, raw):
        return pronounce_url_es(raw)
    if re.match(EMAIL_RE, raw):
        return pronounce_email_es(raw)
    m = DOMAIN_WITH_PATH_RE.match(raw)
    if m:
        return pronounce_url_es(m.group(1) + m.group(2))
    if re.match(DOMAIN_HOST_RE, raw):
        return pronounce_host_es(raw)
    return raw

def unprotect_addresses_as_spoken(text: str, mapping: dict, mapping_types: dict) -> str:
    if not mapping:
        return text
    for token, original in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        typ = mapping_types.get(token, "")
        if typ in {"url", "email", "domain_path", "domain"}:
            text = text.replace(token, to_spoken_spanish_from_raw_address(original))
        else:
            text = text.replace(token, original)  # tech exacto
    return re.sub(r'\s{2,}', ' ', text).strip()

def fix_spanish_intro(es_text: str) -> str:
    """Nick aquí / aquí Nick -> Soy Nick (más natural)."""
    if not es_text:
        return es_text
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

def clean_spaces(s: str) -> str:
    s = re.sub(r'\s*\n+\s*', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def post_edit_es(s: str) -> str:
    s = clean_spaces(s)
    s = fix_spanish_intro(s)
    s = clean_spaces(s)
    return s


# =============================================================================
# Traducción local (M2M100) + reglas protect/unprotect
# =============================================================================
@st.cache_resource(show_spinner=False)
def _load_m2m100():
    model_id = "facebook/m2m100_418M"  # MIT
    tok = M2M100Tokenizer.from_pretrained(model_id)
    model = M2M100ForConditionalGeneration.from_pretrained(model_id)
    return tok, model

def _split_sentences_for_mt(text: str, max_chars: int = 900) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = []
    buf = []
    size = 0
    # split suave por signos finales
    for chunk in re.split(r'([.!?…])\s+', text):
        if chunk is None:
            continue
        if size + len(chunk) > max_chars and buf:
            parts.append("".join(buf).strip())
            buf, size = [], 0
        buf.append(chunk)
        size += len(chunk)
    if buf:
        parts.append("".join(buf).strip())
    # fallback por palabras si alguno es muy largo
    out = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            words = p.split()
            curr = []
            sz = 0
            for w in words:
                wlen = len(w) + 1
                if sz + wlen > max_chars and curr:
                    out.append(" ".join(curr))
                    curr, sz = [w], len(w)
                else:
                    curr.append(w); sz += wlen
            if curr:
                out.append(" ".join(curr))
    return [x for x in out if x.strip()]

def _m2m_translate_text(en_text: str) -> str:
    tok, model = _load_m2m100()
    tok.src_lang = "en"
    pieces = _split_sentences_for_mt(en_text, max_chars=900)
    outs = []
    for ch in pieces:
        inputs = tok(ch, return_tensors="pt", truncation=True, max_length=1024)
        gen = model.generate(
            **inputs,
            forced_bos_token_id=tok.get_lang_id("es"),
            max_length=1024,
            num_beams=4,
            length_penalty=1.05
        )
        outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0])
    return " ".join(outs).strip()

def translate_en2es_with_rules(en_text: str) -> str:
    # 1) Canonicaliza acrónimos tech
    norm = canonicalize_tech_acronyms(en_text or "")
    # 2) Protege direcciones y tech
    tech_re = compile_tech_regex(set(BASE_TECH_TERMS), protect_all_caps=True)
    prot, mapping, mapping_types = protect_spans_with_types(norm, tech_re=tech_re)
    # 3) Traduce local (M2M100)
    es = _m2m_translate_text(prot)
    # 4) Desprotege: URLs/emails -> hablado, tech -> exacto
    es = unprotect_addresses_as_spoken(es, mapping, mapping_types)
    # 5) Post edición ligera
    es = post_edit_es(es)
    return es

def translate_list_en2es_with_rules(texts: List[str], batch_size: int = 8, progress=None) -> List[str]:
    """Traduce lista (segmentos). Para no parecer bloqueado, actualiza progreso."""
    out = []
    n = len(texts)
    for i, t in enumerate(texts):
        out.append(translate_en2es_with_rules(t or ""))
        if progress is not None and n > 0:
            progress.progress((i + 1) / n)
    return out


# =============================================================================
# TTS Azure: sincro por clústeres con ajuste de rate SSML
# =============================================================================
def get_azure_creds() -> Tuple[str | None, str | None]:
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if (not key or not region):
        try:
            key = key or st.secrets.get("AZURE_SPEECH_KEY")
            region = region or st.secrets.get("AZURE_SPEECH_REGION")
        except Exception:
            pass
    return key, region

# parámetros de sincro
SYNC_OFFSET_MS       = 120
JOIN_GAP_MS          = 475
CLUSTER_MAX_MS       = 11000
GUARD_MS             = 30
TAIL_MARGIN_MS       = 60
MIN_WINDOW_MS        = 300
RATE_MIN_PCT         = -12
RATE_MAX_PCT         = +20
ATEMPO_LAST_RESORT   = 1.20

SENT_END_RE = re.compile(r'[.!?…:;]\s*$', re.UNICODE)

def _ends_with_strong_punct(txt: str) -> bool:
    return bool(SENT_END_RE.search((txt or "").strip()))

def _calc_gap_ms(seg_prev, seg_next) -> int:
    return int((float(seg_next["start"]) - float(seg_prev["end"])) * 1000)

def _cluster_segments(segments_en: List[dict], texts_es: List[str]) -> List[dict]:
    clusters = []
    i = 0
    n = len(segments_en)
    while i < n:
        start_ms = int(float(segments_en[i]["start"]) * 1000)
        end_ms   = int(float(segments_en[i]["end"])   * 1000)
        parts = [texts_es[i].strip()]
        j = i
        while j + 1 < n:
            gap = _calc_gap_ms(segments_en[j], segments_en[j+1])
            end_ms_next = int(float(segments_en[j+1]["end"]) * 1000)
            dur_if_join = (end_ms_next - start_ms)
            if _ends_with_strong_punct(parts[-1]): break
            if gap > JOIN_GAP_MS: break
            if dur_if_join > CLUSTER_MAX_MS: break
            j += 1
            end_ms = end_ms_next
            parts.append(texts_es[j].strip())
        clusters.append({"start_ms": start_ms, "end_ms": end_ms, "text": " ".join(p for p in parts if p)})
        i = j + 1
    return clusters

def _azure_tts_bytes(text: str, voice: str, rate_pct: int) -> bytes:
    key, region = get_azure_creds()
    if not key or not region:
        raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")

    rate = f"{rate_pct:+d}%"
    ssml = f"""<speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-ES">
<voice name="{voice}">
<mstts:express-as style="newscast-casual">
<prosody rate="{rate}">{text}</prosody>
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
    return bytes(res.audio_data)

def _tts_cluster_fit(text: str, voice: str, window_ms: int) -> AudioSegment:
    """Ajusta rate SSML para encajar en ventana; atempo suave como último recurso."""
    attempt_rates = [-4, None, None]
    audio_seg = None
    last_bytes = None
    used_rate = -4

    for rate in attempt_rates:
        if rate is None:
            dur_ms = len(audio_seg) if audio_seg else 0
            if dur_ms == 0:
                rate = -4
            else:
                ratio = dur_ms / float(window_ms)
                if ratio > 1.05:
                    inc = min(RATE_MAX_PCT, int(min(25, (ratio - 1.0) * 100 * 0.85)))
                    rate = max(-4, inc)
                elif ratio < 0.85:
                    dec = max(RATE_MIN_PCT, -int(min(12, (1.0 - ratio) * 100 * 0.85)))
                    rate = min(-4, dec)
                else:
                    rate = used_rate

        rate = int(max(RATE_MIN_PCT, min(RATE_MAX_PCT, rate)))
        used_rate = rate

        data = _azure_tts_bytes(text, voice, rate)
        last_bytes = data
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="wav")
        dur_ms = len(audio_seg)

        if abs(dur_ms - window_ms) <= 200:
            if dur_ms < window_ms:
                audio_seg = audio_seg + AudioSegment.silent(duration=(window_ms - dur_ms))
            elif dur_ms > window_ms:
                audio_seg = audio_seg[:window_ms]
            return audio_seg

    # atempo suave si sigue largo
    if len(audio_seg) > window_ms:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf_out:
            AudioSegment.from_file(io.BytesIO(last_bytes), format="wav").export(tf_in.name, format="wav")
            factor = min(ATEMPO_LAST_RESORT, len(audio_seg) / float(window_ms))
            subprocess.run([FFMPEG_BIN, "-y", "-i", tf_in.name, "-filter:a", f"atempo={factor:.3f}", tf_out.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            audio_seg = AudioSegment.from_file(tf_out.name)
            if len(audio_seg) > window_ms:
                audio_seg = audio_seg[:window_ms]
            try:
                os.remove(tf_in.name); os.remove(tf_out.name)
            except Exception:
                pass
        return audio_seg

    # si quedó corto, silenciamos
    return audio_seg + AudioSegment.silent(duration=(window_ms - len(audio_seg)))

def build_dubbed_audio_clusters(video_path: str,
                                segments_en: List[dict],
                                translations_es: List[str],
                                voice: str,
                                out_wav: str = "tts_timeline.wav",
                                progress=None) -> str:
    if not segments_en:
        raise RuntimeError("No hay segmentos de Whisper.")
    if len(translations_es) != len(segments_en):
        raise RuntimeError("Desajuste segments↔translations.")

    video_ms = int(_probe_duration(video_path) * 1000)
    if video_ms <= 0:
        fixed = ensure_video_ok(video_path)
        video_ms = int(_probe_duration(fixed) * 1000)
        if video_ms <= 0:
            raise RuntimeError("No se pudo medir la duración del vídeo.")

    clusters = _cluster_segments(segments_en, translations_es)
    final = AudioSegment.silent(duration=video_ms + 200)

    for idx, cl in enumerate(clusters):
        start_nom = cl["start_ms"] + SYNC_OFFSET_MS
        end_nom   = cl["end_ms"]
        next_start = (clusters[idx + 1]["start_ms"] + SYNC_OFFSET_MS) if (idx + 1 < len(clusters)) else video_ms

        place_ms = max(0, start_nom)
        end_allowed = min(end_nom - TAIL_MARGIN_MS, next_start - GUARD_MS)
        if end_allowed < place_ms + MIN_WINDOW_MS:
            end_allowed = place_ms + MIN_WINDOW_MS
        if end_allowed > video_ms:
            end_allowed = video_ms
        window_ms = max(150, end_allowed - place_ms)

        speech = _tts_cluster_fit(cl["text"], voice, window_ms)
        final = final.overlay(speech, position=place_ms)

        if progress is not None and len(clusters) > 0:
            progress.progress((idx + 1) / len(clusters))

    if len(final) > video_ms:
        final = final[:video_ms]
    elif len(final) < video_ms:
        final = final + AudioSegment.silent(duration=(video_ms - len(final)))

    final.export(out_wav, format="wav")
    return out_wav

def mux_video_audio(video: str, audio: str, out: str = "video_doblado.mp4") -> str:
    subprocess.run([FFMPEG_BIN, "-y", "-i", video, "-i", audio,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out


# =============================================================================
# UI (Streamlit)
# =============================================================================
st.set_page_config(page_title="Doblador EN→ES", page_icon="🎬", layout="centered")
# Evita que Chrome/Google Translate “rompa” el DOM
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

render_title_text_first()
st.caption("por Miguel Ángel Gómez Ortiz")

st.markdown("**Entrada de vídeo** (recomendado: subir archivo o usar enlace directo).")
col_u, _ = st.columns([2, 1])
with col_u:
    source = st.text_input("Pega una URL directa (Drive/Dropbox/OneDrive/MP4) o de YouTube (menos estable):")
uploaded = st.file_uploader("…o sube un vídeo (mp4/webm/mkv/mov)", type=["mp4", "webm", "mkv", "mov"])

accion = st.radio("Acción", ["Obtener el texto en inglés", "Obtener la traducción a español", "Hacer el doblaje del video"], index=2)

colA, colB = st.columns(2)
with colA:
    model_size = st.selectbox("Modelo Whisper", ["small", "base", "medium"], index=0)
with colB:
    voice = st.selectbox("Voz TTS (Azure)", [
        "es-ES-DarioNeural", "es-ES-AlvaroNeural", "es-ES-TeoNeural",
        "es-ES-ArnauNeural", "es-ES-ElviraNeural", "es-ES-LiaNeural"
    ], index=0)

if st.button("Procesar"):
    try:
        tmp_path = None
        if uploaded is not None:
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())

        with st.spinner("Descargando/preparando vídeo..."):
            video = resolve_source(source, tmp_path)

        with st.spinner("Extrayendo audio..."):
            audio_wav = extract_audio(video)

        with st.spinner("Transcribiendo (Whisper)..."):
            segments, full_en = transcribe_segments(audio_wav, model_size=model_size)

        if accion == "Obtener el texto en inglés":
            st.success("✅ Transcripción (EN) lista.")
            st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                               file_name="transcripcion_en.txt", mime="text/plain")

        elif accion == "Obtener la traducción a español":
            with st.spinner("Traduciendo EN→ES (Hugging Face local)..."):
                full_es = translate_en2es_with_rules(full_en)
            st.success("✅ Traducción (ES) lista.")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                                   file_name="transcripcion_en.txt", mime="text/plain")
            with c2:
                st.download_button("⬇️ ES (.txt)", (full_es or "").encode("utf-8"),
                                   file_name="traduccion_es.txt", mime="text/plain")

        else:  # doblaje
            if not AZURE_OK:
                st.error("Instala azure-cognitiveservices-speech para doblar.")
            else:
                # 1) Traducción por segmentos (con progreso)
                st.write("🔁 Traduciendo segmentos (local)…")
                prog_t = st.progress(0.0)
                texts_en = [(s.get("text") or "").strip() for s in segments]
                texts_es = translate_list_en2es_with_rules(texts_en, progress=prog_t)
                prog_t.empty()

                # 2) TTS + sincro por clústeres (con progreso)
                st.write("🎙️ Generando doblaje y sincronizando…")
                prog_a = st.progress(0.0)
                wav_timeline = build_dubbed_audio_clusters(
                    video_path=video,
                    segments_en=segments,
                    translations_es=texts_es,
                    voice=voice,
                    out_wav="tts_timeline.wav",
                    progress=prog_a
                )
                prog_a.empty()

                # 3) Mux final
                with st.spinner("🎬 Montando vídeo final..."):
                    out = mux_video_audio(video, wav_timeline, "video_doblado.mp4")

                st.success("✅ Doblaje listo")
                st.video(out)
                st.download_button("⬇️ Descargar video doblado", open(out, "rb"),
                                   file_name="video_doblado.mp4")

    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg or "Fallo descarga YouTube" in msg:
            st.error("YouTube ha bloqueado la descarga en la nube. Sube el archivo MP4 o usa un enlace directo (Drive/Dropbox/OneDrive).")
        else:
            st.error(msg)

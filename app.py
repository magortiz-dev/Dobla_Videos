# app.py (v11) — Streamlit Cloud ready (sin depender de ffmpeg del sistema)
# EN → ES con Azure Translator + Doblaje con Azure Speech + Whisper (ASR)
#
# Mejoras v11 (para tus ejemplos):
#  - La traducción que SE VE y la que SE OYE es la MISMA (se traduce por "frases/clústeres", no por segmentos sueltos).
#    Esto evita casos tipo: en pantalla "o incluso" pero en audio "ni siquiera".
#  - Clustering EN más inteligente: une segmentos cuando NO hay fin de frase real (evita pausas en mitad).
#  - Mantiene sincronización por timestamps, y ajusta ritmo si el TTS se pasa de su ventana.

import os
import re
import io
import uuid
import time
import wave
import shutil
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import requests
import streamlit as st
import numpy as np
import yt_dlp
import whisper

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_OK = True
except Exception:
    AZURE_SPEECH_OK = False

from pydub import AudioSegment


# =============================================================================
# FFmpeg portátil (no depende del sistema)
# =============================================================================
FFMPEG_BIN: Optional[str] = None

def setup_portable_ffmpeg() -> str:
    global FFMPEG_BIN
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
    else:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        ff_dir = str(Path(FFMPEG_BIN).parent)
        os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN

    AudioSegment.converter = FFMPEG_BIN
    warnings.filterwarnings(
        "ignore",
        message="Couldn't find ffmpeg or avconv*",
        category=RuntimeWarning,
    )
    return FFMPEG_BIN

setup_portable_ffmpeg()

def run_ffmpeg(args: List[str]):
    cmd = [FFMPEG_BIN] + args
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def parse_duration_from_ffmpeg(path: str) -> float:
    """Duración sin ffprobe: parsea stderr de `ffmpeg -i`."""
    try:
        p = subprocess.run([FFMPEG_BIN, "-hide_banner", "-i", path],
                           capture_output=True, text=True)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr or "")
        if not m:
            return 0.0
        hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return 0.0

def ensure_video_ok(video_path: str) -> str:
    dur = parse_duration_from_ffmpeg(video_path)
    if dur > 0.1 and video_path.lower().endswith(".mp4"):
        return video_path

    remux = str(Path(video_path).with_suffix("")) + "_genpts.mp4"
    run_ffmpeg(["-y", "-fflags", "+genpts", "-i", video_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", remux])
    if parse_duration_from_ffmpeg(remux) > 0.1:
        return remux

    rec = str(Path(video_path).with_suffix("")) + "_reencode.mp4"
    run_ffmpeg(["-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", rec])
    return rec


# =============================================================================
# Secrets / ENV
# =============================================================================
def get_secret(name: str) -> Optional[str]:
    """
    Lee secretos desde ENV o st.secrets (Streamlit Cloud).
    Soporta:
      - claves en raíz: st.secrets["AZURE_TRANSLATOR_KEY"]
      - claves dentro de secciones: [azure] AZURE_TRANSLATOR_KEY="..."
    """
    v = os.getenv(name)
    if isinstance(v, str) and v.strip():
        return v.strip()

    try:
        v2 = st.secrets[name]  # type: ignore[index]
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    except Exception:
        pass

    try:
        for k in st.secrets:  # type: ignore[operator]
            try:
                section = st.secrets[k]  # type: ignore[index]
            except Exception:
                continue
            if isinstance(section, str):
                continue
            try:
                if name in section:
                    v3 = section[name]
                    if isinstance(v3, str) and v3.strip():
                        return v3.strip()
            except Exception:
                continue
    except Exception:
        pass

    return None

AZURE_TRANSLATOR_KEY = get_secret("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = get_secret("AZURE_TRANSLATOR_REGION")  # westeurope
AZURE_TRANSLATOR_ENDPOINT = get_secret("AZURE_TRANSLATOR_ENDPOINT")  # opcional (custom subdomain)

AZURE_SPEECH_KEY = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = get_secret("AZURE_SPEECH_REGION")


# =============================================================================
# UI: título con banderas
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
# Entrada vídeo: URL/ruta o subir archivo
# =============================================================================
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url or "", re.I))

def download_youtube(url: str) -> str:
    outtmpl = "%(id)s.%(ext)s"
    ydl_opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": UA},
        "ffmpeg_location": str(Path(FFMPEG_BIN).parent),
        "merge_output_format": "mp4",
        "format": "bv*+ba/b[ext=mp4]/b[ext=mp4]/best",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        fn = ydl.prepare_filename(info)
        if not fn.endswith(".mp4"):
            fn_mp4 = str(Path(fn).with_suffix(".mp4"))
            if Path(fn_mp4).exists():
                fn = fn_mp4
    return ensure_video_ok(fn)

def download_direct_http(url: str) -> str:
    tmp = Path(tempfile.gettempdir()) / f"video_{int(time.time())}.bin"
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return ensure_video_ok(str(tmp))

def resolve_source(source: str, uploaded_path: Optional[str]) -> str:
    if uploaded_path:
        return ensure_video_ok(uploaded_path)

    s = (source or "").strip().strip('"').strip("'")
    if not s:
        raise RuntimeError("Proporciona una URL/ruta o sube un archivo.")

    if re.match(r"^https?://", s, re.I):
        if _is_youtube(s):
            return download_youtube(s)
        return download_direct_http(s)

    if Path(s).exists():
        return ensure_video_ok(str(Path(s).resolve()))

    raise RuntimeError("No se pudo resolver la fuente. Usa URL válida o sube un archivo.")


# =============================================================================
# Audio / Whisper (sin scipy)
# =============================================================================
def extract_audio(video_file: str, audio_file="audio.wav") -> str:
    run_ffmpeg(["-y", "-i", video_file, "-ac", "1", "-ar", "16000", "-vn",
                "-acodec", "pcm_s16le", audio_file])
    return audio_file

def read_wav_mono16k(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    if sw != 2 or sr != 16000 or ch != 1:
        fixed = path + ".fixed.wav"
        run_ffmpeg(["-y", "-i", path, "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", fixed])
        path = fixed
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    audio_i16 = np.frombuffer(frames, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    return whisper.load_model(model_size, device="cpu")

def transcribe_with_segments(audio_wav: str, model_size: str) -> Tuple[List[Dict], str]:
    audio = read_wav_mono16k(audio_wav)
    model = load_whisper_model(model_size)
    res = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        temperature=0.0,
        best_of=5,
        beam_size=5,
        condition_on_previous_text=False,
        fp16=False,
    )
    segments = res.get("segments", []) or []
    full_text = res.get("text", "") or ""
    return segments, full_text


# =============================================================================
# Clustering EN (anti pausas) + Traducción Azure por frases
# =============================================================================
EN_STRONG_END_RE = re.compile(r'[.!?…]\s*$')
EN_CONTINUATION_RE = re.compile(r'^(?:or|and|but|so|because|while|when|that|which|who|to|for|with|in|on|at)\b', re.I)

def ends_strong_punct_en(s: str) -> bool:
    return bool(EN_STRONG_END_RE.search((s or "").strip()))

def looks_continuation_start_en(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return (s[:1].islower()) or bool(EN_CONTINUATION_RE.search(s))

JOIN_GAP_MS = 1200
CLUSTER_MAX_MS = 18000

def cluster_segments_en(segments: List[Dict]) -> List[Dict]:
    clusters: List[Dict] = []
    i = 0
    n = len(segments)

    while i < n:
        start_ms = int(float(segments[i]["start"]) * 1000)
        end_ms = int(float(segments[i]["end"]) * 1000)
        parts = [((segments[i].get("text") or "").strip())]
        j = i

        while j + 1 < n:
            gap_ms = int((float(segments[j+1]["start"]) - float(segments[j]["end"])) * 1000)
            next_end = int(float(segments[j+1]["end"]) * 1000)
            dur_if = next_end - start_ms

            en_prev = (segments[j].get("text") or "").strip()
            en_next = (segments[j+1].get("text") or "").strip()

            # fin real de frase -> cortar (excepto '.' artificial)
            if ends_strong_punct_en(en_prev) and not (gap_ms <= 650 and looks_continuation_start_en(en_next)):
                break
            if gap_ms > JOIN_GAP_MS:
                break
            if dur_if > CLUSTER_MAX_MS:
                break

            j += 1
            end_ms = next_end
            parts.append(en_next)

        text_en = " ".join(p for p in parts if p).strip()
        text_en = re.sub(r"\s{2,}", " ", text_en)
        clusters.append({"start_ms": start_ms, "end_ms": end_ms, "text_en": text_en})
        i = j + 1

    return clusters



# --- Glosario/terminología (mejora traducción en contexto GenAI) ---
# Objetivo: evitar traducciones erróneas tipo "prompts" -> "propone temas".
# Puedes ampliar este glosario con otros términos internos de la empresa.
GLOSSARY_PHRASES_EN_ES = {
    "meaningful prompts": "prompts con sentido",
    "meaningful prompt": "prompt con sentido",
}
GLOSSARY_TERMS_EN_ES = {
    "prompts": "prompts",
    "prompt": "prompt",
    "generative ai": "IA generativa",
}

def _protect_glossary_text(en_text: str, text_idx: int) -> tuple[str, dict[str, str]]:
    """
    Reemplaza términos/frases por tokens ASCII (no traducibles) para mantenerlos
    y reinsertarlos en español después de traducir.
    """
    t = (en_text or "").strip()
    mapping: dict[str, str] = {}

    # Frases primero (más específicas)
    for j, (en_phrase, es_rep) in enumerate(GLOSSARY_PHRASES_EN_ES.items()):
        tok = f"__GPH_{text_idx}_{j}__"
        mapping[tok] = es_rep
        t = re.sub(rf"(?i)(?<!\w){re.escape(en_phrase)}(?!\w)", tok, t)

    # Términos sueltos
    base = len(GLOSSARY_PHRASES_EN_ES)
    for j, (en_term, es_rep) in enumerate(GLOSSARY_TERMS_EN_ES.items()):
        tok = f"__GTR_{text_idx}_{base + j}__"
        mapping[tok] = es_rep
        t = re.sub(rf"(?i)\b{re.escape(en_term)}\b", tok, t)

    return t, mapping

def _unprotect_glossary_text(es_text: str, mapping: dict[str, str]) -> str:
    t = es_text or ""
    for tok, rep in mapping.items():
        t = t.replace(tok, rep)
    return t

def _post_edit_es(es_text: str) -> str:
    """
    Ajustes ligeros de estilo para ES (mejorar naturalidad sin romper meaning).
    """
    t = (es_text or "").strip()
    # "úsalos con reflexión" -> "úsalos con criterio"
    t = re.sub(r"(?i)\b(úsalos|úsalas|úsa(?:los|las))\s+con\s+reflexión\b", r"\1 con criterio", t)
    # AI -> IA
    t = re.sub(r"(?i)\bAI\b", "IA", t)
    # espacios
    t = re.sub(r"\s{2,}", " ", t)
    return t

# =============================================================================
# Azure Translator
# =============================================================================
def azure_translate_batch(texts: List[str], from_lang="en", to_lang="es") -> List[str]:
    if not AZURE_TRANSLATOR_KEY:
        raise RuntimeError("Falta AZURE_TRANSLATOR_KEY (en Secrets/ENV).")
    if not AZURE_TRANSLATOR_REGION:
        raise RuntimeError("Falta AZURE_TRANSLATOR_REGION (en Secrets/ENV).")

    endpoint = (AZURE_TRANSLATOR_ENDPOINT or "https://api.cognitive.microsofttranslator.com").rstrip("/")
    if "cognitiveservices.azure.com" in endpoint:
        url = endpoint + "/translator/text/v3.0/translate"
    else:
        url = endpoint + "/translate"

    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    body = [{"text": (t or "")} for t in texts]
    r = requests.post(url, params=params, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Azure Translator falló ({r.status_code}): {r.text[:800]}")
    data = r.json()
    return [item["translations"][0]["text"] for item in data]

def translate_list_azure(texts: List[str], progress=None) -> List[str]:
    """
    Traduce una lista EN→ES con Azure Translator.
    Protege glosario (tokens) para evitar errores de sentido y aplica post-edición ligera.
    """
    protected: List[str] = []
    maps: List[dict[str, str]] = []
    for i, t in enumerate(texts):
        pt, mp = _protect_glossary_text(t or "", i)
        protected.append(pt)
        maps.append(mp)

    BATCH = 40
    raw_out: List[str] = []
    n = len(protected)

    for i in range(0, n, BATCH):
        chunk = protected[i:i+BATCH]
        raw_out.extend(azure_translate_batch(chunk, "en", "es"))
        if progress is not None and n:
            progress.progress(min(1.0, (i + len(chunk)) / n))

    out: List[str] = []
    for tr, mp in zip(raw_out, maps):
        tr2 = _unprotect_glossary_text(tr, mp)
        tr2 = _post_edit_es(tr2)
        out.append(tr2)

    return out


def translate_clusters_azure(clusters_en: List[Dict], progress=None) -> List[str]:
    texts = [(c.get("text_en") or "").strip() for c in clusters_en]
    return translate_list_azure(texts, progress=progress)

# --- Merge ES clústeres si la frase continúa (evita pausas tipo "imágenes ... o vídeos") ---
ES_CONTINUATION_START_RE = re.compile(r'^(?:o|y|e|u|pero|sino|porque|aunque|mientras|cuando|que|con|para|por|en)\b', re.I)
ES_STRONG_END_RE = re.compile(r'[.!?…]\s*$')
ES_WEAK_END_RE = re.compile(r'\b(?:con|de|del|para|a|en|por|sin|sobre|entre|hacia|hasta|como)\s*$', re.I)
EN_WEAK_END_RE = re.compile(r'\b(?:with|to|for|and|or|in|on|at|of|from|by)\s*$', re.I)

def merge_clusters_for_continuity(clusters_en: List[Dict], clusters_es: List[str]) -> tuple[List[Dict], List[str]]:
    """
    Une clústeres adyacentes cuando en español suena como continuación (sin punto/coma real),
    para evitar pausas artificiales en mitad de una frase.
    """
    if not clusters_en or not clusters_es:
        return clusters_en, clusters_es
    if len(clusters_en) != len(clusters_es):
        return clusters_en, clusters_es

    out_en: List[Dict] = []
    out_es: List[str] = []

    i = 0
    n = len(clusters_en)
    while i < n:
        cur_en = dict(clusters_en[i])
        cur_es = (clusters_es[i] or "").strip()

        while i + 1 < n:
            nxt_en = clusters_en[i+1]
            nxt_es = (clusters_es[i+1] or "").strip()

            gap_ms = int(nxt_en["start_ms"]) - int(cur_en["end_ms"])
            prev_strong = bool(ES_STRONG_END_RE.search(cur_es))
            nxt_cont = (nxt_es[:1].islower()) or bool(ES_CONTINUATION_START_RE.search(nxt_es))

            weak_end = bool(ES_WEAK_END_RE.search(cur_es))
            if ((gap_ms <= 900) and (not prev_strong) and nxt_cont) or ((gap_ms <= 2500) and (not prev_strong) and weak_end):
                cur_en["end_ms"] = int(nxt_en["end_ms"])
                cur_en["text_en"] = (cur_en.get("text_en","").rstrip() + " " + (nxt_en.get("text_en","").lstrip())).strip()
                cur_es = re.sub(r'[;:,.]\s*$', '', cur_es).strip()
                cur_es = (cur_es + " " + nxt_es).strip()
                i += 1
            else:
                break

        out_en.append(cur_en)
        out_es.append(re.sub(r'\s{2,}', ' ', cur_es).strip())
        i += 1

    return out_en, out_es

# --- Reparación de huecos grandes dentro de una MISMA frase ---
# Si Whisper mete un gap grande (p.ej. 4s) pero la frase continúa, adelantamos el inicio del siguiente clúster.
REPAIR_GAP_MS = 900          # a partir de aquí consideramos "hueco sospechoso"
REPAIR_TARGET_GAP_MS = 90    # hueco objetivo entre clústeres tras reparar (ms)

def repair_long_gaps_for_continuity(clusters_en: List[Dict], clusters_es: List[str]) -> List[Dict]:
    """
    Ajusta start_ms de clústeres EN cuando hay un hueco grande pero la frase continúa.
    Mantiene el orden temporal y evita solapes (deja un pequeño hueco).
    """
    if not clusters_en or len(clusters_en) < 2:
        return clusters_en
    out = [dict(c) for c in clusters_en]

    def es_continuation_start(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        return (s[:1].islower()) or bool(ES_CONTINUATION_START_RE.search(s))

    for i in range(len(out) - 1):
        prev = out[i]
        nxt = out[i + 1]
        gap_ms = int(nxt["start_ms"]) - int(prev["end_ms"])

        prev_es = (clusters_es[i] or "").strip() if i < len(clusters_es) else ""
        nxt_es = (clusters_es[i+1] or "").strip() if i+1 < len(clusters_es) else ""

        prev_es_strong = bool(ES_STRONG_END_RE.search(prev_es))
        cont_es = es_continuation_start(nxt_es)

        prev_en = (prev.get("text_en") or "").strip()
        nxt_en = (nxt.get("text_en") or "").strip()
        prev_en_strong = ends_strong_punct_en(prev_en)
        cont_en = looks_continuation_start_en(nxt_en)

        weak_es_end = bool(ES_WEAK_END_RE.search(prev_es))
        weak_en_end = bool(EN_WEAK_END_RE.search(prev_en))
        # Reparar si hay hueco grande y la frase continúa, o si acaba en preposición/conector
        should_repair = (gap_ms >= REPAIR_GAP_MS) and (not prev_es_strong) and (not prev_en_strong) and (cont_es or cont_en or weak_es_end or weak_en_end)

        if should_repair:
            new_start = int(prev["end_ms"]) + REPAIR_TARGET_GAP_MS
            if int(nxt["end_ms"]) - new_start < MIN_WINDOW_MS:
                new_start = max(0, int(nxt["end_ms"]) - MIN_WINDOW_MS)
            out[i + 1]["start_ms"] = new_start

    for i in range(1, len(out)):
        if int(out[i]["start_ms"]) <= int(out[i-1]["end_ms"]):
            out[i]["start_ms"] = int(out[i-1]["end_ms"]) + REPAIR_TARGET_GAP_MS

    return out


# =============================================================================
# Azure Speech TTS + sincro por clústeres
# =============================================================================
def ensure_azure_speech_ok():
    if not AZURE_SPEECH_OK:
        raise RuntimeError("No está instalado azure-cognitiveservices-speech.")
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (Secrets/ENV).")

def tts_ssml_bytes(text: str, voice: str, rate_pct: int) -> bytes:
    ensure_azure_speech_ok()
    rate = f"{rate_pct:+d}%"
    ssml = f"""<speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-ES">
<voice name="{voice}">
<mstts:express-as style="newscast-casual">
<prosody rate="{rate}">{text}</prosody>
</mstts:express-as>
</voice>
</speak>"""
    cfg = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    res = synth.speak_ssml_async(ssml).get()
    if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS falló: {res.reason}")
    return bytes(res.audio_data)

SYNC_OFFSET_MS = 0
GUARD_MS = 30
TAIL_MARGIN_MS = 60
MIN_WINDOW_MS = 300

RATE_MIN_PCT = -12
RATE_MAX_PCT = +20
ATEMPO_LAST_RESORT = 1.20

def atempo_chain(factor: float) -> str:
    if factor <= 0:
        factor = 1.0
    chain = []
    f = factor
    while f < 0.5 or f > 2.0:
        step = 0.5 if f < 0.5 else 2.0
        chain.append(f"atempo={step}")
        f = f / step
    chain.append(f"atempo={f}")
    return ",".join(chain)

def tts_fit_to_window(text: str, voice: str, window_ms: int) -> AudioSegment:
    """
    Evita solapes: nunca sobrepasa la ventana.
    Evita pausas artificiales: si queda corto, NO rellena con silencio.
    """
    attempt_rates = [-4, None, None]
    audio_seg: Optional[AudioSegment] = None
    last_bytes: Optional[bytes] = None
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

        data = tts_ssml_bytes(text, voice, rate)
        last_bytes = data
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="wav")
        dur_ms = len(audio_seg)

        if dur_ms <= window_ms:
            return audio_seg
        if dur_ms - window_ms <= 180:
            return audio_seg[:window_ms]

    # último recurso: atempo suave
    if audio_seg is not None and len(audio_seg) > window_ms and last_bytes is not None:
        tmp_in = str(Path(tempfile.gettempdir()) / f"tts_in_{uuid.uuid4().hex}.wav")
        tmp_out = str(Path(tempfile.gettempdir()) / f"tts_out_{uuid.uuid4().hex}.wav")
        AudioSegment.from_file(io.BytesIO(last_bytes), format="wav").export(tmp_in, format="wav")
        factor = min(ATEMPO_LAST_RESORT, len(audio_seg) / float(window_ms))
        run_ffmpeg(["-y", "-i", tmp_in, "-filter:a", atempo_chain(factor), tmp_out])
        audio_seg2 = AudioSegment.from_file(tmp_out)
        try:
            os.remove(tmp_in); os.remove(tmp_out)
        except Exception:
            pass
        if len(audio_seg2) > window_ms:
            audio_seg2 = audio_seg2[:window_ms]
        return audio_seg2

    return audio_seg if audio_seg is not None else AudioSegment.silent(duration=min(200, window_ms))

def build_dubbed_timeline(video_file: str, clusters_en: List[Dict], clusters_es: List[str], voice: str, progress=None) -> str:
    vdur = parse_duration_from_ffmpeg(video_file)
    if vdur <= 0:
        raise RuntimeError("No se pudo medir la duración del vídeo (ffmpeg).")
    video_ms = int(vdur * 1000)

    if not clusters_en:
        raise RuntimeError("No hay clústeres para doblaje.")
    if len(clusters_en) != len(clusters_es):
        raise RuntimeError("Desfase: número de clústeres EN y ES no coincide.")

    final = AudioSegment.silent(duration=video_ms + 200)

    for idx, cl in enumerate(clusters_en):
        start_nom = int(cl["start_ms"]) + SYNC_OFFSET_MS
        end_nom = int(cl["end_ms"])
        next_start = (int(clusters_en[idx+1]["start_ms"]) + SYNC_OFFSET_MS) if idx + 1 < len(clusters_en) else video_ms

        place_ms = max(0, start_nom)
        end_allowed = min(end_nom - TAIL_MARGIN_MS, next_start - GUARD_MS)
        if end_allowed < place_ms + MIN_WINDOW_MS:
            end_allowed = place_ms + MIN_WINDOW_MS
        if end_allowed > video_ms:
            end_allowed = video_ms
        window_ms = max(150, end_allowed - place_ms)

        text_es = re.sub(r"\s{2,}", " ", (clusters_es[idx] or "").strip())
        speech = tts_fit_to_window(text_es, voice, window_ms)
        final = final.overlay(speech, position=place_ms)

        if progress is not None and len(clusters_en):
            progress.progress((idx + 1) / len(clusters_en))

    if len(final) > video_ms:
        final = final[:video_ms]
    elif len(final) < video_ms:
        final += AudioSegment.silent(duration=(video_ms - len(final)))

    out_wav = "tts_timeline.wav"
    final.export(out_wav, format="wav")
    return out_wav

def mux_video_audio(video_file: str, audio_wav: str, output="video_doblado.mp4") -> str:
    run_ffmpeg(["-y", "-i", video_file, "-i", audio_wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", output])
    return output


# =============================================================================
# Streamlit UI
# =============================================================================
APP_VERSION = "v15"
st.set_page_config(page_title="Doblador EN→ES (Azure)", page_icon="🎬", layout="centered")
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

render_title_text_first()
st.caption("por Miguel Ángel Gómez Ortiz")
st.caption(f"build: {APP_VERSION}")

fuente = st.radio("Fuente del vídeo", ["URL / ruta", "Subir archivo"], horizontal=True)

source = ""
uploaded_path = None

if fuente == "URL / ruta":
    source = st.text_input("🔗 URL (YouTube o mp4 directo) o 📁 ruta local al vídeo")
else:
    up = st.file_uploader("Sube un vídeo (mp4/mkv/webm/mov/m4v)", type=["mp4", "mkv", "webm", "mov", "m4v"])
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.read())
        uploaded_path = str(tmp)

accion = st.radio("Acción", ["Obtener el texto en inglés", "Obtener la traducción a español", "Hacer el doblaje del video"], index=2)

colA, colB = st.columns(2)
with colA:
    model_size = st.selectbox("Modelo Whisper", ["base", "small", "medium"], index=1)
with colB:
    voice = st.selectbox("Voz Azure (ES)", ["es-ES-DarioNeural", "es-ES-AlvaroNeural", "es-ES-ElviraNeural", "es-ES-TeoNeural"], index=0)

def reset_session_outputs():
    for k in ["video_file", "audio_file", "segments", "clusters_en", "clusters_es", "transcript_en", "transcript_es_full", "video_out"]:
        st.session_state.pop(k, None)

if st.button("Procesar"):
    reset_session_outputs()
    try:
        with st.spinner("Preparando vídeo..."):
            video_file = resolve_source(source, uploaded_path)
            st.session_state["video_file"] = video_file

        with st.spinner("Extrayendo audio..."):
            audio_file = extract_audio(video_file, "audio.wav")
            st.session_state["audio_file"] = audio_file

        with st.spinner("Transcribiendo (Whisper)..."):
            segments, transcript_en = transcribe_with_segments(audio_file, model_size)
            st.session_state["segments"] = segments
            st.session_state["transcript_en"] = transcript_en
            st.session_state["clusters_en"] = cluster_segments_en(segments)

        st.success("✅ Transcripción lista")

        if accion == "Obtener el texto en inglés":
            st.subheader("📝 Transcripción (EN)")
            st.write(st.session_state["transcript_en"])
            st.download_button("⬇️ Descargar EN (.txt)",
                               st.session_state["transcript_en"].encode("utf-8"),
                               file_name="transcripcion_en.txt",
                               mime="text/plain")

        elif accion == "Obtener la traducción a español":
            with st.spinner("Traduciendo frases (Azure Translator)..."):
                prog = st.progress(0.0)
                clusters_es = translate_clusters_azure(st.session_state["clusters_en"], progress=prog)
                prog.empty()
                st.session_state["clusters_es"] = clusters_es
                st.session_state["clusters_en"], st.session_state["clusters_es"] = merge_clusters_for_continuity(
                    st.session_state["clusters_en"], st.session_state["clusters_es"]
                )
                st.session_state["clusters_en"], st.session_state["clusters_es"] = merge_clusters_for_continuity(
                    st.session_state["clusters_en"], st.session_state["clusters_es"]
                )
                st.session_state["transcript_es_full"] = " ".join(st.session_state["clusters_es"]).strip()

            st.subheader("🌍 Traducción (ES)")
            st.write(st.session_state["transcript_es_full"])
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ EN (.txt)",
                                   st.session_state["transcript_en"].encode("utf-8"),
                                   file_name="transcripcion_en.txt",
                                   mime="text/plain")
            with c2:
                st.download_button("⬇️ ES (.txt)",
                                   st.session_state["transcript_es_full"].encode("utf-8"),
                                   file_name="traduccion_es.txt",
                                   mime="text/plain")

        else:
            with st.spinner("Traduciendo frases (Azure Translator)..."):
                prog = st.progress(0.0)
                clusters_es = translate_clusters_azure(st.session_state["clusters_en"], progress=prog)
                prog.empty()
                st.session_state["clusters_es"] = clusters_es

            with st.spinner("Generando doblaje y sincronizando..."):
                prog2 = st.progress(0.0)
                wav_tl = build_dubbed_timeline(st.session_state["video_file"],
                                               st.session_state["clusters_en"],
                                               st.session_state["clusters_es"],
                                               voice,
                                               progress=prog2)
                prog2.empty()

            with st.spinner("Montando vídeo final..."):
                out = mux_video_audio(st.session_state["video_file"], wav_tl, "video_doblado.mp4")
                st.session_state["video_out"] = out

            st.success("✅ Doblaje listo")
            st.video(st.session_state["video_out"])
            st.download_button("⬇️ Descargar video doblado",
                               open(st.session_state["video_out"], "rb"),
                               file_name="video_doblado.mp4")

    except Exception as e:
        st.error(str(e))

# Doblaje sin reprocesar (si ya transcribiste o tradujiste)
can_dub = ("video_file" in st.session_state) and ("clusters_en" in st.session_state) and ("transcript_en" in st.session_state)
if can_dub:
    st.divider()
    st.markdown("### 🎙️ Doblaje sin reprocesar")
    if st.button("Hacer doblaje ahora (usando lo ya transcrito/traducido)"):
        try:
            if "clusters_es" not in st.session_state:
                with st.spinner("Traduciendo frases (Azure Translator)..."):
                    prog = st.progress(0.0)
                    st.session_state["clusters_es"] = translate_clusters_azure(st.session_state["clusters_en"], progress=prog)
                    prog.empty()
                    st.session_state["clusters_en"], st.session_state["clusters_es"] = merge_clusters_for_continuity(
                        st.session_state["clusters_en"], st.session_state["clusters_es"]
                    )

            with st.spinner("Generando doblaje y sincronizando..."):
                prog2 = st.progress(0.0)
                wav_tl = build_dubbed_timeline(st.session_state["video_file"],
                                               st.session_state["clusters_en"],
                                               st.session_state["clusters_es"],
                                               voice,
                                               progress=prog2)
                prog2.empty()

            with st.spinner("Montando vídeo final..."):
                out = mux_video_audio(st.session_state["video_file"], wav_tl, "video_doblado.mp4")
                st.session_state["video_out"] = out

            st.success("✅ Doblaje listo")
            st.video(st.session_state["video_out"])
            st.download_button("⬇️ Descargar video doblado",
                               open(st.session_state["video_out"], "rb"),
                               file_name="video_doblado.mp4")
        except Exception as e:
            st.error(str(e))

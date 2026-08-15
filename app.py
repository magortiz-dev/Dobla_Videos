# app.py — v18 (2026)
# Streamlit Cloud ready — EN → ES
#
# Cambios principales frente a v17:
# 1) faster-whisper + CTranslate2 INT8 en CPU: menos RAM y más velocidad.
# 2) Modelo inglés dedicado por defecto: small.en (mejor que small multilingüe para EN).
#    Opciones: distil-large-v3 / turbo para más calidad si el entorno tiene memoria.
# 3) Timestamps A NIVEL DE PALABRA y reconstrucción de frases completas:
#    evita partir "with meaningful prompts..." en dos audios y elimina pausas artificiales.
# 4) La misma frase traducida es la que se muestra, descarga y sintetiza.
# 5) Azure Translator mantiene glosario + admite opcionalmente Custom Translator:
#       AZURE_TRANSLATOR_CATEGORY = "tu-category-id"
# 6) Azure Speech: voces Dragon HD disponibles en West Europe + voces estándar.
#    SSML simplificado (sin style="newscast-casual") para una prosodia más predecible.
# 7) TTS ajustado a la ventana temporal de CADA FRASE completa.
# 8) Directorio temporal por ejecución para evitar colisiones entre usuarios.
#
# Secrets mínimos:
# AZURE_TRANSLATOR_KEY = "..."
# AZURE_TRANSLATOR_REGION = "westeurope"
# AZURE_SPEECH_KEY = "..."
# AZURE_SPEECH_REGION = "westeurope"
#
# Opcional:
# AZURE_TRANSLATOR_ENDPOINT = "https://....cognitiveservices.azure.com"
# AZURE_TRANSLATOR_CATEGORY = "xxxxxxxx-...."  # Custom Translator desplegado

import os
import re
import io
import uuid
import time
import shutil
import warnings
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from xml.sax.saxutils import escape as xml_escape

import requests
import streamlit as st
import yt_dlp
from pydub import AudioSegment
from faster_whisper import WhisperModel

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_OK = True
except Exception:
    AZURE_SPEECH_OK = False


# =============================================================================
# FFmpeg portátil
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
    subprocess.run(
        [FFMPEG_BIN] + args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def parse_duration_from_ffmpeg(path: str) -> float:
    """Obtiene duración sin ffprobe."""
    try:
        p = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-i", path],
            capture_output=True,
            text=True,
        )
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

    base = str(Path(video_path).with_suffix(""))
    remux = base + "_genpts.mp4"

    try:
        run_ffmpeg([
            "-y", "-fflags", "+genpts", "-i", video_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", remux
        ])
        if parse_duration_from_ffmpeg(remux) > 0.1:
            return remux
    except Exception:
        pass

    rec = base + "_reencode.mp4"
    run_ffmpeg([
        "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", rec
    ])
    return rec


# =============================================================================
# Secrets / ENV
# =============================================================================
def get_secret(name: str) -> Optional[str]:
    v = os.getenv(name)
    if isinstance(v, str) and v.strip():
        return v.strip()

    try:
        v2 = st.secrets[name]
        if isinstance(v2, str) and v2.strip():
            return v2.strip()
    except Exception:
        pass

    # Soporte opcional para secciones TOML
    try:
        for k in st.secrets:
            try:
                section = st.secrets[k]
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
AZURE_TRANSLATOR_REGION = get_secret("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = get_secret("AZURE_TRANSLATOR_ENDPOINT")
AZURE_TRANSLATOR_CATEGORY = get_secret("AZURE_TRANSLATOR_CATEGORY")

AZURE_SPEECH_KEY = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = get_secret("AZURE_SPEECH_REGION")


# =============================================================================
# UI: título
# =============================================================================
def render_title_text_first():
    GB = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ec-1f1e7.svg"
    ES = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ea-1f1f8.svg"
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
        unsafe_allow_html=True,
    )


# =============================================================================
# Entrada vídeo
# =============================================================================
HTTP = requests.Session()

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def _is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtube\.com|youtu\.be)", url or "", re.I))

def download_youtube(url: str, workdir: str) -> str:
    outtmpl = str(Path(workdir) / "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "retries": 8,
        "fragment_retries": 8,
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
        mp4 = str(Path(fn).with_suffix(".mp4"))
        if Path(mp4).exists():
            fn = mp4

    return ensure_video_ok(fn)

def download_direct_http(url: str, workdir: str) -> str:
    tmp = Path(workdir) / "video_download.bin"

    with HTTP.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    return ensure_video_ok(str(tmp))

def resolve_source(source: str, uploaded_file, workdir: str) -> str:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        target = Path(workdir) / f"uploaded{suffix}"
        target.write_bytes(uploaded_file.getvalue())
        return ensure_video_ok(str(target))

    s = (source or "").strip().strip('"').strip("'")
    if not s:
        raise RuntimeError("Proporciona una URL/ruta o sube un archivo.")

    if re.match(r"^https?://", s, re.I):
        if _is_youtube(s):
            return download_youtube(s, workdir)
        return download_direct_http(s, workdir)

    if Path(s).exists():
        return ensure_video_ok(str(Path(s).resolve()))

    raise RuntimeError("No se pudo resolver la fuente. Usa URL válida o sube un archivo.")


# =============================================================================
# ASR: faster-whisper INT8 + timestamps por palabra
# =============================================================================
ASR_MODELS = {
    "small.en — recomendado Cloud": "small.en",
    "distil-large-v3 — más calidad": "distil-large-v3",
    "turbo — máxima calidad (más pesado)": "turbo",
}

def _cpu_threads() -> int:
    # Evita pedir demasiados threads en Streamlit Cloud.
    return max(2, min(4, os.cpu_count() or 4))

@st.cache_resource(show_spinner=False)
def load_asr_model(model_name: str):
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=_cpu_threads(),
        num_workers=1,
    )

def transcribe_word_level(media_path: str, model_name: str) -> Tuple[List[Dict], List[Dict], str]:
    model = load_asr_model(model_name)

    segments_gen, info = model.transcribe(
        media_path,
        language="en",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=True,
        # Conservador: solo trata como silencio huecos bastante largos.
        vad_parameters={"min_silence_duration_ms": 1800},
    )

    fw_segments = list(segments_gen)

    segments: List[Dict] = []
    words: List[Dict] = []
    full_parts: List[str] = []

    for seg in fw_segments:
        txt = (seg.text or "").strip()
        if txt:
            full_parts.append(txt)

        seg_dict = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": txt,
        }

        seg_words: List[Dict] = []
        for w in (seg.words or []):
            if w.start is None or w.end is None:
                continue

            wd = {
                "start": float(w.start),
                "end": float(w.end),
                "text": (w.word or "").strip(),
                "probability": float(getattr(w, "probability", 0.0) or 0.0),
            }

            if wd["text"]:
                words.append(wd)
                seg_words.append(wd)

        seg_dict["words"] = seg_words
        segments.append(seg_dict)

    full_text = re.sub(r"\s{2,}", " ", " ".join(full_parts)).strip()
    return segments, words, full_text


# =============================================================================
# Construcción de frases usando timestamps de palabra
# =============================================================================
TERMINAL_EN_RE = re.compile(r'[.!?…]["\']?$')
CONTINUATION_EN = {
    "and", "or", "but", "so", "because", "while", "when",
    "that", "which", "who", "with", "to", "for", "of", "in", "on", "at"
}

MAX_SENTENCE_MS = 30000
LONG_GAP_MS = 2500

def _word_clean(t: str) -> str:
    return (t or "").strip()

def _starts_as_continuation(token: str) -> bool:
    t = re.sub(r"^[^A-Za-z]+", "", _word_clean(token)).lower()
    return t in CONTINUATION_EN

def build_sentence_units(words: List[Dict], segments: List[Dict]) -> List[Dict]:
    """
    Crea UNIDADES DE FRASE completas.
    La idea clave: mientras no haya un final de frase claro, NO se parte el TTS.
    Esto evita silencios internos como:
        "...guide the tools with" [pausa] "meaningful prompts..."
    """
    if not words:
        # Fallback por segmentos
        return [
            {
                "start_ms": int(float(s["start"]) * 1000),
                "end_ms": int(float(s["end"]) * 1000),
                "text_en": (s.get("text") or "").strip(),
            }
            for s in segments
            if (s.get("text") or "").strip()
        ]

    units: List[Dict] = []
    current: List[Dict] = []

    def flush():
        nonlocal current
        if not current:
            return

        text = " ".join(_word_clean(w["text"]) for w in current if _word_clean(w["text"]))
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        if text:
            units.append({
                "start_ms": int(float(current[0]["start"]) * 1000),
                "end_ms": int(float(current[-1]["end"]) * 1000),
                "text_en": text,
            })

        current = []

    n = len(words)
    for i, w in enumerate(words):
        current.append(w)

        token = _word_clean(w["text"])
        start_ms = int(float(current[0]["start"]) * 1000)
        end_ms = int(float(w["end"]) * 1000)
        dur_ms = end_ms - start_ms

        nxt = words[i + 1] if i + 1 < n else None
        next_gap_ms = 0
        next_cont = False

        if nxt is not None:
            next_gap_ms = int((float(nxt["start"]) - float(w["end"])) * 1000)
            next_cont = _starts_as_continuation(nxt["text"])

        terminal = bool(TERMINAL_EN_RE.search(token))

        # Final normal de frase.
        if terminal and not next_cont:
            flush()
            continue

        # Si Whisper crea un hueco grande y NO hay final de frase, mantenemos
        # la misma unidad si lo siguiente parece continuación.
        if nxt is not None and next_gap_ms >= LONG_GAP_MS and not terminal:
            if not next_cont:
                flush()
                continue

        # Freno de seguridad para frases excepcionalmente largas.
        if dur_ms >= MAX_SENTENCE_MS:
            flush()

    flush()
    return units


# =============================================================================
# Glosario / post-edición
# =============================================================================
GLOSSARY_PHRASES_EN_ES = {
    "meaningful prompts": "prompts con sentido",
    "meaningful prompt": "prompt con sentido",
}

GLOSSARY_TERMS_EN_ES = {
    "prompts": "prompts",
    "prompt": "prompt",
    "generative ai": "IA generativa",
}

def _protect_glossary_text(en_text: str, text_idx: int) -> Tuple[str, Dict[str, str]]:
    t = (en_text or "").strip()
    mapping: Dict[str, str] = {}

    for j, (en_phrase, es_rep) in enumerate(GLOSSARY_PHRASES_EN_ES.items()):
        tok = f"__GPH_{text_idx}_{j}__"
        mapping[tok] = es_rep
        t = re.sub(rf"(?i)(?<!\w){re.escape(en_phrase)}(?!\w)", tok, t)

    base = len(GLOSSARY_PHRASES_EN_ES)
    for j, (en_term, es_rep) in enumerate(GLOSSARY_TERMS_EN_ES.items()):
        tok = f"__GTR_{text_idx}_{base + j}__"
        mapping[tok] = es_rep
        t = re.sub(rf"(?i)\b{re.escape(en_term)}\b", tok, t)

    return t, mapping

def _unprotect_glossary_text(es_text: str, mapping: Dict[str, str]) -> str:
    t = es_text or ""
    for tok, rep in mapping.items():
        # Azure normalmente conserva el token; contemplamos alguna separación rara.
        t = t.replace(tok, rep)
        compact = tok.replace("_", "")
        t = t.replace(compact, rep)
    return t

def _post_edit_es(es_text: str) -> str:
    t = (es_text or "").strip()

    t = re.sub(
        r"(?i)\b(úsalos|úsalas|úsa(?:los|las))\s+con\s+reflexión\b",
        r"\1 con criterio",
        t,
    )

    t = re.sub(r"(?i)\bAI\b", "IA", t)

    # "IA generativa hace..." -> "La IA generativa hace..."
    verbs = (
        r"(?:hace|permite|ayuda|facilita|convierte|transforma|mejora|impulsa|"
        r"habilita|ofrece|aporta|crea|genera|reduce|aumenta|acelera|automatiza|"
        r"optimiza|puede|debe|está|es|tiene)"
    )
    subject_pat = re.compile(rf"(?i)(?<!\bLa\s)\bIA generativa\b\s+(?P<verb>{verbs})\b")
    t = subject_pat.sub(lambda m: "La IA generativa " + m.group("verb"), t)

    t = re.sub(r"\s{2,}", " ", t)
    return t


# =============================================================================
# Azure Translator
# =============================================================================
def azure_translate_batch(texts: List[str], from_lang="en", to_lang="es") -> List[str]:
    if not AZURE_TRANSLATOR_KEY:
        raise RuntimeError("Falta AZURE_TRANSLATOR_KEY en Secrets/ENV.")
    if not AZURE_TRANSLATOR_REGION:
        raise RuntimeError("Falta AZURE_TRANSLATOR_REGION en Secrets/ENV.")

    endpoint = (
        AZURE_TRANSLATOR_ENDPOINT
        or "https://api.cognitive.microsofttranslator.com"
    ).rstrip("/")

    if "cognitiveservices.azure.com" in endpoint:
        url = endpoint + "/translator/text/v3.0/translate"
    else:
        url = endpoint + "/translate"

    params = {
        "api-version": "3.0",
        "from": from_lang,
        "to": to_lang,
    }

    # Si algún día despliegas un Custom Translator, basta con añadir el secret.
    if AZURE_TRANSLATOR_CATEGORY:
        params["category"] = AZURE_TRANSLATOR_CATEGORY

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-Type": "application/json; charset=UTF-8",
        "X-ClientTraceId": str(uuid.uuid4()),
    }

    body = [{"text": t or ""} for t in texts]

    r = HTTP.post(
        url,
        params=params,
        headers=headers,
        json=body,
        timeout=120,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Azure Translator falló ({r.status_code}): {r.text[:800]}")

    data = r.json()
    return [item["translations"][0]["text"] for item in data]

def translate_units_azure(units_en: List[Dict], progress=None) -> List[str]:
    protected: List[str] = []
    mappings: List[Dict[str, str]] = []

    for i, u in enumerate(units_en):
        pt, mp = _protect_glossary_text(u.get("text_en", ""), i)
        protected.append(pt)
        mappings.append(mp)

    out_raw: List[str] = []
    batch_size = 40
    n = len(protected)

    for i in range(0, n, batch_size):
        chunk = protected[i:i + batch_size]
        out_raw.extend(azure_translate_batch(chunk, "en", "es"))

        if progress is not None and n:
            progress.progress(min(1.0, (i + len(chunk)) / n))

    out: List[str] = []
    for translated, mapping in zip(out_raw, mappings):
        translated = _unprotect_glossary_text(translated, mapping)
        translated = _post_edit_es(translated)
        out.append(translated)

    return out


# =============================================================================
# Azure Speech TTS
# =============================================================================
VOICE_OPTIONS = {
    "Tristán HD — recomendado": "es-es-Tristan:DragonHDLatestNeural",
    "Ximena HD": "es-es-Ximena:DragonHDLatestNeural",
    "Dario Neural": "es-ES-DarioNeural",
    "Álvaro Neural": "es-ES-AlvaroNeural",
    "Elvira Neural": "es-ES-ElviraNeural",
    "Teo Neural": "es-ES-TeoNeural",
}

RATE_MIN_PCT = -20
RATE_MAX_PCT = +30
TARGET_FILL = 0.97
MAX_ATEMPO = 1.25
MIN_ATEMPO = 0.82

def ensure_azure_speech_ok():
    if not AZURE_SPEECH_OK:
        raise RuntimeError("No está instalado azure-cognitiveservices-speech.")
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION.")

def tts_ssml_bytes(text: str, voice: str, rate_pct: int) -> bytes:
    """
    SSML simple para una prosodia más predecible.
    Eliminamos express-as/newscast-casual, que puede introducir pausas estilísticas.
    """
    ensure_azure_speech_ok()

    safe_text = xml_escape(text or "")
    rate = f"{int(rate_pct):+d}%"

    ssml = (
        '<speak version="1.0" xml:lang="es-ES">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}">{safe_text}</prosody>'
        '</voice>'
        '</speak>'
    )

    cfg = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION,
    )
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )

    synth = speechsdk.SpeechSynthesizer(
        speech_config=cfg,
        audio_config=None,
    )
    res = synth.speak_ssml_async(ssml).get()

    if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = ""
        try:
            details = str(
                speechsdk.SpeechSynthesisCancellationDetails.from_result(res).error_details
            )
        except Exception:
            pass
        raise RuntimeError(f"Azure TTS falló: {res.reason}. {details}")

    return bytes(res.audio_data)

def _atempo_chain(factor: float) -> str:
    factor = max(0.01, float(factor))
    chain: List[str] = []

    while factor < 0.5:
        chain.append("atempo=0.5")
        factor /= 0.5

    while factor > 2.0:
        chain.append("atempo=2.0")
        factor /= 2.0

    chain.append(f"atempo={factor:.5f}")
    return ",".join(chain)

def _apply_atempo(seg: AudioSegment, factor: float, workdir: str) -> AudioSegment:
    if abs(factor - 1.0) < 0.015:
        return seg

    inp = str(Path(workdir) / f"tempo_in_{uuid.uuid4().hex}.wav")
    out = str(Path(workdir) / f"tempo_out_{uuid.uuid4().hex}.wav")

    seg.export(inp, format="wav")
    run_ffmpeg([
        "-y", "-i", inp,
        "-filter:a", _atempo_chain(factor),
        out,
    ])

    result = AudioSegment.from_file(out, format="wav")

    for p in (inp, out):
        try:
            os.remove(p)
        except Exception:
            pass

    return result

def tts_fit_sentence(
    text: str,
    voice: str,
    window_ms: int,
    workdir: str,
) -> AudioSegment:
    """
    Sintetiza UNA FRASE COMPLETA y ajusta su duración a la ventana inglesa.
    Normalmente 1-2 llamadas a Azure TTS.
    """
    window_ms = max(350, int(window_ms))
    target_ms = max(300, int(window_ms * TARGET_FILL))

    # 1ª síntesis: velocidad natural, ligeramente relajada.
    rate1 = -2
    data1 = tts_ssml_bytes(text, voice, rate1)
    seg1 = AudioSegment.from_file(io.BytesIO(data1), format="wav")
    dur1 = len(seg1)

    ratio1 = dur1 / float(target_ms)

    # Si ya encaja bien, evitamos una segunda llamada.
    if 0.93 <= ratio1 <= 1.07:
        seg = seg1
    else:
        # Si dura demasiado -> aumentar rate.
        # Si dura poco -> rate negativo.
        delta_pct = int(round((ratio1 - 1.0) * 100 * 0.90))
        rate2 = max(RATE_MIN_PCT, min(RATE_MAX_PCT, delta_pct))

        data2 = tts_ssml_bytes(text, voice, rate2)
        seg = AudioSegment.from_file(io.BytesIO(data2), format="wav")

    # Ajuste fino local (sin otra llamada a Azure).
    if len(seg) > 0:
        factor = len(seg) / float(target_ms)
        if MIN_ATEMPO <= factor <= MAX_ATEMPO:
            seg = _apply_atempo(seg, factor, workdir)

    # Nunca invadir la frase siguiente.
    if len(seg) > window_ms:
        seg = seg[:window_ms]

    return seg


# =============================================================================
# Timeline / sincronización por FRASES
# =============================================================================
SYNC_OFFSET_MS = 0
GUARD_MS = 25
MIN_WINDOW_MS = 400

def build_dubbed_timeline(
    video_file: str,
    units_en: List[Dict],
    units_es: List[str],
    voice: str,
    workdir: str,
    progress=None,
) -> str:
    vdur = parse_duration_from_ffmpeg(video_file)
    if vdur <= 0:
        raise RuntimeError("No se pudo medir la duración del vídeo.")

    video_ms = int(vdur * 1000)

    if not units_en:
        raise RuntimeError("No hay frases para doblaje.")
    if len(units_en) != len(units_es):
        raise RuntimeError("El número de frases EN y ES no coincide.")

    timeline = AudioSegment.silent(duration=video_ms + 100)

    for idx, unit in enumerate(units_en):
        start_ms = max(0, int(unit["start_ms"]) + SYNC_OFFSET_MS)

        if idx + 1 < len(units_en):
            next_start = int(units_en[idx + 1]["start_ms"]) + SYNC_OFFSET_MS
        else:
            next_start = video_ms

        # La ventana de una frase llega hasta justo antes de la siguiente frase.
        window_ms = max(
            MIN_WINDOW_MS,
            min(video_ms, next_start - GUARD_MS) - start_ms,
        )

        text_es = re.sub(r"\s{2,}", " ", (units_es[idx] or "").strip())

        speech = tts_fit_sentence(
            text=text_es,
            voice=voice,
            window_ms=window_ms,
            workdir=workdir,
        )

        timeline = timeline.overlay(speech, position=start_ms)

        if progress is not None:
            progress.progress((idx + 1) / len(units_en))

    timeline = timeline[:video_ms]

    out_wav = str(Path(workdir) / "tts_timeline.wav")
    timeline.export(out_wav, format="wav")
    return out_wav

def mux_video_audio(
    video_file: str,
    audio_wav: str,
    workdir: str,
) -> str:
    output = str(Path(workdir) / "video_doblado.mp4")

    run_ffmpeg([
        "-y",
        "-i", video_file,
        "-i", audio_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output,
    ])
    return output


# =============================================================================
# Streamlit
# =============================================================================
APP_VERSION = "v18"

st.set_page_config(
    page_title="Doblador EN→ES (Azure)",
    page_icon="🎬",
    layout="centered",
)
st.markdown(
    '<meta name="google" content="notranslate">',
    unsafe_allow_html=True,
)

render_title_text_first()
st.caption("por Miguel Ángel Gómez Ortiz")
st.caption(f"build: {APP_VERSION}")

fuente = st.radio(
    "Fuente del vídeo",
    ["URL / ruta", "Subir archivo"],
    horizontal=True,
)

source = ""
uploaded = None

if fuente == "URL / ruta":
    source = st.text_input(
        "🔗 URL (YouTube o mp4 directo) o 📁 ruta local al vídeo"
    )
else:
    uploaded = st.file_uploader(
        "Sube un vídeo (mp4/mkv/webm/mov/m4v)",
        type=["mp4", "mkv", "webm", "mov", "m4v"],
    )

accion = st.radio(
    "Acción",
    [
        "Obtener el texto en inglés",
        "Obtener la traducción a español",
        "Hacer el doblaje del video",
    ],
    index=2,
)

colA, colB = st.columns(2)

with colA:
    asr_label = st.selectbox(
        "Modelo de transcripción",
        list(ASR_MODELS.keys()),
        index=0,
        help=(
            "small.en es el recomendado para Streamlit Cloud. "
            "distil-large-v3 y turbo pueden dar más calidad, pero consumen más memoria."
        ),
    )
    asr_model_name = ASR_MODELS[asr_label]

with colB:
    voice_label = st.selectbox(
        "Voz Azure (ES)",
        list(VOICE_OPTIONS.keys()),
        index=0,
    )
    voice = VOICE_OPTIONS[voice_label]

def reset_session_outputs():
    for k in [
        "workdir",
        "video_file",
        "segments",
        "words",
        "units_en",
        "units_es",
        "transcript_en",
        "transcript_es_full",
        "video_out",
    ]:
        st.session_state.pop(k, None)

def prepare_run():
    workdir = tempfile.mkdtemp(prefix="dobla_")
    st.session_state["workdir"] = workdir

    with st.spinner("Preparando vídeo..."):
        video_file = resolve_source(source, uploaded, workdir)
        st.session_state["video_file"] = video_file

    with st.spinner("Transcribiendo con faster-whisper..."):
        segments, words, transcript_en = transcribe_word_level(
            video_file,
            asr_model_name,
        )

        units_en = build_sentence_units(words, segments)

        st.session_state["segments"] = segments
        st.session_state["words"] = words
        st.session_state["units_en"] = units_en
        st.session_state["transcript_en"] = transcript_en

if st.button("Procesar"):
    reset_session_outputs()

    try:
        prepare_run()
        st.success("✅ Transcripción lista")

        if accion == "Obtener el texto en inglés":
            st.subheader("📝 Transcripción (EN)")
            st.write(st.session_state["transcript_en"])

            st.download_button(
                "⬇️ Descargar EN (.txt)",
                st.session_state["transcript_en"].encode("utf-8"),
                file_name="transcripcion_en.txt",
                mime="text/plain",
            )

        elif accion == "Obtener la traducción a español":
            with st.spinner("Traduciendo frases (Azure Translator)..."):
                prog = st.progress(0.0)

                units_es = translate_units_azure(
                    st.session_state["units_en"],
                    progress=prog,
                )

                prog.empty()

                st.session_state["units_es"] = units_es
                st.session_state["transcript_es_full"] = (
                    " ".join(units_es).strip()
                )

            st.subheader("🌍 Traducción (ES)")
            st.write(st.session_state["transcript_es_full"])

            c1, c2 = st.columns(2)

            with c1:
                st.download_button(
                    "⬇️ EN (.txt)",
                    st.session_state["transcript_en"].encode("utf-8"),
                    file_name="transcripcion_en.txt",
                    mime="text/plain",
                )

            with c2:
                st.download_button(
                    "⬇️ ES (.txt)",
                    st.session_state["transcript_es_full"].encode("utf-8"),
                    file_name="traduccion_es.txt",
                    mime="text/plain",
                )

        else:
            with st.spinner("Traduciendo frases (Azure Translator)..."):
                prog = st.progress(0.0)

                units_es = translate_units_azure(
                    st.session_state["units_en"],
                    progress=prog,
                )

                prog.empty()
                st.session_state["units_es"] = units_es

            with st.spinner("Generando voz y sincronizando por frases..."):
                prog2 = st.progress(0.0)

                wav_timeline = build_dubbed_timeline(
                    video_file=st.session_state["video_file"],
                    units_en=st.session_state["units_en"],
                    units_es=st.session_state["units_es"],
                    voice=voice,
                    workdir=st.session_state["workdir"],
                    progress=prog2,
                )

                prog2.empty()

            with st.spinner("Montando vídeo final..."):
                out = mux_video_audio(
                    st.session_state["video_file"],
                    wav_timeline,
                    st.session_state["workdir"],
                )

                st.session_state["video_out"] = out

            st.success("✅ Doblaje listo")
            st.video(st.session_state["video_out"])

            with open(st.session_state["video_out"], "rb") as f:
                st.download_button(
                    "⬇️ Descargar video doblado",
                    f.read(),
                    file_name="video_doblado.mp4",
                )

    except Exception as e:
        st.error(str(e))


# Doblaje sin reprocesar
can_dub = (
    "video_file" in st.session_state
    and "units_en" in st.session_state
    and "transcript_en" in st.session_state
)

if can_dub:
    st.divider()
    st.markdown("### 🎙️ Doblaje sin reprocesar")

    if st.button("Hacer doblaje ahora (usando lo ya transcrito/traducido)"):
        try:
            if "units_es" not in st.session_state:
                with st.spinner("Traduciendo frases (Azure Translator)..."):
                    prog = st.progress(0.0)
                    st.session_state["units_es"] = translate_units_azure(
                        st.session_state["units_en"],
                        progress=prog,
                    )
                    prog.empty()

            with st.spinner("Generando voz y sincronizando por frases..."):
                prog2 = st.progress(0.0)

                wav_timeline = build_dubbed_timeline(
                    video_file=st.session_state["video_file"],
                    units_en=st.session_state["units_en"],
                    units_es=st.session_state["units_es"],
                    voice=voice,
                    workdir=st.session_state["workdir"],
                    progress=prog2,
                )

                prog2.empty()

            with st.spinner("Montando vídeo final..."):
                out = mux_video_audio(
                    st.session_state["video_file"],
                    wav_timeline,
                    st.session_state["workdir"],
                )
                st.session_state["video_out"] = out

            st.success("✅ Doblaje listo")
            st.video(st.session_state["video_out"])

            with open(st.session_state["video_out"], "rb") as f:
                st.download_button(
                    "⬇️ Descargar video doblado",
                    f.read(),
                    file_name="video_doblado.mp4",
                )

        except Exception as e:
            st.error(str(e))

# app.py — Streamlit Cloud ready (sin depender de ffmpeg del sistema)
# EN → ES (Azure Translator) + Doblaje (Azure Speech TTS) + Whisper (ASR)
#
# Incluye:
#  - Título con banderas (Twemoji) estable en cualquier SO/Cloud
#  - Fuente de vídeo: URL/ruta, carpeta local (si ejecutas en tu PC), o subir archivo
#  - Acciones: Obtener texto EN, obtener traducción ES, o doblar vídeo
#  - Tras transcribir/traducir, permite doblar SIN volver a pulsar "Procesar"
#  - ffmpeg portátil con imageio-ffmpeg
#  - Whisper sin ffmpeg interno (le pasamos audio como numpy)
#  - Doblaje sincronizado por clústeres (timestamps Whisper) + ajuste de rate SSML

import os
import re
import io
import uuid
import shutil
import time
import wave
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
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        return None

AZURE_TRANSLATOR_KEY = get_secret("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = get_secret("AZURE_TRANSLATOR_REGION")

AZURE_SPEECH_KEY = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = get_secret("AZURE_SPEECH_REGION")

# Si no defines claves específicas del traductor, reutiliza Speech Key/Region (si tu recurso lo permite)
if not AZURE_TRANSLATOR_KEY:
    AZURE_TRANSLATOR_KEY = AZURE_SPEECH_KEY
if not AZURE_TRANSLATOR_REGION:
    AZURE_TRANSLATOR_REGION = AZURE_SPEECH_REGION


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
# Entrada vídeo: URL/ruta, carpeta local, subir archivo
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
# Azure Translator (batch)
# =============================================================================
def azure_translate_batch(texts: List[str], from_lang="en", to_lang="es") -> List[str]:
    if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_REGION:
        raise RuntimeError("Faltan AZURE_TRANSLATOR_KEY / AZURE_TRANSLATOR_REGION (Secrets/ENV). Puedes usar las mismas que AZURE_SPEECH_KEY / AZURE_SPEECH_REGION si tu configuración lo permite.")

    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = "/translate"
    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    body = [{"text": t or ""} for t in texts]
    r = requests.post(endpoint + path, params=params, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Azure Translator falló ({r.status_code}): {r.text[:800]}")
    data = r.json()
    return [item["translations"][0]["text"] for item in data]

def translate_segments_azure(texts: List[str], progress=None) -> List[str]:
    BATCH = 40
    out: List[str] = []
    n = len(texts)
    for i in range(0, n, BATCH):
        chunk = texts[i:i+BATCH]
        out.extend(azure_translate_batch(chunk, "en", "es"))
        if progress is not None and n:
            progress.progress(min(1.0, (i + len(chunk)) / n))
    return out

def translate_fulltext_azure(text: str) -> str:
    words = (text or "").split()
    chunks = []
    cur = []
    size = 0
    for w in words:
        if size + len(w) + 1 > 4500 and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            size = len(w) + 1
        else:
            cur.append(w)
            size += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return " ".join(azure_translate_batch(chunks, "en", "es"))


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

SYNC_OFFSET_MS = 120
JOIN_GAP_MS = 450
CLUSTER_MAX_MS = 11000
GUARD_MS = 30
TAIL_MARGIN_MS = 60
MIN_WINDOW_MS = 300
RATE_MIN_PCT = -12
RATE_MAX_PCT = +20
ATEMPO_LAST_RESORT = 1.20

SENT_END_RE = re.compile(r'[.!?…:;]\s*$')

def ends_strong_punct(s: str) -> bool:
    return bool(SENT_END_RE.search((s or "").strip()))

def cluster_by_english(segments: List[Dict], texts_es: List[str]) -> List[Dict]:
    clusters = []
    i = 0
    n = len(segments)
    while i < n:
        start_ms = int(float(segments[i]["start"]) * 1000)
        end_ms = int(float(segments[i]["end"]) * 1000)
        parts = [texts_es[i].strip()]
        j = i
        while j + 1 < n:
            gap_ms = int((float(segments[j+1]["start"]) - float(segments[j]["end"])) * 1000)
            next_end = int(float(segments[j+1]["end"]) * 1000)
            dur_if = next_end - start_ms
            en_prev = (segments[j].get("text") or "").strip()
            if ends_strong_punct(en_prev):
                break
            if gap_ms > JOIN_GAP_MS:
                break
            if dur_if > CLUSTER_MAX_MS:
                break
            j += 1
            end_ms = next_end
            parts.append(texts_es[j].strip())
        clusters.append({"start_ms": start_ms, "end_ms": end_ms, "text": " ".join(p for p in parts if p)})
        i = j + 1
    return clusters

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

        data = tts_ssml_bytes(text, voice, rate)
        last_bytes = data
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="wav")
        dur_ms = len(audio_seg)

        if abs(dur_ms - window_ms) <= 200:
            if dur_ms < window_ms:
                audio_seg += AudioSegment.silent(duration=(window_ms - dur_ms))
            elif dur_ms > window_ms:
                audio_seg = audio_seg[:window_ms]
            return audio_seg

    if len(audio_seg) > window_ms:
        tmp_in = str(Path(tempfile.gettempdir()) / f"tts_in_{uuid.uuid4().hex}.wav")
        tmp_out = str(Path(tempfile.gettempdir()) / f"tts_out_{uuid.uuid4().hex}.wav")
        AudioSegment.from_file(io.BytesIO(last_bytes), format="wav").export(tmp_in, format="wav")
        factor = min(ATEMPO_LAST_RESORT, len(audio_seg) / float(window_ms))
        run_ffmpeg(["-y", "-i", tmp_in, "-filter:a", atempo_chain(factor), tmp_out])
        audio_seg = AudioSegment.from_file(tmp_out)
        if len(audio_seg) > window_ms:
            audio_seg = audio_seg[:window_ms]
        try:
            os.remove(tmp_in); os.remove(tmp_out)
        except Exception:
            pass
        return audio_seg

    return audio_seg + AudioSegment.silent(duration=(window_ms - len(audio_seg)))

def build_dubbed_timeline(video_file: str, segments: List[Dict], texts_es: List[str], voice: str, progress=None) -> str:
    vdur = parse_duration_from_ffmpeg(video_file)
    if vdur <= 0:
        raise RuntimeError("No se pudo medir la duración del vídeo (ffmpeg).")
    video_ms = int(vdur * 1000)

    clusters = cluster_by_english(segments, texts_es)
    final = AudioSegment.silent(duration=video_ms + 200)

    for idx, cl in enumerate(clusters):
        start_nom = cl["start_ms"] + SYNC_OFFSET_MS
        end_nom = cl["end_ms"]
        next_start = (clusters[idx+1]["start_ms"] + SYNC_OFFSET_MS) if idx+1 < len(clusters) else video_ms

        place_ms = max(0, start_nom)
        end_allowed = min(end_nom - TAIL_MARGIN_MS, next_start - GUARD_MS)
        if end_allowed < place_ms + MIN_WINDOW_MS:
            end_allowed = place_ms + MIN_WINDOW_MS
        if end_allowed > video_ms:
            end_allowed = video_ms
        window_ms = max(150, end_allowed - place_ms)

        speech = tts_fit_to_window(cl["text"], voice, window_ms)
        final = final.overlay(speech, position=place_ms)

        if progress is not None and len(clusters):
            progress.progress((idx + 1) / len(clusters))

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
st.set_page_config(page_title="Doblador EN→ES (Azure)", page_icon="🎬", layout="centered")
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

render_title_text_first()
st.caption("por Miguel Ángel Gómez Ortiz")


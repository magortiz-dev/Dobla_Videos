# app.py (Streamlit Cloud ready) — EN -> ES
# - Descarga YouTube con yt-dlp (usa ffmpeg portátil de imageio-ffmpeg)
# - Transcripción Whisper (sin llamar a ffmpeg dentro de whisper: le pasamos el audio como numpy)
# - Traducción con Azure AI Translator
# - Doblaje con Azure Speech (es-ES)
# - Mux final con ffmpeg portátil, ajustando el audio a la duración del vídeo (sin recortar vídeo)

import os
import re
import io
import uuid
import shutil
import subprocess
from pathlib import Path

import requests
import streamlit as st
import numpy as np
from scipy.io import wavfile

import whisper
import yt_dlp

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_OK = True
except Exception:
    AZURE_SPEECH_OK = False


# =============================================================================
# FFmpeg portátil (NO depende del sistema)
# =============================================================================
FFMPEG_BIN = None

def setup_portable_ffmpeg() -> str:
    global FFMPEG_BIN
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
        return FFMPEG_BIN

    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

    ff_dir = str(Path(FFMPEG_BIN).parent)
    os.environ["PATH"] = ff_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = FFMPEG_BIN
    return FFMPEG_BIN

setup_portable_ffmpeg()

def run_ffmpeg(args: list[str]):
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


# =============================================================================
# Secrets / ENV
# =============================================================================
def get_secret(name: str) -> str | None:
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


# =============================================================================
# 1) Descargar vídeo (yt-dlp) — usa ffmpeg_location apuntando al portátil
# =============================================================================
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def download_video(url: str) -> str:
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
    return fn


# =============================================================================
# 2) Extraer audio a WAV 16k mono PCM
# =============================================================================
def extract_audio(video_file: str, audio_file="audio.wav") -> str:
    run_ffmpeg(["-y", "-i", video_file, "-ac", "1", "-ar", "16000", "-vn",
                "-acodec", "pcm_s16le", audio_file])
    return audio_file


# =============================================================================
# 3) Transcribir Whisper (sin ffmpeg interno)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    return whisper.load_model(model_size, device="cpu")

def transcribe_audio(audio_wav: str, model_size="small") -> str:
    sr, pcm = wavfile.read(audio_wav)
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if sr != 16000:
        fixed = audio_wav + ".16k.wav"
        run_ffmpeg(["-y", "-i", audio_wav, "-ac", "1", "-ar", "16000",
                    "-acodec", "pcm_s16le", fixed])
        sr, pcm = wavfile.read(fixed)
    audio = pcm.astype(np.float32) / 32768.0

    model = load_whisper(model_size)
    result = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        temperature=0.0,
        best_of=5,
        beam_size=5,
        condition_on_previous_text=False,
        fp16=False,
    )
    return result.get("text", "")


# =============================================================================
# 4) Traducir con Azure Translator (Text API v3)
# =============================================================================
def translate_text_azure(text: str, from_lang="en", to_lang="es") -> str:
    if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_REGION:
        raise RuntimeError("Faltan AZURE_TRANSLATOR_KEY / AZURE_TRANSLATOR_REGION (ENV o st.secrets).")

    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = "/translate"
    params = {"api-version": "3.0", "from": from_lang, "to": to_lang}
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    body = [{"text": text}]
    r = requests.post(endpoint + path, params=params, headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Azure Translator falló ({r.status_code}): {r.text[:500]}")
    data = r.json()
    return data[0]["translations"][0]["text"]


# =============================================================================
# 5) TTS Azure (SSML) -> WAV
# =============================================================================
def text_to_speech_azure(text: str, output="final_es.wav", voice="es-ES-DarioNeural") -> str:
    if not AZURE_SPEECH_OK:
        raise RuntimeError("No está instalado azure-cognitiveservices-speech.")
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (ENV o st.secrets).")

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    ssml = f"""<speak version="1.0" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="es-ES">
<voice name="{voice}">
<mstts:express-as style="newscast-casual">
<prosody rate="-4%">{text}</prosody>
</mstts:express-as>
</voice>
</speak>"""

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(f"Azure TTS falló: {result.reason}")
    return output


# =============================================================================
# 6) Ajustar audio a duración del vídeo y mux
# =============================================================================
def fit_audio_to_video(video_file: str, tts_wav: str, out_wav: str = "tts_fit.wav") -> str:
    vdur = parse_duration_from_ffmpeg(video_file)
    if vdur <= 0:
        raise RuntimeError("No se pudo medir la duración del vídeo (ffmpeg).")

    run_ffmpeg([
        "-y", "-i", tts_wav,
        "-af", f"apad,atrim=0:{vdur:.3f}",
        "-t", f"{vdur:.3f}",
        out_wav
    ])
    return out_wav

def replace_audio(video_file: str, audio_wav: str, output="video_doblado.mp4") -> str:
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
        output
    ])
    return output


# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="Doblador EN→ES (Azure)", page_icon="🎬", layout="centered")
st.title("🎬 Traductor y Doblador de Videos (EN ➝ ES) — Azure")

with st.expander("✅ Estado de credenciales / ffmpeg", expanded=False):
    st.write(f"FFmpeg portátil: {'✅' if FFMPEG_BIN else '❌'}")
    st.write(f"Azure Translator: {'✅' if (AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION) else '❌'}")
    st.write(f"Azure Speech (TTS): {'✅' if (AZURE_SPEECH_OK and AZURE_SPEECH_KEY and AZURE_SPEECH_REGION) else '❌'}")

url = st.text_input("Introduce la URL del video (YouTube):")
model_size = st.selectbox("Modelo Whisper", ["base", "small", "medium"], index=1)
voice = st.selectbox("Voz Azure (ES)", ["es-ES-DarioNeural", "es-ES-AlvaroNeural", "es-ES-ElviraNeural"], index=0)

if st.button("Procesar"):
    if not url:
        st.error("Introduce una URL.")
        st.stop()

    try:
        with st.spinner("Descargando video..."):
            video_file = download_video(url)

        with st.spinner("Extrayendo audio..."):
            audio_file = extract_audio(video_file)

        with st.spinner("Transcribiendo con Whisper (EN)..."):
            transcript_en = transcribe_audio(audio_file, model_size=model_size)
            st.subheader("📝 Transcripción (EN):")
            st.write(transcript_en)

        with st.spinner("Traduciendo al español (Azure Translator)..."):
            transcript_es = translate_text_azure(transcript_en, from_lang="en", to_lang="es")
            st.subheader("🌍 Traducción (ES):")
            st.write(transcript_es)

        with st.spinner("Generando doblaje (Azure TTS)..."):
            tts_wav = text_to_speech_azure(transcript_es, output="final_es.wav", voice=voice)

        with st.spinner("Ajustando audio a duración del vídeo..."):
            tts_fit = fit_audio_to_video(video_file, tts_wav, out_wav="tts_fit.wav")

        with st.spinner("Montando video final con doblaje..."):
            video_final = replace_audio(video_file, tts_fit, output="video_doblado.mp4")

        st.success("✅ Proceso completado")
        st.video(video_final)
        st.download_button("⬇️ Descargar video doblado", open(video_final, "rb"), file_name="video_doblado.mp4")

    except Exception as e:
        st.error(str(e))

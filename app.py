# app.py — Entrada estable (archivo/Drive/Dropbox/OneDrive/HTTP) + YouTube plan C
# Traducción Google Cloud (si hay credenciales) o fallback, TTS Azure (es-ES),
# sincronía exacta audio↔vídeo, cookies/proxy desde st.secrets para yt-dlp.

import os, re, io, glob, shutil, tempfile, subprocess, pathlib, json, time
import requests
import numpy as np
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile
from deep_translator import GoogleTranslator

# ---------- ASR ----------
import whisper

# ---------- TTS Azure ----------
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_OK = True
except Exception:
    AZURE_OK = False

# ---------- Descargas YouTube ----------
import yt_dlp

# ---------- Preferir ffmpeg del sistema; fallback a imageio-ffmpeg ----------
FFPROBE_BIN = None
FFMPEG_BIN = None

def _setup_ffmpeg():
    """Intenta usar ffmpeg del sistema; si no existe, usa el portátil de imageio-ffmpeg."""
    global FFMPEG_BIN
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
    else:
        # Fallback portátil
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        # Asegura PATH para librerías que llaman por nombre
        os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN

    # Registrar en PyDub
    AudioSegment.converter = FFMPEG_BIN

def _ffmpeg_ok():
    import shutil as _sh
    # ✅ usa FFMPEG_BIN correcto, no "_BIN"
    return bool(FFMPEG_BIN) or (_sh.which("ffmpeg") is not None)

_setup_ffmpeg()

def _to_float(s: str) -> float:
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except:
        return 0.0

def _ffprobe_text(args):
    if not FFPROBE_BIN:
        return ""
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        return out.decode(errors="ignore")
    except Exception:
        return ""

def _probe_duration(path: str) -> float:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return 0.0
        seg = AudioSegment.from_file(path)
        return len(seg) / 1000.0
    except Exception:
        return 0.0

def ensure_video_ok(video_path: str) -> str:
    try:
        d = _probe_duration(video_path)
        if d > 0.1 and video_path.lower().endswith(".mp4"):
            return video_path
    except Exception:
        pass
    remux = os.path.splitext(video_path)[0] + "_genpts.mp4"
    subprocess.run(
        [_BIN or "", "-y", "-i", video_path,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", remux],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return remux if os.path.exists(remux) and os.path.getsize(remux) > 0 else video_path

# ---------- yt-dlp: cookies/proxy desde Secrets/ENV ----------
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

# ---------- Detectores de origen ----------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _is_youtube(url: str) -> bool:
    return bool(re.search(r'(youtube\.com|youtu\.be)', url or '', re.I))

def _is_gdrive(url: str) -> bool:
    return 'drive.google.com' in (url or '') or 'docs.google.com/uc' in (url or '')

def _is_dropbox(url: str) -> bool:
    return 'dropbox.com' in (url or '')

def _is_onedrive(url: str) -> bool:
    return ('1drv.ms' in (url or '')) or ('onedrive.live.com' in (url or '')) or ('sharepoint.com' in (url or ''))

# ---------- Descargas estables ----------
def _download_http(url: str, out_path: str, chunk=1<<20) -> str:
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)
    return out_path

def _gdrive_file_id(url: str) -> str | None:
    m = re.search(r'/d/([A-Za-z0-9_-]{10,})', url)
    if m: return m.group(1)
    m = re.search(r'[?&]id=([A-Za-z0-9_-]{10,})', url)
    if m: return m.group(1)
    return None

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
    if 'dl=' in url:
        return re.sub(r'dl=\d', 'dl=1', url)
    sep = '&' if '?' in url else '?'
    return f"{url}{sep}dl=1"

def _onedrive_direct(url: str) -> str:
    if re.search(r'[?&]download=1', url):
        return url
    sep = '&' if '?' in url else '?'
    return f"{url}{sep}download=1"

def download_video_stable(url: str) -> str:
    """Stable: Drive/Dropbox/OneDrive/HTTP directo (sin cookies)."""
    tmp_dir = tempfile.gettempdir()
    base = f"video_in_{int(time.time())}"
    # 1) Google Drive
    if _is_gdrive(url):
        path = os.path.join(tmp_dir, base + ".bin")
        download_from_gdrive(url, path)
        return ensure_video_ok(path)
    # 2) Dropbox
    if _is_dropbox(url):
        durl = _dropbox_direct(url)
        path = os.path.join(tmp_dir, base + ".mp4")
        _download_http(durl, path)
        return ensure_video_ok(path)
    # 3) OneDrive/SharePoint
    if _is_onedrive(url):
        durl = _onedrive_direct(url)
        path = os.path.join(tmp_dir, base + ".mp4")
        _download_http(durl, path)
        return ensure_video_ok(path)
    # 4) HTTP/HTTPS directo
    if re.match(r'^https?://', url, re.I):
        ext = '.mp4' if '.mp4' in url.lower() else ('.webm' if '.webm' in url.lower() else '.bin')
        path = os.path.join(tmp_dir, base + ext)
        _download_http(url, path)
        return ensure_video_ok(path)
    raise RuntimeError("URL no soportada para descarga directa.")

def download_youtube(url: str) -> str:
    """Plan C para YouTube. Usa cookies/proxy si están en Secrets/ENV."""
    if not __ok():
        raise RuntimeError(" no encontrado.")
    ydl_base = {
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": True,
        "noplaylist": True,
        "retries": 8,
        "fragment_retries": 8,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "http_headers": {"User-Agent": UA},
        "extractor_args": {"youtube": {"player_client": ["web","android","ios","tv"]}},
        **({"_location": os.path.dirname(_BIN)} if _BIN else {}),
        "postprocessors": [{"key": "VideoConvertor", "preferedformat": "mp4"}],
        "postprocessor_args": {"VideoConvertor": ["-movflags", "faststart"]},
        "allow_multiple_video_streams": False,
        "allow_multiple_audio_streams": False,
        "format_sort": [
            "proto:https", "ext:mp4:m4a", "vcodec:h264:avc1", "acodec:aac:mp4a", "res", "tbr"
        ],
        "compat_opts": ["format-sort-force"],
        "sleep_interval_requests": 0.5,
        "throttled_rate": 1024 * 1024,
    }
    if YTDLP_COOKIEFILE:
        ydl_base["cookiefile"] = YTDLP_COOKIEFILE
    if YTDLP_PROXY:
        ydl_base["proxy"] = YTDLP_PROXY

    attempts = [
        "bv*+ba/b[ext=mp4]/b[ext=mp4]",
        "bestvideo*+bestaudio*/best",
        "best",
    ]
    last_err = None
    for fmt in attempts:
        try:
            opts = dict(ydl_base); opts["format"] = fmt
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
    # 1) Si el usuario sube archivo, priorizarlo
    if uploaded_path:
        p = pathlib.Path(uploaded_path)
        if p.suffix.lower() in {".mp4",".webm",".mkv",".mov"}:
            return ensure_video_ok(str(p.resolve()))
        out = p.with_suffix(".mp4")
        subprocess.run([_BIN or "","-y","-i",str(p),
                        "-c:v","copy","-c:a","aac","-b:a","192k",
                        "-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))
    # 2) URL
    s = (user_in or "").strip().strip('"').strip("'")
    if not s:
        raise RuntimeError("Proporciona una URL o sube un archivo de vídeo.")
    if re.match(r'^https?://', s, re.I):
        if _is_youtube(s):
            # Menos estable en cloud; si falla → sugerir subir archivo
            return download_youtube(s)
        return download_video_stable(s)
    # 3) Ruta local (solo ejecución local)
    if pathlib.Path(s).exists():
        p = pathlib.Path(s)
        if p.suffix.lower() in {".mp4",".webm",".mkv",".mov"}:
            return ensure_video_ok(str(p.resolve()))
        out = p.with_suffix(".mp4")
        subprocess.run([_BIN or "","-y","-i",str(p),
                        "-c:v","copy","-c:a","aac","-b:a","192k",
                        "-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))
    raise RuntimeError("Entrada no válida. Sube un archivo o pega una URL directa/YouTube.")

# ---------- Audio / ASR / Traducción / TTS ----------
def extract_audio(video: str, out="audio.wav") -> str:
    subprocess.run([FFMPEG_BIN or "ffmpeg","-y","-i",video,"-ac","1","-ar","16000","-vn",
                    "-acodec","pcm_s16le", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def transcribe_segments(audio: str, model_size='small'):
    model = whisper.load_model(model_size, device='cpu')
    res = model.transcribe(audio, language='en', task='transcribe',
                           temperature=0.0, best_of=5, beam_size=5,
                           condition_on_previous_text=False, fp16=False)
    return res.get('segments', []), res.get('text', '')

def translate_google_cloud(texts):
    from google.cloud import translate_v2 as translate
    client = translate.Client()
    if isinstance(texts, str):
        texts = [texts]
    outs = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            outs.append(""); continue
        parts, chunk, total = [], [], 0
        for w in t.split():
            lw = len(w) + 1
            if total + lw > 4500:
                parts.append(" ".join(chunk)); chunk=[w]; total=lw
            else:
                chunk.append(w); total += lw
        if chunk: parts.append(" ".join(chunk))
        segs=[]
        for p in parts:
            r = client.translate(p, source_language='en', target_language='es', format_='text')
            segs.append(r['translatedText'])
        outs.append(" ".join(segs))
    return outs

def translate_fallback(texts):
    gt = GoogleTranslator(source="en", target="es")
    if isinstance(texts, str):
        texts = [texts]
    outs=[]
    for t in texts:
        t=(t or "").strip()
        if not t:
            outs.append(""); continue
        parts, chunk, total = [], [], 0
        for w in t.split():
            lw = len(w) + 1
            if total + lw > 4500:
                parts.append(" ".join(chunk)); chunk=[w]; total=lw
            else:
                chunk.append(w); total += lw
        if chunk: parts.append(" ".join(chunk))
        outs.append(" ".join(gt.translate(p) for p in parts if p.strip()))
    return outs

def translate_en2es(text: str) -> str:
    # Usa Google Cloud si hay credenciales, si no fallback
    has_gcp_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    # También soporta secrets[gcp_service_account]
    if not has_gcp_env:
        try:
            svc = st.secrets["gcp_service_account"]
            # Escribir a fichero temporal para la lib oficial
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
            json.dump(dict(svc), tf); tf.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
            has_gcp_env = True
        except Exception:
            has_gcp_env = False
    if has_gcp_env:
        try:
            return translate_google_cloud(text)[0]
        except Exception:
            pass
    return translate_fallback(text)[0]

def get_azure_creds():
    # lee de ENV o de st.secrets (sin UI)
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if (not key or not region) and hasattr(st, "secrets"):
        try:
            key = key or st.secrets.get("AZURE_SPEECH_KEY")
            region = region or st.secrets.get("AZURE_SPEECH_REGION")
        except Exception:
            pass
    return key, region

def tts_azure(text: str, voice="es-ES-DarioNeural", outfile="tts_es.wav", rate_pct=-4):
    if not AZURE_OK:
        raise RuntimeError("azure-cognitiveservices-speech no instalado.")
    key, region = get_azure_creds()
    if not key or not region:
        raise RuntimeError("Faltan AZURE_SPEECH_KEY / AZURE_SPEECH_REGION")
    text = (text or "").strip()
    if not text:
        wavfile.write(outfile, 24000, np.zeros(int(0.05*24000), dtype=np.int16))
        return outfile
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
    audio = bytes(res.audio_data)
    AudioSegment.from_file(io.BytesIO(audio), format="wav").export(outfile, format="wav")
    return outfile

def _atempo_chain(factor: float):
    if factor <= 0: factor = 1.0
    chain=[]; f=factor
    while f < 0.5 or f > 2.0:
        step = 0.5 if f < 0.5 else 2.0
        chain.append(f"atempo={step}"); f = f/step
    chain.append(f"atempo={f}")
    return ",".join(chain)

def fit_audio_to_video(video: str, audio_in: str, audio_out: str) -> str:
    v = _probe_duration(video); a = _probe_duration(audio_in)
    if v <= 0:
        video = ensure_video_ok(video); v = _probe_duration(video)
    if v <= 0: raise RuntimeError("No se pudo medir duración del vídeo.")
    if a <= 0: raise RuntimeError("Audio TTS vacío.")
    if abs(a - v) < 0.01:
        shutil.copyfile(audio_in, audio_out); return audio_out
    if a > v:
        factor = a / v
        subprocess.run([FFMPEG_BIN or "ffmpeg","-y","-i",audio_in,"-filter:a",_atempo_chain(factor), audio_out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        pad = (v - a) + 0.05
        tmp = audio_out + ".tmp.wav"
        subprocess.run([FFMPEG_BIN or "ffmpeg","-y","-i",audio_in,"-af",f"apad=pad_dur={pad:.3f}", "-t", f"{v:.3f}", tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(tmp, audio_out)
    subprocess.run([FFMPEG_BIN or "ffmpeg","-y","-i",audio_out,"-t",f"{v:.3f}", audio_out + ".fix.wav"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    shutil.move(audio_out + ".fix.wav", audio_out)
    return audio_out

def mux_video_audio(video: str, audio: str, out="video_doblado.mp4") -> str:
    subprocess.run([FFMPEG_BIN or "ffmpeg","-y","-i",video,"-i",audio,
                    "-map","0:v:0","-map","1:a:0","-c:v","copy",
                    "-c:a","aac","-b:a","192k","-movflags","+faststart", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- UI ----------
st.set_page_config(page_title="Doblador EN→ES estable", page_icon="🎬", layout="centered")
st.title("🎬 Doblador EN→ES ")

st.markdown("**Entrada de vídeo** (recomendado: subir archivo o usar enlace directo).")
col_u, col_o = st.columns([2,1])
with col_u:
    source = st.text_input("Pega una URL directa (Drive/Dropbox/OneDrive/MP4) o de YouTube (menos estable):")
uploaded = st.file_uploader("…o sube un vídeo (mp4/webm/mkv/mov)", type=["mp4","webm","mkv","mov"])

st.caption(f"ffmpeg: {'✅' if _ffmpeg_ok() else '❌'}  |  Cookies YouTube: {'✅' if YTDLP_COOKIEFILE else '—'}")

accion = st.radio("Acción", ["Obtener el texto en inglés","Obtener la traducción a español","Hacer el doblaje del video"], index=2)

colA,colB = st.columns(2)
with colA:
    model = st.selectbox("Modelo Whisper", ["medium","small","base"], index=1)
with colB:
    voice = st.selectbox("Voz TTS (Azure)", ["es-ES-DarioNeural","es-ES-AlvaroNeural","es-ES-TeoNeural",
                                             "es-ES-ArnauNeural","es-ES-ElviraNeural","es-ES-LiaNeural"], index=0)

if st.button("Procesar"):
    try:
        # Resolver fuente estable
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
            segments, full_en = transcribe_segments(audio_wav, model_size=model)

        if accion == "Obtener el texto en inglés":
            st.success("✅ Transcripción (EN) lista.")
            st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                               file_name="transcripcion_en.txt", mime="text/plain")

        elif accion == "Obtener la traducción a español":
            with st.spinner("Traduciendo con Google..."):
                full_es = translate_en2es(full_en)
            st.success("✅ Traducción (ES) lista.")
            c1,c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                                   file_name="transcripcion_en.txt", mime="text/plain")
            with c2:
                st.download_button("⬇️ ES (.txt)", (full_es or "").encode("utf-8"),
                                   file_name="traduccion_es.txt", mime="text/plain")
            if AZURE_OK and (os.getenv("AZURE_SPEECH_KEY") or st.secrets.get("AZURE_SPEECH_KEY", None)):
                if st.button("Doblaje ahora con Azure TTS"):
                    with st.spinner("Sintetizando y ajustando al tiempo del vídeo..."):
                        wav = tts_azure(full_es, voice=voice, outfile="tts_es.wav", rate_pct=-4)
                        fit = fit_audio_to_video(video, wav, "tts_fit.wav")
                        out = mux_video_audio(video, fit, "video_doblado.mp4")
                    st.success("✅ Doblaje listo")
                    st.video(out)
                    st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                       file_name="video_doblado.mp4")

        else:  # Hacer el doblaje del video
            with st.spinner("Traduciendo con Google..."):
                full_es = translate_en2es(full_en)
            if not AZURE_OK:
                st.error("Instala azure-cognitiveservices-speech para doblar.")
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
        msg = str(e)
        if "403" in msg or "Forbidden" in msg or "Fallo descarga YouTube" in msg:
            st.error("YouTube ha bloqueado la descarga en la nube. Sube el archivo MP4 o usa un enlace directo (Drive/Dropbox/OneDrive).")
        else:
            st.error(msg)

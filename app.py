# app.py — Entrada estable (archivo/Drive/Dropbox/OneDrive/HTTP) + YouTube plan C
# Traducción Google Cloud (si hay credenciales) o fallback, TTS Azure (es-ES),
# SINCRONÍA por CLÚSTERES (fusión de segmentos + ajuste de rate SSML),
# cookies/proxy desde st.secrets para yt-dlp.

import os, re, io, glob, shutil, tempfile, subprocess, pathlib, json, time
import requests
import numpy as np
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile
from deep_translator import GoogleTranslator

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
    global FFMPEG_BIN, FFPROBE_BIN
    sys_ffmpeg  = shutil.which("ffmpeg")
    sys_ffprobe = shutil.which("ffprobe")

    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
        FFPROBE_BIN = sys_ffprobe
    else:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        guess_probe = os.path.join(os.path.dirname(FFMPEG_BIN), "ffprobe")
        FFPROBE_BIN = guess_probe if os.path.exists(guess_probe) else sys_ffprobe
        os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH","")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN

    AudioSegment.converter = FFMPEG_BIN
    if FFPROBE_BIN:
        AudioSegment.ffprobe = FFPROBE_BIN

def _ffmpeg_ok():
    return bool(FFMPEG_BIN) or (shutil.which("ffmpeg") is not None)

_setup_ffmpeg()

# --- Duración robusta: ffprobe -> parseo de ffmpeg -> pydub ---
def _to_float(s: str) -> float:
    s = (s or "").strip().replace(",", "."); 
    try: return float(s)
    except: return 0.0

def _duration_from_ffmpeg_stderr(txt: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt or "")
    if not m: return 0.0
    hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return hh*3600 + mm*60 + ss

def _probe_duration(path: str) -> float:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0: return 0.0
        if FFPROBE_BIN and os.path.exists(FFPROBE_BIN):
            for args in (
                [FFPROBE_BIN,"-v","error","-select_streams","v:0","-show_entries","stream=duration","-of","default=nokey=1:noprint_wrappers=1",path],
                [FFPROBE_BIN,"-v","error","-show_entries","format=duration","-of","default=nokey=1:noprint_wrappers=1",path],
            ):
                p=subprocess.run(args,capture_output=True,text=True)
                val=_to_float(p.stdout)
                if val>0: return val
        if FFMPEG_BIN:
            p=subprocess.run([FFMPEG_BIN,"-hide_banner","-i",path],capture_output=True,text=True)
            val=_duration_from_ffmpeg_stderr(p.stderr or "")
            if val>0: return val
        seg=AudioSegment.from_file(path)
        return len(seg)/1000.0
    except Exception:
        return 0.0

def ensure_video_ok(video_path: str) -> str:
    dur=_probe_duration(video_path)
    if dur>0.1 and video_path.lower().endswith(".mp4"): return video_path
    remux=os.path.splitext(video_path)[0]+"_genpts.mp4"
    subprocess.run([FFMPEG_BIN,"-y","-fflags","+genpts","-i",video_path,"-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart",remux],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if _probe_duration(remux)>0.1: return remux
    rec=os.path.splitext(video_path)[0]+"_reencode.mp4"
    subprocess.run([FFMPEG_BIN,"-y","-i",video_path,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","192k","-movflags","+faststart",rec],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return rec if _probe_duration(rec)>0.1 else video_path

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
        blob = cookies_blob.replace("\r\n","\n")
        if "Netscape HTTP Cookie File" not in (blob.splitlines() or [""])[0]:
            blob = "# Netscape HTTP Cookie File\n" + blob.lstrip()
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".cookies.txt", mode="w", encoding="utf-8")
        tf.write(blob); tf.close()
        YTDLP_COOKIEFILE = tf.name

_bootstrap_ytdlp_auth()

# ---------- Detectores ----------
UA=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
def _is_youtube(url:str)->bool: return bool(re.search(r'(youtube\.com|youtu\.be)', url or '', re.I))
def _is_gdrive(url:str)->bool:  return 'drive.google.com' in (url or '') or 'docs.google.com/uc' in (url or '')
def _is_dropbox(url:str)->bool: return 'dropbox.com' in (url or '')
def _is_onedrive(url:str)->bool:return ('1drv.ms' in (url or '')) or ('onedrive.live.com' in (url or '')) or ('sharepoint.com' in (url or ''))

# ---------- Descargas estables ----------
def _download_http(url: str, out_path: str, chunk=1<<20) -> str:
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": UA}) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b: f.write(b)
    return out_path

def _gdrive_file_id(url: str) -> str | None:
    m=re.search(r'/d/([A-Za-z0-9_-]{10,})', url); 
    if m: return m.group(1)
    m=re.search(r'[?&]id=([A-Za-z0-9_-]{10,})', url)
    return m.group(1) if m else None

def download_from_gdrive(url: str, out_path: str) -> str:
    try:
        import gdown
    except Exception:
        raise RuntimeError("Falta gdown. Añade gdown==5.2.0 a requirements.txt")
    fid=_gdrive_file_id(url)
    if fid: gdown.download(id=fid, output=out_path, quiet=True)
    else:   gdown.download(url=url, output=out_path, quiet=True)
    if not os.path.exists(out_path) or os.path.getsize(out_path)==0:
        raise RuntimeError("Descarga de Google Drive falló (archivo vacío).")
    return out_path

def _dropbox_direct(url: str) -> str:
    if 'dl=' in url: return re.sub(r'dl=\d','dl=1',url)
    sep='&' if '?' in url else '?'
    return f"{url}{sep}dl=1"

def _onedrive_direct(url: str) -> str:
    if re.search(r'[?&]download=1', url): return url
    sep='&' if '?' in url else '?'
    return f"{url}{sep}download=1"

def download_video_stable(url: str) -> str:
    tmp_dir=tempfile.gettempdir(); base=f"video_in_{int(time.time())}"
    if _is_gdrive(url):
        path=os.path.join(tmp_dir,base+".bin"); download_from_gdrive(url,path); return ensure_video_ok(path)
    if _is_dropbox(url):
        path=os.path.join(tmp_dir,base+".mp4"); _download_http(_dropbox_direct(url),path); return ensure_video_ok(path)
    if _is_onedrive(url):
        path=os.path.join(tmp_dir,base+".mp4"); _download_http(_onedrive_direct(url),path); return ensure_video_ok(path)
    if re.match(r'^https?://', url, re.I):
        ext='.mp4' if '.mp4' in url.lower() else ('.webm' if '.webm' in url.lower() else '.bin')
        path=os.path.join(tmp_dir,base+ext); _download_http(url,path); return ensure_video_ok(path)
    raise RuntimeError("URL no soportada para descarga directa.")

def download_youtube(url: str) -> str:
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
        **({"_location": os.path.dirname(FFMPEG_BIN)} if FFMPEG_BIN else {}),
        "postprocessors": [{"key": "VideoConvertor", "preferedformat": "mp4"}],
        "postprocessor_args": {"VideoConvertor": ["-movflags", "faststart"]},
        "allow_multiple_video_streams": False,
        "allow_multiple_audio_streams": False,
        "format_sort": ["proto:https","ext:mp4:m4a","vcodec:h264:avc1","acodec:aac:mp4a","res","tbr"],
        "compat_opts": ["format-sort-force"],
        "sleep_interval_requests": 0.5,
        "throttled_rate": 1024*1024,
    }
    if YTDLP_COOKIEFILE: ydl_base["cookiefile"]=YTDLP_COOKIEFILE
    if YTDLP_PROXY:      ydl_base["proxy"]=YTDLP_PROXY

    attempts=["bv*+ba/b[ext=mp4]/b[ext=mp4]","bestvideo*+bestaudio*/best","best"]
    last_err=None
    for fmt in attempts:
        try:
            opts=dict(ydl_base); opts["format"]=fmt
            with yt_dlp.YoutubeDL(opts) as ydl:
                info=ydl.extract_info(url, download=True)
                fn=ydl.prepare_filename(info)
                base,_=os.path.splitext(fn); mp4=base+".mp4"
                final=mp4 if os.path.exists(mp4) else fn
                return ensure_video_ok(final)
        except Exception as e:
            last_err=e
    raise RuntimeError(f"Fallo descarga YouTube: {last_err}")

def resolve_source(user_in: str, uploaded_path: str | None) -> str:
    if uploaded_path:
        p=pathlib.Path(uploaded_path)
        if p.suffix.lower() in {".mp4",".webm",".mkv",".mov"}:
            return ensure_video_ok(str(p.resolve()))
        out=p.with_suffix(".mp4")
        subprocess.run([FFMPEG_BIN,"-y","-i",str(p),"-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))
    s=(user_in or "").strip().strip('"').strip("'")
    if not s: raise RuntimeError("Proporciona una URL o sube un archivo de vídeo.")
    if re.match(r'^https?://', s, re.I):
        if _is_youtube(s): return download_youtube(s)
        return download_video_stable(s)
    if pathlib.Path(s).exists():
        p=pathlib.Path(s)
        if p.suffix.lower() in {".mp4",".webm",".mkv",".mov"}: return ensure_video_ok(str(p.resolve()))
        out=p.with_suffix(".mp4")
        subprocess.run([FFMPEG_BIN,"-y","-i",str(p),"-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart",str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ensure_video_ok(str(out))
    raise RuntimeError("Entrada no válida. Sube un archivo o pega una URL directa/YouTube.")

# ---------- Audio / ASR / Traducción / TTS ----------
import whisper

def extract_audio(video: str, out="audio.wav") -> str:
    subprocess.run([FFMPEG_BIN,"-y","-i",video,"-ac","1","-ar","16000","-vn","-acodec","pcm_s16le",out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def transcribe_segments(audio_wav_path: str, model_size: str = "small"):
    sr, pcm = wavfile.read(audio_wav_path)
    if pcm.ndim>1: pcm=pcm[:,0]
    if sr!=16000:
        fixed=audio_wav_path+".16k.wav"
        subprocess.run([FFMPEG_BIN,"-y","-i",audio_wav_path,"-ac","1","-ar","16000","-acodec","pcm_s16le",fixed],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        sr, pcm = wavfile.read(fixed)
    audio = pcm.astype(np.float32)/32768.0
    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(audio, language="en", task="transcribe",
                              temperature=0.0, best_of=5, beam_size=5,
                              condition_on_previous_text=False, fp16=False)
    return result.get("segments", []), result.get("text", "")

def translate_google_cloud(texts):
    from google.cloud import translate_v2 as translate
    client=translate.Client()
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
        segs=[]
        for p in parts:
            r=client.translate(p, source_language='en', target_language='es', format_='text')
            segs.append(r['translatedText'])
        outs.append(" ".join(segs))
    return outs

def translate_fallback(texts):
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

def translate_en2es(text: str) -> str:
    has_gcp_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    if not has_gcp_env:
        try:
            svc = st.secrets["gcp_service_account"]
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
            json.dump(dict(svc), tf); tf.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
            has_gcp_env = True
        except Exception:
            has_gcp_env = False
    if has_gcp_env:
        try: return translate_google_cloud(text)[0]
        except Exception: pass
    return translate_fallback(text)[0]

def translate_list_en2es(texts: list[str]) -> list[str]:
    has_gcp_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    if not has_gcp_env:
        try:
            svc = st.secrets["gcp_service_account"]
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
            json.dump(dict(svc), tf); tf.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
            has_gcp_env = True
        except Exception:
            has_gcp_env = False
    if has_gcp_env:
        try: return translate_google_cloud(texts)
        except Exception: pass
    return translate_fallback(texts)

def get_azure_creds():
    key=os.getenv("AZURE_SPEECH_KEY")
    region=os.getenv("AZURE_SPEECH_REGION")
    if (not key or not region) and hasattr(st,"secrets"):
        try:
            key = key or st.secrets.get("AZURE_SPEECH_KEY")
            region = region or st.secrets.get("AZURE_SPEECH_REGION")
        except Exception:
            pass
    return key, region

# =======================
#     NUEVA SINCRO PRO
# =======================

# --- Constantes de sincro fina / clústeres ---
SYNC_OFFSET_MS       = 120   # retardo global (evitar que la voz se adelante)
JOIN_GAP_MS          = 350   # si la separación entre segmentos es <= esto, se pueden unir
CLUSTER_MAX_MS       = 9000  # no generar clústeres demasiado largos
GUARD_MS             = 30    # separación mínima entre clústeres
TAIL_MARGIN_MS       = 60    # no ocupar el final exacto del último seg de clúster
MIN_WINDOW_MS        = 300   # mínima ventana de clúster
RATE_MIN_PCT         = -12   # hasta 12% más lento como máx (evitar voz pastosa)
RATE_MAX_PCT         = +25   # hasta 25% más rápido como máx
ATEMPO_LAST_RESORT   = 1.20  # compresión ligera final (máx) si aún no cabe

SENT_END_RE = re.compile(r'[.!?…:;]\s*$', re.UNICODE)

def _ends_with_strong_punct(txt: str) -> bool:
    return bool(SENT_END_RE.search((txt or "").strip()))

def _calc_gap_ms(seg_prev, seg_next) -> int:
    return int((float(seg_next["start"]) - float(seg_prev["end"])) * 1000)

def _cluster_segments(segments_en: list[dict], texts_es: list[str]):
    """
    Une segmentos consecutivos si:
      - NO termina con puntuación fuerte
      - gap <= JOIN_GAP_MS
      - duración total del clúster no supera CLUSTER_MAX_MS
    """
    clusters = []
    i = 0
    n = len(segments_en)
    while i < n:
        start_i = i
        start_ms = int(float(segments_en[i]["start"]) * 1000)
        end_ms   = int(float(segments_en[i]["end"])   * 1000)
        parts = [texts_es[i].strip()]
        j = i
        while j + 1 < n:
            gap = _calc_gap_ms(segments_en[j], segments_en[j+1])
            end_ms_next = int(float(segments_en[j+1]["end"]) * 1000)
            dur_if_join = (end_ms_next - start_ms)
            if _ends_with_strong_punct(parts[-1]):
                break
            if gap > JOIN_GAP_MS:
                break
            if dur_if_join > CLUSTER_MAX_MS:
                break
            # unir
            j += 1
            end_ms = end_ms_next
            parts.append(texts_es[j].strip())
        text_cluster = " ".join(p for p in parts if p)
        clusters.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text_cluster
        })
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
    """
    Intenta encajar el TTS en window_ms ajustando el rate SSML.
    1) rate inicial -4%
    2) si dur>window → subir rate (más rápido)
    3) si dur<window → si falta poco, rellenar con silencio; si falta mucho, bajar rate (más lento)
    Máx 3 intentos. Último recurso: atempo suave.
    """
    attempt_rates = [ -4 ]
    # heurística de 2 ajustes según ratio esperado
    # (no conocemos duración hasta sintetizar, así que iteramos)
    for _ in range(2):
        attempt_rates.append(None)  # será calculado según la medida

    audio_seg = None
    last_bytes = None
    used_rate = -4

    for k, rate in enumerate(attempt_rates):
        if rate is None:
            # calcular siguiente rate basándonos en la duración previa
            dur_ms = len(audio_seg) if audio_seg else 0
            if dur_ms == 0:
                rate = -4
            else:
                ratio = dur_ms / float(window_ms)
                if ratio > 1.05:   # demasiado largo → acelerar
                    # escalar a % pero suavizado
                    inc = min(RATE_MAX_PCT, int(min(25, (ratio-1.0)*100*0.85)))
                    rate = max(-4, inc)  # no bajamos por debajo de -4 en este caso
                elif ratio < 0.85: # demasiado corto → ralentizar
                    dec = max(RATE_MIN_PCT, - int(min(12, (1.0-ratio)*100*0.85)))
                    rate = min(-4, dec)  # mantener en rango [-12, -4]
                else:
                    rate = used_rate  # ya estamos cerca
        rate = int(max(RATE_MIN_PCT, min(RATE_MAX_PCT, rate)))
        used_rate = rate

        # sintetizar
        data = _azure_tts_bytes(text, voice, rate)
        last_bytes = data
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="wav")
        dur_ms = len(audio_seg)

        if abs(dur_ms - window_ms) <= 200:
            # perfecto o casi
            if dur_ms < window_ms:
                audio_seg = audio_seg + AudioSegment.silent(duration=(window_ms - dur_ms))
            elif dur_ms > window_ms:
                audio_seg = audio_seg[:window_ms]
            return audio_seg

        # si es más largo, intentaremos de nuevo con rate mayor; si más corto, con rate menor
        # (el bucle calcula el siguiente rate)

    # Último recurso: atempo suave si sigue largo
    if len(audio_seg) > window_ms:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf_out:
            AudioSegment.from_file(io.BytesIO(last_bytes), format="wav").export(tf_in.name, format="wav")
            factor = min(ATEMPO_LAST_RESORT, len(audio_seg)/float(window_ms))
            subprocess.run([FFMPEG_BIN,"-y","-i",tf_in.name,"-filter:a",f"atempo={factor:.3f}", tf_out.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            audio_seg = AudioSegment.from_file(tf_out.name)
            if len(audio_seg) > window_ms: audio_seg = audio_seg[:window_ms]
            try: os.remove(tf_in.name); os.remove(tf_out.name)
            except: pass
        return audio_seg
    # Si quedó corto, silenciamos
    return audio_seg + AudioSegment.silent(duration=(window_ms - len(audio_seg)))

def build_dubbed_audio_clusters(video_path: str,
                                segments_en: list[dict],
                                translations_es: list[str],
                                voice: str,
                                out_wav: str = "tts_timeline.wav") -> str:
    if not segments_en: raise RuntimeError("No hay segmentos de Whisper.")
    if len(translations_es)!=len(segments_en): raise RuntimeError("Desajuste segments↔translations.")
    video_ms = int(_probe_duration(video_path)*1000)
    if video_ms<=0:
        fixed=ensure_video_ok(video_path); video_ms=int(_probe_duration(fixed)*1000)
        if video_ms<=0: raise RuntimeError("No se pudo medir la duración del vídeo.")

    # 1) Preparar textos y fusionar
    texts_es=[(t or "").strip() for t in translations_es]
    clusters=_cluster_segments(segments_en, texts_es)

    # 2) Timeline base
    final = AudioSegment.silent(duration=video_ms + 200)

    for idx, cl in enumerate(clusters):
        start_nom = cl["start_ms"] + SYNC_OFFSET_MS
        end_nom   = cl["end_ms"]

        # límite superior: siguiente clúster (si existe)
        if idx+1 < len(clusters):
            next_start = clusters[idx+1]["start_ms"] + SYNC_OFFSET_MS
        else:
            next_start = video_ms

        place_ms = max(0, start_nom)
        end_allowed = min(end_nom - TAIL_MARGIN_MS, next_start - GUARD_MS)
        if end_allowed < place_ms + MIN_WINDOW_MS:
            end_allowed = place_ms + MIN_WINDOW_MS
        if end_allowed > video_ms:
            end_allowed = video_ms
        window_ms = max(150, end_allowed - place_ms)

        # 3) Sintetizar y ajustar al hueco con rate SSML
        speech = _tts_cluster_fit(cl["text"], voice, window_ms)

        # 4) Colocar en timeline
        final = final.overlay(speech, position=place_ms)

    # 5) Duración exacta
    if len(final) > video_ms: final = final[:video_ms]
    elif len(final) < video_ms: final = final + AudioSegment.silent(duration=(video_ms - len(final)))

    final.export(out_wav, format="wav")
    return out_wav

# --- Mux ---
def mux_video_audio(video: str, audio: str, out="video_doblado.mp4") -> str:
    subprocess.run([FFMPEG_BIN,"-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0",
                    "-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- UI ----------
st.set_page_config(page_title="Doblador EN→ES", page_icon="🎬", layout="centered")
st.title("🎬 Doblador Vídeos EN→ES ")
st.caption("por Miguel Ángel Gómez Ortiz")

st.markdown("**Entrada de vídeo** (recomendado: subir archivo o usar enlace directo).")
col_u, col_o = st.columns([2,1])
with col_u:
    source = st.text_input("Pega una URL directa (Drive/Dropbox/OneDrive/MP4) o de YouTube (menos estable):")
uploaded = st.file_uploader("…o sube un vídeo (mp4/webm/mkv/mov)", type=["mp4","webm","mkv","mov"])

accion = st.radio("Acción", ["Obtener el texto en inglés","Obtener la traducción a español","Hacer el doblaje del video"], index=2)

colA,colB = st.columns(2)
with colA:
    model = st.selectbox("Modelo Whisper", ["medium","small","base"], index=1)
with colB:
    voice = st.selectbox("Voz TTS (Azure)", ["es-ES-DarioNeural","es-ES-AlvaroNeural","es-ES-TeoNeural",
                                             "es-ES-ArnauNeural","es-ES-ElviraNeural","es-ES-LiaNeural"], index=0)

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
                if st.button("Doblaje segmentado (recomendado)"):
                    with st.spinner("Generando doblaje por clústeres y sincronizado..."):
                        texts_en = [ (s.get("text") or "").strip() for s in segments ]
                        texts_es = translate_list_en2es(texts_en)
                        wav_timeline = build_dubbed_audio_clusters(
                            video_path = video,
                            segments_en = segments,
                            translations_es = texts_es,
                            voice = voice,
                            out_wav = "tts_timeline.wav"
                        )
                        out = mux_video_audio(video, wav_timeline, "video_doblado.mp4")
                    st.success("✅ Doblaje listo")
                    st.video(out)
                    st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                       file_name="video_doblado.mp4")

        else:  # Hacer el doblaje del video
            with st.spinner("Traduciendo con Google por segmentos..."):
                texts_en = [ (s.get("text") or "").strip() for s in segments ]
                texts_es = translate_list_en2es(texts_en)
            if not AZURE_OK:
                st.error("Instala azure-cognitiveservices-speech para doblar.")
            else:
                with st.spinner("Generando doblaje por clústeres y sincronizado..."):
                    wav_timeline = build_dubbed_audio_clusters(
                        video_path = video,
                        segments_en = segments,
                        translations_es = texts_es,
                        voice = voice,
                        out_wav = "tts_timeline.wav"
                    )
                    out = mux_video_audio(video, wav_timeline, "video_doblado.mp4")
                st.success("✅ Doblaje listo")
                st.video(out)
                st.download_button("⬇️ Descargar video doblado", open(out,"rb"),
                                   file_name="video_doblado.mp4")
    except Exception as e:
        msg=str(e)
        if "403" in msg or "Forbidden" in msg or "Fallo descarga YouTube" in msg:
            st.error("YouTube ha bloqueado la descarga en la nube. Sube el archivo MP4 o usa un enlace directo (Drive/Dropbox/OneDrive).")
        else:
            st.error(msg)

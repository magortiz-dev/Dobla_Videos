# app.py — EN→ES privado con Hugging Face NLLB (sin Google), TTS Azure es-ES,
# entrada estable (archivo/Drive/Dropbox/OneDrive/HTTP/YouTube plan-C),
# sincronía por clústeres + SSML rate + atempo de último recurso.

import os, re, io, glob, shutil, tempfile, subprocess, pathlib, json, time
import requests
import numpy as np
import streamlit as st
from pydub import AudioSegment
from scipy.io import wavfile

# ---------- TTS Azure ----------
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_OK = True
except Exception:
    AZURE_OK = False

# ---------- ASR ----------
import yt_dlp
import whisper

# ---------- HF NLLB (Hugging Face) ----------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Preferir ffmpeg del sistema; fallback a imageio-ffmpeg
FFMPEG_BIN = None

def _setup_ffmpeg():
    global FFMPEG_BIN
    sys_ffmpeg  = shutil.which("ffmpeg")
    

    if sys_ffmpeg:
        FFMPEG_BIN = sys_ffmpeg
    else:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["PATH"] = os.path.dirname(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH","")
        os.environ["FFMPEG_BINARY"] = FFMPEG_BIN

    # Registrar SOLO ffmpeg en pydub
    AudioSegment.converter = FFMPEG_BIN

def _ffmpeg_ok():
    return bool(FFMPEG_BIN) or (shutil.which("ffmpeg") is not None)

_setup_ffmpeg()

# --- Título ---
def render_title_with_flags():
    GB = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ec-1f1e7.svg"  # 🇬🇧
    ES = "https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1ea-1f1f8.svg"  # 🇪🇸
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; margin-top:6px; margin-bottom:10px;">
          <span style="font-size:2rem; line-height:1;">🎬</span>
          <span style="font-size:1.8rem; font-weight:700; letter-spacing:0.2px;">
            Doblador de Vídeos
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

# --- Duración robusta ---
def _to_float(s: str) -> float:
    s = (s or "").strip().replace(",", ".")
    try: return float(s)
    except: return 0.0

def _duration_from_ffmpeg_stderr(txt: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt or "")
    if not m: return 0.0
    hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return hh*3600 + mm*60 + ss


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

UA=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
def _is_youtube(url:str)->bool: return bool(re.search(r'(youtube\.com|youtu\.be)', url or '', re.I))
def _is_gdrive(url:str)->bool:  return 'drive.google.com' in (url or '') or 'docs.google.com/uc' in (url or '')
def _is_dropbox(url:str)->bool: return 'dropbox.com' in (url or '')
def _is_onedrive(url:str)->bool:return ('1drv.ms' in (url or '')) or ('onedrive.live.com' in (url or '')) or ('sharepoint.com' in (url or ''))

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

# ---------- Audio / ASR ----------
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

# ---------- Traducción EN→ES con Hugging Face NLLB ----------
@st.cache_resource(show_spinner=False)
def _load_hf_translator():
    model_id = "facebook/nllb-200-distilled-600M"
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        try:
            hf_token = st.secrets.get("HF_TOKEN", None)
        except Exception:
            hf_token = None

    # ✅ clave: usar tokenizer "slow" para que soporte bien códigos NLLB
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=hf_token)

    tokenizer.src_lang = "eng_Latn"

    # ✅ NO usar lang_code_to_id; usar conversión de token a id
    forced_bos_id = tokenizer.convert_tokens_to_ids("spa_Latn")
    if forced_bos_id is None or forced_bos_id == tokenizer.unk_token_id:
        forced_bos_id = tokenizer.get_vocab().get("spa_Latn")

    if forced_bos_id is None:
        raise RuntimeError("No se pudo obtener forced_bos_token_id para spa_Latn (NLLB).")

    return model, tokenizer, forced_bos_id

def _chunk_by_tokens(text: str, tokenizer, max_tokens: int = 768):
    # Partimos por frases para mantener naturalidad
    sents = re.split(r'(?<=[\.\!\?\;\:\n])\s+', (text or "").strip())
    batches = []
    curr, curr_tok = [], 0
    for s in sents:
        if not s.strip(): continue
        n_tok = len(tokenizer.tokenize(s))
        if curr and (curr_tok + n_tok > max_tokens):
            batches.append(" ".join(curr))
            curr, curr_tok = [s], n_tok
        else:
            curr.append(s)
            curr_tok += n_tok
    if curr: batches.append(" ".join(curr))
    return batches

def translate_hf(text: str) -> str:
    model, tokenizer, forced_bos_id = _load_hf_translator()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    gen = model.generate(**enc, forced_bos_token_id=forced_bos_id, max_length=1024, num_beams=4)
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    outs=[]
    pieces = _chunk_by_tokens(text, tokenizer, max_tokens=768)
    for chunk in pieces:
        if not chunk.strip():
            outs.append("")
            continue
                  
    return " ".join(outs).strip()

def translate_list_hf(texts: list[str]) -> list[str]:
    # Procesa uno a uno para no romper contexto y mantener naturalidad por segmento
    return [translate_hf(t or "") for t in texts]

# ---------- TTS Azure ----------
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
#     SINCRO POR CLÚSTER
# =======================
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

def _cluster_segments(segments_en: list[dict], texts_es: list[str]):
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
        text_cluster = " ".join(p for p in parts if p)
        clusters.append({"start_ms": start_ms, "end_ms": end_ms, "text": text_cluster})
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
    attempt_rates = [ -4, None, None ]
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
                    inc = min(RATE_MAX_PCT, int(min(25, (ratio-1.0)*100*0.85)))
                    rate = max(-4, inc)
                elif ratio < 0.85:
                    dec = max(RATE_MIN_PCT, - int(min(12, (1.0-ratio)*100*0.85)))
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

    texts_es=[(t or "").strip() for t in translations_es]
    clusters=_cluster_segments(segments_en, texts_es)
    final = AudioSegment.silent(duration=video_ms + 200)

    for idx, cl in enumerate(clusters):
        start_nom = cl["start_ms"] + SYNC_OFFSET_MS
        end_nom   = cl["end_ms"]
        next_start = (clusters[idx+1]["start_ms"] + SYNC_OFFSET_MS) if (idx+1 < len(clusters)) else video_ms

        place_ms = max(0, start_nom)
        end_allowed = min(end_nom - TAIL_MARGIN_MS, next_start - GUARD_MS)
        if end_allowed < place_ms + MIN_WINDOW_MS:
            end_allowed = place_ms + MIN_WINDOW_MS
        if end_allowed > video_ms:
            end_allowed = video_ms
        window_ms = max(150, end_allowed - place_ms)

        speech = _tts_cluster_fit(cl["text"], voice, window_ms)
        final = final.overlay(speech, position=place_ms)

    if len(final) > video_ms: final = final[:video_ms]
    elif len(final) < video_ms: final = final + AudioSegment.silent(duration=(video_ms - len(final)))

    final.export(out_wav, format="wav")
    return out_wav

def mux_video_audio(video: str, audio: str, out="video_doblado.mp4") -> str:
    subprocess.run([FFMPEG_BIN,"-y","-i",video,"-i",audio,"-map","0:v:0","-map","1:a:0",
                    "-c:v","copy","-c:a","aac","-b:a","192k","-movflags","+faststart", out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

# ---------- UI ----------
st.set_page_config(page_title="Doblador EN→ES", page_icon="🎬", layout="centered")
render_title_with_flags()
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
            with st.spinner("Traduciendo (Hugging Face NLLB)…"):
                full_es = translate_hf(full_en)
            st.success("✅ Traducción (ES) lista.")
            c1,c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ EN (.txt)", (full_en or "").encode("utf-8"),
                                   file_name="transcripcion_en.txt", mime="text/plain")
            with c2:
                st.download_button("⬇️ ES (.txt)", (full_es or "").encode("utf-8"),
                                   file_name="traduccion_es.txt", mime="text/plain")

            if AZURE_OK and (os.getenv("AZURE_SPEECH_KEY") or (hasattr(st,"secrets") and st.secrets.get("AZURE_SPEECH_KEY", None))):
                if st.button("Doblaje segmentado (recomendado)"):
                    with st.spinner("Generando doblaje por clústeres y sincronizado..."):
                        texts_en = [ (s.get("text") or "").strip() for s in segments ]
                        texts_es = translate_list_hf(texts_en)
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
            with st.spinner("Traduciendo por segmentos (Hugging Face NLLB)…"):
                texts_en = [ (s.get("text") or "").strip() for s in segments ]
                texts_es = translate_list_hf(texts_en)
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

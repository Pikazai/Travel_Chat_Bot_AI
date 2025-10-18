import os
import io
import tempfile
import base64
import subprocess
import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment  # requires ffmpeg installed
from gtts import gTTS

# ---------- helper ----------
def detect_audio_type_header(b):
    # returns 'wav','ogg','webm','mp3','flac', or None
    if len(b) < 12:
        return None
    if b[0:4] == b'RIFF' and b[8:12] == b'WAVE':
        return 'wav'
    if b[0:4] == b'fLaC':
        return 'flac'
    if b[0:4] == b'OggS':
        return 'ogg'
    if b[0:4] == b'\x1f\x8b\x08\x00':  # gzip-ish (not typical)
        return None
    # webm container starts with 0x1A45DFA3
    if b[0:4] == b'\x1A\x45\xDF\xA3':
        return 'webm'
    # mp3 often has ID3 tag or frame header - check for 'ID3'
    if b[0:3] == b'ID3' or b[0] == 0xFF:
        return 'mp3'
    return None

def write_temp_file_and_convert_to_wav(audio_bytes):
    """
    - Save audio_bytes to temp file with detected extension.
    - If not wav, convert to wav using pydub (ffmpeg).
    - Return path_to_wav_file or raise Exception.
    """
    header = audio_bytes[:64]
    atype = detect_audio_type_header(header)
    # fallback extension
    ext_map = {
        'wav': '.wav',
        'ogg': '.ogg',
        'webm': '.webm',
        'mp3': '.mp3',
        'flac': '.flac'
    }
    ext = ext_map.get(atype, '.webm')

    tmp_dir = tempfile.mkdtemp()
    src_path = os.path.join(tmp_dir, "input" + ext)
    wav_path = os.path.join(tmp_dir, "converted.wav")

    with open(src_path, "wb") as f:
        f.write(audio_bytes)

    # If it's already wav, just return
    if atype == 'wav':
        return src_path

    # Try pydub (requires ffmpeg)
    try:
        audio = AudioSegment.from_file(src_path)  # autodetect by extension / file header
        audio = audio.set_frame_rate(16000).set_channels(1)  # recommended for speech_recognition
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        # Last resort: try ffmpeg subprocess directly (if installed)
        try:
            cmd = [
                "ffmpeg", "-y", "-i", src_path,
                "-ar", "16000", "-ac", "1", wav_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return wav_path
        except Exception as e2:
            raise RuntimeError(f"Không thể chuyển đổi audio sang WAV: {e} | {e2}")

# ---------- Streamlit UI ----------
st.title("Voice test - robust audio handling")

audio = mic_recorder(
    start_prompt="🎙️ Nhấn để nói",
    stop_prompt="🛑 Dừng",
    just_once=True,
    key="rec1"
)

if audio:
    st.info("Đã nhận dữ liệu âm thanh, đang xử lý...")
    audio_bytes = audio["bytes"]

    try:
        wav_file = write_temp_file_and_convert_to_wav(audio_bytes)
    except Exception as e:
        st.error(f"Không thể chuyển đổi audio: {e}")
        wav_file = None

    if wav_file:
        r = sr.Recognizer()
        try:
            with sr.AudioFile(wav_file) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language="vi-VN")
                st.success(f"🗣️ Bạn nói: {text}")

                # ví dụ: TTS phản hồi
                reply = f"Tôi nghe được: {text}. Chúc bạn một ngày vui vẻ!"
                tts = gTTS(reply, lang="vi")
                bio = io.BytesIO()
                tts.write_to_fp(bio)
                bio.seek(0)
                b64 = base64.b64encode(bio.read()).decode()
                st.markdown(f'<audio autoplay controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

        except sr.UnknownValueError:
            st.error("Không thể nhận diện giọng nói (UnknownValueError).")
        except Exception as e:
            st.error(f"Lỗi nhận diện: {e}")

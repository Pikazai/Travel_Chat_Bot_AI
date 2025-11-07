"""
Voice service for Speech-to-Text (STT) and Text-to-Speech (TTS).
Handles audio recording, conversion, and processing.
"""

import io
import base64
import tempfile
import subprocess
from typing import Optional
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

from config.settings import Settings


class VoiceService:
    """Service for voice input/output processing."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.recognizer = sr.Recognizer()
    
    def detect_audio_type(self, audio_bytes: bytes) -> Optional[str]:
        """Detect audio format from header bytes."""
        if len(audio_bytes) < 12:
            return None
        
        if audio_bytes[0:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        if audio_bytes[0:4] == b'fLaC':
            return 'flac'
        if audio_bytes[0:4] == b'OggS':
            return 'ogg'
        if audio_bytes[0:4] == b'\x1A\x45\xDF\xA3':
            return 'webm'
        if audio_bytes[0:3] == b'ID3' or audio_bytes[0] == 0xFF:
            return 'mp3'
        return None
    
    def convert_to_wav(self, audio_bytes: bytes) -> Optional[str]:
        """Convert audio bytes to WAV format."""
        header = audio_bytes[:64]
        atype = self.detect_audio_type(header)
        
        ext_map = {'wav': '.wav', 'ogg': '.ogg', 'webm': '.webm', 
                   'mp3': '.mp3', 'flac': '.flac'}
        ext = ext_map.get(atype, '.webm')
        
        tmp_dir = tempfile.mkdtemp()
        src_path = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=tmp_dir).name
        wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=tmp_dir).name
        
        try:
            with open(src_path, "wb") as f:
                f.write(audio_bytes)
            
            if atype == 'wav':
                return src_path
            
            # Try pydub conversion
            try:
                audio = AudioSegment.from_file(src_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(wav_path, format="wav")
                return wav_path
            except Exception:
                # Fallback to ffmpeg
                cmd = ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return wav_path
        except Exception as e:
            print(f"[WARN] Audio conversion failed: {e}")
            return None
    
    def speech_to_text(self, audio_bytes: bytes, language: str = None) -> Optional[str]:
        """Convert speech to text."""
        language = language or self.settings.ASR_LANGUAGE
        wav_file = self.convert_to_wav(audio_bytes)
        
        if not wav_file:
            return None
        
        try:
            with sr.AudioFile(wav_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=language)
                return text
        except sr.UnknownValueError:
            print("[WARN] Could not understand audio")
            return None
        except Exception as e:
            print(f"[WARN] Speech recognition error: {e}")
            return None
    
    def text_to_speech(self, text: str, language: str = None) -> Optional[str]:
        """Convert text to speech audio (base64 encoded)."""
        language = language or self.settings.TTS_LANGUAGE
        try:
            tts = gTTS(text, lang=language)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            return base64.b64encode(bio.read()).decode()
        except Exception as e:
            print(f"[WARN] TTS error: {e}")
            return None


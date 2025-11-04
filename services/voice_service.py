"""
Voice service for speech-to-text and text-to-speech functionality.
"""
import os
import io
import base64
import tempfile
import subprocess
from typing import Optional
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS


class VoiceService:
    """Service for voice input/output processing."""
    
    def __init__(self):
        """Initialize the voice service."""
        self.recognizer = sr.Recognizer()
    
    @staticmethod
    def detect_audio_type_header(audio_bytes: bytes) -> Optional[str]:
        """
        Detect audio format from header bytes.
        
        Args:
            audio_bytes: Audio file bytes
            
        Returns:
            Audio type ('wav', 'ogg', 'webm', 'mp3', 'flac') or None
        """
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
    
    def convert_to_wav(self, audio_bytes: bytes) -> str:
        """
        Convert audio bytes to WAV format.
        
        Args:
            audio_bytes: Audio file bytes
            
        Returns:
            Path to WAV file
            
        Raises:
            RuntimeError: If conversion fails
        """
        header = audio_bytes[:64]
        atype = self.detect_audio_type_header(header)
        
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
        
        # If already WAV, return
        if atype == 'wav':
            return src_path
        
        # Try pydub first
        try:
            audio = AudioSegment.from_file(src_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            # Fallback to ffmpeg subprocess
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", src_path,
                    "-ar", "16000", "-ac", "1", wav_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return wav_path
            except Exception as e2:
                raise RuntimeError(f"Không thể chuyển đổi audio sang WAV: {e} | {e2}")
    
    def speech_to_text(self, audio_bytes: bytes, language: str = "vi-VN") -> Optional[str]:
        """
        Convert speech to text.
        
        Args:
            audio_bytes: Audio file bytes
            language: Language code (e.g., "vi-VN", "en-US")
            
        Returns:
            Transcribed text or None
        """
        try:
            wav_file = self.convert_to_wav(audio_bytes)
            with sr.AudioFile(wav_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=language)
                return text
        except sr.UnknownValueError:
            return None
        except Exception:
            return None
    
    def text_to_speech(self, text: str, language: str = "vi") -> Optional[str]:
        """
        Convert text to speech audio (base64 encoded).
        
        Args:
            text: Text to convert
            language: Language code (e.g., "vi", "en")
            
        Returns:
            Base64 encoded audio string or None
        """
        try:
            tts = gTTS(text, lang=language)
            bio = io.BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            b64 = base64.b64encode(bio.read()).decode()
            return b64
        except Exception:
            return None


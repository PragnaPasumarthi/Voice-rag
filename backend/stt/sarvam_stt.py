"""
Speech-to-Text module using local Whisper model.
Completely free - runs locally, no API key needed.
Supports 99 languages including Hindi, English, and Indian languages.
"""
import io
import tempfile
import os
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class WhisperSTT:
    """
    Local Whisper Speech-to-Text.
    Supports 99 languages via faster-whisper or openai-whisper.
    Completely free, no API key required.
    
    Models: tiny, base, small, medium, large-v3
    - base: ~1GB RAM, good accuracy
    - small: ~2GB RAM, better accuracy
    - medium: ~5GB RAM, great accuracy
    """

    def __init__(self, config=None):
        from ..config import STTConfig
        self.config = config or STTConfig()
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        model_size = self.config.whisper_model
        device = self.config.device

        if FASTER_WHISPER_AVAILABLE:
            self._model = WhisperModel(model_size, device=device, compute_type="int8")
        elif WHISPER_AVAILABLE:
            self._model = whisper.load_model(model_size, device=device)
        else:
            raise RuntimeError(
                "No Whisper library available. Install: pip install faster-whisper"
            )

        return self._model

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "en",
        filename: str = "audio.wav",
    ) -> Dict[str, Any]:
        """
        Transcribe audio bytes to text using local Whisper.
        
        Args:
            audio_bytes: Raw audio data (WAV, MP3, OGG, etc.)
            language_code: Language code (en, hi, bn, ta, te, etc.)
            filename: Original filename (for format detection)
        
        Returns:
            {
                "text": "transcribed text",
                "language": "detected language",
                "confidence": 0.0-1.0,
                "success": bool,
                "error": str or None
            }
        """
        try:
            model = self._get_model()

            # Save to temp file for processing
            suffix = os.path.splitext(filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # Convert to WAV if needed
                wav_path = tmp_path
                if suffix.lower() not in (".wav", ".wave"):
                    if PYDUB_AVAILABLE:
                        audio = AudioSegment.from_file(tmp_path)
                        wav_path = tmp_path + ".wav"
                        audio.export(wav_path, format="wav")
                    else:
                        # Try reading directly
                        wav_path = tmp_path

                # Run transcription
                lang = language_code.split("-")[0] if "-" in language_code else language_code

                if FASTER_WHISPER_AVAILABLE:
                    segments, info = model.transcribe(
                        wav_path,
                        language=lang,
                        beam_size=5,
                        vad_filter=True,
                    )
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text.strip())
                    text = " ".join(text_parts)
                    detected_lang = info.language
                    confidence = info.language_probability

                elif WHISPER_AVAILABLE:
                    result = model.transcribe(wav_path, language=lang)
                    text = result["text"].strip()
                    detected_lang = result.get("language", lang)
                    confidence = 0.9

                else:
                    return {
                        "text": "",
                        "language": language_code,
                        "confidence": 0.0,
                        "success": False,
                        "error": "No Whisper library available",
                    }

                # Clean up converted file
                if wav_path != tmp_path and os.path.exists(wav_path):
                    os.unlink(wav_path)

                return {
                    "text": text,
                    "language": detected_lang,
                    "confidence": confidence,
                    "success": True,
                    "error": None,
                }

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            return {
                "text": "",
                "language": language_code,
                "confidence": 0.0,
                "success": False,
                "error": f"Whisper STT error: {str(e)}",
            }


# Keep alias for backward compatibility
SarvamSTT = WhisperSTT

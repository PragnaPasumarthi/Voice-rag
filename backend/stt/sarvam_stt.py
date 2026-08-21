"""
Speech-to-Text module using ElevenLabs Scribe API.
High-accuracy transcription in 90+ languages.
Docs: https://elevenlabs.io/docs/capabilities/speech-to-text
"""
import os
import aiohttp
from typing import Dict, Any, Optional


class ElevenLabsSTT:
    """
    ElevenLabs Scribe Speech-to-Text.
    Supports 90+ languages with word-level timestamps.
    API: POST https://api.elevenlabs.io/v1/speech-to-text
    """

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
    MODEL_ID = "scribe_v1"

    MIME_MAP = {
        ".wav": "audio/wav", ".wave": "audio/wav",
        ".mp3": "audio/mpeg", ".mpeg": "audio/mpeg",
        ".ogg": "audio/ogg", ".oga": "audio/ogg",
        ".webm": "audio/webm",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".amr": "audio/amr",
        ".opus": "audio/opus",
    }

    def __init__(self, config=None):
        from ..config import STTConfig
        self.config = config or STTConfig()
        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str = "en",
        filename: str = "audio.wav",
    ) -> Dict[str, Any]:
        if not self._api_key:
            return {
                "text": "",
                "language": language_code,
                "confidence": 0.0,
                "success": False,
                "error": "ELEVENLABS_API_KEY not set in .env",
            }

        ext = os.path.splitext(filename)[1].lower()
        mime = self.MIME_MAP.get(ext, "audio/wav")

        try:
            form = aiohttp.FormData()
            form.add_field("model_id", self.MODEL_ID)
            form.add_field(
                "file",
                audio_bytes,
                filename=filename,
                content_type=mime,
            )

            headers = {"xi-api-key": self._api_key}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(self.API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {
                            "text": "",
                            "language": language_code,
                            "confidence": 0.0,
                            "success": False,
                            "error": f"ElevenLabs API error {resp.status}: {body[:200]}",
                        }

                    result = await resp.json()

            transcript = result.get("text", "")
            detected = result.get("language_code", language_code)

            return {
                "text": transcript,
                "language": detected,
                "confidence": 0.95,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "text": "",
                "language": language_code,
                "confidence": 0.0,
                "success": False,
                "error": f"ElevenLabs STT error: {str(e)}",
            }


WhisperSTT = ElevenLabsSTT
SarvamSTT = ElevenLabsSTT

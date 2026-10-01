"""WhisperX-ROCm backend — the previous implementation, kept as the fallback.

Behaviour is unchanged from the pre-migration `_transcribe_path`, including
returning "" when Silero VAD decides there was no speech.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from .base import STTBackend

logger = logging.getLogger("uvicorn.error")

MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ja")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "8"))
VAD_METHOD = os.getenv("WHISPER_VAD_METHOD", "silero")


class WhisperXBackend(STTBackend):
    name = "whisperx"

    def __init__(self) -> None:
        self._model = None
        self._whisperx = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        whisperx = self._module()
        t0 = time.perf_counter()
        self._model = whisperx.load_model(
            MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE, vad_method=VAD_METHOD
        )
        logger.info("whisperX loaded in %.1fs (%s, %s)", time.perf_counter() - t0, MODEL, DEVICE)

    def transcribe_path(self, path: str) -> str:
        self.load()
        whisperx = self._module()

        t0 = time.perf_counter()
        audio = whisperx.load_audio(path)  # ffmpeg decode to 16kHz mono
        decode_ms = (time.perf_counter() - t0) * 1000
        audio_sec = len(audio) / 16000

        t1 = time.perf_counter()
        try:
            with self._lock:
                result = self._model.transcribe(audio, batch_size=BATCH_SIZE)
        except IndexError:
            # Silero VAD found no speech; WhisperX then indexes inputs[0] and raises.
            logger.info("STT[whisperx] no speech detected (audio %.2fs)", audio_sec)
            return ""
        infer_ms = (time.perf_counter() - t1) * 1000

        segments = result.get("segments", []) if isinstance(result, dict) else []
        text = "".join(seg.get("text", "") for seg in segments).strip()
        rtf = (infer_ms / 1000) / audio_sec if audio_sec else 0
        logger.info(
            "STT[whisperx] decode %.0fms + infer %.0fms (audio %.2fs, RTF %.2f): %r",
            decode_ms, infer_ms, audio_sec, rtf, text[:40],
        )
        return text

    def info(self) -> dict:
        return {
            "model": MODEL,
            "language": LANGUAGE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "batch_size": BATCH_SIZE,
            "vad_method": VAD_METHOD,
            "loaded": self._model is not None,
            "supports_streaming": self.supports_streaming,
        }

    def _module(self):
        if self._whisperx is None:
            import whisperx  # noqa: PLC0415 — lazy so NeMo-only runs never touch ctranslate2

            self._whisperx = whisperx
        return self._whisperx

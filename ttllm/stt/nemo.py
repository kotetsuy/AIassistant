"""NeMo Speech backend — nemotron-3.5-asr-streaming-0.6b on ROCm (gfx1151).

Deliberately does NOT use `model.transcribe()`: that rebuilds a Lhotse dataloader
on every call and costs 0.2-0.4s per request, which would eat the entire latency
win over whisperX. We call the encoder and the RNNT decoder directly, the same
way Speech/rocm-inference/scripts/benchmark.py does (0.05-0.09s for 1.4-4.0s of
Japanese).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time

from .audio import SAMPLE_RATE, decode_to_16k_mono
from .base import STTBackend

logger = logging.getLogger("uvicorn.error")

MODEL = os.getenv("NEMO_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b")
LANGUAGE = os.getenv("NEMO_LANGUAGE", "ja-JP")
DEVICE = os.getenv("NEMO_DEVICE", "cuda")
# [left, right] context in 80ms frames; chunk length = (right + 1) * 80ms.
# "[56,13]" = 1120ms, "[56,1]" = 160ms. Empty string keeps the model default.
ATT_CONTEXT_SIZE = os.getenv("NEMO_ATT_CONTEXT_SIZE", "[56,13]")

# The model appends a "<xx-XX>" tag after the terminal punctuation.
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")


class NeMoBackend(STTBackend):
    name = "nemo"

    def __init__(self) -> None:
        self._model = None
        self._prompt_id = None
        # The model is a single shared object; serialize forward passes through it.
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import nemo.collections.asr as nemo_asr

        t0 = time.perf_counter()
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location=DEVICE)
        model.eval()
        model.to(torch.device(DEVICE))

        prompt_dict = model.cfg.model_defaults.get("prompt_dictionary") or {}
        if LANGUAGE not in prompt_dict:
            raise RuntimeError(
                f"NEMO_LANGUAGE={LANGUAGE!r} is not in the model's prompt_dictionary. "
                f"Japanese is 'ja-JP' (a bare 'ja' is rejected)."
            )

        # Streaming reads the prompt off the model rather than from a tensor
        # argument, so it has to be set before any conformer_stream_step call.
        if hasattr(model, "set_inference_prompt"):
            model.set_inference_prompt(LANGUAGE)

        self._att_context_size = _parse_att_context_size(ATT_CONTEXT_SIZE)
        if self._att_context_size is not None:
            model.encoder.set_default_att_context_size(att_context_size=self._att_context_size)

        self._model = model
        self._prompt_id = prompt_dict[LANGUAGE]
        logger.info(
            "NeMo loaded in %.1fs (%s, %s, prompt %s=%d, att_context_size=%s)",
            time.perf_counter() - t0, type(model).__name__, DEVICE, LANGUAGE,
            self._prompt_id, self._att_context_size or "default",
        )
        self._warmup()

    @property
    def supports_streaming(self) -> bool:
        return True

    def new_stream(self):
        """Start a cache-aware streaming session. Each caller gets its own state."""
        from .streaming import StreamSession

        self.load()
        return StreamSession(self._model, self._lock)

    def _warmup(self) -> None:
        """Run one throwaway pass. Without it the first real request pays ~2.7s of
        kernel compilation and allocator warmup instead of the usual ~90ms."""
        import numpy as np
        import torch

        t0 = time.perf_counter()
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            self._infer(silence)
        except Exception as e:  # noqa: BLE001 — warmup must never break startup
            logger.warning("NeMo warmup pass failed (continuing): %s", e)
            return
        logger.info("NeMo warmup done in %.1fs", time.perf_counter() - t0)

    def transcribe_path(self, path: str) -> str:
        self.load()

        t0 = time.perf_counter()
        audio = decode_to_16k_mono(path)
        decode_ms = (time.perf_counter() - t0) * 1000
        audio_sec = len(audio) / SAMPLE_RATE
        if audio_sec <= 0:
            return ""

        t1 = time.perf_counter()
        text = self._infer(audio)
        infer_ms = (time.perf_counter() - t1) * 1000

        rtf = (infer_ms / 1000) / audio_sec
        logger.info(
            "STT[nemo] decode %.0fms + infer %.0fms (audio %.2fs, RTF %.2f): %r",
            decode_ms, infer_ms, audio_sec, rtf, text[:40],
        )
        return text

    def _infer(self, audio) -> str:
        import numpy as np
        import torch

        device = torch.device(DEVICE)
        signal = torch.from_numpy(np.expand_dims(audio, 0)).to(device)
        length = torch.tensor([audio.shape[0]], dtype=torch.long, device=device)
        prompts = torch.tensor([self._prompt_id], dtype=torch.long, device=device)

        with self._lock, torch.inference_mode():
            encoded, encoded_len = self._model.forward(
                input_signal=signal, input_signal_length=length, prompt_indices=prompts
            )
            hyps = self._model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)
        return _hyp_text(hyps)

    def info(self) -> dict:
        return {
            "model": MODEL,
            "language": LANGUAGE,
            "device": DEVICE,
            "att_context_size": ATT_CONTEXT_SIZE or None,
            "loaded": self._model is not None,
            "supports_streaming": self.supports_streaming,
        }


def _parse_att_context_size(raw: str):
    """"[56,13]" -> [56, 13]. Empty/invalid keeps the model default."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        parts = [int(p) for p in raw.strip("[]").split(",")]
    except ValueError:
        logger.warning("NEMO_ATT_CONTEXT_SIZE=%r is not parseable; using the model default", raw)
        return None
    if len(parts) != 2:
        logger.warning("NEMO_ATT_CONTEXT_SIZE=%r must have 2 values; using the model default", raw)
        return None
    return parts


def _hyp_text(hyps) -> str:
    hyp = hyps[0]
    if isinstance(hyp, list):  # some configs return (best, all-candidates)
        hyp = hyp[0]
    text = getattr(hyp, "text", hyp)
    return LANG_TAG_RE.sub("", text).strip()

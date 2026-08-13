"""STT backend selection and fallback.

STT_BACKEND=nemo|whisperx|auto  (default: auto)
    auto   -- use NeMo; on load or inference failure, fall back to STT_FALLBACK
              and stay there for the rest of the process.
    nemo   -- NeMo only. A failure is a failure.
    whisperx -- whisperX only.

STT_FALLBACK=whisperx|none      (default: whisperx)
STT_EAGER_FALLBACK=1|0          (default: 1)
    Load the fallback during warmup too. Costs ~2s and some VRAM, but means the
    first request after a failure is not stuck behind a model load.

A silent fallback is the dangerous outcome — the assistant would keep answering
with a quietly different model. Every fallback is logged at WARNING and shown in
/health as `fallback_active`.
"""

from __future__ import annotations

import logging
import os

from .base import STTBackend

logger = logging.getLogger("uvicorn.error")

REQUESTED = os.getenv("STT_BACKEND", "auto").lower()
FALLBACK = os.getenv("STT_FALLBACK", "whisperx").lower()
EAGER_FALLBACK = os.getenv("STT_EAGER_FALLBACK", "1") not in ("0", "false", "no")

_VALID = ("nemo", "whisperx", "auto")


class STTRouter:
    """Holds the active backend and performs the fallback."""

    def __init__(self, requested: str = REQUESTED, fallback: str = FALLBACK) -> None:
        if requested not in _VALID:
            raise ValueError(f"STT_BACKEND must be one of {_VALID}, got {requested!r}")
        self.requested = requested
        self._fallback_name = fallback if requested == "auto" and fallback != "none" else None
        self._primary = _make("nemo" if requested == "auto" else requested)
        self._fallback = _make(self._fallback_name) if self._fallback_name else None
        self._active = self._primary
        self.fallback_active = False
        self.last_error: str | None = None

    def load(self) -> None:
        try:
            self._primary.load()
        except Exception as e:  # noqa: BLE001 — any load failure should trigger fallback
            self._activate_fallback(f"load failed: {e}")
        if self._fallback is not None and EAGER_FALLBACK and not self.fallback_active:
            try:
                self._fallback.load()
            except Exception as e:  # noqa: BLE001
                logger.warning("STT fallback (%s) failed to preload: %s", self._fallback.name, e)

    def transcribe_path(self, path: str) -> str:
        try:
            return self._active.transcribe_path(path)
        except Exception as e:  # noqa: BLE001
            if self._active is self._primary and self._fallback is not None:
                self._activate_fallback(f"inference failed: {e}")
                return self._active.transcribe_path(path)
            raise

    @property
    def active(self) -> STTBackend:
        return self._active

    def info(self) -> dict:
        return {
            "backend": self._active.name,
            "requested": self.requested,
            "fallback": self._fallback_name,
            "fallback_active": self.fallback_active,
            "last_error": self.last_error,
            **self._active.info(),
        }

    def _activate_fallback(self, reason: str) -> None:
        self.last_error = reason
        if self._fallback is None:
            logger.error("STT backend %s failed and no fallback is configured: %s",
                         self._primary.name, reason)
            raise RuntimeError(f"STT backend {self._primary.name} failed: {reason}")
        logger.warning(
            "STT falling back from %s to %s and staying there — %s",
            self._primary.name, self._fallback.name, reason,
        )
        self._active = self._fallback
        self.fallback_active = True


def _make(name: str) -> STTBackend:
    if name == "nemo":
        from .nemo import NeMoBackend

        return NeMoBackend()
    if name == "whisperx":
        from .whisperx import WhisperXBackend

        return WhisperXBackend()
    raise ValueError(f"unknown STT backend {name!r}")


__all__ = ["STTBackend", "STTRouter"]

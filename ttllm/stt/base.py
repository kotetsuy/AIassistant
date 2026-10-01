"""STT backend interface.

`transcribe_path` keeps the contract the old `_transcribe_path` had: take a path
to an audio file in whatever format the browser sent, return the transcript as a
plain string ("" when there is no speech). That is what lets the three FastAPI
endpoints stay untouched.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Iterator

logger = logging.getLogger("uvicorn.error")


class STTBackend(ABC):
    """One speech-to-text implementation."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load the model. Called from /warmup; safe to call repeatedly."""

    @abstractmethod
    def transcribe_path(self, path: str) -> str:
        """Audio file path -> transcript. Returns "" when nothing was spoken."""

    @property
    def supports_streaming(self) -> bool:
        return False

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[str]:
        """PCM chunks -> incremental transcripts. Implemented in Phase 2."""
        raise NotImplementedError(f"{self.name} backend does not support streaming")

    @abstractmethod
    def info(self) -> dict:
        """Details for /health."""

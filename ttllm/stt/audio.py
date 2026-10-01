"""Audio decoding for the NeMo backend.

The browser sends webm/opus (MediaRecorder), not 16kHz WAV. whisperX hides this
behind `whisperx.load_audio()`, which shells out to ffmpeg; the NeMo path needs
its own equivalent.
"""

from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16000
# Cache-aware streaming needs right context to emit the final token, and a
# push-to-talk recording ends the instant the button is released. Half a second
# of silence is what stopped the sentence-final particle from being dropped
# during the Speech validation (see Speech/rocm-inference/docs/PHASE1.md).
TAIL_PAD_SEC = 0.5


def decode_to_16k_mono(path: str, pad_tail: bool = True) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 mono PCM at 16kHz."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace")[-500:]
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}") from e

    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if pad_tail:
        audio = pad_tail_silence(audio)
    return audio


def pad_tail_silence(audio: np.ndarray, seconds: float = TAIL_PAD_SEC) -> np.ndarray:
    return np.concatenate([audio, np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)])

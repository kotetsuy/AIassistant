#!/usr/bin/env python
"""Phase 2 check: cache-aware streaming against the known-good transcripts.

Feeds each evaluation wav to StreamSession in 100ms slices (roughly what the
browser will send) and compares the result with both the Speech-repo streaming
validation and this backend's own batch path.

    ttllm/.venv/bin/python ttllm/verify_streaming.py
"""

import torch  # noqa: F401  isort:skip  — must precede ctranslate2/whisperx

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stt.nemo import NeMoBackend  # noqa: E402

AUDIO_DIR = Path.home() / "nemo-rocm-verify" / "audio" / "16k"  # unpadded: the session pads
CHUNK_SEC = 0.1
# Expected streaming output at att_context_size=[56,13] (Speech/rocm-inference/docs/PHASE1.md).
EXPECTED = {
    "human_greeting": "こんにちは",
    "human_mountain": "日本で二番目に高い山は",
    "human_river": "日本で一番長い川は",
    "zundamon_greeting": "こんにちは",
    "zundamon_mountain": "日本で二番目に高い山は",
    "zundamon_river": "日本で一番長い川は",
    "ritsu_greeting": "こんにちは",
    "ritsu_mountain": "日本で二番目に高い山は",
    "ritsu_river": "日本で一番長い川は",
}


def main() -> int:
    backend = NeMoBackend()
    backend.load()

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    mismatches = []
    print(f"\n{'音源':22s} {'確定転写':24s} {'部分':>4s} {'発話終了→確定':>13s}")

    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        if sr != 16000:
            raise SystemExit(f"{wav} is {sr}Hz")

        session = backend.new_stream()
        step = int(16000 * CHUNK_SEC)
        partials = []

        # Feed in real-time-sized slices (without sleeping — we want compute time).
        for i in range(0, len(audio), step):
            text = session.add_audio(audio[i : i + step])
            if text and (not partials or text != partials[-1]):
                partials.append(text)

        t0 = time.perf_counter()
        final = session.finish()
        finalize_ms = (time.perf_counter() - t0) * 1000

        expected = EXPECTED.get(wav.stem)
        mark = "OK  " if final == expected else "DIFF"
        if final != expected:
            mismatches.append((wav.stem, expected, final))
        print(f"{mark} {wav.stem:20s} {final:24s} {len(partials):4d} {finalize_ms:11.0f}ms")

    print("\n=== バッチ経路との比較 (att_context_size 変更の影響確認) ===")
    for wav in wavs:
        batch = backend.transcribe_path(str(wav))
        expected = EXPECTED.get(wav.stem)
        # ritsu_river is the one the model gets wrong in batch mode; see PHASE1.md
        note = "" if batch == expected else "  <- バッチ固有の差異"
        print(f"  {wav.stem:20s} {batch}{note}")

    print("\n=== 判定 ===")
    if mismatches:
        for stem, exp, got in mismatches:
            print(f"NG: {stem}: expected {exp!r}, got {got!r}")
        return 1
    print("OK: ストリーミング 9/9 が既知の正解と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

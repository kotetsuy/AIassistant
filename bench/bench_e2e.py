#!/usr/bin/env python
"""Phase 3 (end-to-end): speech end -> first spoken audio, for both paths.

Measures what the user actually perceives: the gap between releasing the mic
button and Koteko starting to talk. Subscribes to three-vrm's broadcast socket
and timestamps the first `speak` chunk.

    batch  : POST /voice_chat_speak_stream  (webm upload, the original path)
    stream : WS  /stt_chat_stream           (PCM during speech, Phase 2)

LLM sampling makes the reply length vary between runs, and the first sentence
has to be generated and synthesised before any audio exists — so this number is
noisier than the STT-only benchmark and is reported as a range, not a verdict.

    ttllm/.venv/bin/python bench/bench_e2e.py --runs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

BENCH_DIR = Path(__file__).resolve().parent
THREE_VRM = "http://localhost:8000"
CHUNK_SEC = 0.1


async def main() -> None:
    args = _parse_args()
    wavs = [BENCH_DIR / "audio" / f"{n}.wav" for n in args.audio]

    results = []
    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        assert sr == 16000, wav
        webm = _to_webm(wav)
        dur = len(audio) / sr

        batch = [await _measure_batch(webm) for _ in range(args.runs)]
        batch = [x for x in batch if x is not None]
        row = {
            "audio": wav.stem,
            "duration_sec": round(dur, 2),
            "batch_ms": _stats(batch),
        }
        line = (f"{wav.stem:16s} ({dur:5.2f}s)  "
                f"batch {row['batch_ms']['median']:6.0f}ms "
                f"[{row['batch_ms']['min']:.0f}-{row['batch_ms']['max']:.0f}] "
                f"over1s={row['batch_ms']['over_1s']}/{row['batch_ms']['n']}")

        if args.stream:
            stream = [await _measure_stream(audio) for _ in range(args.runs)]
            stream = [x for x in stream if x is not None]
            row["stream_ms"] = _stats(stream)
            line += (f"   stream {row['stream_ms']['median']:6.0f}ms "
                     f"[{row['stream_ms']['min']:.0f}-{row['stream_ms']['max']:.0f}] "
                     f"over1s={row['stream_ms']['over_1s']}/{row['stream_ms']['n']}")

        results.append(row)
        print(line)

    # Results are keyed by backend label so the whisperX baseline and the NeMo
    # numbers can be collected across separate runs (the backend is chosen when
    # ttllm starts, so one process cannot measure both).
    out = BENCH_DIR / "results_e2e.json"
    merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "rows" in merged:      # migrate the old single-backend format
        merged = {}
    merged[args.label] = {"runs": args.runs, "rows": results}
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} (label: {args.label})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--audio", nargs="+", default=["human_river", "long1"])
    p.add_argument(
        "--label", default="nemo",
        help="key to store these results under, e.g. the active STT_BACKEND",
    )
    p.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help="skip the streaming path (whisperX has none — it would hang waiting for audio)",
    )
    return p.parse_args()


async def _first_speak(ws_ready: asyncio.Event) -> float:
    """Timestamp of the first `speak` broadcast after the caller signals start."""
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"ws://localhost:8000/ws") as ws:
            ws_ready.set()
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                if json.loads(msg.data).get("type") == "speak":
                    return time.perf_counter()
    return 0.0


async def _measure_batch(webm: bytes):
    ready = asyncio.Event()
    watcher = asyncio.create_task(_first_speak(ready))
    await ready.wait()

    form = aiohttp.FormData()
    form.add_field("audio", webm, filename="utterance.webm",
                   content_type="application/octet-stream")
    form.add_field("speaker_id", "3")

    t_end = time.perf_counter()  # the upload starts the moment the mic stops
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
        async with s.post(f"{THREE_VRM}/voice_chat_speak_stream", data=form) as r:
            await r.read()
    t_speak = await asyncio.wait_for(watcher, timeout=60)
    return (t_speak - t_end) * 1000 if t_speak else None


async def _measure_stream(audio: np.ndarray):
    ready = asyncio.Event()
    watcher = asyncio.create_task(_first_speak(ready))
    await ready.wait()

    step = int(16000 * CHUNK_SEC)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
        async with s.ws_connect(f"ws://localhost:8000/stt_chat_stream?speaker_id=3") as ws:
            for i in range(0, len(audio), step):
                await ws.send_bytes(audio[i : i + step].astype("<f4").tobytes())
                await asyncio.sleep(CHUNK_SEC)  # real time, as the browser would
            t_end = time.perf_counter()
            await ws.send_str(json.dumps({"type": "end"}))
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                if json.loads(msg.data).get("type") in ("done", "error"):
                    break
    t_speak = await asyncio.wait_for(watcher, timeout=60)
    return (t_speak - t_end) * 1000 if t_speak else None


def _to_webm(wav: Path) -> bytes:
    import subprocess

    out = wav.with_suffix(".webm")
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
             "-c:a", "libopus", "-b:a", "32k", "-ac", "1", str(out)],
            check=True,
        )
    return out.read_bytes()


def _stats(xs: list) -> dict:
    if not xs:
        return {"median": 0, "min": 0, "max": 0, "n": 0, "over_1s": 0, "samples": []}
    return {
        "median": round(statistics.median(xs), 1),
        "mean": round(statistics.fmean(xs), 1),
        "min": round(min(xs), 1),
        "max": round(max(xs), 1),
        "n": len(xs),
        # How often the user waits more than a second is more legible than the
        # spread alone, given how wide LLM generation makes the distribution.
        "over_1s": sum(1 for x in xs if x > 1000),
        "samples": [round(x, 1) for x in sorted(xs)],
    }


if __name__ == "__main__":
    asyncio.run(main())

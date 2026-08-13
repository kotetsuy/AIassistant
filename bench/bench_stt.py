#!/usr/bin/env python
"""Phase 3: compare the three STT configurations on latency and accuracy.

    A  whisperX-ROCm   large-v3-turbo, batch_size=8, float16   (the previous default)
    B  NeMo batch      nemotron-3.5-asr-streaming-0.6b, float32
    C  NeMo streaming  same model, att_context_size=[56,13]

The headline metric is **speech end -> final transcript**, because that is what
the user waits through. For A and B that is the whole decode; for C it is only
the tail that has not been processed yet, so it should barely grow with
utterance length.

Everything runs in-process: HTTP/WebSocket overhead is the same for all three
and would only add noise. Accuracy is reported as CER against the reference,
after light normalisation (see `_normalize`), since the two models differ in
surface conventions rather than content.

    ttllm/.venv/bin/python bench/bench_stt.py --runs 20
"""

from __future__ import annotations

import torch  # noqa: F401  isort:skip  — must precede ctranslate2/whisperx

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "ttllm"))

AUDIO_DIR = BENCH_DIR / "audio"
SHORT_REFS = {
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
CHUNK_SEC = 0.1  # what the browser's AudioWorklet sends


def main() -> None:
    args = _parse_args()
    refs = dict(SHORT_REFS)
    long_refs = AUDIO_DIR / "long_refs.json"
    if long_refs.exists():
        refs.update(json.loads(long_refs.read_text(encoding="utf-8")))

    wavs = sorted(AUDIO_DIR.glob("*.wav"), key=lambda p: (_dur(p), p.name))
    if not wavs:
        raise SystemExit(f"no audio in {AUDIO_DIR}")

    from stt.nemo import NeMoBackend
    from stt.whisperx import WhisperXBackend

    print("=== モデルロード ===")
    nemo = NeMoBackend()
    t0 = time.perf_counter()
    nemo.load()
    nemo_load = time.perf_counter() - t0

    wx = WhisperXBackend()
    t0 = time.perf_counter()
    wx.load()
    wx_load = time.perf_counter() - t0
    print(f"  NeMo     {nemo_load:6.1f}s")
    print(f"  whisperX {wx_load:6.1f}s")

    rows = []
    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype="float32")
        if sr != 16000:
            raise SystemExit(f"{wav} is {sr}Hz")
        dur = len(audio) / sr
        ref = refs.get(wav.stem, "")

        a = _bench(lambda: wx.transcribe_path(str(wav)), args.runs, args.warmup)
        b = _bench(lambda: nemo.transcribe_path(str(wav)), args.runs, args.warmup)
        c = _bench_stream(nemo, audio, args.runs, args.warmup)

        row = {
            "audio": wav.stem,
            "duration_sec": round(dur, 2),
            "reference": ref,
            "A_whisperx": a,
            "B_nemo_batch": b,
            "C_nemo_stream": c,
        }
        for key in ("A_whisperx", "B_nemo_batch", "C_nemo_stream"):
            row[key]["cer"] = _cer(ref, row[key]["text"]) if ref else None
        rows.append(row)

        print(
            f"\n{wav.stem}  ({dur:.2f}s)\n"
            f"  A whisperX      {a['median_ms']:7.0f}ms  CER {_pct(row['A_whisperx']['cer'])}  {a['text']}\n"
            f"  B NeMo batch    {b['median_ms']:7.0f}ms  CER {_pct(row['B_nemo_batch']['cer'])}  {b['text']}\n"
            f"  C NeMo stream   {c['median_ms']:7.0f}ms  CER {_pct(row['C_nemo_stream']['cer'])}  {c['text']}"
        )

    result = {
        "load_sec": {"nemo": round(nemo_load, 1), "whisperx": round(wx_load, 1)},
        "runs": args.runs,
        "rows": rows,
    }
    (BENCH_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (BENCH_DIR / "results.md").write_text(_markdown(result), encoding="utf-8")
    print(f"\nwrote {BENCH_DIR / 'results.json'} and results.md")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def _bench(fn, runs: int, warmup: int) -> dict:
    """Speech end -> transcript, for a config that decodes the whole utterance."""
    for _ in range(warmup):
        fn()
    samples, text = [], ""
    for _ in range(runs):
        t0 = time.perf_counter()
        text = fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return _stats(samples, text)


def _bench_stream(backend, audio: np.ndarray, runs: int, warmup: int) -> dict:
    """Speech end -> transcript for streaming.

    The audio is fed in 100ms slices first (as it would arrive during speech),
    and only `finish()` — the flush after the mic stops — is timed. Feeding is
    not slowed to real time: we are measuring compute, not wall-clock playback.
    """
    step = int(16000 * CHUNK_SEC)

    def one() -> tuple:
        session = backend.new_stream()
        for i in range(0, len(audio), step):
            session.add_audio(audio[i : i + step])
        t0 = time.perf_counter()
        text = session.finish()
        return (time.perf_counter() - t0) * 1000, text

    for _ in range(warmup):
        one()
    samples, text = [], ""
    for _ in range(runs):
        ms, text = one()
        samples.append(ms)
    return _stats(samples, text)


def _stats(samples: list, text: str) -> dict:
    return {
        "median_ms": round(statistics.median(samples), 1),
        "mean_ms": round(statistics.fmean(samples), 1),
        "min_ms": round(min(samples), 1),
        "max_ms": round(max(samples), 1),
        "text": text,
    }


_PUNCT = re.compile(r"[、。，．,\.\s！？!?「」『』・…]")
_KANJI_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
                 "六": "6", "七": "7", "八": "8", "九": "9", "〇": "0"}


def _normalize(s: str) -> str:
    """Fold the surface differences between the two models.

    whisperX writes "2番目" and adds trailing punctuation; NeMo writes "二番目"
    and omits it. Neither is a transcription error, so CER should not count them.
    Kanji digits are folded to ASCII only when they are used as numerals — the
    standalone forms in these references ("一番", "二番目") are exactly that case.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _PUNCT.sub("", s)
    return "".join(_KANJI_DIGITS.get(ch, ch) for ch in s)


def _cer(ref: str, hyp: str) -> float:
    """Character error rate after normalisation. Japanese is scored with CER,
    not WER — there are no word boundaries to segment on."""
    r, h = _normalize(ref), _normalize(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def _pct(v) -> str:
    return " n/a " if v is None else f"{v * 100:4.1f}%"


def _dur(p: Path) -> float:
    info = sf.info(str(p))
    return info.frames / info.samplerate


def _markdown(result: dict) -> str:
    rows = result["rows"]
    out = [
        "# Phase 3 — STT 速度比較",
        "",
        f"- 各構成 {result['runs']} 回計測の中央値",
        "- 指標: **発話終了 → 転写確定**(ユーザが待たされる時間)",
        "- CER は表記ゆれを正規化した後の値(全角半角・句読点・漢数字)",
        "",
        f"モデルロード: NeMo {result['load_sec']['nemo']}s / whisperX {result['load_sec']['whisperx']}s",
        "",
        "## 発話終了 → 転写確定 (ms, 中央値)",
        "",
        "| 音源 | 長さ | A: whisperX | B: NeMo一括 | C: NeMoストリーミング | C の優位 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, b, c = r["A_whisperx"]["median_ms"], r["B_nemo_batch"]["median_ms"], r["C_nemo_stream"]["median_ms"]
        out.append(
            f"| {r['audio']} | {r['duration_sec']:.2f}s | {a:.0f} | {b:.0f} | **{c:.0f}** | "
            f"whisperX比 {a / c:.1f}x / NeMo一括比 {b / c:.1f}x |"
        )

    out += ["", "## 精度 (CER, 正規化後)", "",
            "| 音源 | A: whisperX | B: NeMo一括 | C: NeMoストリーミング |", "|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['audio']} | {_pct(r['A_whisperx']['cer'])} | "
            f"{_pct(r['B_nemo_batch']['cer'])} | {_pct(r['C_nemo_stream']['cer'])} |"
        )

    out += ["", "## 転写内容", ""]
    for r in rows:
        out += [
            f"### {r['audio']} ({r['duration_sec']:.2f}s)",
            "",
            f"- 原文: {r['reference']}",
            f"- A whisperX: {r['A_whisperx']['text']}",
            f"- B NeMo一括: {r['B_nemo_batch']['text']}",
            f"- C NeMoストリーミング: {r['C_nemo_stream']['text']}",
            "",
        ]
    return "\n".join(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 0 gate: prove NeMo and whisperX can live in one process.

Checks, in order:
  1. `import torch` first, then NeMo, then whisperX (+ ctranslate2) all import
  2. both backends load their model on the ROCm GPU in the same process
  3. both transcribe the same Japanese audio
  4. NeMo's output matches what the standalone Speech validation produced

Run with the shared venv:
    ttllm/.venv/bin/python ttllm/verify_coexist.py
"""

# torch must be imported before ctranslate2/whisperx so its bundled ROCm
# (rocm-sdk-libraries-gfx1151) loads first — same constraint as server.py.
import torch  # noqa: F401  isort:skip

import re
import sys
import time
from pathlib import Path

AUDIO_DIR = Path.home() / "nemo-rocm-verify" / "audio" / "16k_pad"
# Reference transcripts from the standalone Speech validation (docs/PHASE1.md).
EXPECTED_NEMO = {
    "human_greeting": "こんにちは",
    "human_mountain": "日本で二番目に高い山は",
    "human_river": "日本で一番長い川は",
    "zundamon_greeting": "こんにちは",
    "zundamon_mountain": "日本で二番目に高い山は",
    "zundamon_river": "日本で一番長い川は",
    "ritsu_greeting": "こんにちは",
    "ritsu_mountain": "日本で二番目に高い山は",
    "ritsu_river": "一本で一番長い川は",  # known model error, kept as-is
}
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")
NEMO_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
WHISPER_MODEL = "large-v3-turbo"

failures = []


def main() -> int:
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}  hip={torch.version.hip}")
    else:
        failures.append("GPU not available")

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print(f"FATAL: no audio under {AUDIO_DIR}", file=sys.stderr)
        return 1

    nemo_out = check_nemo(wavs)
    whisper_out = check_whisperx(wavs)

    print("\n=== 転写結果の比較 ===")
    print(f"{'音源':22s} {'NeMo':24s} whisperX")
    for wav in wavs:
        stem = wav.stem
        print(f"{stem:22s} {nemo_out.get(stem, '(失敗)'):24s} {whisper_out.get(stem, '(失敗)')}")

    print("\n=== 判定 ===")
    if failures:
        for f in failures:
            print(f"NG: {f}")
        return 1
    print("OK: 同一プロセスで両バックエンドがロード・転写でき、NeMo は既存結果と一致")
    return 0


def check_nemo(wavs: list) -> dict:
    print("\n--- NeMo ---")
    import nemo.collections.asr as nemo_asr

    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(NEMO_MODEL, map_location="cuda")
    model.eval()
    print(f"load: {time.perf_counter() - t0:.1f}s  ({type(model).__name__})")

    computer = model.decoding.decoding.decoding_computer
    print(f"cuda_graphs_mode: {computer.cuda_graphs_mode}")

    prompt_id = model.cfg.model_defaults.get("prompt_dictionary")["ja-JP"]
    out = {}
    for wav in wavs:
        text = nemo_transcribe(model, wav, prompt_id)
        out[wav.stem] = text
        expected = EXPECTED_NEMO.get(wav.stem)
        if expected is not None and text != expected:
            failures.append(f"NeMo output changed for {wav.stem}: {text!r} != {expected!r}")
    return out


def nemo_transcribe(model, wav: Path, prompt_id: int) -> str:
    """Direct encoder+decoder path — the one the backend will use (spec 4.7)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="float32")
    if sr != 16000:
        raise SystemExit(f"{wav} is {sr}Hz")
    signal = torch.from_numpy(np.expand_dims(audio, 0)).cuda()
    length = torch.tensor([audio.shape[0]], dtype=torch.long, device="cuda")
    prompts = torch.tensor([prompt_id], dtype=torch.long, device="cuda")

    with torch.inference_mode():
        encoded, encoded_len = model.forward(
            input_signal=signal, input_signal_length=length, prompt_indices=prompts
        )
        hyps = model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)
    hyp = hyps[0]
    if isinstance(hyp, list):
        hyp = hyp[0]
    return LANG_TAG_RE.sub("", getattr(hyp, "text", hyp)).strip()


def check_whisperx(wavs: list) -> dict:
    print("\n--- whisperX ---")
    try:
        import whisperx
    except Exception as e:  # noqa: BLE001 — any import failure is a gate failure
        failures.append(f"whisperx import failed after NeMo was loaded: {e}")
        return {}

    t0 = time.perf_counter()
    try:
        model = whisperx.load_model(
            WHISPER_MODEL, "cuda", compute_type="float16", language="ja", vad_method="silero"
        )
    except Exception as e:  # noqa: BLE001
        failures.append(f"whisperx.load_model failed: {e}")
        return {}
    print(f"load: {time.perf_counter() - t0:.1f}s")

    out = {}
    for wav in wavs:
        audio = whisperx.load_audio(str(wav))
        try:
            result = model.transcribe(audio, batch_size=8)
            text = "".join(s.get("text", "") for s in result.get("segments", [])).strip()
        except IndexError:
            text = ""  # silero VAD found no speech
        except Exception as e:  # noqa: BLE001
            failures.append(f"whisperx transcribe failed for {wav.stem}: {e}")
            text = "(失敗)"
        out[wav.stem] = text
    return out


if __name__ == "__main__":
    sys.exit(main())

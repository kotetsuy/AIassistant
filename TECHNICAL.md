# AIassistant — Technical Guide / Tuning

For setup and startup instructions, see [`README.md`](./README.md).
This document covers the internal architecture, latency optimization, model selection,
endpoint specifications, VRM effects, and known limitations.

## Pipeline overview

```
Browser (three-vrm)
  └─ Mic capture (MediaRecorder webm/opus)
         ↓ POST /voice_chat_speak_stream
    three-vrm server (port 8000)
         ↓ POST /voice_chat_stream
       ttllm bridge (port 8001)
         ├─ STT: NeMo Speech (default) / WhisperX-ROCm (fallback)
         └─ llama-server (Qwen3.6-35B-A3B MoE, port 8080)
         ↓ Token stream over SSE
    three-vrm: split at sentence boundaries → VOICEVOX (port 50021) → push over WS
         ↓ WS (audio + visemes)
 Browser: AudioContext continuous playback + VRM lip sync + background + idle motion
```

For short utterances (around "hello"), the target is roughly 1 second perceived latency;
even for long replies, the goal is **first audio in the 1-second range**.

## Latency optimization

### 1. Disable Qwen3 thinking mode

By default, Qwen3 emits several hundred tokens of `reasoning_content` (internal monologue)
before its answer, which adds a few seconds of perceived delay. ttllm passes
`chat_template_kwargs: {"enable_thinking": false}` to llama-server to disable it
(see `ttllm/server.py:_call_llama`). This single line shaves 4–8 seconds off the LLM stage.

### 2. Pipeline LLM → VOICEVOX

- Added `/voice_chat_stream` (SSE) to ttllm, which calls llama-server with `stream: true`
  and returns the flow `{transcript}` → `{token}×N` → `{done}`.
- `/voice_chat_speak_stream` on three-vrm consumes the SSE. It splits sentences at
  `[。！？\n]`, and as a safety net for long lines, it also splits at `[、]` past 60 characters.
  TTS is serialized through an `asyncio.Queue` + consumer task (to preserve WS order),
  while LLM decoding continues in parallel.
- The client resets the playhead on `turn_start`, and queues each `speak` chunk for
  continuous playback at `startAt = max(playheadTime, now)`. Visemes are scheduled
  on absolute time, so they don't collide across chunks.

Result (measured, long-form 8-sentence reply):

| Metric | Before (non-streaming) | After (pipeline) |
|---|---|---|
| Time to first audio | **3.32 s** | **1.06 s** |
| Total completion time | 3.32 s | 2.98 s |

### 3. Switch WhisperX from large-v3 to large-v3-turbo

Switching the STT stage to the turbo model roughly halves transcription time. Measured
in `/warmup`-ed steady state (2.56-second audio sample, float16, batch 8, Silero VAD):

| Metric | large-v3 | large-v3-turbo | Improvement |
|---|---|---|---|
| Transcription time (steady median) | 474 ms | **247 ms** | **-48% (1.92x faster)** |
| Transcription time (cold first) | 664 ms | 440 ms | -34% |
| Model load | 6.51 s | 4.83 s | -26% |

**Effect on the "first utterance"**: the STT stage gets **about 227 ms shorter**, so the
time to first audio improves by that much (it helps TTFT). Recognition accuracy is on par
(short sentences return the same text).

### 4. Stop the previous utterance immediately when a new turn starts

The moment the mic is pressed, the client `stop(0)`s every currently scheduled
`AudioBufferSourceNode` and flushes the viseme queue (`stopAllPlayback`). Because it
doesn't wait for the server's `turn_start` to arrive, the UI feels instantly responsive.

## The MTP-vs-MoE decision (LLM history)

The current default LLM is **Qwen3.6-35B-A3B (MoE, Q4_K_XL, ~21GB)**, which `start_all.sh`
launches with `-fit off` (no `--spec-type`). This is a switch away from the earlier dense
Qwen3.6-27B + MTP speculative decoding setup. The rationale is recorded below.

### Old setup: Qwen3.6-27B + MTP speculative decoding

Qwen3.6-27B ships with one MTP layer, and llama.cpp's `--spec-type draft-mtp` enables
speculative decoding. The MTP head predicts up to 3 draft tokens ahead, and the target
model advances in one step by however many get accepted.

Measured (same gguf, same prompt, 142 tokens generated, temperature 0.7, seed 42):

| Metric | Without MTP | With MTP | Improvement |
|---|---|---|---|
| Generation tokens/sec | 7.71 | **10.15** | **+31.7% (1.32x)** |
| 142-token response time | 18.42 s | **13.99 s** | -24% |
| TTFT (first token) | 0.46 s | 0.48 s | ≈ same |
| Draft acceptance | — | 24.7% (60/243) | — |

**Important caveat**: MTP speeds up the **per-token generation rate**, but
**TTFT (time to first token) is unchanged**. So "time to first audio" (already at 1.06 s
thanks to the streaming pipeline) is **not improved by MTP**; the gain shows up in
"total completion time for long responses". The shorter the reply, the smaller the effect.

### Why we switched: the MoE model is much faster on bandwidth-limited iGPUs

On a smaller chip such as the Ryzen AI HX 370 (16 CU, ~120 GB/s, 32GB unified), MTP gives no
benefit and the dense 27B is slow to begin with. Switching to the
**Qwen3.6-35B-A3B (Q4_K_XL, ~21GB)** MoE model — only ~3B active parameters per token —
is dramatically faster:

| Metric | dense 27B | MoE 35B-A3B | Improvement |
|---|---|---|---|
| TTFT | ~360 ms | **~88 ms** | ~4x |
| Generation tokens/sec | ~5.0 | **~19.8** | ~4x |

The MoE wins on both TTFT and generation rate even without MTP, so the current setup drops
MTP entirely. If you want to re-evaluate the dense 27B + MTP, build a llama.cpp master that
supports `--spec-type draft-mtp` and use `Qwen3.6-27B-MTP-Q8_0.gguf` (includes the MTP
layer, ~29GB). You can confirm the MTP layer with:

```bash
cd ~/llama.cpp
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors \
  ~/qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf | grep -E "nextn|architecture"
# → qwen35.nextn_predict_layers = 1
```

MTP support was merged in llama.cpp PR
[#22673](https://github.com/ggml-org/llama.cpp/pull/22673).

## VRM viewer effects

### Random background rotation

- Images are auto-detected from `~/AIassistant/images/*.{jpg,png,webp}` (override with the `IMAGES_DIR` env var)
- `GET /images_list` returns the file list; `GET /images/<name>` serves the image
- One is picked at page load, and a different image is swapped in **every 5 minutes** (`zundamon.html`)
- Images aren't bundled. To add some, just drop files into the directory — no server restart needed.

### Idle motion

To avoid the T-pose stand-still look, a small rotation is applied every frame
(`zundamon.html:applyIdlePose`).

| Body part | Frequency | Amplitude |
|---|---|---|
| spine / chest (X axis, breathing) | 0.25 Hz | ±0.7° |
| spine / chest (Z axis, side sway) | 0.13 Hz (different phase) | ±1.1° |
| head (X axis) | 0.10 Hz | ±0.9° |
| head (Y axis) | 0.08 Hz | ±1.7° |

The pose is set before `vrm.update(delta)`, so the VRM's spring bones (hair, skirt, etc.)
follow naturally as secondary motion.

### Lower both arms

VRM's default is T-pose, so right after loading, `applyRestPose()` drops both arms
into a natural standing position and bends the elbows by about 14° (`zundamon.html`).

## Main endpoints

### ttllm (port 8001)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | self + llama-server reachability |
| POST | `/warmup` | Load the STT model and run a warmup inference |
| POST | `/transcribe` | Audio → text |
| POST | `/chat` | Text → LLM response (non-streaming) |
| POST | `/voice_chat` | Audio → response (non-streaming) |
| POST | `/voice_chat_stream` | Audio → SSE (transcript + token + done) |
| POST | `/chat_stream` | Text → SSE (token + done); for callers that did STT themselves **new** |
| WS | `/transcribe_stream` | Raw PCM (float32 16kHz mono) → partial / final **new** |

### three-vrm (port 8000)

| Method | Path | Purpose |
|---|---|---|
| GET | `/zundamon.html` | Viewer |
| GET | `/ws` | WebSocket (turn_start / speak / turn_end / transcript / error) |
| POST | `/speak` | Speak given text |
| POST | `/voice_chat_speak` | Audio → one-shot response (non-streaming) |
| POST | `/voice_chat_speak_stream` | Audio → pipelined response |
| GET | `/stt_chat_stream` | WebSocket: PCM during speech → partials → LLM → VOICEVOX **new** |
| GET | `/images_list` | List background images |
| GET | `/images/{name}` | Serve background image |
| GET | `/vrm/{name}` | Serve VRM file |
| GET | `/status` | Connected client count |

## STT backend design

### Why two backends

Before the migration this was WhisperX-ROCm only. Moving to NeMo Speech did not remove it:
both live in one process with a **runtime fallback**, for two reasons.

1. NeMo runs here on a configuration NVIDIA does not support (ROCm). If it breaks, the
   conversation should not stop.
2. WhisperX is **more accurate on long, proper-noun-heavy speech** (numbers below), so
   having the choice is worth the extra VRAM.

### Layout

```
ttllm/stt/
  base.py       STTBackend ABC — transcribe_path(path) -> str is the whole contract
  audio.py      ffmpeg: webm/ogg/mp4 → 16kHz mono float32, plus 0.5s of trailing silence
  nemo.py       NeMoBackend (default)
  whisperx.py   WhisperXBackend (fallback)
  streaming.py  StreamSession — per-utterance cache-aware streaming state
  __init__.py   STTRouter — backend selection and fallback
```

`_transcribe_path()` survives in `server.py` as a thin wrapper, so `/transcribe`,
`/voice_chat` and `/voice_chat_stream` did not change at all.

### Decisions worth knowing

- **`model.transcribe()` is not used.** It rebuilds a Lhotse dataloader per call, costing
  0.2-0.4s per request. Calling the encoder and `rnnt_decoder_predictions_tensor()` directly
  takes a 3.5s utterance from 119ms to 88ms.
- **Audio decoding is ours to do.** The browser sends webm/opus, not 16kHz WAV. WhisperX hid
  this inside `load_audio()`.
- **0.5s of trailing silence is appended.** Cache-aware decoding cannot emit the final token
  without right context, and a push-to-talk recording ends the instant the button is
  released. Without the padding, VOICEVOX-generated audio lost its sentence-final particle.
- **The language prompt is `ja-JP`.** A bare `ja` is not in the model's `prompt_dictionary`.
- **Every fallback is logged at WARNING and surfaced in `/health`.** Silently switching
  models is the worst possible failure mode.

### Streaming path

```
Browser  getUserMedia → AudioWorklet (raw PCM) → downsample to 16kHz mono
   │ WebSocket (float32 binary)
three-vrm  /stt_chat_stream  ── relay ──▶  ttllm  /transcribe_stream
   │                                        NeMo cache-aware streaming
   │                                        att_context_size=[56,13] (1120ms chunks)
   ◀── partial / final ─────────────────────┘
   └─ final transcript → ttllm /chat_stream (SSE) → sentence split → VOICEVOX → existing WS
```

**MediaRecorder's `start(timeslice)` cannot be used**: webm/opus chunks after the first are
not independently decodable, so the server cannot process them one at a time. Raw PCM via
AudioWorklet is the way.

**Mel features are recomputed over the whole utterance on every append.** Appending 100ms
slices makes the mel preprocessor pad each call independently, putting a window artifact at
every slice boundary — that turned 「川は」 into 「かは」 in testing. Accumulating the raw
PCM and rebuilding the mel buffer (while preserving `buffer_idx`) gives frames identical to
the offline path. `online_normalization=True` goes with it, because offline normalisation
draws statistics from the entire utterance and is fundamentally incompatible with streaming.

Per-session state (encoder cache, partial hypotheses) lives in `StreamSession`; only the
model forward is serialised, through `NeMoBackend._lock`.

### Measurements (speech end → final transcript, median of 20)

| Audio length | A: whisperX | B: NeMo batch | C: NeMo streaming |
|---|---|---|---|
| 1.06s | 254ms | 84ms | **48ms** |
| 3.50s | 287ms | 119ms | **49ms** |
| 10.44s | 432ms | 221ms | **53ms** |
| 12.84s | 545ms | 274ms | **50ms** |
| 14.22s | 460ms | 255ms | **95ms** |

**Streaming's finalisation time barely depends on utterance length** — only the last chunk
is left to process. The one-shot paths scale with length, so the gap widens: 10.8x over
whisperX at 12.8 seconds.

Accuracy (CER, after normalising surface conventions) is 0.0% for all three configs on the
nine short utterances, and **B and C match exactly on every clip** — streaming costs no
accuracy. On long clips **whisperX wins** (9.9% vs 21.1% on the proper-noun-heavy one).

End-to-end (speech end → first audio) went 717ms → 676ms at 3.5s and 616ms → 412ms at
10.4s: a much weaker effect, because generating and synthesising the first sentence
dominates. The honest summary is that streaming **shortens the slow tail** rather than
making every turn faster.

Details in `docs/STT移植_PHASE3.md`; raw data in `bench/results.json`.

## Known limitations

- **PyTorch (ROCm) requires the gfx1151-specific wheels.** Use `torch==2.8.0+rocm7.12.0` /
  `torchaudio==2.8.0a0+rocm7.12.0` from `repo.amd.com/rocm/whl/gfx1151/`. The generic
  `whl-multi-arch` build fails at runtime with `hipErrorInvalidImage`
  (`kpack_load_code_object failed with error: 13`) on **every GPU op**. torch bundles its own
  `rocm-sdk-libraries-gfx1151` (7.12), separate from the system ROCm 7.14, but it coexists in
  one process with CTranslate2 (which uses the system ROCm 7.14) as long as `LD_LIBRARY_PATH`
  includes `/opt/rocm/lib`.
- **torch must be imported before ctranslate2/whisperx.** `whisperx.load_model` imports
  ctranslate2 first (via `whisperx.asr`), and ctranslate2 imports torch during its own init.
  In that order ctranslate2 loads the system ROCm first, and torch's bundled `libhipblaslt.so.1`
  cannot resolve rocRoller symbols (`OSError: undefined symbol: _ZN9rocRoller...`). As a fix,
  `ttllm/server.py` does `import torch` at the very top.
- **Pin torchaudio to < 2.9.** pyannote-audio uses `torchaudio.info` / `AudioMetaData`, both
  removed in torchaudio 2.9, so 2.9+ fails at import with `AttributeError`.
- **Run three-vrm with the venv python.** system python (3.14) has no `aiohttp`.
- **WhisperX hits a GPU memory fault past 60 seconds** (a known issue with ROCm + PyTorch).
  `vtt` works around it by force-cutting at 55 seconds via VAD. Avoid
  long recordings on the browser side as well.
- **Silent utterances previously caused a 500 error**. The WhisperX `IndexError` thrown
  when Silero VAD returns "No active speech" is now caught inside `_transcribe_path` and
  reduced to an empty string (`ttllm/server.py`).
- **VOICEVOX runs CPU inference**. This avoids VRAM contention with ROCm; short utterances
  are comfortably real-time, but long responses may become TTS-bound.
- **Chrome's AudioContext** requires an initial click (user-gesture requirement).
- **Qwen3 thinking** is always OFF when going through ttllm, but if you call llama-server
  directly, you'll need to add `chat_template_kwargs` yourself.

## About paths

Every hard-coded path in shell scripts and Python has been replaced with `$USER` /
`os.path.expanduser("~/...")` — there's no remaining `/home/<someone>` hardcoding. It
works for other users too, as long as the directory layout
(`~/AIassistant/`, `~/llama.cpp/`, `~/whisperx/whisperX-rocm/.venv/`) is in place.

> :pencil: The pipeline's venv is **`~/AIassistant/ttllm/.venv`**
> (`ttllm/run.sh` / `install.sh`; override with `TTLLM_VENV`). It holds both STT backends,
> and three-vrm runs on the same python. The standalone
> `~/whisperx/whisperX-rocm/.venv` is kept only for using whisperX from the CLI.
> The `whisperX-rocm` symlink
> under AIassistant points there too. It previously used `~/AIzunda/whisperX-rocm`, but after
> upgrading to Ubuntu 26.04 (system python 3.14) the old venv's interpreter broke, so it was
> consolidated onto `~/whisperx`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Nothing happens when pressing 🎤 | Click the screen to enable AudioContext. Also check the browser's mic permission |
| Koteko doesn't speak / 500 error | Check ttllm logs via `tmux attach -t aiassistant`. Also test llama reachability with `curl :8001/health` |
| STT fails with `undefined symbol: _ZN9rocRoller...` | torch is imported after ctranslate2. Confirm the `import torch` at the top of `ttllm/server.py` |
| torch fails with `hipErrorInvalidImage` / `kpack_load_code_object failed` | The generic multi-arch torch is installed. Replace it with the gfx1151-specific wheels (README step 3) |
| `module 'torchaudio' has no attribute 'AudioMetaData'` | torchaudio is 2.9+. Downgrade to 2.8.x (`2.8.0a0+rocm7.12.0`) |
| three-vrm fails with `ModuleNotFoundError: aiohttp` | Run it with the venv python (`start_all.sh` already does). By hand: `$TTLLM_VENV/bin/python server.py`. If the venv lacks aiohttp: `VIRTUAL_ENV=$TTLLM_VENV uv pip install aiohttp` |
| CTranslate2 cmake fails on `cmake_minimum_required` | CMake 4.x. Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` |
| First utterance is slow | Preload the STT model with `curl -X POST :8001/warmup` (load + warmup pass, ~33s) |
| STT quietly switched to whisperX | NeMo failed and the router fell back. `curl -s :8001/health \| jq .stt.last_error` |
| NeMo dies with `Cannot reach https://huggingface.co/...: offline mode` | `run.sh` sets `HF_HUB_OFFLINE=1`. Run `ttllm/install.sh` to cache the model |
| Arms point the wrong way (after swapping VRM) | Flip the sign of `rotation.z` in `zundamon.html:applyRestPose` |
| Background doesn't change | Check the `/images_list` response in DevTools console. Reload the browser after adding images |
| VRM doesn't load | Verify `VRM_DIR` in `server.py` against the actual file path. The filename has to match the `VRM_URL` in `zundamon.html` |
| Stop everything | `~/AIassistant/stop_all.sh` |

## Summary

The goal is to run a "Koteko you can talk to with your voice" entirely locally on a single
AMD Ryzen AI Max+ 395 + ROCm machine, with no dependency on cloud APIs. By suppressing
Qwen3.6-35B-A3B (MoE) thinking, pipelining LLM→TTS, and using the turbo STT model, we hit
roughly 1 second to first audio while adding unobtrusive idle motion and background effects
with minimal code. Even on bandwidth-limited iGPUs, the ~3B-active MoE keeps both TTFT and
generation rate comfortable.

Possible extensions:

- Conversation history (currently stateless per turn — just pass it via the `history` parameter)
- Loading idle animations in VRMA format (currently procedural)
- Swapping VOICEVOX for a GPU build (to speed up TTS for long responses)
- Switching to a smaller STT model (medium can cut another 200–300 ms)
- Linking hand-gesture motion to live LLM streaming
- Leveraging multimodal (image input) support (available via `mmproj-F16.gguf`)
</content>

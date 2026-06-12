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
         ├─ WhisperX-ROCm (STT, large-v3-turbo)
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
| POST | `/warmup` | Preload WhisperX model |
| POST | `/transcribe` | Audio → text |
| POST | `/chat` | Text → LLM response (non-streaming) |
| POST | `/voice_chat` | Audio → response (non-streaming) |
| POST | `/voice_chat_stream` | Audio → SSE (transcript + token + done) **new** |

### three-vrm (port 8000)

| Method | Path | Purpose |
|---|---|---|
| GET | `/zundamon.html` | Viewer |
| GET | `/ws` | WebSocket (turn_start / speak / turn_end / transcript / error) |
| POST | `/speak` | Speak given text |
| POST | `/voice_chat_speak` | Audio → one-shot response (non-streaming) |
| POST | `/voice_chat_speak_stream` | Audio → pipelined response **new** |
| GET | `/images_list` | List background images |
| GET | `/images/{name}` | Serve background image |
| GET | `/vrm/{name}` | Serve VRM file |
| GET | `/status` | Connected client count |

## Known limitations

- **WhisperX hits a GPU memory fault past 60 seconds** (a known issue with ROCm 7.x +
  PyTorch nightly). `vtt` works around it by force-cutting at 55 seconds via VAD. Avoid
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
(`~/AIassistant/`, `~/llama.cpp/`, `~/AIzunda/whisperX-rocm/.venv/`) is in place.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Nothing happens when pressing 🎤 | Click the screen to enable AudioContext. Also check the browser's mic permission |
| Koteko doesn't speak / 500 error | Check ttllm logs via `tmux attach -t aiassistant`. Also test llama reachability with `curl :8001/health` |
| First utterance is slow | Preload WhisperX with `curl -X POST :8001/warmup` |
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

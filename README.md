# AIassistant — A Voice-Driven AI Pipeline for Talking with Koteko

> Voice: VOICEVOX:Zundamon

A fully local stack that runs **Voice → STT → LLM → TTS → VRM Lip Sync** end-to-end
on Ubuntu + AMD Ryzen AI Max+ 395 (ROCm). Press the 🎤 button in your browser and
Koteko replies in her own voice.

<img width="1219" height="1140" alt="https---qiita-image-store s3 ap-northeast-1 amazonaws com-0-263486-86fd1211-a196-4c6d-bf7b-e4ff53d8c5ba" src="https://github.com/user-attachments/assets/4292a4f1-5239-4a83-8c9e-3c3d4610fed2" />


```
Browser (three-vrm)
  └─ Mic capture (MediaRecorder webm/opus)
         ↓ POST /voice_chat_speak_stream
    three-vrm server (port 8000)
         ↓ POST /voice_chat_stream
       ttllm bridge (port 8001)
         ├─ WhisperX-ROCm (STT, large-v3-turbo)
         └─ llama-server (Qwen3.6-27B MTP, port 8080)
         ↓ Token stream over SSE
    three-vrm: split at sentence boundaries → VOICEVOX (port 50021) → push over WS
         ↓ WS (audio + visemes)
 Browser: AudioContext continuous playback + VRM lip sync + background + idle motion
```

## Components

| Path | Role | Port |
|---|---|---|
| `voicevox/` | VOICEVOX Engine (Docker, CPU inference) | 50021 |
| `~/llama.cpp/build/bin/llama-server` | Qwen3.6 inference (MTP speculative decoding) | 8080 |
| `qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf` | LLM model (includes one MTP layer) | — |
| `ttllm/` | FastAPI bridge (WhisperX + llama.cpp) | 8001 |
| `three-vrm/` | aiohttp server + VRM viewer (HTML/three-vrm) | 8000 |
| `vtt/` | CLI PTT mic (optional) | — |
| `images/` | VRM viewer backgrounds (rotated every 5 minutes) | — |
| `vroid/koteko.vrm` | Koteko VRM 1.0 model | — |
| `whisperX-rocm/` | ROCm fork of WhisperX (symlink to `~/AIzunda/whisperX-rocm`) | — |

### Prerequisites

- **OS** : Ubuntu 24.04.4 LTS
- **GPU** : AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151, 48GB VRAM)
- **ROCm** : 7.2.1 (`/opt/rocm`)
- **Python** : 3.12.3
- **Docker** : 29.x (for VOICEVOX)
- **Browser** : Google Chrome (Firefox also works since it uses `AudioContext`)
- **tmux / curl / uv / huggingface_hub (hf CLI)** : used by the startup script

For detailed setup, refer to the `READMEJ.md` in each subdirectory:
`ttllm/READMEJ.md` / `vtt/READMEJ.md` / `three-vrm/READMEJ.md` / `voicevox/READMEJ.md` /
`whisperX-rocm/READMEJ.md`.

## From `git clone` to Koteko speaking via `./start_all.sh`

## 1. Fetch the repository and dependencies

The main repository references `whisperX-rocm` / `llama.cpp` / `qwen3.6` via symlinks,
so first place the main repo and its dependencies **directly under your home directory**.

```bash
cd ~
git clone https://github.com/kotetsuy/AIassistant.git
git clone https://github.com/ggml-org/llama.cpp.git
```

You also need the ROCm forks of WhisperX and CTranslate2:

```bash
mkdir -p ~/whisperx && cd ~/whisperx
git clone https://github.com/<your_whisperx_rocm_fork>/whisperX-rocm.git
git clone https://github.com/<your_ctranslate2_rocm_fork>/ctranslate2-rocm.git
```

> :pencil: On the actual machine, `whisperX-rocm` is placed at `~/AIzunda/whisperX-rocm`,
> but for a fresh setup `~/whisperx/whisperX-rocm` works just as well. The `whisperX-rocm`
> entry inside AIassistant is a **symlink**, so re-point it to match your environment.

Refer this URL also.

https://qiita.com/kotetsu_yama/items/449e0d0527ab3a233fb8

---

## 2. Build CTranslate2-ROCm from source

Build the CTranslate2 backend that `faster-whisper` calls, with ROCm/HIP support.

```bash
cd ~/whisperx/ctranslate2-rocm
mkdir -p build && cd build

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMDGPU_TARGETS=gfx1151

cmake .. -DWITH_HIP=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_CLI=OFF
make -j$(nproc) && sudo make install
```

If `/usr/local/lib/libctranslate2.so` is installed, the build succeeded.

---

## 3. Create a venv for WhisperX-ROCm

```bash
cd ~/whisperx/whisperX-rocm
uv venv && uv pip install -e .

# Reinstall the Python bindings of the ROCm build of ctranslate2
rm -rf .venv/lib/python3.12/site-packages/ctranslate2*
export CTRANSLATE2_ROOT=/usr/local
uv pip install --reinstall pybind11 ~/whisperx/ctranslate2-rocm/python
```

Verify:

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# → CUDA: True  (ROCm's HIP layer translates the CUDA API)
.venv/bin/python -c "import ctranslate2; print(ctranslate2.__version__)"
```

---

## 4. Build llama.cpp from the **latest MTP-capable master**

**MTP (Multi-Token Prediction)** support for the Qwen3.5/3.6 family was merged in PR
[#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (as of May 2026). It's used
later in [§3](#3-mtp-投機デコードの仕組みと効果), so pull the latest master.

```bash
cd ~/llama.cpp
git pull --ff-only origin master

mkdir -p build && cd build
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMDGPU_TARGETS=gfx1151

cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Post-build check:

```bash
./bin/llama-server --version
# version: 9294 (...) etc.

./bin/llama-server --help | grep -A2 spec-type
# --spec-type none,draft-simple,draft-eagle3,draft-mtp,...
```

If `draft-mtp` appears, this build supports MTP.

---

## 5. Download the Qwen3.6 MTP model

Fetch the model with `hf` (the huggingface CLI, formerly `huggingface-cli`):

```bash
mkdir -p ~/qwen3.6
hf download am17an/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-MTP-Q8_0.gguf \
  --local-dir ~/qwen3.6
```

It's 29 GB. Confirm that the MTP layer is included:

```bash
cd ~/llama.cpp
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors \
  ~/qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf | grep -E "nextn|architecture"
# → qwen35.nextn_predict_layers = 1
```

If `nextn_predict_layers = 1` shows up, the gguf includes the MTP layer.

---

## 6. Create the symlinks inside AIassistant

Make `llama.cpp` / `whisperX-rocm` / `qwen3.6` reachable from `~/AIassistant` via relative paths.

```bash
cd ~/AIassistant
ln -sf ../llama.cpp llama.cpp
ln -sf ../whisperx/whisperX-rocm whisperX-rocm
ln -sf ../qwen3.6 qwen3.6

ls -la
# llama.cpp -> ../llama.cpp
# qwen3.6 -> ../qwen3.6
# whisperX-rocm -> ../whisperx/whisperX-rocm
```

---

## 7. Add ttllm bridge dependencies to the venv

```bash
cd ~/AIassistant/ttllm
./install.sh
```

This **adds `fastapi` / `uvicorn` / `httpx` / `python-multipart` / `pydantic` to the
WhisperX-ROCm venv** (no dedicated venv is created — the venv is shared).

---

## 8. Place the VRM model (Koteko)

Drop a VRM 1.0 model built with VRoid Studio or similar:

```bash
mkdir -p ~/AIassistant/vroid
cp /path/to/your_avatar.vrm ~/AIassistant/vroid/koteko.vrm
```

If you change the filename, update these two locations to match:

```python
# three-vrm/server.py
VRM_DIR = os.path.expanduser("~/AIassistant/vroid")
```
```html
<!-- three-vrm/TalkingHead/zundamon.html -->
const VRM_URL = "http://localhost:8000/vrm/koteko.vrm";
```

---

## 9. Pull the VOICEVOX Docker image

```bash
docker pull voicevox/voicevox_engine:cpu-ubuntu20.04-latest
# start_all.sh handles the launch, so a plain pull is enough here
```

The CPU inference build is used. The GPU is occupied by the LLM and STT, so leaving TTS on CPU is the safe choice.

---

## 10. Launch everything at once

```bash
cd ~/AIassistant
./start_all.sh
```

The following services come up serially, with HTTP health checks gating each step:

1. VOICEVOX (Docker, port 50021)
2. llama-server (Qwen3.6-27B-MTP, port 8080, with `--spec-type draft-mtp`)
3. ttllm bridge (port 8001)
4. WhisperX warmup (POSTs to `/warmup` to finish the first model load up front)
5. three-vrm server (port 8000)
6. Chrome auto-opens `http://localhost:8000/zundamon.html`
7. vtt (CLI PTT, optional)

All windows live inside the tmux session `aiassistant`, so:

```bash
tmux attach -t aiassistant   # view logs
~/AIassistant/stop_all.sh    # stop everything
```

is enough to operate the stack.

---

## 11. Verify operation

Once the browser opens, **click the screen once** to enable AudioContext (Chrome's
user-gesture requirement). Flow for the 🎤 button at the bottom right:

- **Long press (≥ 250 ms)**: records only while pressed, auto-submits on release
- **Short click**: starts recording → click again to submit

User utterances appear as light-blue subtitles, Koteko's replies as white subtitles.
If the first audio comes back in roughly 1 second, you're good.

## Start / stop everything

```bash
~/AIassistant/start_all.sh   # full stack startup + health check + WhisperX warmup + Chrome open
~/AIassistant/stop_all.sh    # stop the tmux session and VOICEVOX
~/AIassistant/stop_all.sh --keep-voicevox   # leave the VOICEVOX container running
```

`start_all.sh` creates the tmux session `aiassistant` and runs each service in its own window.

| window | command |
|---|---|
| 0 voicevox | `docker logs -f voicevox_engine` |
| 1 llama | `llama-server -m Qwen3.6-27B-MTP-Q8_0.gguf --port 8080 -ngl 99 -c 8192 --spec-type draft-mtp` |
| 2 ttllm | `ttllm/run.sh` (uvicorn) |
| 3 three-vrm | `python3 three-vrm/server.py` |
| 4 vtt | `vtt/run.sh --device USB` (CLI PTT, optional) |

View logs: `tmux attach -t aiassistant`  
Stop everything: `~/AIassistant/stop_all.sh`

The startup order is serialized to follow the dependency graph, with HTTP health-check
waits at each stage (only the llama-server model load has a generous 600-second timeout).
Right after ttllm comes up, `/warmup` is called to preload the WhisperX model, so the
very first utterance isn't slow.

## Using the browser UI

1. `start_all.sh` auto-opens Chrome at `http://localhost:8000/zundamon.html`
2. Click the screen once to enable AudioContext (browser user-gesture requirement)
3. The **🎤 button** at the bottom right
   - **Long press (≥ 250ms)**: records only while held, sends on release
   - **Short click**: starts recording → click again to send
4. User speech appears as light-blue subtitles, Koteko's replies as white subtitles

## Latency optimization

For short utterances (around "hello"), the target is roughly 1 second perceived latency;
even for long replies, the goal is **first audio in the 1-second range**.

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

### 4. MTP (Multi-Token Prediction) speculative decoding

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

### 5. Stop the previous utterance immediately when a new turn starts

The moment the mic is pressed, the client `stop(0)`s every currently scheduled
`AudioBufferSourceNode` and flushes the viseme queue (`stopAllPlayback`). Because it
doesn't wait for the server's `turn_start` to arrive, the UI feels instantly responsive.

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
Qwen3.6-27B (MTP) thinking, pipelining LLM→TTS, and using MTP speculative decoding, we
hit roughly 1 second to first audio with +32% generation speed, while adding unobtrusive
idle motion and background effects with minimal code.

Possible extensions:

- Conversation history (currently stateless per turn — just pass it via the `history` parameter)
- Loading idle animations in VRMA format (currently procedural)
- Swapping VOICEVOX for a GPU build (to speed up TTS for long responses)
- Switching to a smaller STT model (medium can cut another 200–300 ms)
- Linking hand-gesture motion to live LLM streaming

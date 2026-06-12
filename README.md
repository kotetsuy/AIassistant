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
         └─ llama-server (Qwen3.6-35B-A3B MoE, port 8080)
         ↓ Token stream over SSE
    three-vrm: split at sentence boundaries → VOICEVOX (port 50021) → push over WS
         ↓ WS (audio + visemes)
 Browser: AudioContext continuous playback + VRM lip sync + background + idle motion
```

> **The technical details (latency optimization, the MTP-vs-MoE decision, endpoint
> specifications, VRM effects, known limitations, troubleshooting) are collected in
> [`TECHNICAL.md`](./TECHNICAL.md).** This document is the step-by-step guide from
> `git clone` to Koteko speaking via `./start_all.sh`.

## Components

| Path | Role | Port |
|---|---|---|
| `voicevox/` | VOICEVOX Engine (Docker, CPU inference) | 50021 |
| `~/llama.cpp/build/bin/llama-server` | Qwen3.6 inference (MoE, ~3B active) | 8080 |
| `qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | LLM model (MoE, ~21GB) | — |
| `qwen3.6/mmproj-F16.gguf` | Vision encoder (optional, for multimodal) | — |
| `ttllm/` | FastAPI bridge (WhisperX + llama.cpp) | 8001 |
| `three-vrm/` | aiohttp server + VRM viewer (HTML/three-vrm) | 8000 |
| `vtt/` | CLI PTT mic (optional) | — |
| `images/` | VRM viewer backgrounds (rotated every 5 minutes) | — |
| `vroid/koteko.vrm` | Koteko VRM 1.0 model | — |
| `whisperX-rocm/` | ROCm fork of WhisperX (symlink to `~/AIzunda/whisperX-rocm`) | — |

> :pencil: The current default LLM is **Qwen3.6-35B-A3B (MoE)**. We previously used a dense
> Qwen3.6-27B with MTP speculative decoding, but switched because the MoE model is faster on
> bandwidth-limited iGPUs. See [`TECHNICAL.md`](./TECHNICAL.md) for the rationale and
> measurements.

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

Refer to this URL also:

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

## 4. Build llama.cpp with ROCm support

Qwen3.6-35B-A3B is a **MoE + Mamba hybrid**, so it needs a reasonably recent llama.cpp that
supports it. Pull the latest master and build.

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
```

> :bulb: The current default configuration (MoE) does not use MTP speculative decoding. You
> only need a `--spec-type draft-mtp`-capable master if you want to try the dense Qwen3.6-27B
> + MTP. See "The MTP-vs-MoE decision" in [`TECHNICAL.md`](./TECHNICAL.md) for details.

---

## 5. Download the Qwen3.6-35B-A3B (MoE) model

Fetch the model with `hf` (the huggingface CLI, formerly `huggingface-cli`). We use the
Unsloth Dynamic Q4_K_XL quantization:

```bash
mkdir -p ~/qwen3.6
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mmproj-F16.gguf \
  --local-dir ~/qwen3.6
```

The main model is about 21GB; the vision encoder (`mmproj-F16.gguf`, ~858MB) is only needed
if you use multimodal input. Verify:

```bash
ls -lh ~/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# ~20.8 GiB
```

> Only about 3B active parameters are used per token, so it is fast despite the 34.66B total
> (tg128 ≈ 50 t/s on the Ryzen AI Max+ 395). See `qwen3.6/READMEJ.md` for detailed benchmarks.

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

I made a sample VRM model. Feel free to use it.

https://hub.vroid.com/characters/2782544841139509367

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
2. llama-server (Qwen3.6-35B-A3B MoE, port 8080)
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
| 1 llama | `llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8080 -ngl 99 -c 8192 -fit off` |
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

## What to read next

- **Technical guide / tuning**: [`TECHNICAL.md`](./TECHNICAL.md)
  (latency optimization, the MTP-vs-MoE decision, endpoint specifications, VRM effects,
  known limitations, troubleshooting, summary)
- Per-component details: `ttllm/READMEJ.md` / `three-vrm/READMEJ.md` /
  `voicevox/READMEJ.md` / `vtt/READMEJ.md` / `qwen3.6/READMEJ.md`
</content>

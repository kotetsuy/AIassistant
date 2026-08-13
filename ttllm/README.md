# ttllm — STT ↔ llama.cpp bridge

A minimal FastAPI bridge that wires speech recognition to llama.cpp
(`llama-server`). POST audio in, get a transcript and an LLM reply back in one
call. Meant to be called directly from the `three-vrm` frontend.

STT runs on **NeMo Speech** by default with **WhisperX-ROCm** as a runtime
fallback; see [STT backends](#stt-backends) below.

## Layout

```
ttllm/
├── server.py    # FastAPI app
├── stt/         # STT backends
│   ├── base.py       # STTBackend ABC
│   ├── audio.py      # ffmpeg decode → 16kHz mono + tail padding
│   ├── nemo.py       # NeMoBackend (default)
│   ├── whisperx.py   # WhisperXBackend (fallback)
│   ├── streaming.py  # cache-aware streaming session
│   └── __init__.py   # STTRouter (selection + fallback)
├── install.sh   # builds the shared venv with BOTH backends
├── run.sh       # sets ROCm env vars and launches uvicorn
└── README.md    # this file
```

`install.sh` creates `ttllm/.venv`, which holds both STT stacks plus the bridge
and the three-vrm server. torch is pinned to 2.8.x because WhisperX needs
`torchaudio<2.9`; NeMo only requires `>=2.6`.

## Prerequisites

- ROCm 7.x at `/opt/rocm`, gfx1151 GPU
- The `Speech` symlink in `~/AIassistant` pointing at a NeMo Speech checkout on
  the **`rocm-inference`** branch
- The `whisperX-rocm` symlink, and ctranslate2-rocm built and installed to
  `/usr/local`
- `~/llama.cpp/build/bin/llama-server` already built
- Qwen3.6 model at `~/AIassistant/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

## Setup

```bash
cd ~/AIassistant/ttllm
./install.sh
```

This adds `fastapi` / `uvicorn` / `httpx` / `python-multipart` / `pydantic`
to the WhisperX venv.

## Launch

**1. Start llama-server** (in another terminal)

```bash
cd ~/llama.cpp/build/bin
./llama-server \
    -m ~/AIassistant/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl 99 -c 8192
```

**2. Start the bridge**

```bash
cd ~/AIassistant/ttllm
./run.sh
```

Listens on `http://0.0.0.0:8001` by default. Swagger UI is at
`http://localhost:8001/docs`.

## Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/health`     | Reachability of self / WhisperX / llama-server |
| POST   | `/warmup`     | Preload the WhisperX model to remove first-call latency |
| POST   | `/transcribe` | Audio → text (no LLM) |
| POST   | `/chat`       | Text → LLM reply |
| POST   | `/voice_chat` | Audio → transcript + LLM reply, in one call |

### `/voice_chat` (multipart/form-data)

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `audio`       | file            | —       | wav / mp3 / m4a etc. |
| `system`      | str             | Zundamon persona | Override system prompt |
| `history`     | str (JSON list) | `[]`    | `[{"role":"user","content":"..."}]` |
| `temperature` | float           | `0.7`   | |
| `max_tokens`  | int             | `512`   | |

Response:

```json
{ "transcript": "こんにちは", "reply": "こんにちはなのだ！" }
```

### `/chat` (application/json)

```json
{
  "text": "自己紹介して",
  "history": [],
  "system": null,
  "temperature": 0.7,
  "max_tokens": 512
}
```

### Examples

```bash
# From an audio file all the way to a reply
curl -X POST http://localhost:8001/voice_chat \
    -F "audio=@sample.wav"

# Text-only LLM call
curl -X POST http://localhost:8001/chat \
    -H 'Content-Type: application/json' \
    -d '{"text":"ずんだ餅について教えてなのだ"}'

# Preload the model (kills first-call latency)
curl -X POST http://localhost:8001/warmup
```

## Environment variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WHISPER_MODEL`        | `large-v3`              | WhisperX model name |
| `WHISPER_LANGUAGE`     | `ja`                    | Recognition language |
| `WHISPER_COMPUTE_TYPE` | `float16`               | `float16` / `int8_float16` etc. |
| `WHISPER_DEVICE`       | `cuda`                  | GPU is used through ROCm's HIP layer |
| `WHISPER_BATCH_SIZE`   | `8`                     | |
| `WHISPER_VAD_METHOD`   | `silero`                | `silero` / `pyannote` |
| `LLAMA_SERVER_URL`     | `http://localhost:8080` | URL of llama-server |
| `LLAMA_TIMEOUT`        | `120`                   | seconds |
| `SYSTEM_PROMPT`        | Zundamon persona        | Default system prompt |
| `BRIDGE_HOST`          | `0.0.0.0`               | |
| `BRIDGE_PORT`          | `8001`                  | |
| `TTLLM_VENV`           | `~/AIassistant/ttllm/.venv` | Path of the shared venv (holds both STT backends) |
| `STT_BACKEND`          | `auto`                  | `nemo` / `whisperx` / `auto` |
| `STT_FALLBACK`         | `whisperx`              | Fallback for `auto`; `none` disables it |
| `STT_EAGER_FALLBACK`   | `1`                     | Preload the fallback during warmup |
| `NEMO_MODEL`           | `nvidia/nemotron-3.5-asr-streaming-0.6b` | |
| `NEMO_LANGUAGE`        | `ja-JP`                 | A bare `ja` is rejected — it is not a `prompt_dictionary` key |
| `NEMO_DEVICE`          | `cuda`                  | ROCm reports itself as `cuda` |
| `NEMO_ATT_CONTEXT_SIZE`| `[56,13]`               | Streaming chunk: `[left,right]` in 80ms frames, chunk = (right+1)×80ms |

## STT backends

`STT_BACKEND=auto` uses NeMo and switches to WhisperX if NeMo fails to load or
raises mid-request, then stays there. The failing request itself is retried on
the fallback, so nothing is dropped.

Every fallback is logged at WARNING and reflected in `/health`:

```bash
curl -s localhost:8001/health | jq .stt
# { "backend": "nemo", "requested": "auto", "fallback_active": false, ... }
```

Silent degradation is the thing this is designed to avoid — a quietly different
model answering questions is worse than a visible failure.

## Calling from a frontend

The server starts with permissive CORS, so browsers (e.g. `talkinghead` /
`zundavrm`) can `fetch` it directly:

```javascript
const fd = new FormData();
fd.append("audio", blob, "utterance.wav");
const res = await fetch("http://localhost:8001/voice_chat", {
  method: "POST",
  body: fd,
});
const { transcript, reply } = await res.json();
```

## Caveats

- Audio longer than 60 s can trigger ROCm memory faults (see `~/CLAUDE.md`).
  Chunk on the client side for long inputs.
- `/chat` and `/voice_chat` are stateless. Keep history on the caller side
  and pass it via the `history` field.
- Bridging to TTS (VOICEVOX) is out of scope here. The receiver of `reply`
  is responsible for synthesis.

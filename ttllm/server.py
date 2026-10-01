import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# torch must be imported before ctranslate2/whisperx so its bundled ROCm
# (rocm-sdk-libraries-gfx1151) loads first. Otherwise ctranslate2 preloads the
# system ROCm libs and torch's bundled libhipblaslt fails to resolve rocRoller
# symbols (OSError: undefined symbol _ZN9rocRoller...).
import torch  # noqa: F401

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from stt import STTRouter

# uvicorn 配下なので uvicorn.error ロガーに乗せれば tmux の ttllm ウィンドウに出る。
logger = logging.getLogger("uvicorn.error")

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:9931").rstrip("/")
LLAMA_TIMEOUT = float(os.getenv("LLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "あなたはオリジナルキャラです。名前はコテコ。一人称は「コテコ」、元気いっぱいの明るい女の子として、「〜だよ！」「〜だね！」のような弾んだ口調で、親しみやすく簡潔に話してください。",
)

_stt = STTRouter()


def get_model():
    """Load the active STT backend (and the fallback, if eager)."""
    _stt.load()
    return _stt.active


app = FastAPI(title="ttllm bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    text: str
    history: List[Message] = []
    system: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512


class ChatResponse(BaseModel):
    reply: str


class TranscribeResponse(BaseModel):
    transcript: str


class VoiceChatResponse(BaseModel):
    transcript: str
    reply: str


def _transcribe_path(path: str) -> str:
    """音声ファイルパス → 転写テキスト。バックエンドの違いはここから先には出ない。"""
    try:
        return _stt.transcribe_path(path)
    except Exception as e:  # noqa: BLE001 — 502 の方が「無言で空文字」より安全
        logger.exception("STT failed on all backends")
        raise HTTPException(503, f"STT failed: {e}") from e


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await upload.read())
        return f.name


def _build_messages(
    user_text: str,
    system: Optional[str],
    history: List[dict],
) -> List[dict]:
    messages: List[dict] = []
    sys_msg = system if system is not None else SYSTEM_PROMPT
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


async def _call_llama(messages: List[dict], temperature: float, max_tokens: int) -> str:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # Qwen3 系は既定で thinking を吐くので、chat template 側で切る。
        # これを渡さないと reasoning_content に数百トークン食われて
        # content が空のまま max_tokens に到達する。
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
            r = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"llama-server error: {e}") from e

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise HTTPException(502, f"unexpected llama-server response: {data}") from e


@app.get("/health")
async def health():
    llama_reachable = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LLAMA_SERVER_URL}/health")
            llama_reachable = r.status_code < 500
    except httpx.HTTPError:
        pass
    return {
        "ok": True,
        "stt": _stt.info(),
        "llama": {"url": LLAMA_SERVER_URL, "reachable": llama_reachable},
    }


@app.post("/warmup")
async def warmup():
    # load_model() は同期ブロッキングなので、スレッドプールに逃がして
    # イベントループ (単一ワーカー) を止めない。これを直に await 無しで
    # 呼ぶと warmup 中は /health すら応答できなくなる。
    await run_in_threadpool(get_model)
    return {"loaded": True}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    path = await _save_upload(audio)
    try:
        text = await run_in_threadpool(_transcribe_path, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return TranscribeResponse(transcript=text)


@app.websocket("/transcribe_stream")
async def transcribe_stream(ws: WebSocket):
    """Cache-aware streaming STT.

    Client -> server: binary frames of float32 little-endian PCM, 16kHz mono.
                      A text frame {"type":"end"} closes the utterance.
    Server -> client: {"type":"partial","text":...} while speaking,
                      {"type":"final","text":...} once, then the socket closes.

    Only the NeMo backend streams; whisperX has no equivalent, so the client is
    told to fall back to the one-shot /voice_chat_stream path.
    """
    await ws.accept()

    backend = await run_in_threadpool(get_model)
    if not backend.supports_streaming:
        await ws.send_json({
            "type": "error",
            "error": f"backend {backend.name} does not support streaming",
            "fallback": "batch",
        })
        await ws.close()
        return

    session = await run_in_threadpool(backend.new_stream)
    t_started = time.perf_counter()
    t_last_audio = None
    last_sent = ""

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

            if (data := msg.get("bytes")) is not None:
                pcm = np.frombuffer(data, dtype="<f4")
                if not pcm.size:
                    continue
                t_last_audio = time.perf_counter()
                text = await run_in_threadpool(session.add_audio, pcm)
                if text and text != last_sent:
                    last_sent = text
                    await ws.send_json({"type": "partial", "text": text})
                continue

            if (raw := msg.get("text")) is not None:
                try:
                    kind = json.loads(raw).get("type")
                except json.JSONDecodeError:
                    kind = None
                if kind != "end":
                    continue

                t_end = time.perf_counter()
                text = await run_in_threadpool(session.finish)
                finalize_ms = (time.perf_counter() - t_end) * 1000
                logger.info(
                    "STT[stream] finalize %.0fms (utterance %.2fs): %r",
                    finalize_ms,
                    (t_last_audio - t_started) if t_last_audio else 0.0,
                    text[:40],
                )
                await ws.send_json({"type": "final", "text": text, "finalize_ms": round(finalize_ms)})
                await ws.close()
                return
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("streaming STT failed")
        try:
            await ws.send_json({"type": "error", "error": str(e), "fallback": "batch"})
            await ws.close()
        except RuntimeError:
            pass


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = _build_messages(
        req.text, req.system, [m.model_dump() for m in req.history]
    )
    reply = await _call_llama(messages, req.temperature, req.max_tokens)
    return ChatResponse(reply=reply)


@app.post("/voice_chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    system: Optional[str] = Form(None),
    history: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: int = Form(512),
):
    path = await _save_upload(audio)
    try:
        transcript = await run_in_threadpool(_transcribe_path, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if not transcript:
        return VoiceChatResponse(transcript="", reply="")

    parsed_history: List[dict] = []
    if history:
        try:
            raw = json.loads(history)
            parsed_history = [
                {"role": m["role"], "content": m["content"]} for m in raw
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise HTTPException(400, f"invalid history JSON: {e}")

    messages = _build_messages(transcript, system, parsed_history)
    reply = await _call_llama(messages, temperature, max_tokens)
    return VoiceChatResponse(transcript=transcript, reply=reply)


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/voice_chat_stream")
async def voice_chat_stream(
    audio: UploadFile = File(...),
    system: Optional[str] = Form(None),
    history: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    max_tokens: int = Form(512),
):
    """STT → LLM をストリームし、SSE で {transcript, token..., done} を返す。"""
    path = await _save_upload(audio)
    try:
        transcript = await run_in_threadpool(_transcribe_path, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    parsed_history: List[dict] = []
    if history:
        try:
            raw = json.loads(history)
            parsed_history = [
                {"role": m["role"], "content": m["content"]} for m in raw
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise HTTPException(400, f"invalid history JSON: {e}")

    async def event_stream():
        yield _sse({"type": "transcript", "text": transcript})

        if not transcript:
            yield _sse({"type": "done", "reply": ""})
            return

        messages = _build_messages(transcript, system, parsed_history)
        async for event in _llm_sse(messages, temperature, max_tokens):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat_stream")
async def chat_stream(req: ChatRequest):
    """テキスト入力版の SSE。STT をストリーミングで済ませた呼び出し元が使う。

    /voice_chat_stream と同じ {token..., done} を返す (transcript は出さない)。
    """
    messages = _build_messages(
        req.text, req.system, [m.model_dump() for m in req.history]
    )

    async def event_stream():
        async for event in _llm_sse(messages, req.temperature, req.max_tokens):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _llm_sse(messages: List[dict], temperature: float, max_tokens: int):
    """llama-server のトークンストリームを SSE イベントに変換する。"""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # Qwen3 系は既定で thinking を吐くので、chat template 側で切る。
        "chat_template_kwargs": {"enable_thinking": False},
    }

    reply_parts: List[str] = []
    llm_t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    n_chunks = 0
    try:
        async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        delta = chunk["choices"][0].get("delta") or {}
                    except (KeyError, IndexError):
                        continue
                    text = delta.get("content") or ""
                    if text:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - llm_t0) * 1000
                        n_chunks += 1
                        reply_parts.append(text)
                        yield _sse({"type": "token", "text": text})
    except httpx.HTTPError as e:
        yield _sse({"type": "error", "error": f"llama-server: {e}"})

    gen_ms = (time.perf_counter() - llm_t0) * 1000
    if ttft_ms is not None and n_chunks > 1 and gen_ms > ttft_ms:
        tps = (n_chunks - 1) / ((gen_ms - ttft_ms) / 1000)
    else:
        tps = 0.0
    logger.info(
        "LLM TTFT %sms, total %.0fms, %d chunks, %.1f chunk/s",
        f"{ttft_ms:.0f}" if ttft_ms is not None else "n/a",
        gen_ms, n_chunks, tps,
    )

    yield _sse({"type": "done", "reply": "".join(reply_parts)})

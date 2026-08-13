#!/usr/bin/env python3
"""
TalkingHead サーバー
- http://localhost:8000/zundamon.html  ← ブラウザで開く
- POST /speak  {"text": "...", "speaker_id": 3}  ← パイプラインから呼ぶ
"""
import asyncio
import base64
import json
import os
import re
import uuid
import weakref

import aiohttp
from aiohttp import web

VOICEVOX_URL = os.getenv("VOICEVOX_URL", "http://localhost:50021")
TTLLM_URL = os.getenv("TTLLM_URL", "http://localhost:8001")
TTLLM_TIMEOUT = float(os.getenv("TTLLM_TIMEOUT", "180"))
VRM_DIR = os.path.expanduser("~/AIassistant/vroid")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TalkingHead")
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.expanduser("~/AIzunda/images"))

clients: weakref.WeakSet = weakref.WeakSet()

VOWEL_TO_VISEME = {
    "a": "aa", "i": "I", "u": "U", "e": "E", "o": "O",
    "N": "nn", "cl": "sil", "pau": "sil",
}

CONSONANT_TO_VISEME = {
    "p": "PP",  "b": "PP",  "m": "PP",
    "py": "PP", "by": "PP", "my": "PP",
    "f": "FF",
    "s": "SS",  "z": "SS",  "sh": "SS",
    "t": "DD",  "d": "DD",  "ts": "DD",
    "k": "kk",  "g": "kk",  "ky": "kk", "gy": "kk",
    "ch": "CH", "j": "CH",
    "n": "nn",  "ny": "nn",
    "r": "RR",  "ry": "RR",
    "h": "sil", "hy": "sil", "w": "sil", "y": "sil",
}


def mora_to_visemes(accent_phrases: list) -> tuple[list, list, list]:
    """VOICEVOXのaccentPhrasesをTalkingHeadのvisemeデータに変換する。"""
    visemes, vtimes, vdurations = [], [], []
    t = 0.0

    for phrase in accent_phrases:
        for mora in phrase.get("moras", []):
            c = mora.get("consonant")
            cl = mora.get("consonant_length") or 0.0
            v = mora.get("vowel", "pau")
            vl = mora.get("vowel_length") or 0.0

            if c and cl > 0:
                visemes.append(CONSONANT_TO_VISEME.get(c, "sil"))
                vtimes.append(int(t * 1000))
                vdurations.append(max(1, int(cl * 1000)))
                t += cl

            if v and vl > 0:
                visemes.append(VOWEL_TO_VISEME.get(v, "sil"))
                vtimes.append(int(t * 1000))
                vdurations.append(max(1, int(vl * 1000)))
                t += vl

        pause = phrase.get("pause_mora")
        if pause:
            pl = pause.get("vowel_length") or 0.0
            if pl > 0:
                visemes.append("sil")
                vtimes.append(int(t * 1000))
                vdurations.append(max(1, int(pl * 1000)))
                t += pl

    return visemes, vtimes, vdurations


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.ERROR:
                break
    finally:
        clients.discard(ws)
    return ws


async def _broadcast(message: dict) -> int:
    """接続中の全 WS クライアントに JSON を送る。"""
    payload = json.dumps(message, ensure_ascii=False)
    dead = []
    for ws in list(clients):
        try:
            await ws.send_str(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)
    return len(clients)


async def _synth_chunk(
    session: aiohttp.ClientSession, text: str, speaker_id: int
) -> tuple[bytes, list, list, list]:
    """1 文ぶんの WAV + visemes を合成して返す (ブロードキャストはしない)。"""
    async with session.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
    ) as resp:
        if resp.status != 200:
            raise web.HTTPBadGateway(
                reason=f"audio_query failed ({resp.status}): {await resp.text()}"
            )
        query = await resp.json()

    async with session.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
        headers={"Content-Type": "application/json"},
    ) as resp:
        if resp.status != 200:
            raise web.HTTPBadGateway(
                reason=f"synthesis failed ({resp.status}): {await resp.text()}"
            )
        wav_bytes = await resp.read()

    visemes, vtimes, vdurations = mora_to_visemes(query.get("accent_phrases", []))
    return wav_bytes, visemes, vtimes, vdurations


async def _synthesize_and_broadcast(text: str, speaker_id: int) -> dict:
    """VOICEVOX → WAV + visemes を生成し、接続中の WS クライアントに配信。"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": speaker_id},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise web.HTTPBadGateway(
                    reason=f"audio_query failed ({resp.status}): {body}"
                )
            query = await resp.json()

        async with session.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": speaker_id},
            json=query,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise web.HTTPBadGateway(
                    reason=f"synthesis failed ({resp.status}): {body}"
                )
            wav_bytes = await resp.read()

    visemes, vtimes, vdurations = mora_to_visemes(query.get("accent_phrases", []))

    message = json.dumps({
        "type": "speak",
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
        "visemes": visemes,
        "vtimes": vtimes,
        "vdurations": vdurations,
        "text": text,
    })

    dead = []
    for ws in list(clients):
        try:
            await ws.send_str(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

    return {"visemes": len(visemes), "clients": len(clients)}


async def speak_handler(request: web.Request) -> web.Response:
    data = await request.json()
    text: str = data.get("text", "").strip()
    speaker_id: int = data.get("speaker_id", 3)

    if not text:
        return web.json_response({"error": "no text"}, status=400)

    try:
        result = await _synthesize_and_broadcast(text, speaker_id)
    except web.HTTPBadGateway as e:
        return web.json_response({"error": e.reason}, status=502)

    return web.json_response({"ok": True, **result})


async def voice_chat_speak_handler(request: web.Request) -> web.Response:
    """音声 → ttllm (/voice_chat) → VOICEVOX 合成 → WS 配信 をワンショットで実行。"""
    reader = await request.multipart()

    audio_field = None
    speaker_id = 3
    system: str | None = None
    history: str | None = None
    temperature = 0.7
    max_tokens = 512

    async for part in reader:
        if part.name == "audio":
            audio_field = (part.filename or "audio.wav", await part.read(decode=False))
        elif part.name == "speaker_id":
            try:
                speaker_id = int((await part.text()).strip())
            except ValueError:
                pass
        elif part.name == "system":
            system = await part.text()
        elif part.name == "history":
            history = await part.text()
        elif part.name == "temperature":
            try:
                temperature = float((await part.text()).strip())
            except ValueError:
                pass
        elif part.name == "max_tokens":
            try:
                max_tokens = int((await part.text()).strip())
            except ValueError:
                pass

    if not audio_field:
        return web.json_response({"error": "audio field required"}, status=400)

    filename, audio_bytes = audio_field

    form = aiohttp.FormData()
    form.add_field("audio", audio_bytes, filename=filename,
                   content_type="application/octet-stream")
    form.add_field("temperature", str(temperature))
    form.add_field("max_tokens", str(max_tokens))
    if system is not None:
        form.add_field("system", system)
    if history is not None:
        form.add_field("history", history)

    timeout = aiohttp.ClientTimeout(total=TTLLM_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{TTLLM_URL}/voice_chat", data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return web.json_response(
                        {"error": f"ttllm /voice_chat failed ({resp.status}): {body}"},
                        status=502,
                    )
                chat = await resp.json()
    except aiohttp.ClientError as e:
        return web.json_response({"error": f"ttllm unreachable: {e}"}, status=502)

    transcript = (chat.get("transcript") or "").strip()
    reply = (chat.get("reply") or "").strip()

    if not reply:
        return web.json_response({
            "ok": True,
            "transcript": transcript,
            "reply": "",
            "visemes": 0,
            "clients": len(clients),
            "note": "empty reply (no transcript or LLM returned empty)",
        })

    try:
        result = await _synthesize_and_broadcast(reply, speaker_id)
    except web.HTTPBadGateway as e:
        return web.json_response(
            {"error": e.reason, "transcript": transcript, "reply": reply},
            status=502,
        )

    return web.json_response({
        "ok": True,
        "transcript": transcript,
        "reply": reply,
        **result,
    })


_SENTENCE_END = re.compile(r"[。！？!?\n]")
_SOFT_BREAK = re.compile(r"[、,]")
_MAX_CHUNK_CHARS = 60  # 読点も句点も来ない長文の保険


def _split_sentences(buf: str, flush: bool = False) -> tuple[list[str], str]:
    """buf から完成した文を切り出す。flush=True なら残りも全部返す。"""
    out: list[str] = []
    while True:
        m = _SENTENCE_END.search(buf)
        if m:
            end = m.end()
            piece = buf[:end].strip()
            buf = buf[end:]
            if piece:
                out.append(piece)
            continue
        if len(buf) >= _MAX_CHUNK_CHARS:
            m2 = _SOFT_BREAK.search(buf, _MAX_CHUNK_CHARS // 2)
            cut = m2.end() if m2 else _MAX_CHUNK_CHARS
            piece = buf[:cut].strip()
            buf = buf[cut:]
            if piece:
                out.append(piece)
            continue
        break
    if flush and buf.strip():
        out.append(buf.strip())
        buf = ""
    return out, buf


async def voice_chat_speak_stream_handler(request: web.Request) -> web.Response:
    """音声 → ttllm /voice_chat_stream → 文単位で VOICEVOX + WS ブロードキャスト。

    LLM デコードと TTS 合成を並列化することで体感遅延を縮める。
    """
    reader = await request.multipart()

    audio_field = None
    speaker_id = 3
    system: str | None = None
    history: str | None = None
    temperature = 0.7
    max_tokens = 512

    async for part in reader:
        if part.name == "audio":
            audio_field = (part.filename or "audio.wav", await part.read(decode=False))
        elif part.name == "speaker_id":
            try:
                speaker_id = int((await part.text()).strip())
            except ValueError:
                pass
        elif part.name == "system":
            system = await part.text()
        elif part.name == "history":
            history = await part.text()
        elif part.name == "temperature":
            try:
                temperature = float((await part.text()).strip())
            except ValueError:
                pass
        elif part.name == "max_tokens":
            try:
                max_tokens = int((await part.text()).strip())
            except ValueError:
                pass

    if not audio_field:
        return web.json_response({"error": "audio field required"}, status=400)

    filename, audio_bytes = audio_field

    turn_id = uuid.uuid4().hex
    transcript = ""
    reply_accum = ""
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
    chunks_sent = 0

    await _broadcast({"type": "turn_start", "turn_id": turn_id})

    async def tts_consumer():
        """sentence_q を順に VOICEVOX → WS へ流す (順序保証のため直列)。"""
        nonlocal chunks_sent
        async with aiohttp.ClientSession() as session:
            while True:
                sentence = await sentence_q.get()
                if sentence is None:
                    return
                try:
                    wav, visemes, vtimes, vdurations = await _synth_chunk(
                        session, sentence, speaker_id
                    )
                except web.HTTPBadGateway as e:
                    await _broadcast({"type": "error", "turn_id": turn_id, "error": e.reason})
                    continue
                await _broadcast({
                    "type": "speak",
                    "turn_id": turn_id,
                    "seq": chunks_sent,
                    "audio": base64.b64encode(wav).decode("ascii"),
                    "visemes": visemes,
                    "vtimes": vtimes,
                    "vdurations": vdurations,
                    "text": sentence,
                })
                chunks_sent += 1

    consumer_task = asyncio.create_task(tts_consumer())

    form = aiohttp.FormData()
    form.add_field("audio", audio_bytes, filename=filename,
                   content_type="application/octet-stream")
    form.add_field("temperature", str(temperature))
    form.add_field("max_tokens", str(max_tokens))
    if system is not None:
        form.add_field("system", system)
    if history is not None:
        form.add_field("history", history)

    timeout = aiohttp.ClientTimeout(total=TTLLM_TIMEOUT)
    buf = ""
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{TTLLM_URL}/voice_chat_stream", data=form
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    await sentence_q.put(None)
                    await consumer_task
                    await _broadcast({"type": "turn_end", "turn_id": turn_id})
                    return web.json_response(
                        {"error": f"ttllm /voice_chat_stream failed ({resp.status}): {body}"},
                        status=502,
                    )
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    t = msg.get("type")
                    if t == "transcript":
                        transcript = msg.get("text", "") or ""
                        await _broadcast({
                            "type": "transcript",
                            "turn_id": turn_id,
                            "text": transcript,
                        })
                    elif t == "token":
                        buf += msg.get("text", "") or ""
                        sentences, buf = _split_sentences(buf, flush=False)
                        for s in sentences:
                            reply_accum += s
                            await sentence_q.put(s)
                    elif t == "error":
                        await _broadcast({
                            "type": "error",
                            "turn_id": turn_id,
                            "error": msg.get("error", ""),
                        })
                    elif t == "done":
                        final_reply = msg.get("reply", "")
                        if final_reply:
                            reply_accum = final_reply
                        break
    except aiohttp.ClientError as e:
        await sentence_q.put(None)
        await consumer_task
        await _broadcast({"type": "turn_end", "turn_id": turn_id})
        return web.json_response({"error": f"ttllm unreachable: {e}"}, status=502)

    tail, _ = _split_sentences(buf, flush=True)
    for s in tail:
        if not reply_accum.endswith(s):
            reply_accum += s
        await sentence_q.put(s)

    await sentence_q.put(None)
    await consumer_task

    await _broadcast({
        "type": "turn_end",
        "turn_id": turn_id,
        "chunks": chunks_sent,
    })

    return web.json_response({
        "ok": True,
        "transcript": transcript,
        "reply": reply_accum,
        "chunks": chunks_sent,
        "turn_id": turn_id,
    })


async def stt_chat_stream_handler(request: web.Request) -> web.WebSocketResponse:
    """ストリーミング STT 版の会話経路。

    ブラウザ (WS, float32 PCM 16kHz mono)
      → ttllm /transcribe_stream (WS)          … 発話中から部分転写
      → 確定したら ttllm /chat_stream (SSE)     … LLM
      → 文単位で VOICEVOX → 既存の WS ブロードキャスト

    一括経路 (/voice_chat_speak_stream) はそのまま残してあり、ブラウザ側が
    どちらを使うかを選ぶ。ttllm 側が streaming 非対応 (whisperX 稼働中など) の
    場合は type=error, fallback=batch を返すので、ブラウザは一括経路に倒す。
    """
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    speaker_id = int(request.query.get("speaker_id", "3"))
    turn_id = uuid.uuid4().hex
    transcript = ""

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TTLLM_TIMEOUT)
        ) as session:
            async with session.ws_connect(f"{TTLLM_URL}/transcribe_stream") as up:
                transcript = await _relay_stt(ws, up, turn_id)

            if transcript is None:          # streaming unsupported upstream
                await ws.send_json({"type": "error", "error": "streaming unsupported",
                                    "fallback": "batch"})
                return ws
            if not transcript:
                await ws.send_json({"type": "done", "transcript": "", "reply": "", "chunks": 0})
                return ws

            await ws.send_json({"type": "final", "text": transcript})
            await _broadcast({"type": "turn_start", "turn_id": turn_id})
            reply, chunks = await _llm_to_speech(session, transcript, speaker_id, turn_id)

        await _broadcast({"type": "turn_end", "turn_id": turn_id, "chunks": chunks})
        await ws.send_json({"type": "done", "transcript": transcript,
                            "reply": reply, "chunks": chunks, "turn_id": turn_id})
    except aiohttp.ClientError as e:
        await _broadcast({"type": "turn_end", "turn_id": turn_id})
        if not ws.closed:
            await ws.send_json({"type": "error", "error": f"ttllm unreachable: {e}",
                                "fallback": "batch"})
    return ws


async def _relay_stt(down: web.WebSocketResponse, up, turn_id: str):
    """ブラウザ ↔ ttllm の PCM / 転写中継。確定転写を返す (非対応なら None)。

    上流の読み出しは専用タスクにする。PCM 送信ループの合間にポーリングする形だと
    部分転写が溜まってしまい、発話終了時にまとめて届いて字幕がライブにならない。
    """
    result: asyncio.Future = asyncio.get_running_loop().create_future()

    async def pump_up():
        async for reply in up:
            if reply.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(reply.data)
            kind = data.get("type")
            if kind == "partial":
                if not down.closed:
                    await down.send_json(data)
                await _broadcast({"type": "transcript", "turn_id": turn_id,
                                  "text": data.get("text", ""), "partial": True})
            elif kind == "final":
                if not result.done():
                    result.set_result(data.get("text", "") or "")
                return
            elif kind == "error":
                if not result.done():
                    result.set_result(None)
                return
        if not result.done():
            result.set_result("")

    pump = asyncio.create_task(pump_up())
    try:
        async for msg in down:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if result.done():      # upstream already gave up
                    break
                await up.send_bytes(msg.data)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                if json.loads(msg.data).get("type") != "end":
                    continue
                await up.send_str(json.dumps({"type": "end"}))
                return await result
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break
        return await result if result.done() else ""
    finally:
        pump.cancel()


async def _llm_to_speech(session, transcript: str, speaker_id: int, turn_id: str):
    """ttllm /chat_stream のトークンを文に切って VOICEVOX → WS へ流す。

    /voice_chat_speak_stream の後半と同じ処理だが、入力が音声ではなく確定転写。
    """
    sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
    chunks_sent = 0
    reply_accum = ""

    async def tts_consumer():
        nonlocal chunks_sent
        async with aiohttp.ClientSession() as tts_session:
            while True:
                sentence = await sentence_q.get()
                if sentence is None:
                    return
                try:
                    wav, visemes, vtimes, vdurations = await _synth_chunk(
                        tts_session, sentence, speaker_id
                    )
                except web.HTTPBadGateway as e:
                    await _broadcast({"type": "error", "turn_id": turn_id, "error": e.reason})
                    continue
                await _broadcast({
                    "type": "speak",
                    "turn_id": turn_id,
                    "seq": chunks_sent,
                    "audio": base64.b64encode(wav).decode("ascii"),
                    "visemes": visemes,
                    "vtimes": vtimes,
                    "vdurations": vdurations,
                    "text": sentence,
                })
                chunks_sent += 1

    consumer_task = asyncio.create_task(tts_consumer())
    buf = ""

    async with session.post(f"{TTLLM_URL}/chat_stream", json={"text": transcript}) as resp:
        if resp.status != 200:
            await sentence_q.put(None)
            await consumer_task
            raise aiohttp.ClientError(f"/chat_stream failed ({resp.status})")
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "token":
                buf += msg.get("text", "") or ""
                sentences, buf = _split_sentences(buf, flush=False)
                for s in sentences:
                    reply_accum += s
                    await sentence_q.put(s)
            elif t == "error":
                await _broadcast({"type": "error", "turn_id": turn_id,
                                  "error": msg.get("error", "")})
            elif t == "done":
                if msg.get("reply"):
                    reply_accum = msg["reply"]
                break

    tail, _ = _split_sentences(buf, flush=True)
    for s in tail:
        await sentence_q.put(s)
    await sentence_q.put(None)
    await consumer_task
    return reply_accum, chunks_sent


async def vrm_handler(request: web.Request) -> web.Response:
    filename = os.path.basename(request.match_info["filename"])
    filepath = os.path.join(VRM_DIR, filename)
    if not os.path.isfile(filepath):
        raise web.HTTPNotFound()
    return web.FileResponse(filepath)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


async def images_list_handler(request: web.Request) -> web.Response:
    if not os.path.isdir(IMAGES_DIR):
        return web.json_response({"images": []})
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(_IMAGE_EXTS)
    )
    return web.json_response({
        "images": [f"/images/{f}" for f in files],
    })


async def status_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "clients": len(clients),
        "voicevox": VOICEVOX_URL,
        "vrm_dir": VRM_DIR,
    })


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/speak", speak_handler)
    app.router.add_post("/voice_chat_speak", voice_chat_speak_handler)
    app.router.add_post("/voice_chat_speak_stream", voice_chat_speak_stream_handler)
    app.router.add_get("/stt_chat_stream", stt_chat_stream_handler)
    app.router.add_get("/vrm/{filename}", vrm_handler)
    app.router.add_get("/images_list", images_list_handler)
    app.router.add_get("/status", status_handler)
    if os.path.isdir(IMAGES_DIR):
        app.router.add_static("/images", IMAGES_DIR)
    app.router.add_static("/", STATIC_DIR, show_index=True)
    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 50)
    print("TalkingHead server: http://localhost:8000")
    print("Avatar page:        http://localhost:8000/zundamon.html")
    print("Speak endpoint:     POST http://localhost:8000/speak")
    print('  body: {"text": "ずんだもんなのだ", "speaker_id": 3}')
    print("Voice chat endpoint: POST http://localhost:8000/voice_chat_speak")
    print("  multipart: audio=<file> [speaker_id=3] [system=...] [history=...]")
    print(f"  ttllm: {TTLLM_URL}  (WhisperX + llama.cpp)")
    print(f"  voicevox: {VOICEVOX_URL}")
    print("=" * 50)
    web.run_app(app, host="0.0.0.0", port=8000)

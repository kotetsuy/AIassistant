# Phase 2 記録 — ストリーミング STT

- 実施日: 2026-08-13
- 結果: **完了**(完了条件3つとも達成)

## 完了条件の判定

| 条件 | 結果 |
|---|---|
| ① 発話中に部分転写が返ってくる | ✅ 約1.1秒(チャンク長)ごとに更新 |
| ② 発話終了→確定が一括経路より明確に短い | ✅ **56ms vs 230ms**(10.4秒発話、STT のみ) |
| ③ フラグ off で従来経路が完全に元どおり動く | ✅ 一括経路は無改修・無影響 |

## 経路

```
Browser  getUserMedia → AudioWorklet(生PCM) → 16kHz mono へ間引き
   │ WebSocket (float32 binary)
   ↓
three-vrm  /stt_chat_stream
   │ WebSocket 中継
   ↓
ttllm  /transcribe_stream
   │ NeMo cache-aware streaming (att_context_size=[56,13] = 1120ms チャンク)
   ↓ partial / final
three-vrm  確定転写で ttllm /chat_stream (SSE) → 文分割 → VOICEVOX → 既存WS配信
```

一括経路(`/voice_chat_speak_stream`)は**一切変更していない**。ブラウザ側のフラグで選択する。

## 実装

### 新設: `ttllm/stt/streaming.py`

`StreamSession` が 1 発話分の状態を持つ。encoder cache(`cache_last_*`)と部分仮説は
セッション固有で、モデルの forward だけが `NeMoBackend._lock` で直列化される。

### ttllm

| 追加 | 内容 |
|---|---|
| `/transcribe_stream` (WebSocket) | binary float32 PCM を受け、`{"type":"partial"/"final"}` を返す。`{"type":"end"}` で確定 |
| `/chat_stream` (POST, SSE) | テキスト入力の LLM ストリーム。STT を別途済ませた呼び出し元用 |
| `_llm_sse()` | `/voice_chat_stream` から LLM 部分を切り出した共通ヘルパ |

whisperX バックエンド稼働時は `supports_streaming=False` なので、
`{"type":"error","fallback":"batch"}` を返してクライアントを一括経路へ倒す。

### three-vrm

| 追加 | 内容 |
|---|---|
| `/stt_chat_stream` (WebSocket) | ブラウザ ↔ ttllm の PCM/転写中継 + 確定後の LLM→VOICEVOX |
| `_relay_stt()` | 中継。上流読み出しは専用タスク(後述) |
| `_llm_to_speech()` | `/chat_stream` のトークンを文に切って VOICEVOX へ |

### ブラウザ (`zundamon.html`)

- `STT_STREAMING` フラグ。**既定 off**。`?stt=stream` / `?stt=batch` か
  `localStorage.sttStreaming="1"` で切り替え
- `AudioWorklet`(`pcm-tap`)で生 PCM を取り出し、`downsampleTo16k()` で 16kHz へ間引いて送信
- ストリーミング開始に失敗したら**その場で一括経路にフォールバック**する

## 実装中に踏んだ問題

### 1. `append_audio` の stream_id が -1 のまま返る

最初の append(`buffer is None` の分岐)では `stream_id` が更新されず **-1 が返る**。
戻り値をそのまま次回に渡すと、同じストリームを伸ばすのではなく**2 本目のストリームが作られ**、
`Sizes of tensors must match except in dimension 1` で落ちる。初回以降は `0` を明示する。

### 2. 100ms ずつ append すると認識が劣化する

mel プリプロセッサは append のたびに独立してパディングするため、**スライス境界ごとに窓の
アーティファクト**が入る。9 音源中 1 件で「川は」が**「かは」**に化けた。

対策として、生 PCM を蓄積して**毎回すべてから mel を作り直し**、`buffer_idx` を保ったまま
バッファ内容を差し替える方式に変更。あわせて `online_normalization=True` にした
(オフライン正規化だと統計が発話全体に依存し、audio が増えるたびに既存フレームがずれるため、
ストリーミングとは原理的に両立しない)。

この変更で **9/9 が既知の正解と一致**。mel の再計算コストは数秒の音声なら encoder 1 ステップに
比べて無視できる。

### 3. 部分転写がまとめて届く

three-vrm の中継で `asyncio.wait_for(up.receive(), timeout=0.0)` のポーリングをしていたが、
これでは上流からの受信がイベントループに回らず、**部分転写が全部発話終了時にまとめて届いていた**
(字幕がライブにならない)。上流の読み出しを**専用タスク**に分離して解決。

修正前: partial が全部 +3.55s に到着 → 修正後: +1.18s / +2.20s / +3.32s に順次到着。

### 4. ストリーミング経路で言語タグが除去されていなかった

バッチ経路は `_hyp_text()` で `<ja-JP>` を剥がしていたが、ストリーミング側に同じ処理が無く、
**`<ja-JP>` が付いたまま LLM に渡っていた**。`streaming.py` にも `LANG_TAG_RE` を適用。

## 実測

### 部分転写のタイミング(3.5秒発話、実時間で送信)

```
+1.18s  日本
+2.20s  日本で一番
+3.32s  日本で一番長い川は
FINAL   発話終了から +0.05s
```

チャンク長 1120ms なので、約 1.1 秒ごとに更新される。

### 発話終了 → 転写確定(STT のみ)

| 音声長 | ストリーミング | 一括 (`/transcribe`) |
|---|---|---|
| 3.50s | 52〜56ms | 約 120ms |
| 10.44s | **56 / 57 / 57 ms** | **230 / 240 / 230 ms** |

**ストリーミングの確定時間は音声長にほぼ依存しない**(処理済みの分を待たなくてよいため)。
一括は音声長に比例するので、**発話が長いほど差が開く**。10.4 秒で約 4 倍。

### エンドツーエンド(発話終了 → 応答完了・VOICEVOX 合成込み)

| 音声長 | ストリーミング | 一括 |
|---|---|---|
| 3.50s | 1.47s (chunks=3) | 2.19s (chunks=5) |
| 10.44s | 2.69〜3.16s (chunks=9〜10) | 2.51〜4.10s (chunks=5) |

LLM の生成長がばらつくのでエンドツーエンドの差は STT ほど明確ではない。
**Phase 3 では STT 単体の指標を主軸にし、エンドツーエンドは参考値として扱う**のが妥当。

### 従来経路の非回帰

フラグ off の `/voice_chat_speak_stream` は変更なしで動作:

```
ok      : True
転写    : 日本で一番長い川は
応答    : それは**利根川（とねがわ）**だよ！...
chunks  : 5
```

## 副次的な発見: att_context_size がバッチ精度も改善した

ストリーミングのために `NEMO_ATT_CONTEXT_SIZE=[56,13]` をモデルに適用したところ、
**バッチ経路の精度も上がった**。Phase 1 で唯一外していた `ritsu_river` が直っている。

| 音源 | Phase 1 (既定 `[56,3]`) | Phase 2 (`[56,13]`) |
|---|---|---|
| ritsu_river | 「**一本**で一番長い川は」 | 「**日本**で一番長い川は」 ✅ |

これで **9 音源すべてがバッチ・ストリーミング両方で正解**になった。
Phase 3 の比較では、この設定であることを明記して測る。

## Phase 3 への引き継ぎ

- 比較指標は **「発話終了 → 転写確定」を主軸**に。純粋推論時間だけではストリーミングの利点が出ない
- 音声は 10 秒級を必ず含める(3.5 秒では差が 2 倍、10.4 秒では 4 倍と、長いほど差が開く)
- whisperX にはストリーミングが無いので、比較は
  **A: whisperX 一括 / B: NeMo 一括 / C: NeMo ストリーミング** の 3 構成
- 精度の正規化(whisperX は「2番目」+文末句読点、NeMo は「二番目」)は Phase 0 の指摘どおり必要

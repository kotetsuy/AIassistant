# AIassistant — 技術解説 / チューニング

セットアップと起動手順は [`READMEJ.md`](./READMEJ.md) を参照してください。
こちらは内部構成・レイテンシ最適化・モデル選定・エンドポイント仕様・VRM 演出・既知の制約を
まとめた技術ドキュメントです。

## パイプライン全体像

```
ブラウザ (three-vrm)
  └─ マイク録音 (MediaRecorder webm/opus)
         ↓ POST /voice_chat_speak_stream
    three-vrm サーバ (port 8000)
         ↓ POST /voice_chat_stream
       ttllm ブリッジ (port 8001)
         ├─ WhisperX-ROCm (STT, large-v3-turbo)
         └─ llama-server (Qwen3.6-35B-A3B MoE, port 8080)
         ↓ SSE で token ストリーム
    three-vrm: 文境界で分割 → VOICEVOX (port 50021) → WS 配信
         ↓ WS (audio + visemes)
 ブラウザ: AudioContext 連続再生 + VRM リップシンク + 背景 + idle motion
```

短い発話 (「こんにちは」程度) で体感 1 秒前後、長文応答でも**初音 1 秒台**を目標にしています。

## レイテンシ最適化

### 1. Qwen3 thinking モードを切る

既定で Qwen3 は返答前に `reasoning_content` (内部独白) を数百トークン吐き、
これで数秒の体感遅延が出ます。ttllm から llama-server に
`chat_template_kwargs: {"enable_thinking": false}` を渡して無効化しています
(`ttllm/server.py:_call_llama`)。この 1 行だけで LLM 段を 4〜8 秒短縮。

### 2. LLM → VOICEVOX のパイプライン化

- ttllm に `/voice_chat_stream` (SSE) を追加し、llama-server を `stream: true` で叩いて
  `{transcript}` → `{token}×N` → `{done}` の流れで返す。
- three-vrm の `/voice_chat_speak_stream` が SSE を消費。`[。！？\n]` で文分割、
  長文保険で 60 文字超は `[、]` でも切る。TTS は `asyncio.Queue` + consumer task で
  直列化 (WS 順序保証)、LLM デコードは並列継続。
- クライアントは `turn_start` で playhead をリセット、各 `speak` チャンクを
  `startAt = max(playheadTime, now)` でキュー末尾に連続再生。viseme は絶対時刻で
  スケジュールするので複数チャンクでも干渉しない。

結果 (実測、長文 8 文応答):

| 指標 | 改善前 (非streaming) | 改善後 (pipeline) |
|---|---|---|
| 初音までの時間 | **3.32 s** | **1.06 s** |
| 全体完了時間 | 3.32 s | 2.98 s |

### 3. WhisperX を large-v3 → large-v3-turbo に変更

STT 段を turbo モデルに切り替えると、転写時間がほぼ半減します。`/warmup` 済みの
steady state で測定 (2.56 秒の音声サンプル、float16、batch 8、Silero VAD):

| 指標 | large-v3 | large-v3-turbo | 改善 |
|---|---|---|---|
| 転写時間 (steady median) | 474 ms | **247 ms** | **-48% (1.92x 速い)** |
| 転写時間 (cold first) | 664 ms | 440 ms | -34% |
| モデルロード | 6.51 s | 4.83 s | -26% |

**「最初の発話」への効果**: STT 段が **約 227 ms 短縮** されるので、初音までの時間が
そのぶん早くなります (TTFT に効く)。認識精度は同等(短文では同じテキストを返す)。

### 4. 新ターン開始時に前の発話を即停止

マイクを押した時点で、クライアントは現在スケジュール済みの全 `AudioBufferSourceNode` を
`stop(0)` → viseme キューも消す、という処理を入れています (`stopAllPlayback`)。
サーバの `turn_start` 到着を待たないので体感が即応。

## MTP と MoE の選択 (LLM の経緯)

現在の既定 LLM は **Qwen3.6-35B-A3B (MoE, Q4_K_XL, 約 21GB)** で、`start_all.sh` はこれを
`-fit off` で起動します (`--spec-type` は付けません)。これは以前の dense な
Qwen3.6-27B + MTP 投機デコード構成から切り替えたものです。経緯を残しておきます。

### 旧構成: Qwen3.6-27B + MTP 投機デコード

Qwen3.6-27B には MTP 層が 1 つ付属しており、llama.cpp の `--spec-type draft-mtp`
で投機的デコードができます。MTP ヘッドが draft トークンを 3 つまで先読みし、
ターゲットモデルが accept したぶんだけ 1 ステップで進めます。

実測 (同じ gguf、同一プロンプト、142 トークン生成、温度 0.7、seed 42):

| 指標 | MTP なし | MTP 有効 | 改善 |
|---|---|---|---|
| 生成 tokens/sec | 7.71 | **10.15** | **+31.7% (1.32x)** |
| 142 トークン応答時間 | 18.42 s | **13.99 s** | -24% |
| TTFT (初トークン) | 0.46 s | 0.48 s | ≒ 同等 |
| Draft acceptance | — | 24.7% (60/243) | — |

**重要な注意**: MTP は **生成中の速度** を上げる仕組みで、**TTFT (初トークン到達時間) は変わりません**。
よって「初音までの時間」(streaming pipelining で 1.06 s 達成) は **MTP では短縮されず**、
効果が出るのは「長文応答の完走時間」です。短い応答ほど効果が薄れます。

### 現構成に切り替えた理由: 帯域が細い iGPU では MoE が断然速い

Ryzen AI HX 370 (16 CU, 約 120 GB/s, 32GB) のような小型チップでは MTP の効果が無く、
dense な 27B はそもそも遅い。1 トークンあたりアクティブ約 3B パラメータのみの
**Qwen3.6-35B-A3B (Q4_K_XL, 約 21GB)** MoE モデルにすると劇的に速くなります:

| 指標 | dense 27B | MoE 35B-A3B | 改善 |
|---|---|---|---|
| TTFT | 約 360 ms | **約 88 ms** | 約 4x |
| 生成 tokens/sec | 約 5.0 | **約 19.8** | 約 4x |

MoE は MTP を使わなくても TTFT・生成速度の両方で勝るため、現構成では MTP を外しています。
dense 27B + MTP を再評価したい場合は、`--spec-type draft-mtp` 対応の llama.cpp master を
ビルドし、`Qwen3.6-27B-MTP-Q8_0.gguf` (MTP 層入り、約 29GB) を使ってください。MTP 層は
以下で確認できます:

```bash
cd ~/llama.cpp
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors \
  ~/qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf | grep -E "nextn|architecture"
# → qwen35.nextn_predict_layers = 1
```

MTP サポートは llama.cpp の PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)
でマージされています。

## VRM ビューアの演出

### 背景ランダムローテーション

- 画像は `~/AIassistant/images/*.{jpg,png,webp}` を自動検出 (環境変数 `IMAGES_DIR` で上書き可)
- `GET /images_list` でファイル一覧、`GET /images/<name>` で配信
- ページ読み込み時に 1 枚選択、**5 分ごと**にランダムで別の画像に切替 (`zundamon.html`)
- 画像は同梱されていません。追加する場合はディレクトリに放り込むだけ (サーバ再起動不要)

### Idle モーション

T-pose 棒立ちを避けるため、毎フレーム微小な回転を加えています
(`zundamon.html:applyIdlePose`)。

| 部位 | 周波数 | 振幅 |
|---|---|---|
| spine / chest (X 軸、呼吸) | 0.25 Hz | ±0.7° |
| spine / chest (Z 軸、左右揺れ) | 0.13 Hz (位相違い) | ±1.1° |
| head (X 軸) | 0.10 Hz | ±0.9° |
| head (Y 軸) | 0.08 Hz | ±1.7° |

`vrm.update(delta)` の前にポーズを設定しているので、VRM の spring bones (髪・スカート等)
が自然に二次追従します。

### 両手を下ろす

VRM のデフォルトは T-pose なので、ロード直後に `applyRestPose()` で
両腕を自然立ちの位置に落とし、肘も約 14° 曲げています (`zundamon.html`)。

## 主要エンドポイント

### ttllm (port 8001)

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/health` | 自身 + llama-server 到達性 |
| POST | `/warmup` | WhisperX モデル先読み |
| POST | `/transcribe` | 音声 → テキスト |
| POST | `/chat` | テキスト → LLM 応答 (非streaming) |
| POST | `/voice_chat` | 音声 → 応答 (非streaming) |
| POST | `/voice_chat_stream` | 音声 → SSE (transcript + token + done) **new** |

### three-vrm (port 8000)

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/zundamon.html` | ビューア |
| GET | `/ws` | WebSocket (turn_start / speak / turn_end / transcript / error) |
| POST | `/speak` | テキスト指定で発話 |
| POST | `/voice_chat_speak` | 音声 → ワンショット応答 (非streaming) |
| POST | `/voice_chat_speak_stream` | 音声 → パイプライン応答 **new** |
| GET | `/images_list` | 背景画像一覧 |
| GET | `/images/{name}` | 背景画像配信 |
| GET | `/vrm/{name}` | VRM ファイル配信 |
| GET | `/status` | クライアント数 |

## 既知の制約

- **PyTorch (ROCm) は gfx1151 専用ホイールが必須**。`repo.amd.com/rocm/whl/gfx1151/` の
  `torch==2.8.0+rocm7.12.0` / `torchaudio==2.8.0a0+rocm7.12.0` を使う。汎用の
  `whl-multi-arch` 版は実行時に `hipErrorInvalidImage` (`kpack_load_code_object failed with
  error: 13`) で **全 GPU 操作が落ちる**。torch はシステム ROCm 7.14 とは別に自前の
  `rocm-sdk-libraries-gfx1151` (7.12) を同梱するが、CTranslate2 (システム ROCm 7.14) と
  同一プロセスで共存できる (`LD_LIBRARY_PATH` に `/opt/rocm/lib` が必要)。
- **torch は ctranslate2/whisperx より先に import** する必要がある。`whisperx.load_model`
  は `whisperx.asr` 経由で ctranslate2 を先に読み、その中で torch が読まれる。この順序だと
  ctranslate2 がシステム ROCm を先に載せ、torch 同梱の `libhipblaslt.so.1` が rocRoller
  シンボルを解決できず `OSError: undefined symbol: _ZN9rocRoller...` で落ちる。対策として
  `ttllm/server.py` の先頭で `import torch` している。
- **torchaudio は 2.9 未満に固定**。pyannote-audio が `torchaudio.info` / `AudioMetaData`
  を使うが torchaudio 2.9 で削除されたため、2.9 以上だと `AttributeError` で import が落ちる。
- **three-vrm は venv python で起動**する。system python (3.14) には `aiohttp` が無い。
- **WhisperX は 60 秒超で GPU memory fault** (ROCm + PyTorch の既知問題)。
  vtt は VAD で 55 秒に強制カットして回避しています。ブラウザ側の録音も長尺は避けてください。
- **無音発話で以前 500 エラー** が出ていましたが、Silero VAD が "No active speech" を
  返したときの WhisperX IndexError を `_transcribe_path` で捕捉して空文字に落とすように
  修正済 (`ttllm/server.py`)。
- **VOICEVOX は CPU 推論**。ROCm との VRAM 競合を避けるための選択で、
  短文なら十分リアルタイム。長文では合成が律速になる可能性あり。
- **Chrome の AudioContext** は初回クリックが必須 (user-gesture 要件)。
- **Qwen3 の thinking** は ttllm 経由では常に OFF ですが、llama-server を直叩きする場合は
  `chat_template_kwargs` を自分で付与する必要があります。

## パスについて

全 shell script / Python のハードコードパスは `$USER` / `os.path.expanduser("~/...")`
に置換済で、`/home/<someone>` の決め打ちは残っていません。他ユーザーで動かす場合でも、
`~/AIassistant/`, `~/llama.cpp/`, `~/whisperx/whisperX-rocm/.venv/` のディレクトリ構造さえ揃えれば
動きます。

> :pencil: WhisperX venv の既定パスは `~/whisperx/whisperX-rocm/.venv` です
> (`ttllm/run.sh` / `install.sh`、`WHISPERX_VENV` で上書き可)。AIassistant 直下の
> `whisperX-rocm` シンボリックリンクもこのパスを指します。以前は `~/AIzunda/whisperX-rocm`
> を使っていましたが、OS を Ubuntu 26.04 (system python 3.14) に更新した際に旧 venv の
> インタプリタ参照が切れたため、動作確認済みの `~/whisperx` 側に統一しました。

## トラブルシュート

| 現象 | 対処 |
|---|---|
| 🎤 を押しても無音 | 画面をクリックして AudioContext を有効化。ブラウザの mic 権限も確認 |
| コテコが喋らない / 500 エラー | `tmux attach -t aiassistant` で ttllm のログ確認。`curl :8001/health` で llama 到達性もチェック |
| STT で `undefined symbol: _ZN9rocRoller...` | torch が ctranslate2 より後に import されている。`ttllm/server.py` 冒頭の `import torch` を確認 |
| torch で `hipErrorInvalidImage` / `kpack_load_code_object failed` | 汎用 multi-arch の torch が入っている。gfx1151 専用インデックス版に入れ替える (READMEJ 手順3) |
| `module 'torchaudio' has no attribute 'AudioMetaData'` | torchaudio が 2.9 以上。2.8.x (`2.8.0a0+rocm7.12.0`) に下げる |
| three-vrm が `ModuleNotFoundError: aiohttp` | venv python で起動する (`$VENV/bin/python server.py`)。または system python に aiohttp を入れる |
| CTranslate2 の cmake が `cmake_minimum_required` で失敗 | CMake 4.x。`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` を付ける |
| 初回発話が遅い | `curl -X POST :8001/warmup` で WhisperX 先読み |
| 腕の向きがおかしい (VRM 差し替え時) | `zundamon.html:applyRestPose` の `rotation.z` 符号を反転 |
| 背景が切り替わらない | DevTools console で `/images_list` のレスポンスを確認。画像を置いたらブラウザリロード |
| VRM が読めない | `server.py` の `VRM_DIR` と実ファイルパスを確認。ファイル名は `zundamon.html` の `VRM_URL` に一致させる |
| 全部止めたい | `~/AIassistant/stop_all.sh` |

## まとめ

ローカル完結で、クラウド API に依存しない「声で会話できるコテコ」を、
AMD Ryzen AI Max+ 395 + ROCm のワンマシン上で動かすことをゴールにしています。
Qwen3.6-35B-A3B (MoE) の thinking 抑制、LLM→TTS パイプライン化、STT の turbo 化で、
初音まで約 1 秒を達成しつつ、違和感のない待機モーションと背景演出を最小コードで付けています。
帯域が細い iGPU でも、アクティブ約 3B の MoE により TTFT・生成速度の両方で快適に動きます。

拡張の余地は以下あたりです。

- 会話履歴の保持 (現在は毎ターンステートレス、`history` パラメタで渡すだけ)
- VRMA 形式の idle アニメ読み込み (現在はプロシージャル)
- VOICEVOX を GPU ビルドに差し替え (長文応答の合成を高速化)
- smaller STT model への切替 (medium で 200〜300 ms 短縮可能)
- LLM ストリーミング中の手振りジェスチャ連動
- マルチモーダル (画像入力) の活用 (`mmproj-F16.gguf` で対応可能)
</content>

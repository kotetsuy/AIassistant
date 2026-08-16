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
         ├─ STT: NeMo Speech (既定) / WhisperX-ROCm (フォールバック)
         └─ llama-server (Qwen3.6-35B-A3B MoE, port 9931)
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

### 評価して見送り: Qwen3.8-27B (2026-08-16)

`Qwen/Qwen3.8-27B` を後継として評価し、**現時点では採用を見送りました**。これは MoE ではなく
**dense** モデルです。名前ではなく GGUF のメタデータで確認しています:

| | Qwen3.8-27B | Qwen3.6-35B-A3B |
|---|---|---|
| `general.architecture` | `qwen35` | `qwen35moe` |
| `expert_count` | キーなし | 256 (使用 8) |
| `*_exps` テンソル | 0 個 | 120 個 |

つまり 1 トークンごとに約 19GB の重みを全部読みます。本機での実測は
**89.9 ms/token (11.1 tok/s)**、MoE の約 50 tok/s に対して大幅に低下します。

初音までの時間への影響 (`bench/bench_e2e.py`、30 回、`human_river` 3.5 秒、NeMo STT。
生データは `bench/results_e2e.json` の `nemo-qwen3.8-27B`):

| 経路 | モデル | 中央値 | 平均 | 最小 | 最大 | 1 秒超 |
|---|---|---|---|---|---|---|
| batch | Qwen3.6-35B-A3B (MoE) | 735 ms | 723 ms | 415 ms | 939 ms | **0/30** |
| batch | Qwen3.8-27B (dense) | 829 ms | 1204 ms | 799 ms | 2222 ms | **14/30** |
| stream | Qwen3.6-35B-A3B (MoE) | 615 ms | 608 ms | 353 ms | 903 ms | **0/30** |
| stream | Qwen3.8-27B (dense) | 772 ms | 1171 ms | 743 ms | 3721 ms | **12/30** |

中央値の悪化は小さいのですが、分布が明確に割れます (batch で約 810 / 1445 / 2100 ms の 3 山)。
クラスタ間隔の約 640 ms は 90 ms/token で約 7 トークンぶんです。初音は「第 1 文が出るまで」で
決まるので、11 tok/s では第 1 文の長さが支配項になります。MoE は 30 回すべて 1 秒以内でしたが、
dense 27B は 16 回しか収まりません。

**結論: このクラスで Qwen3.8 の MoE 版が出るまで Qwen3.6-35B-A3B を継続します。**
(`Qwen/Qwen3.8-2.4T-A95B` は MoE ですが本機には大きすぎます。) MoE 版が出たときに
再検討する価値がある項目が 2 つあり、どちらも dense でも効きます:

- `ttllm/run.sh` に第 1 文を 15 文字以内に縛る `SYSTEM_PROMPT` がコメントアウトされたまま
  残っています。上のばらつきにちょうど効く設定です。
- `ggml-org/Qwen3.8-27B-GGUF` に `--spec-type draft-mtp` 用の
  `mtp-Qwen3.8-27B-Q4_0.gguf` (約 1.7GB) があります。

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
| POST | `/warmup` | STT モデル先読み + ウォームアップ推論 |
| POST | `/transcribe` | 音声 → テキスト |
| POST | `/chat` | テキスト → LLM 応答 (非streaming) |
| POST | `/voice_chat` | 音声 → 応答 (非streaming) |
| POST | `/voice_chat_stream` | 音声 → SSE (transcript + token + done) |
| POST | `/chat_stream` | テキスト → SSE (token + done)。STT を別途済ませた呼び出し元用 **new** |
| WS | `/transcribe_stream` | 生 PCM (float32 16kHz mono) → partial / final **new** |

### three-vrm (port 8000)

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/zundamon.html` | ビューア |
| GET | `/ws` | WebSocket (turn_start / speak / turn_end / transcript / error) |
| POST | `/speak` | テキスト指定で発話 |
| POST | `/voice_chat_speak` | 音声 → ワンショット応答 (非streaming) |
| POST | `/voice_chat_speak_stream` | 音声 → パイプライン応答 |
| GET | `/stt_chat_stream` | WebSocket。発話中の PCM → 部分転写 → LLM → VOICEVOX **new** |
| GET | `/images_list` | 背景画像一覧 |
| GET | `/images/{name}` | 背景画像配信 |
| GET | `/vrm/{name}` | VRM ファイル配信 |
| GET | `/status` | クライアント数 |

## STT バックエンドの設計

### なぜ 2 系統なのか

移植前は WhisperX-ROCm 一択だった。NeMo Speech に切り替えるにあたり WhisperX を消さず
**同一プロセス内で実行時フォールバックできる形**にしてある。理由は 2 つ:

1. NeMo は ROCm 公式サポート外の構成で動かしている。壊れたときに会話が止まるのは困る
2. 長く固有名詞の多い発話では **WhisperX の方が正確**(実測は後述)。用途によって選べる方がいい

### 構成

```
ttllm/stt/
  base.py       STTBackend 抽象基底 — transcribe_path(path) -> str が契約の中心
  audio.py      ffmpeg で webm/ogg/mp4 → 16kHz mono float32 + 末尾 0.5 秒パディング
  nemo.py       NeMoBackend (既定)
  whisperx.py   WhisperXBackend (フォールバック)
  streaming.py  StreamSession — cache-aware ストリーミングのセッション状態
  __init__.py   STTRouter — バックエンド選択とフォールバック
```

`server.py` の `_transcribe_path()` は薄いラッパとして残してあるので、
`/transcribe` `/voice_chat` `/voice_chat_stream` の 3 エンドポイントは無改修。

### 設計上の判断

- **`model.transcribe()` は使わない。** 呼び出しごとに Lhotse dataloader を再構築し、
  1 リクエストあたり 0.2〜0.4 秒を浪費する。encoder forward と
  `rnnt_decoder_predictions_tensor()` を直接叩くことで 3.5 秒の発話が 119ms → 88ms になる
- **音声デコードは自前で持つ。** ブラウザから来るのは webm/opus であって 16kHz WAV ではない。
  WhisperX は `load_audio()` が内部で ffmpeg を呼んでいたので意識せずに済んでいた
- **末尾に 0.5 秒の無音を足す。** cache-aware は発話終端の右 context が無いと最後のトークンを
  出せず、PTT 録音はボタンを離した瞬間に切れる。VOICEVOX 音源では実際に文末の助詞が落ちた
- **言語プロンプトは `ja-JP`。** `ja` 単体は `prompt_dictionary` に無く拒否される
- **フォールバックは必ず WARNING と `/health` に出す。** 無言で別モデルに変わるのが最悪

### ストリーミング経路

```
Browser  getUserMedia → AudioWorklet(生PCM) → 16kHz mono へ間引き
   │ WebSocket (float32 binary)
three-vrm  /stt_chat_stream  ── WS 中継 ──▶  ttllm  /transcribe_stream
   │                                            NeMo cache-aware streaming
   │                                            att_context_size=[56,13] (1120ms)
   ◀── partial / final ────────────────────────┘
   └─ 確定転写で ttllm /chat_stream (SSE) → 文分割 → VOICEVOX → 既存 WS 配信
```

**MediaRecorder の `start(timeslice)` は使えない。** webm/opus のチャンクは先頭以外が
単独でデコードできず、サーバ側で 1 つずつ扱えないため。AudioWorklet で生 PCM を取る。

**mel は毎回全体から作り直す。** 100ms ずつ `append_audio` すると、mel プリプロセッサが
append ごとに独立してパディングするためスライス境界に窓のアーティファクトが入り、
実際に「川は」が「かは」に化けた。生 PCM を蓄積して毎回全体から mel を計算し直し、
`buffer_idx` を保ったままバッファを差し替えることで、オフライン経路と同じフレームになる。
あわせて `online_normalization=True`(オフライン正規化は統計が発話全体に依存するので
ストリーミングと原理的に両立しない)。

セッション状態(encoder cache・部分仮説)は **StreamSession ごとに独立**で、
モデル本体の forward だけを `NeMoBackend._lock` で直列化している。

### 実測 (発話終了 → 転写確定、20 回の中央値)

| 音源長 | A: whisperX | B: NeMo 一括 | C: NeMo ストリーミング |
|---|---|---|---|
| 1.06s | 254ms | 84ms | **48ms** |
| 3.50s | 287ms | 119ms | **49ms** |
| 10.44s | 432ms | 221ms | **53ms** |
| 12.84s | 545ms | 274ms | **50ms** |
| 14.22s | 460ms | 255ms | **95ms** |

**ストリーミングの確定時間は音声長にほぼ依存しない**(未処理なのは最後のチャンク分だけ)。
一括は音声長に比例するので、発話が長いほど差が開く(12.8 秒で whisperX 比 10.8 倍)。

精度(CER、表記ゆれ正規化後)は短文 9 本すべてで 3 構成とも 0.0%。
**B と C は全音源で完全一致**しており、ストリーミング化による劣化は無い。
一方 **長文では whisperX が優位**(固有名詞の多い文で 9.9% vs 21.1%)。

エンドツーエンド(発話終了 → 最初の音声、3.5秒発話で各30回)は
**whisperX 895ms → NeMo 一括 735ms / ストリーミング 614ms**(中央値)。
効いているのは中央値より分布で、**whisperX では 11/30 (37%) が1秒超だったのが NeMo では 0/30**。
ただしこの指標はセッションごとに中央値が 895〜1074ms とぶれるので、
**単発の中央値ではなく「1秒を超える割合」で見る**方が頑健。
LLM の最初の一文の生成と VOICEVOX 合成が支配的なため。

詳細は `docs/STT移植_PHASE3.md`、生データは `bench/results.json`。

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
- **NeMo は cache-aware モデルのため float32 固定**。`compute_dtype != float32` は
  `NotImplementedError` で明示的に拒否される。WhisperX は float16 で動くので、
  速度比較はこの条件差込みで見ること。
- **NeMo はロードだけでは足りずウォームアップ推論が要る**。モデルをロードしただけだと
  最初のリクエストがカーネルコンパイルで約 2.8 秒かかる。`NeMoBackend.load()` が
  1 秒の無音で捨て推論を回している。
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
`~/AIassistant/`, `~/llama.cpp/`, `~/Speech/`, `~/whisperx/whisperX-rocm/` のディレクトリ構造さえ
揃えれば動きます。

> :pencil: パイプラインが使う venv は **`~/AIassistant/ttllm/.venv`** です
> (`ttllm/run.sh` / `install.sh`、`TTLLM_VENV` で上書き可)。NeMo と whisperX の両方が
> ここに入っており、three-vrm もこの python で起動します。
> `~/whisperx/whisperX-rocm/.venv` の単体 venv は whisperX を CLI で使うとき用に残してあり、
> パイプラインからは参照しません。
>
> AIassistant 直下の `Speech` / `whisperX-rocm` シンボリックリンク経由でのみ外部リポジトリを
> 参照します(絶対パスを書かない)。`Speech` は **`rocm-inference` ブランチ**である必要があります。

## トラブルシュート

| 現象 | 対処 |
|---|---|
| 🎤 を押しても無音 | 画面をクリックして AudioContext を有効化。ブラウザの mic 権限も確認 |
| コテコが喋らない / 500 エラー | `tmux attach -t aiassistant` で ttllm のログ確認。`curl :8001/health` で llama 到達性もチェック |
| STT で `undefined symbol: _ZN9rocRoller...` | torch が ctranslate2 より後に import されている。`ttllm/server.py` 冒頭の `import torch` を確認 |
| torch で `hipErrorInvalidImage` / `kpack_load_code_object failed` | 汎用 multi-arch の torch が入っている。gfx1151 専用インデックス版に入れ替える (READMEJ 手順3) |
| `module 'torchaudio' has no attribute 'AudioMetaData'` | torchaudio が 2.9 以上。2.8.x (`2.8.0a0+rocm7.12.0`) に下げる |
| three-vrm が `ModuleNotFoundError: aiohttp` | venv python で起動する (`start_all.sh` は対応済み)。手動起動なら `$TTLLM_VENV/bin/python server.py`。venv に aiohttp が無ければ `VIRTUAL_ENV=$TTLLM_VENV uv pip install aiohttp` |
| CTranslate2 の cmake が `cmake_minimum_required` で失敗 | CMake 4.x。`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` を付ける |
| 初回発話が遅い | `curl -X POST :8001/warmup` で STT 先読み(ロード + ウォームアップ推論で約 33 秒) |
| STT が急に whisperX になっている | NeMo が落ちてフォールバックした。`curl -s :8001/health \| jq .stt.last_error` で理由を確認 |
| NeMo が `Cannot reach https://huggingface.co/...: offline mode` で落ちる | `run.sh` が `HF_HUB_OFFLINE=1` で起動するため。`ttllm/install.sh` を実行してモデルをキャッシュする |
| ストリーミングにしても部分転写が出ない | `/health` の `stt.supports_streaming` を確認。whisperX バックエンドでは非対応で一括経路に倒れる |
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

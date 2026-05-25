# AIassistant — コテコと声で会話できる AI パイプライン

> 音声: VOICEVOX:ずんだもん

Ubuntu + AMD Ryzen AI Max+ 395 (ROCm) 上で、**音声 → STT → LLM → TTS → VRM リップシンク** を
一気通貫で動かすローカルスタック。ブラウザの 🎤 ボタンを押すとコテコが声で返します。

<img width="1219" height="1140" alt="https---qiita-image-store s3 ap-northeast-1 amazonaws com-0-263486-86fd1211-a196-4c6d-bf7b-e4ff53d8c5ba" src="https://github.com/user-attachments/assets/4292a4f1-5239-4a83-8c9e-3c3d4610fed2" />


```
ブラウザ (three-vrm)
  └─ マイク録音 (MediaRecorder webm/opus)
         ↓ POST /voice_chat_speak_stream
    three-vrm サーバ (port 8000)
         ↓ POST /voice_chat_stream
       ttllm ブリッジ (port 8001)
         ├─ WhisperX-ROCm (STT, large-v3-turbo)
         └─ llama-server (Qwen3.6-27B MTP, port 8080)
         ↓ SSE で token ストリーム
    three-vrm: 文境界で分割 → VOICEVOX (port 50021) → WS 配信
         ↓ WS (audio + visemes)
 ブラウザ: AudioContext 連続再生 + VRM リップシンク + 背景 + idle motion
```

## 構成

| パス | 役割 | ポート |
|---|---|---|
| `voicevox/` | VOICEVOX Engine (Docker, CPU 推論) | 50021 |
| `~/llama.cpp/build/bin/llama-server` | Qwen3.6 推論 (MTP 投機デコード対応) | 8080 |
| `qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf` | LLM モデル (MTP 層 1 つ含む) | — |
| `ttllm/` | FastAPI ブリッジ (WhisperX + llama.cpp) | 8001 |
| `three-vrm/` | aiohttp サーバ + VRM ビューア (HTML/three-vrm) | 8000 |
| `vtt/` | CLI PTT マイク (任意) | — |
| `images/` | VRM ビューア背景 (5 分ごとにローテーション) | — |
| `vroid/koteko.vrm` | コテコ VRM 1.0 モデル | — |
| `whisperX-rocm/` | WhisperX の ROCm フォーク (`~/AIzunda/whisperX-rocm` へのシンボリックリンク) | — |

### 前提

- **OS** : Ubuntu 24.04.4 LTS
- **GPU** : AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1150、48GB VRAM)
- **ROCm** : 7.2.1 (`/opt/rocm`)
- **Python** : 3.12.3
- **Docker** : 29.x (VOICEVOX 用)
- **ブラウザ** : Google Chrome (`AudioContext` を使うため Firefox でも可)
- **tmux / curl / uv / huggingface_hub (hf CLI)** : 起動スクリプトで使用

詳細なセットアップは各サブディレクトリの `READMEJ.md` を参照:
`ttllm/READMEJ.md` / `vtt/READMEJ.md` / `three-vrm/READMEJ.md` / `voicevox/READMEJ.md` /
`whisperX-rocm/READMEJ.md`。

## git clone から ./start_all.sh でコテコが喋るまでの手順

## 1. リポジトリと依存物の取得

本体リポジトリには `whisperX-rocm` / `llama.cpp` / `qwen3.6` をシンボリックリンクで参照する構造になっているので、まず本体と依存物を **ホームディレクトリ直下** に並べて配置します。

```bash
cd ~
git clone https://github.com/kotetsuy/AIassistant.git
git clone https://github.com/ggml-org/llama.cpp.git
```

WhisperX の ROCm フォークと CTranslate2 の ROCm フォークも別途必要です:

```bash
mkdir -p ~/whisperx && cd ~/whisperx
git clone https://github.com/<your_whisperx_rocm_fork>/whisperX-rocm.git
git clone https://github.com/<your_ctranslate2_rocm_fork>/ctranslate2-rocm.git
```

> :pencil: 実機では `whisperX-rocm` を `~/AIzunda/whisperX-rocm` に置いていますが、新規構築する場合は `~/whisperx/whisperX-rocm` でも構いません。AIassistant 側の `whisperX-rocm` は **シンボリックリンク** なので、リンク先は環境に合わせて貼り直してください。

こちらも参照してください

https://qiita.com/kotetsu_yama/items/449e0d0527ab3a233fb8

---

## 2. CTranslate2-ROCm をソースビルド

`faster-whisper` が呼ぶ CTranslate2 を ROCm/HIP 対応でビルドします。

```bash
cd ~/whisperx/ctranslate2-rocm
mkdir -p build && cd build

export HSA_OVERRIDE_GFX_VERSION=11.5.0
export AMDGPU_TARGETS=gfx1150

cmake .. -DWITH_HIP=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1150 -DCMAKE_BUILD_TYPE=Release \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_CLI=OFF
make -j$(nproc) && sudo make install
```

`/usr/local/lib/libctranslate2.so` が入れば成功です。

---

## 3. WhisperX-ROCm 用の venv を作る

```bash
cd ~/whisperx/whisperX-rocm
uv venv && uv pip install -e .

# ROCm 版 ctranslate2 の Python バインディングを再インストール
rm -rf .venv/lib/python3.12/site-packages/ctranslate2*
export CTRANSLATE2_ROOT=/usr/local
uv pip install --reinstall pybind11 ~/whisperx/ctranslate2-rocm/python
```

確認:

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# → CUDA: True  (ROCm の HIP レイヤーが CUDA API を翻訳している)
.venv/bin/python -c "import ctranslate2; print(ctranslate2.__version__)"
```

---

## 4. llama.cpp を **MTP 対応の最新 master** でビルド

Qwen3.5/3.6 系の **MTP (Multi-Token Prediction)** サポートは PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) でマージされたばかりです (2026-05 時点)。後半 [§3](#3-mtp-投機デコードの仕組みと効果) で使うので、master 最新を pull してください。

```bash
cd ~/llama.cpp
git pull --ff-only origin master

mkdir -p build && cd build
export HSA_OVERRIDE_GFX_VERSION=11.5.0
export AMDGPU_TARGETS=gfx1150

cmake .. -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1150 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

ビルド後の確認:

```bash
./bin/llama-server --version
# version: 9294 (...) など

./bin/llama-server --help | grep -A2 spec-type
# --spec-type none,draft-simple,draft-eagle3,draft-mtp,...
```

`draft-mtp` が出てくれば MTP 対応版です。

---

## 5. Qwen3.6 MTP モデルをダウンロード

`hf` (huggingface CLI、旧 `huggingface-cli`) でモデルを取得します:

```bash
mkdir -p ~/qwen3.6
hf download am17an/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-MTP-Q8_0.gguf \
  --local-dir ~/qwen3.6
```

29 GB あります。MTP 層を含むことを確認:

```bash
cd ~/llama.cpp
python3 gguf-py/gguf/scripts/gguf_dump.py --no-tensors \
  ~/qwen3.6/Qwen3.6-27B-MTP-Q8_0.gguf | grep -E "nextn|architecture"
# → qwen35.nextn_predict_layers = 1
```

`nextn_predict_layers = 1` が出れば MTP 層入り gguf です。

---

## 6. AIassistant 内のシンボリックリンクを張る

`~/AIassistant` 配下から `llama.cpp` / `whisperX-rocm` / `qwen3.6` を相対パスで参照できるようにします。

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

## 7. ttllm ブリッジの依存を venv に追加

```bash
cd ~/AIassistant/ttllm
./install.sh
```

`fastapi` / `uvicorn` / `httpx` / `python-multipart` / `pydantic` が **WhisperX-ROCm の venv に追加** されます (専用 venv は作らず共有)。

---

## 8. VRM モデル (コテコ) を配置

VRoid Studio などで作った VRM 1.0 モデルを置きます:

```bash
mkdir -p ~/AIassistant/vroid
cp /path/to/your_avatar.vrm ~/AIassistant/vroid/koteko.vrm
```

ファイル名を変える場合は以下 2 箇所を合わせて書き換えてください:

```python
# three-vrm/server.py
VRM_DIR = os.path.expanduser("~/AIassistant/vroid")
```
```html
<!-- three-vrm/TalkingHead/zundamon.html -->
const VRM_URL = "http://localhost:8000/vrm/koteko.vrm";
```

サンプルを作りました。自由にお使いください。

https://hub.vroid.com/characters/2782544841139509367

---

## 9. VOICEVOX Docker を取得

```bash
docker pull voicevox/voicevox_engine:cpu-ubuntu20.04-latest
# 起動は start_all.sh が面倒を見るのでここでは pull だけで OK
```

CPU 推論版を使います。GPU は LLM + STT で埋めるので、TTS は CPU が無難。

---

## 10. 一括起動

```bash
cd ~/AIassistant
./start_all.sh
```

以下が直列で立ち上がり、HTTP health check で待ち合わせます:

1. VOICEVOX (Docker, port 50021)
2. llama-server (Qwen3.6-27B-MTP, port 8080, `--spec-type draft-mtp` 付き)
3. ttllm ブリッジ (port 8001)
4. WhisperX warmup (`POST /warmup` を叩いて初回のモデルロードを済ませる)
5. three-vrm サーバ (port 8000)
6. Chrome で `http://localhost:8000/zundamon.html` を自動オープン
7. vtt (CLI PTT、任意)

全ウィンドウは tmux セッション `aiassistant` に入っているので、

```bash
tmux attach -t aiassistant   # ログを見る
~/AIassistant/stop_all.sh    # 全部止める
```

で操作できます。

---

## 11. 動作確認

ブラウザが開いたら **画面を一度クリック** して AudioContext を有効化 (Chrome の user-gesture 要件)。右下の 🎤 ボタンの動線:

- **長押し (≥ 250 ms)**: 押している間だけ録音、離すと自動で送信
- **短クリック**: 録音開始 → もう一度クリックで送信

ユーザー発話は薄青、コテコの返答は白の字幕として出ます。初音まで体感 1 秒前後で返ってくれば成功です。

## 一括起動 / 停止

```bash
~/AIassistant/start_all.sh   # 全段起動 + health check + WhisperX warmup + Chrome オープン
~/AIassistant/stop_all.sh    # tmux セッション + VOICEVOX を停止
~/AIassistant/stop_all.sh --keep-voicevox   # VOICEVOX コンテナは残す
```

`start_all.sh` は tmux セッション `aiassistant` を作り、各サービスを別ウィンドウで走らせます。

| window | コマンド |
|---|---|
| 0 voicevox | `docker logs -f voicevox_engine` |
| 1 llama | `llama-server -m Qwen3.6-27B-MTP-Q8_0.gguf --port 8080 -ngl 99 -c 8192 --spec-type draft-mtp` |
| 2 ttllm | `ttllm/run.sh` (uvicorn) |
| 3 three-vrm | `python3 three-vrm/server.py` |
| 4 vtt | `vtt/run.sh --device USB` (CLI PTT, 任意) |

ログを見る: `tmux attach -t aiassistant`  
全部落とす: `~/AIassistant/stop_all.sh`

起動順序は依存関係に合わせて直列化しており、各段で HTTP health check 待ちを入れています
(llama-server のモデルロードだけ最大 600 秒タイムアウト)。ttllm が上がった直後に
`/warmup` を叩いて WhisperX モデルをあらかじめロードするので、最初の発話が遅くなりません。

## ブラウザでの使い方

1. `start_all.sh` が自動で Chrome を開く (`http://localhost:8000/zundamon.html`)
2. 画面を一度クリックして AudioContext を有効化 (ブラウザの user-gesture 要件)
3. 右下の **🎤 ボタン**
   - **長押し (≥ 250ms)** : 押している間だけ録音、離すと送信
   - **短クリック** : 録音開始 → もう一度クリックで送信
4. ユーザー発話は薄青の字幕、コテコの返答は白の字幕として表示

## レイテンシ最適化

短い発話 (「こんにちは」程度) で体感 1 秒前後、長文応答でも**初音 1 秒台**を目標にしています。

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

### 4. MTP (Multi-Token Prediction) 投機デコード

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

> **帯域が細い iGPU では MoE モデルの方が断然速い。** Ryzen AI HX 370 (16 CU, 約120 GB/s, 32GB)
> のような小型チップでは MTP の効果が無く、dense な 27B は遅い。1 トークンあたりアクティブ約 3B
> パラメータのみの **Qwen3.6-35B-A3B (Q4_K_XL, 約21GB)** MoE モデルに戻すと劇的に速くなる:
> **TTFT 約 88 ms (vs 約 360 ms)、約 19.8 tok/s (vs 約 5.0)** で両方およそ 4 倍。
> `start_all.sh` はこのモデルを指している。

### 5. 新ターン開始時に前の発話を即停止

マイクを押した時点で、クライアントは現在スケジュール済みの全 `AudioBufferSourceNode` を
`stop(0)` → viseme キューも消す、という処理を入れています (`stopAllPlayback`)。
サーバの `turn_start` 到着を待たないので体感が即応。

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

- **WhisperX は 60 秒超で GPU memory fault** (ROCm 7.x + PyTorch nightly の既知問題)。
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
`~/AIassistant/`, `~/llama.cpp/`, `~/AIzunda/whisperX-rocm/.venv/` のディレクトリ構造さえ揃えれば
動きます。

## トラブルシュート

| 現象 | 対処 |
|---|---|
| 🎤 を押しても無音 | 画面をクリックして AudioContext を有効化。ブラウザの mic 権限も確認 |
| コテコが喋らない / 500 エラー | `tmux attach -t aiassistant` で ttllm のログ確認。`curl :8001/health` で llama 到達性もチェック |
| 初回発話が遅い | `curl -X POST :8001/warmup` で WhisperX 先読み |
| 腕の向きがおかしい (VRM 差し替え時) | `zundamon.html:applyRestPose` の `rotation.z` 符号を反転 |
| 背景が切り替わらない | DevTools console で `/images_list` のレスポンスを確認。画像を置いたらブラウザリロード |
| VRM が読めない | `server.py` の `VRM_DIR` と実ファイルパスを確認。ファイル名は `zundamon.html` の `VRM_URL` に一致させる |
| 全部止めたい | `~/AIassistant/stop_all.sh` |

## まとめ

ローカル完結で、クラウド API に依存しない「声で会話できるコテコ」を、
AMD Ryzen AI Max+ 395 + ROCm のワンマシン上で動かすことをゴールにしています。
Qwen3.6-27B (MTP) の thinking 抑制、LLM→TTS パイプライン化、MTP 投機デコードで、
初音まで約 1 秒・生成速度 +32% を達成しつつ、違和感のない待機モーションと背景演出を最小コードで付けています。

拡張の余地は以下あたりです。

- 会話履歴の保持 (現在は毎ターンステートレス、`history` パラメタで渡すだけ)
- VRMA 形式の idle アニメ読み込み (現在はプロシージャル)
- VOICEVOX を GPU ビルドに差し替え (長文応答の合成を高速化)
- smaller STT model への切替 (medium で 200〜300 ms 短縮可能)
- LLM ストリーミング中の手振りジェスチャ連動

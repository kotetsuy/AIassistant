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
         └─ llama-server (Qwen3.6-35B-A3B MoE, port 8080)
         ↓ SSE で token ストリーム
    three-vrm: 文境界で分割 → VOICEVOX (port 50021) → WS 配信
         ↓ WS (audio + visemes)
 ブラウザ: AudioContext 連続再生 + VRM リップシンク + 背景 + idle motion
```

> **技術的な詳細（レイテンシ最適化、MTP と MoE の選択、エンドポイント仕様、VRM 演出、
> 既知の制約、トラブルシュート）は [`TECHNICALJ.md`](./TECHNICALJ.md) にまとめています。**
> こちらは「git clone から `./start_all.sh` でコテコが喋るまで」の手順書です。

## 構成

| パス | 役割 | ポート |
|---|---|---|
| `voicevox/` | VOICEVOX Engine (Docker, CPU 推論) | 50021 |
| `~/llama.cpp/build/bin/llama-server` | Qwen3.6 推論 (MoE, アクティブ約 3B) | 8080 |
| `qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | LLM モデル (MoE, 約 21GB) | — |
| `qwen3.6/mmproj-F16.gguf` | 視覚エンコーダ (任意、マルチモーダル用) | — |
| `ttllm/` | FastAPI ブリッジ (WhisperX + llama.cpp) | 8001 |
| `three-vrm/` | aiohttp サーバ + VRM ビューア (HTML/three-vrm) | 8000 |
| `vtt/` | CLI PTT マイク (任意) | — |
| `images/` | VRM ビューア背景 (5 分ごとにローテーション) | — |
| `vroid/koteko.vrm` | コテコ VRM 1.0 モデル | — |
| `whisperX-rocm/` | WhisperX の ROCm フォーク (`~/whisperx/whisperX-rocm` へのシンボリックリンク) | — |

> :pencil: 現在の既定 LLM は **Qwen3.6-35B-A3B (MoE)** です。以前は dense な
> Qwen3.6-27B + MTP 投機デコードを使っていましたが、帯域が細い iGPU では MoE の方が
> 速いため切り替えました。経緯と実測は [`TECHNICALJ.md`](./TECHNICALJ.md) を参照。

### 前提

- **OS** : Ubuntu 26.04 (resolute)
- **GPU** : AMD Ryzen AI Max+ 395 / Radeon 8060S (gfx1151、48GB VRAM)
- **ROCm** : 7.14.0 (`/opt/rocm`)。Ubuntu 26.04 では apt でネイティブ導入できます
  (`amdrocm-core-sdk7.14-gfx1151` を `repo.amd.com/rocm/packages-multi-arch/ubuntu2604` から)。
  カーネル同梱 amdgpu が gfx1151 対応済みなので DKMS / `amdgpu-install` は不要。
- **Python** : system 3.14 / 各 venv は 3.12 (`.python-version` で固定)
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

> :pencil: `whisperX-rocm` は `~/whisperx/whisperX-rocm` に置きます (AIassistant 側の
> `whisperX-rocm` はそこへのシンボリックリンク)。ttllm の `run.sh` / `install.sh` もこのパスを
> 既定にしています。以前は `~/AIzunda/whisperX-rocm` に置いていましたが、OS 更新 (Ubuntu
> 26.04 / system python 3.14) で旧 venv が壊れたため `~/whisperx` 側に統一しました。
> リンク先を変える場合は `ln -sfn <path> whisperX-rocm` と `WHISPERX_VENV` を合わせてください。

こちらも参照してください

https://qiita.com/kotetsu_yama/items/449e0d0527ab3a233fb8

---

## 2. CTranslate2-ROCm をソースビルド

`faster-whisper` が呼ぶ CTranslate2 を ROCm/HIP 対応でビルドします。

```bash
cd ~/whisperx/ctranslate2-rocm
mkdir -p build && cd build

export HSA_OVERRIDE_GFX_VERSION=11.5.1
export AMDGPU_TARGETS=gfx1151

cmake .. -DWITH_HIP=ON -DWITH_MKL=OFF -DWITH_OPENBLAS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DOPENMP_RUNTIME=COMP -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DBUILD_CLI=OFF
make -j$(nproc) && sudo make install && sudo ldconfig
```

`/usr/local/lib/libctranslate2.so` が入れば成功です。

> :warning: CMake 4.x では同梱 `third_party/cpu_features` の `cmake_minimum_required`
> が古すぎて configure が失敗します。上記の `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` が必須です。

---

## 3. WhisperX-ROCm 用の venv を作る

```bash
cd ~/whisperx/whisperX-rocm
uv venv && uv pip install -e .   # venv は .python-version により 3.12

# uv pip install -e . は既定で NVIDIA CUDA 版 torch を入れてしまうので、
# gfx1151 専用の ROCm ホイールに差し替える (下の「PyTorch (ROCm)」参照)
uv pip uninstall torch torchaudio
uv pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease allow \
  torch==2.8.0+rocm7.12.0 torchaudio==2.8.0a0+rocm7.12.0

# ROCm 版 ctranslate2 の Python バインディングを再インストール
rm -rf .venv/lib/python3.12/site-packages/ctranslate2*
export CTRANSLATE2_ROOT=/usr/local
uv pip install --reinstall --no-deps pybind11 ~/whisperx/ctranslate2-rocm/python
```

> :warning: **PyTorch (ROCm) は gfx1151 専用インデックスを使うこと。**
> 汎用の `whl-multi-arch` 版は実行時に `hipErrorInvalidImage`
> (`kpack_load_code_object failed`) で全 GPU 操作が落ちます。また **torchaudio は 2.9 未満**
> に固定します (pyannote が `torchaudio.info` / `AudioMetaData` を使うが 2.9 で削除された)。
> これらのホイールは cp310 が無いため venv は Python 3.11+ が必須です。

確認:

```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# → CUDA: True  (ROCm の HIP レイヤーが CUDA API を翻訳している)
.venv/bin/python -c "import ctranslate2; print(ctranslate2.__version__)"
```

---

## 4. llama.cpp を ROCm 対応でビルド

Qwen3.6-35B-A3B は **MoE + Mamba ハイブリッド** なので、対応済みの新しめの llama.cpp が必要です。
master 最新を pull してビルドしてください。

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

ビルド後の確認:

```bash
./bin/llama-server --version
# version: 9294 (...) など
```

> :bulb: いまの既定構成 (MoE) は MTP 投機デコードを使いません。dense な Qwen3.6-27B + MTP を
> 試したい場合のみ、`--spec-type draft-mtp` 対応の master が要ります。詳細は
> [`TECHNICALJ.md`](./TECHNICALJ.md) の「MTP と MoE の選択」を参照。

---

## 5. Qwen3.6-35B-A3B (MoE) モデルをダウンロード

`hf` (huggingface CLI、旧 `huggingface-cli`) でモデルを取得します。Unsloth Dynamic の
Q4_K_XL 量子化を使います:

```bash
mkdir -p ~/qwen3.6
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf mmproj-F16.gguf \
  --local-dir ~/qwen3.6
```

メインモデルは約 21GB、視覚エンコーダ (`mmproj-F16.gguf`、約 858MB) はマルチモーダルを
使う場合のみ必要です。確認:

```bash
ls -lh ~/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
# 約 20.8 GiB
```

> アクティブパラメータは 1 トークンあたり約 3B 相当なので、総 34.66B の割に高速です
> (Ryzen AI Max+ 395 で tg128 ≈ 50 t/s)。詳細なベンチは `qwen3.6/READMEJ.md` を参照。

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

`fastapi` / `uvicorn` / `httpx` / `python-multipart` / `pydantic` が **WhisperX-ROCm の venv に追加** されます (専用 venv は作らず共有)。`install.sh` / `run.sh` は既定で
`~/whisperx/whisperX-rocm/.venv` を使います (別の場所なら `WHISPERX_VENV` で上書き)。

> :warning: **three-vrm も同じ venv の python で起動してください。** `start_all.sh` は
> `python3 server.py` (system python) で起動しますが、system python には `aiohttp` が無いため
> 失敗します。venv に `aiohttp` を入れておき、three-vrm を venv python で走らせるのが確実です:
>
> ```bash
> VIRTUAL_ENV=~/whisperx/whisperX-rocm/.venv uv pip install aiohttp
> # start_all.sh の three-vrm 起動を「$VENV/bin/python server.py」に変更するか、
> # system python3 に aiohttp を入れる
> ```

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
2. llama-server (Qwen3.6-35B-A3B MoE, port 8080)
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
| 1 llama | `llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --port 8080 -ngl 99 -c 8192 -fit off` |
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

## 次に読むもの

- **技術解説 / チューニング**: [`TECHNICALJ.md`](./TECHNICALJ.md)
  (レイテンシ最適化、MTP と MoE の選択、エンドポイント仕様、VRM 演出、既知の制約、トラブルシュート、まとめ)
- 各コンポーネントの詳細: `ttllm/READMEJ.md` / `three-vrm/READMEJ.md` /
  `voicevox/READMEJ.md` / `vtt/READMEJ.md` / `qwen3.6/READMEJ.md`
</content>
</invoke>

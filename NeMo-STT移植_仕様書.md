# AIassistant STTバックエンド移植 仕様書 — whisperX-rocm → NeMo Speech (rocm-inference)

- 作成日: 2026-08-13
- 対象: `~/AIassistant` の STT を、WhisperX-ROCm から NeMo Speech (`nemotron-3.5-asr-streaming-0.6b`) へ移植する
- 前提成果: NeMo Speech の ROCm (gfx1151) 移植は完了済み。詳細は `Speech/rocm-inference/README.md`

## 1. 目的

`~/AIassistant` の音声認識を NeMo Speech の `nemotron-3.5-asr-streaming-0.6b` に切り替える。
ただし WhisperX-ROCm を削除せず、**実行時にフォールバックできる2系統構成**とする。

移植の狙いは 2 つ:

1. **レイテンシ削減** — NeMo 側は 1.35〜4.00 秒の日本語発話に対し純粋推論 0.052〜0.086 秒(gfx1151, float32)。
   WhisperX (large-v3-turbo) との実測比較は Phase 3 で行う
2. **ストリーミング STT の獲得** — cache-aware ストリーミングにより、発話終了を待たずに認識を進められる。
   現行のプッシュトゥトーク方式では「録音停止 → 全体を STT」なので、発話長にほぼ比例した待ち時間が発生している

## 2. 要件(確定事項)

| # | 要件 | 決定 |
|---|---|---|
| 1 | whisperX-rocm と Speech rocm-inference の両対応 | **単一 venv に同居させ、実行時フォールバック**する |
| 2 | 既定バックエンド | **NeMo Speech (rocm-inference)** |
| 3 | `~/AIassistant` 内に `ln -s` で `Speech` を作成し、AIassistant 内のファイルは**すべてその symlink を参照**する | `~/AIassistant/Speech -> ../Speech`(既存の `whisperX-rocm`, `llama.cpp` と同じ流儀) |
| 4 | whisperX 使用時と Speech 使用時の速度比較 | **最終フェーズで実施**(Phase 3) |
| 5 | 本仕様書の置き場所 | `~/AIassistant/NeMo-STT移植_仕様書.md` |
| 6 | NeMo の推論モード | **バッチとストリーミングの両方を実装**する |

## 3. 現状(as-is)

### 3.1 パイプライン

```
Browser (three-vrm/TalkingHead/zundamon.html)
  └─ MediaRecorder (webm/opus) — 録音停止までバッファ
         ↓ POST /voice_chat_speak_stream (multipart, 音声まるごと1個)
    three-vrm server (:8000)
         ↓ POST /voice_chat_stream (multipart)
       ttllm bridge (:8001)
         ├─ WhisperX-ROCm (STT)      ← ここを差し替える
         └─ llama-server (:8080)
         ↓ SSE (transcript → token…)
    three-vrm: 文単位で VOICEVOX (:50021) → WS で音声+viseme 配信
```

### 3.2 STT の実装箇所

STT は `ttllm/server.py` の **3 関数に閉じている**。移植範囲はここが中心になる。

| 関数 | 役割 |
|---|---|
| `_load_whisperx()` | `import whisperx` の遅延ロード |
| `get_model()` | `wx.load_model(...)` のシングルトン |
| `_transcribe_path(path) -> str` | 音声ファイルパス → 転写テキスト |

呼び出し元は `/transcribe`, `/voice_chat`, `/voice_chat_stream` の 3 エンドポイントで、
いずれも `run_in_threadpool(_transcribe_path, path)` の形。**`_transcribe_path` のシグネチャを保てば呼び出し側は無改修**。

関連する環境変数(すべて `WHISPER_*` 接頭辞):
`WHISPER_MODEL` / `WHISPER_LANGUAGE` / `WHISPER_COMPUTE_TYPE` / `WHISPER_DEVICE` / `WHISPER_BATCH_SIZE` / `WHISPER_VAD_METHOD`

`/health` は `whisper` キーでモデル名・デバイス・ロード済みかを返している。

### 3.3 venv の現状

| | whisperX-rocm venv | Speech env-rocm |
|---|---|---|
| パス | `~/whisperx/whisperX-rocm/.venv` | `~/Speech/rocm-inference/env-rocm/.venv` |
| Python | 3.12.13 | 3.12.13 |
| torch | **2.8.0+rocm7.12.0** | **2.9.1+rocm7.13.0** |
| torchaudio | 2.8.0a0+rocm7.12.0 | (なし) |
| transformers | 5.14.1 | 5.15.0 |
| numpy | 2.0.2 | 2.5.2 |
| 固有 | ctranslate2 4.6.2 (ROCmビルド), whisperx 3.7.4, pyannote.audio 3.4.0, faster-whisper 1.2.1 | nemo-toolkit 3.1.0+ef41369156, lhotse 2.0.0a3 |

ttllm は現在 whisperX venv 上で動いている(`ttllm/run.sh` の `WHISPERX_VENV`)。
`three-vrm/server.py` も同じ venv の python で起動される(`start_all.sh`)。

---

## 4. 設計

### 4.1 symlink(要件 3)

```bash
ln -s ../Speech ~/AIassistant/Speech
```

以後、AIassistant 配下のファイルから NeMo Speech を参照する際は
**必ず `~/AIassistant/Speech/...` を経由**する。`~/Speech/...` を直接書かない。
これは venv への editable install も含む(`uv pip install -e ~/AIassistant/Speech` の形で行い、
venv 内の `__editable__` パスも symlink 経由にする)。
symlink 自体は既存の `whisperX-rocm` と同様に **git 管理に含める**。

:::note warn
`~/Speech` は **`rocm-inference` ブランチ**である必要がある(ROCm 対応差分がこのブランチにのみ入っているため)。
`start_all.sh` の事前チェックで、symlink 先が存在し、かつ ROCm パッチが入っていることを確認する。
:::

### 4.2 STT バックエンド抽象化

`ttllm/server.py` から STT 実装を切り出し、共通インターフェースの下に 2 実装を置く。

```
ttllm/
  server.py          FastAPI 本体(STT の中身は知らない)
  stt/
    __init__.py      get_backend() — 環境変数でバックエンドを解決
    base.py          STTBackend 抽象基底
    whisperx.py      WhisperXBackend
    nemo.py          NeMoBackend
```

インターフェース:

```python
class STTBackend:
    name: str

    def load(self) -> None:
        """モデルをロードする。/warmup から呼ばれる。"""

    def transcribe_path(self, path: str) -> str:
        """音声ファイルパス → 転写テキスト。既存 _transcribe_path と同じ契約。"""

    def transcribe_stream(self, chunks: Iterable[bytes]) -> Iterator[str]:
        """PCM チャンク列 → 部分転写の逐次出力。Phase 2 で追加。
        whisperX 実装は NotImplementedError を投げる。"""

    def info(self) -> dict:
        """/health に載せる情報。"""
```

`_transcribe_path()` は薄いラッパとして残し、呼び出し元 3 エンドポイントは無改修とする。

### 4.3 バックエンド選択とフォールバック

新設する環境変数:

| 変数 | 既定 | 意味 |
|---|---|---|
| `STT_BACKEND` | `auto` | `nemo` / `whisperx` / `auto` |
| `STT_FALLBACK` | `whisperx` | `auto` 時のフォールバック先。`none` で無効化 |
| `NEMO_MODEL` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | |
| `NEMO_LANGUAGE` | `ja-JP` | **`ja` 単体は不可**。prompt_dictionary のキー |
| `NEMO_DEVICE` | `cuda` | ROCm も `cuda` と名乗る |
| `NEMO_ATT_CONTEXT_SIZE` | `[56,13]` | ストリーミングのチャンク長(1120ms)。160ms は `[56,1]` |

`auto` の挙動:

1. 起動時(`/warmup`)に NeMo のロードを試みる
2. 成功 → 以後 NeMo を使う(**既定はこの経路**)
3. 失敗 → 警告ログを出して whisperX にフォールバックし、以後 whisperX を使う
4. 推論時に NeMo が例外を投げた場合も、その 1 リクエストは whisperX で処理し、以降は whisperX に固定する

「NeMo が落ちたまま無音を返し続ける」のが最悪なので、**フォールバックの発生は必ず WARNING でログに残し、`/health` にも反映**する。

```json
{
  "stt": {
    "backend": "nemo",
    "requested": "auto",
    "fallback_active": false,
    "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "device": "cuda",
    "loaded": true
  }
}
```

既存の `whisper` キーは互換のため残すか、`stt` に一本化するかを Phase 1 で決める(three-vrm 側が参照していないかを確認してから)。

### 4.4 単一 venv 同居戦略(最大のリスク)

**方針: whisperX 側の torch 2.8.0+rocm7.12.0 を土台とし、そこに `nemo-toolkit[asr]` を載せる。**

NeMo Speech の要求は `torch>=2.6.0` なので 2.8.0 で条件は満たす。
逆に whisperX を torch 2.9.1 へ上げる案は、pyannote.audio が `torchaudio.info` / `AudioMetaData` を使う都合で
**torchaudio を 2.9 未満に据え置く必要があり**、torch/torchaudio のバージョン差が開くため採らない。

同居にあたり確認が必要な項目:

| 項目 | 懸念 | 確認方法 |
|---|---|---|
| torch 2.8.0 での NeMo 動作 | 検証済みなのは 2.9.1。2.8.0 は未検証 | 27件の推論を再実行し、既存結果と一致するか |
| transformers | whisperx/pyannote は 5.14.1、NeMo env は 5.15.0 | 共通で動くバージョンを特定 |
| numpy | 2.0.2 (whisperX) vs 2.5.2 (NeMo env)。lhotse が新しい numpy を要求する可能性 | 解決後に両方の推論を実行 |
| lhotse 2.0.0a3 | pre-release。他パッケージとの衝突 | `uv pip install` の解決結果を確認 |
| ctranslate2 の初期化順 | **`import torch` を `import ctranslate2` より先に行う制約**は継続。NeMo の import が順序を壊さないか | `server.py` 冒頭の `import torch  # noqa: F401` を維持し、実際に両バックエンドをロードして確認 |
| ROCm ランタイムの二重化 | torch はバンドル ROCm 7.12、ctranslate2 はシステム ROCm 7.14 を使う既存構成。NeMo も torch のバンドルを使うので新規リスクは低い見込み | 両バックエンドを同一プロセスでロードして推論 |

:::note alert
**Phase 0 のゲート**: この同居が成立しない(依存解決できない / どちらかが動かない)と判明した場合は、
**venv を分離して起動時選択方式(再起動で切り替え)に退避**する。その場合「実行時フォールバック」は
「起動時に NeMo が使えなければ whisperX venv で起動し直す」意味に後退する。判断は Phase 0 で行い、本仕様書に追記する。
:::

同居 venv の置き場所は `~/AIassistant/ttllm/.venv` を新設する案と、whisperX venv をそのまま拡張する案がある。
**新設を採る** — whisperX 単体利用(`~/whisperx` 側の CLI)を壊さないため。
`WHISPERX_VENV` を参照している `ttllm/run.sh` / `ttllm/install.sh` / `start_all.sh` は新 venv を指すよう更新する。

なお `start_all.sh` は **three-vrm サーバも whisperX venv の python で起動している**
(`new_window "three-vrm" ... ${WHISPERX_VENV}/bin/python server.py`)。three-vrm の依存は aiohttp のみなので、
新 venv に aiohttp を含めて同じ venv で起動するよう更新する(venv を 2 つ維持しない)。

### 4.5 モデルキャッシュとオフライン起動

`ttllm/run.sh` は `HF_HUB_OFFLINE=1` を既定にしている(起動時に HF へ問い合わせて詰まるのを防ぐため)。
NeMo の `from_pretrained()` も HF からモデルを取得するので、**事前にキャッシュしておかないとオフラインで起動できない**。

- Phase 0 のセットアップ手順に「モデルの事前ダウンロード」を明記する
- `install.sh` にダウンロードステップを入れる
- 起動時にキャッシュが無ければ、**分かりやすいエラーで即失敗**させる(無言でフォールバックしない)

### 4.6 ストリーミング STT(要件 6)

現行はブラウザが録音停止まで音声を溜め、まとめて送っている。ストリーミング STT を活かすには、
**発話中から音声を送り続ける**必要がある。

:::note warn
**MediaRecorder の `start(timeslice)` でチャンクを送る方法は使えない。**
webm/opus のチャンクは先頭以外が単独でデコードできず、サーバ側で 1 チャンクずつ扱えない。
**Web Audio API (AudioWorklet) で生 PCM を取り出し、16kHz mono にダウンサンプルして送る**方式を採る。
:::

構成:

```
Browser
  └─ getUserMedia → AudioWorklet → Float32 PCM → 16kHz mono へ変換
         ↓ WebSocket (バイナリ PCM チャンク)
    three-vrm server (:8000)  /stt_stream
         ↓ WebSocket 中継
       ttllm (:8001)  /transcribe_stream
         └─ NeMoBackend.transcribe_stream()
              cache-aware streaming (att_context_size=[56,13])
         ↓ 部分転写を逐次返す
    発話終了(ブラウザが送信終了を通知) → 確定転写 → 既存の LLM → VOICEVOX 経路へ
```

- 既存の一括経路(`/voice_chat_stream`)は**残す**。ストリーミングは追加経路とし、
  ブラウザ側にフラグ(既定 off → 検証後に on)を置く
- whisperX バックエンドはストリーミング非対応。`STT_BACKEND=whisperx` 時はストリーミング経路を無効化し、
  自動的に一括経路へ倒す
- 発話終端の扱い: NeMo は右 context が足りないと最終トークンを出せない。
  **送信終了時に 0.5 秒相当の無音 PCM を流し込んでから確定**させる(Phase 1 検証で判明済みの挙動)

ストリーミングの効果測定は Phase 3 の比較に含める。

### 4.7 NeMoBackend 実装上の要点

**音声デコードを自前で持つ。** ttllm に届くのはブラウザの **webm/opus**(MediaRecorder 出力)であり、
16kHz WAV ではない。whisperX は `load_audio()` が内部で ffmpeg デコードするためこれを意識せずに済んでいたが、
NeMo 経路には相当する処理が無い。`NeMoBackend` は ffmpeg で **webm/ogg/mp4 → 16kHz mono float32** に
変換してからモデルへ渡す。あわせて**末尾に 0.5 秒の無音を付加**する
(ストリーミングで必須なのは検証済み。バッチも、ボタンリリースで発話直後に切れる録音への保険として同様に行う)。

**`model.transcribe()` は使わない。** `transcribe()` は呼び出しごとに Lhotse dataloader を再構築し、
**1 リクエストあたり約 0.2〜0.4 秒を浪費する**(ROCm 移植時の計測: `transcribe()` 経由 0.29〜0.5 秒 vs
直接 forward 0.05〜0.09 秒)。これでは whisperX からの移行メリットが消えるため、
`Speech/rocm-inference/scripts/benchmark.py` と同じ **encoder forward + `rnnt_decoder_predictions_tensor()` の
直接経路**で実装する。プロンプト(`ja-JP` = prompt id 10)はテンソルで直接渡す。

**言語タグを除去する。** モデルは終端句読点の後に `<ja-JP>` 形式のタグを付けることがある。
`re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")` で除去してから返す。

**排他制御。** 現行は単一グローバルモデル + `run_in_threadpool` で、whisperX 時代から同時リクエストの
排他は暗黙(実質 PTT で 1 ユーザ)。NeMo でも**バッチ推論はモデル単位のロックで直列化**すれば足りる。
ストリーミングは cache(encoder 状態)がセッションごとに必要なので、**セッションごとに独立した
cache 状態オブジェクトを持ち**、モデル本体の forward だけをロックで守る。同時ストリーミングセッション数は
当面 1 に制限してよい(PTT 前提のため)。Phase 2 で明示的に実装する。

**フォールバック先のロードタイミング。** whisperX を遅延ロードにすると、初回フォールバック時の
リクエストがモデルロード(数十秒)を被る。VRAM は 48GB あり両モデル常駐(NeMo float32 ≈ 2.5GB +
large-v3-turbo)で問題ないため、**`STT_BACKEND=auto` 時は warmup で両方ロード(eager)を既定**とし、
`STT_EAGER_FALLBACK=0` で遅延に切り替えられるようにする。

---

## 5. フェーズ構成

### Phase 0: 準備と同居検証(ゲート)

- `~/AIassistant/Speech -> ../Speech` の symlink 作成。`~/Speech` が `rocm-inference` ブランチであることを確認
- `~/AIassistant/ttllm/.venv` を Python 3.12 で新設し、**whisperx + nemo-toolkit[asr] + ttllm の依存を同居**させる
- モデル(`nemotron-3.5-asr-streaming-0.6b`)を HF から事前キャッシュ
- **完了条件(すべて満たすこと)**:
  1. 同一プロセス内で `import torch` → NeMo モデルロード → whisperX モデルロードが順に成功する
  2. 同一プロセス内で両バックエンドが実際に日本語音声を転写できる
  3. NeMo の出力が既存の検証結果(`Speech/rocm-inference/docs/`)と一致する
- **未達の場合**: 4.4 の退避策(venv 分離 + 起動時選択)に切り替え、本仕様書を更新してから Phase 1 へ
- **実施結果(2026-08-13): 通過**。完了条件3つをすべて満たしたため退避策は不要。
  - 同居 venv: `~/AIassistant/ttllm/.venv`(torch 2.8.0+rocm7.12.0 / whisperx 3.7.4 / nemo-toolkit 3.1.0+cc3a90f12b / numpy 2.0.2)
  - **torch 2.8.0 での NeMo は 2.9.1 と同等**(速度差は全音源で2%以内、転写結果は9/9一致)
  - 記録は `docs/STT移植_PHASE0.md`、検証スクリプトは `ttllm/verify_coexist.py`
  - 先行観察: **whisperX は NeMo が唯一外す `ritsu_river` を正しく認識する**。また whisperX は「2番目」
    (算用数字)+文末句読点、NeMo は「二番目」(漢数字)と表記が揺れる。Phase 3 の精度比較は
    単純完全一致では測れないため、正規化または人手判定を前提にすること

### Phase 1: バッチ推論の両対応

- `ttllm/stt/` を新設し、`STTBackend` / `WhisperXBackend` / `NeMoBackend`(バッチ)を実装
- `server.py` の `_load_whisperx` / `get_model` / `_transcribe_path` を抽象化に置き換え
- `STT_BACKEND` / `STT_FALLBACK` を実装。**既定は NeMo**
- `/health` に `stt` セクションを追加
- `ttllm/run.sh` / `ttllm/install.sh` / `start_all.sh` を新 venv 向けに更新
- NeMoBackend は 4.7 の要点(ffmpeg デコード + 末尾パディング / 直接 forward 経路 / タグ除去 / ロック)を満たすこと
- 完了条件:
  1. `STT_BACKEND=nemo` / `whisperx` / `auto` の 3 通りで `/voice_chat_stream` が動作
  2. NeMo を意図的に失敗させたとき(モデルパスを壊す等)、whisperX に自動フォールバックし WARNING が出る
  3. ブラウザから通しで会話できる(音声 → 応答 → VOICEVOX 発話)。**入力は実際の webm/opus であること**
  4. NeMo バッチの 1 リクエスト STT 時間が `transcribe()` 経由ではなく直接経路の水準(3〜4 秒発話で 0.1 秒台)であること
- **実施結果(2026-08-13): 完了**(条件 1/2/4 は実測、条件 3 はブラウザ通しのみ未実施)
  - NeMo **114〜129ms** vs whisperX **279〜292ms**(同一 webm/opus 入力、3.5秒発話)。約 2.4 倍速い
  - フォールバックはロード失敗・推論失敗の両方で動作。WARNING + `/health` の `fallback_active` に反映
  - **ウォームアップ推論が必須と判明**: モデルロードだけでは初回が 2831ms かかる。
    `NeMoBackend.load()` に捨て推論を追加して初回も 129ms に
  - 記録は `docs/STT移植_PHASE1.md`

### Phase 2: ストリーミング STT

- `NeMoBackend.transcribe_stream()` を cache-aware streaming で実装
- ttllm に `/transcribe_stream`(WebSocket)を追加
- three-vrm に `/stt_stream`(WebSocket 中継)を追加
- ブラウザ側を AudioWorklet + WebSocket 送信に対応(**既定 off のフラグ付き**)
- 送信終了時の無音パディング(0.5秒相当)を実装
- 完了条件:
  1. 発話中に部分転写が返ってくる
  2. 発話終了から確定転写までの時間が、一括経路より明確に短い
  3. フラグ off で従来経路が完全に元どおり動く
- **実施結果(2026-08-13): 完了**(3条件とも達成)
  - 部分転写は約1.1秒(チャンク長)ごとに更新。確定は発話終了から **56ms**
  - 10.4秒発話で **56ms vs 230ms(一括)**。**確定時間は音声長にほぼ依存しない**ため、発話が長いほど差が開く
  - 100ms ずつ mel を計算すると境界のアーティファクトで認識が劣化したため、
    **生PCMを蓄積して毎回全体から mel を作り直す**方式に変更(+ `online_normalization=True`)
  - **副次効果**: `att_context_size=[56,13]` の適用でバッチ経路の精度も改善し、
    Phase 1 で唯一外していた `ritsu_river` が正解に。9音源すべてが両経路で正解になった
  - 記録は `docs/STT移植_PHASE2.md`

### Phase 3: 速度比較(要件 4 / 最終フェーズ)

**比較対象(3 構成)**

| # | 構成 |
|---|---|
| A | WhisperX-ROCm (large-v3-turbo, batch_size=8, float16) — 現行 |
| B | NeMo Speech バッチ (ja-JP, float32) |
| C | NeMo Speech ストリーミング (ja-JP, att_context_size=[56,13]) |

**測定指標**

| 指標 | 定義 |
|---|---|
| STT 純粋推論時間 | モデルロード済み・ウォームアップ後。複数回の**中央値** |
| STT RTF | 推論時間 ÷ 音声長 |
| **発話終了 → 転写確定までの実時間** | ユーザ体感に最も近い。C はここで効くはず |
| 発話終了 → 最初の VOICEVOX 音声再生まで | エンドツーエンド |
| モデルロード時間 | 起動コスト(参考) |

**測定条件**

- 音声は `Speech/rocm-inference` の検証で使った 9 音源(肉声3 / ずんだもん3 / 波音リツ3)に加え、
  **10秒級の長めの発話を数本**追加する(現行検証は 1.35〜4.00 秒のみで、実際の会話より短い)
- 各構成・各音源で 20 回計測、中央値で比較。min/max も記録
- **転写精度も併記**する。速度だけ見て精度が落ちていたら意味がないため
- 計測スクリプトは `~/AIassistant/bench/` に置き、結果を JSON + Markdown で保存

**完了条件**: 3 構成 × 全音源の比較表(速度 + 精度)が出そろい、
どの構成を既定にすべきかの判断材料になっていること。

- **実施結果(2026-08-13): 完了**。13音源(短文9 + 長文4)× 3構成 × 20回
  - **速度**: NeMo一括は whisperX の約2〜3倍。ストリーミングは**音声長によらず 48〜106ms**で確定し、
    3.5秒で whisperX比 5.8倍、12.8秒で 10.8倍
  - **精度**: 短文9本は3構成とも CER 0.0%。**B と C は全音源で完全一致**(ストリーミング化による劣化なし)。
    ただし**長文では whisperX が優位**(long3 で 9.9% vs 21.1%)
  - **エンドツーエンド**: 差は STT ほど明確でない(LLM と TTS が支配的)
  - **判断: 既定は NeMo のままでよい**。実用対象の短い会話文では精度同等・速度2〜3倍。
    精度が要る場面は `STT_BACKEND=whisperx` で切り替え可能
  - 記録は `docs/STT移植_PHASE3.md`、生データは `bench/results.json` / `bench/results_e2e.json`

### Phase 4: ドキュメント更新

- `README.md` / `READMEJ.md` — 構成図の「WhisperX-ROCm (STT)」を更新、コンポーネント表に `Speech/` を追加、
  バックエンド切り替え手順を記載
- `TECHNICAL.md` / `TECHNICALJ.md` — Phase 3 の測定結果、バックエンド抽象化の設計、ストリーミング経路の仕様
- `ttllm/README.md` / `READMEJ.md` — 環境変数とフォールバック挙動

---

## 6. リスクと既知の懸念

| 項目 | 内容 | 対応 |
|---|---|---|
| **venv 同居** | torch / transformers / numpy / lhotse の依存が解決できない可能性。**本件最大のリスク** | Phase 0 をゲートにし、未達なら venv 分離へ退避 |
| torch 2.8.0 での NeMo | 検証済みは 2.9.1。2.8.0 は未検証 | Phase 0 で 27 件の再現確認 |
| ctranslate2 の初期化順 | `import torch` を先に行う制約。NeMo の import が絡んで壊れないか | `server.py` 冒頭の `import torch  # noqa: F401` を維持し実機確認 |
| HF オフライン起動 | `HF_HUB_OFFLINE=1` 下で NeMo モデルが取得できず起動失敗 | 事前キャッシュを手順化し、無ければ即エラー |
| 認識精度の差 | NeMo は 0.6B、whisperX は large-v3-turbo。**語彙・固有名詞で NeMo が劣る可能性**がある | Phase 3 で速度と精度を必ず併記。精度が実用を割るなら既定を再検討 |
| 長尺音声 | NeMo 側の検証は 1.35〜4.00 秒のみ。同一機の whisperX は 60 秒超で GPU memory fault の実績あり | Phase 3 で 10 秒級を追加。長時間ストリーミングは別途 |
| ストリーミングのブラウザ改修 | MediaRecorder では不可、AudioWorklet + WebSocket が必要。**移植の範囲を超える規模** | 既定 off のフラグで段階導入し、従来経路を常に残す |
| float32 固定 | cache-aware モデルは fp16/bf16 不可。whisperX は float16 で動いている | 速度比較はこの条件込みで評価する |
| VAD | whisperX は silero VAD で無音を弾き、無発話時に空文字を返している。**NeMo 経路に同等の処理が無い** | Phase 1 で無発話時の挙動を決める(NeMo は空文字を返す想定だが要確認) |
| webm デコード | NeMo 経路は音声デコードを自前で持つ必要がある(4.7)。ffmpeg 呼び出し分のレイテンシが乗る | デコード時間も Phase 3 の計測に含める(whisperX 側も同等のデコードをしているため公平) |
| `transcribe()` オーバーヘッド | 素直に `model.transcribe()` を使うと 0.2〜0.4 秒/回の dataloader 構築コストで移行メリットが消える | 直接 forward 経路で実装(4.7)。Phase 1 完了条件 4 で担保 |
| 同時リクエスト | ストリーミングの cache 状態はセッション固有。バッチとの並行や複数セッションで壊れうる | モデルロック + セッション別状態(4.7)。当面同時 1 セッションに制限 |

## 7. スコープ外

- **学習・ファインチューニング**(NeMo Speech の ROCm 対応は推論のみ)
- WhisperX-ROCm 側の改良
- LLM (llama-server) / VOICEVOX / VRM 表示まわりの変更(ストリーミング STT の経路追加を除く)
- 長時間(数分以上)の連続ストリーミングにおける cache 挙動の検証

## 8. 参照

- NeMo Speech ROCm 移植: `~/AIassistant/Speech/rocm-inference/README.md`
- 移植時の記録: `~/AIassistant/Speech/rocm-inference/docs/PHASE{0,1,3}.md`
- 現行 STT 実装: `~/AIassistant/ttllm/server.py`
- 起動スクリプト: `~/AIassistant/start_all.sh`, `~/AIassistant/ttllm/run.sh`

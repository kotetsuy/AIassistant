# Phase 1 記録 — バッチ推論の両対応

- 実施日: 2026-08-13
- 結果: **完了**(完了条件4つのうち3つを実測で確認、1つはユーザ操作待ち。下記)

## 完了条件の判定

| 条件 | 結果 |
|---|---|
| ① `nemo` / `whisperx` / `auto` の 3 通りで動作 | ✅ 実測 |
| ② NeMo 失敗時に whisperX へ自動フォールバックし WARNING が出る | ✅ ロード失敗・推論失敗の両方で実測 |
| ③ ブラウザから通しで会話できる(webm/opus 実入力) | ⚠️ **HTTP 経由の webm/opus 転写までは実測済み**。ブラウザ通しは Chrome が起動するため未実施(下記) |
| ④ NeMo バッチの STT 時間が直接経路の水準(3〜4秒発話で 0.1 秒台) | ✅ **114〜129ms**(HTTP 込み 0.12s) |

## 実装

### 新設: `ttllm/stt/`

| ファイル | 役割 |
|---|---|
| `base.py` | `STTBackend` 抽象基底。`transcribe_path(path) -> str` が契約の中心 |
| `audio.py` | ffmpeg で webm/ogg/mp4 → 16kHz mono float32。末尾 0.5 秒の無音付加 |
| `nemo.py` | `NeMoBackend`。直接 forward 経路・言語タグ除去・モデルロック・ウォームアップ |
| `whisperx.py` | `WhisperXBackend`。移植前の挙動をそのまま維持(VAD 無発話時に `""`) |
| `__init__.py` | `STTRouter`。バックエンド選択とフォールバック |

### `server.py` の変更

- `_load_whisperx()` / `get_model()` の whisperX 直結を削除し、`STTRouter` に委譲
- `_transcribe_path()` は薄いラッパとして残したので、**呼び出し元 3 エンドポイントは無改修**
  (`/transcribe`, `/voice_chat`, `/voice_chat_stream`)
- `/health` の `whisper` キーを `stt` に置き換え。
  参照している消費者が無いことを確認済み(`start_all.sh` の `wait_http` は 200 だけを見る)
- 全バックエンド失敗時は**空文字ではなく 503** を返す(無言で無音扱いにしない)

### 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `STT_BACKEND` | `auto` | `nemo` / `whisperx` / `auto` |
| `STT_FALLBACK` | `whisperx` | `auto` 時のフォールバック先。`none` で無効 |
| `STT_EAGER_FALLBACK` | `1` | warmup でフォールバック先も先読みする |
| `NEMO_MODEL` | `nvidia/nemotron-3.5-asr-streaming-0.6b` | |
| `NEMO_LANGUAGE` | `ja-JP` | `ja` 単体は拒否される |
| `NEMO_DEVICE` | `cuda` | ROCm も `cuda` と名乗る |

既存の `WHISPER_*` は whisperX バックエンド用としてそのまま維持。

### 起動スクリプト

- `ttllm/run.sh`: `WHISPERX_VENV` → **`TTLLM_VENV`**(既定 `~/AIassistant/ttllm/.venv`)。
  `STT_BACKEND` / `STT_FALLBACK` の既定値を export
- `ttllm/install.sh`: 共用 venv をゼロから構築する内容に全面改訂。
  symlink の存在確認、`Speech` が `rocm-inference` ブランチかの警告、ctranslate2 の ROCm ビルド上書き、
  **NeMo モデルの事前キャッシュ**まで行う
- `start_all.sh`: `TTLLM_VENV` に変更。three-vrm も同 venv で起動。
  **`Speech` symlink の存在チェックを preflight に追加**

## 実測結果

### バックエンド別の転写(入力は webm/opus、ブラウザと同形式)

音源: 「日本で二番目に高い山は」(肉声、3.5秒)

| モード | 1回目 | 2回目 | 3回目 | 転写 |
|---|---|---|---|---|
| `nemo` | 129ms | 114ms | 117ms | 日本で二番目に高い山は |
| `whisperx` | 434ms | 292ms | 279ms | 日本で2番目に高い山は、 |
| `auto` | 129ms | 114ms | 117ms | 日本で二番目に高い山は |

内訳(NeMo): ffmpeg デコード 28〜36ms + 推論 88ms。RTF 0.02。
内訳(whisperX): デコード 26〜33ms + 推論 252〜404ms。RTF 0.07。

**NeMo は whisperX の約 2.4 倍速い**(Phase 3 で正式に比較する)。

### ウォームアップの効果

初期実装ではモデルをロードするだけで推論していなかったため、**最初の 1 リクエストが 2831ms** かかっていた
(カーネルコンパイルとアロケータの初期化)。`NeMoBackend.load()` に 1 秒の無音で捨て推論を行う
`_warmup()` を追加し、**初回も 129ms** になった。

`/warmup` 全体の所要時間: **33 秒**(NeMo ロード 25.3s + ウォームアップ 2.8s + whisperX ロード 0.7s)。
`start_all.sh` の warmup 呼び出しは `-m 300` なので余裕がある。

### フォールバック

**ロード失敗時**(`NEMO_MODEL` を存在しないモデルに変更):

```
WARNING STT falling back from nemo to whisperx and staying there — load failed:
        Cannot reach https://huggingface.co/nvidia/does-not-exist-xyz/... : offline mode is enabled.
  transcript: '日本で2番目に高い山は、'
  backend= whisperx fallback_active= True
```

**推論失敗時**(`_infer` に例外を注入):

```
  正常時: '日本で二番目に高い山は' nemo
WARNING STT falling back from nemo to whisperx and staying there — inference failed: injected GPU fault
  故障後: '日本で2番目に高い山は、' whisperx
  次回も: '日本で2番目に高い山は、' whisperx
```

いずれも WARNING が出て、`/health` の `fallback_active` と `last_error` に反映される。
**失敗したリクエスト自体もフォールバック先で処理され、転写が返る**(取りこぼさない)。

### `/health`

```json
{
  "ok": true,
  "stt": {
    "backend": "nemo",
    "requested": "auto",
    "fallback": "whisperx",
    "fallback_active": false,
    "last_error": null,
    "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "language": "ja-JP",
    "device": "cuda",
    "loaded": true,
    "supports_streaming": false
  },
  "llama": { "url": "http://localhost:8080", "reachable": false }
}
```

## 完了条件③について

HTTP サーバを起動し、**ブラウザと同じ webm/opus を `/transcribe` に POST して転写できること**は実測した
(0.12s、上記)。一方、**ブラウザからの通し**(🎤 ボタン → three-vrm → ttllm → llama → VOICEVOX → 発話)は
`start_all.sh` が Chrome を新規ウィンドウで開く動作を含むため、**未実施**。

venv 起因のリスクは潰してある:

- `three-vrm/server.py` が共用 venv で `compile()` / `import aiohttp` に成功することを確認済み
- ttllm 側は HTTP レベルで動作確認済み
- `start_all.sh` は `bash -n` で構文確認済み

残っているのは実際に `./start_all.sh` を走らせる操作のみ。

## 新規に判明した注意点

- **NeMo はウォームアップ推論をしないと初回が 2.8 秒かかる**。モデルのロードだけでは足りない
- whisperX は初回転写も 434ms 止まりで、ウォームアップ依存が小さい(NeMo との性格差)
- `torchaudio.list_audio_backends()` の DeprecationWarning が whisperX ロード時に出るが無害

## Phase 2 への引き継ぎ

- `STTBackend.transcribe_stream()` は基底で `NotImplementedError` を投げる形で用意済み。
  `supports_streaming` も `/health` に出るようにしてある
- `NeMoBackend._infer()` を切り出してあるので、ストリーミング実装はここに cache 状態を足す形で入る
- 仕様書 4.7 のとおり、ストリーミングは**セッションごとに独立した cache 状態**を持たせ、
  モデル本体の forward のみ `self._lock` で守る

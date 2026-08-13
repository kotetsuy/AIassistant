# Phase 0 記録 — 準備と同居検証(ゲート)

- 実施日: 2026-08-13
- 結果: **通過**。完了条件3つをすべて満たしたため、退避策(venv 分離)は不要。Phase 1 へ進む

## 完了条件の判定

| 条件 | 結果 |
|---|---|
| ① 同一プロセスで `import torch` → NeMo ロード → whisperX ロードが成功 | ✅ |
| ② 同一プロセスで両バックエンドが日本語を転写できる | ✅ 9音源とも両方成功 |
| ③ NeMo の出力が既存検証結果と一致 | ✅ 9/9 一致(既知の `ritsu_river` 誤認識も含めて同一) |

検証スクリプト: `ttllm/verify_coexist.py`(再実行可能)

## symlink(要件 3)

```bash
ln -s ../Speech ~/AIassistant/Speech
```

- 参照先 `~/Speech` は **`rocm-inference` ブランチ / `cc3a90f12b`**
- ROCm パッチ(`nemo/core/utils/cuda_python_utils.py` の `torch.version.hip` 判定)が入っていることを確認済み
- venv への editable install も symlink 経由のパスで実施(`-e ~/AIassistant/Speech[asr]`)

## 同居 venv

**場所**: `~/AIassistant/ttllm/.venv`(Python 3.12.13)

whisperX 単体利用(`~/whisperx` 側の CLI)を壊さないため、既存 venv の拡張ではなく新設した。

### 構成

| パッケージ | バージョン | 備考 |
|---|---|---|
| torch | **2.8.0+rocm7.12.0** | whisperX の `torch>=2.8.0,<2.9.0` 制約に合わせた |
| torchaudio | 2.8.0a0+rocm7.12.0 | pyannote が `torchaudio<2.9` を要求 |
| triton | 3.4.0+rocm7.12.0 | |
| ctranslate2 | 4.6.2 | `~/whisperx/ctranslate2-rocm/python` のローカル ROCm ビルド |
| whisperx | 3.7.4 | `-e ~/AIassistant/whisperX-rocm` |
| nemo-toolkit | 3.1.0+cc3a90f12b | `-e ~/AIassistant/Speech[asr]` |
| pyannote.audio | 3.4.0 | |
| transformers | 5.15.0 | |
| numpy | **2.0.2** | whisperX の `numpy>=2.0.2,<2.1.0` 制約が支配 |
| lhotse | 2.0.0a3 | |
| faster-whisper | 1.2.1 | |
| fastapi / aiohttp | 0.141.1 / 3.14.3 | ttllm + three-vrm の依存 |

### 構築手順

```bash
cd ~/AIassistant/ttllm
uv venv --python 3.12

VIRTUAL_ENV=~/AIassistant/ttllm/.venv uv pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease allow \
  "torch==2.8.0+rocm7.12.0" "torchaudio==2.8.0a0+rocm7.12.0" \
  -e "$HOME/AIassistant/whisperX-rocm" \
  -e "$HOME/AIassistant/Speech[asr]" \
  "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9" \
  "httpx>=0.27" "pydantic>=2" "aiohttp"

# ctranslate2 は ROCm ローカルビルドで上書きする(PyPI 版は CUDA ビルド)
export CTRANSLATE2_ROOT=/usr/local
VIRTUAL_ENV=~/AIassistant/ttllm/.venv uv pip install --reinstall --no-deps \
  pybind11 ~/whisperx/ctranslate2-rocm/python
```

### 依存解決で必要だったこと

- **`--index-strategy unsafe-best-match`**: gfx1151 インデックスと PyPI をまたいで解決させるため。
  Speech 側の env-rocm と同じ理由(`rocm[libraries]` の sdist が AMD インデックスにしかない)
- **ctranslate2 は最後に `--reinstall --no-deps` で上書き**: 通常解決だと PyPI の 4.8.1(CUDA ビルド)が入る

### 懸念だった項目の結果

| 懸念(仕様書 4.4) | 結果 |
|---|---|
| torch 2.8.0 での NeMo 動作 | **問題なし**。転写結果・速度とも 2.9.1 と同等(下記) |
| transformers の共通版 | 5.15.0 で両方動作 |
| numpy 2.0.2 (whisperX 制約) で NeMo が動くか | **問題なし**。lhotse 2.0.0a3 も 2.0.2 で動作 |
| lhotse 2.0.0a3 の衝突 | なし |
| ctranslate2 の初期化順 | `import torch` を先頭に置く既存の制約を維持すれば問題なし |
| ROCm ランタイムの二重化 | 問題なし。torch はバンドル ROCm 7.12、ctranslate2 はシステム ROCm 7.14 |

## torch 2.8 での NeMo 性能

仕様書のリスク項目「検証済みは 2.9.1、2.8.0 は未検証」に対する確認。
`Speech/rocm-inference/scripts/benchmark.py` を同居 venv で実行(ウォームアップ5回 + 20回の中央値)。

| 音源 | torch 2.9.1+rocm7.13.0 | torch 2.8.0+rocm7.12.0 | 差 |
|---|---|---|---|
| human_greeting | 0.0765s | 0.0771s | +0.8% |
| human_mountain | 0.0861s | 0.0871s | +1.2% |
| human_river | 0.0837s | 0.0844s | +0.8% |
| ritsu_greeting | 0.0519s | 0.0521s | +0.4% |
| ritsu_mountain | 0.0799s | 0.0804s | +0.6% |
| ritsu_river | 0.0624s | 0.0626s | +0.3% |
| zundamon_greeting | 0.0523s | 0.0526s | +0.6% |
| zundamon_mountain | 0.0799s | 0.0815s | +2.0% |
| zundamon_river | 0.0781s | 0.0786s | +0.6% |

**差は全音源で 2% 以内。** torch を 2.8.0 に下げたことによる性能低下は実質無い。
`cuda_graphs_mode` も `no_while_loops` のままで、CUDA graphs のフォールバック挙動も同じ。

## モデルの事前キャッシュ

`HF_HUB_OFFLINE=1` 下でも起動できるよう、両モデルがキャッシュ済みであることを確認。

| モデル | 状態 |
|---|---|
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | snapshot `1c8deaec` にキャッシュ済み |
| `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | キャッシュ済み |
| silero VAD | `~/.cache/torch/hub/snakers4_silero-vad_master` にキャッシュ済み |

## Phase 3(速度比較)に向けた先行観察

ゲート検証で両バックエンドの出力が並んだので、精度比較の材料として記録しておく。

| 音源 | NeMo | whisperX (large-v3-turbo) |
|---|---|---|
| human_greeting | こんにちは | こんにちは。 |
| human_mountain | 日本で二番目に高い山は | 日本で**2番目**に高い山は、 |
| human_river | 日本で一番長い川は | 日本で一番長い川は、 |
| ritsu_greeting | こんにちは | こんにちは。 |
| ritsu_mountain | 日本で二番目に高い山は | 日本で**2番目**に高い山は、 |
| **ritsu_river** | **一本**で一番長い川は | 日本で一番長い川は、 |
| zundamon_greeting | こんにちは | こんにちは |
| zundamon_mountain | 日本で二番目に高い山は | 日本で**2番目**に高い山は |
| zundamon_river | 日本で一番長い川は | 日本で一番長い川は |

読み取れること:

1. **whisperX は `ritsu_river` を正しく認識する**。NeMo が唯一外している音源を large-v3-turbo は取れている。
   Phase 3 の精度比較では NeMo が不利になる可能性があり、**速度だけで既定を決めてはいけない**という
   仕様書の懸念が現実味を帯びた
2. **表記の揺れがある**。whisperX は「2番目」(算用数字)、NeMo は「二番目」(漢数字)。
   また whisperX は文末に「。」「、」を付ける傾向がある。
   **Phase 3 の精度比較は単純な完全一致では測れない**ため、表記正規化を入れるか、
   差分を人手で読む前提にする必要がある
3. **whisperX のロードは 2.1 秒、NeMo は 25.9 秒**。起動コストは NeMo が大幅に不利
   (ただし常駐サービスなので実用上の影響は小さい)

## 新規に判明した注意点

- torch 2.8.0+rocm7.12.0 で NeMo をロードすると
  `warning: xnack 'Off' was requested for a processor that does not support it!` が出る。
  2.9.1+rocm7.13.0 では出なかった。**動作に影響は無い**(転写結果・速度とも同等)が、ログに現れる
- 検証時は `LD_LIBRARY_PATH` に `/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib` が必要
  (ctranslate2 が システム ROCm を参照するため)。`ttllm/run.sh` の既存設定がそのまま使える

## 次フェーズへの引き継ぎ

- Phase 1 では `ttllm/stt/` を新設し、`verify_coexist.py` の `nemo_transcribe()` が
  そのまま `NeMoBackend.transcribe_path()` の中核になる(直接 forward 経路・タグ除去を実装済み)
- `ttllm/run.sh` / `ttllm/install.sh` / `start_all.sh` の `WHISPERX_VENV` を
  新 venv (`~/AIassistant/ttllm/.venv`) に向ける作業が残っている
- three-vrm も同 venv で起動するよう `start_all.sh` を更新する(aiohttp 導入済み)

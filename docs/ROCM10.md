# ROCm 10 への移行と検証

対象は Ubuntu 26.04 / Ryzen AI Max+ 395 (gfx1151)。
[移行時の実測記録](https://qiita.com/kotetsu_yama/items/e4a2d284158e3a13178c) を参考にする。

## 構成

- システム SDK: ROCm 10.0.0 (`amdrocm-core-sdk10.0-gfx1151`)。
- 音声用 `ttllm/.venv`: torch 2.8.0+rocm7.12.0 / torchaudio 2.8.0a0+rocm7.12.0 を維持。
- torch を CTranslate2 / WhisperX より先に import する。
- `HSA_OVERRIDE_GFX_VERSION` は設定しない。

システム更新に合わせて音声 wheel を更新する必要はない。torchaudio 2.9 以降では
既存 pyannote が必要とする API が消える。過去の PHASE0 記録にある
「CTranslate2 はシステム 7.14 を使う」という説明は、実ロード先の検証で訂正された。

## 既存環境を移行する場合

AMD の stable リポジトリは
`https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/`。
既存の鍵は上書き前に比較する。旧 SDK は新 SDK の動作確認まで保持し、
インストール前に `apt-get -s install amdrocm-core-sdk10.0-gfx1151` で変更内容を確認する。
OS のパッケージ更新はアプリの `install.sh` では実行しない。

`readlink -f /opt/rocm/core` と `/opt/rocm/core/.info/version` で有効な SDK を確認する。
`hipconfig --version` は HIP 自体の版であり、製品版とは区別する。
旧 SDK の削除や `apt autoremove` は、検証後に削除対象を確認して別途判断する。

## アプリの確認

リポジトリ直下で実行する。`ROCM_PATH` は両起動スクリプトのライブラリ検索先にも反映される。

```bash
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
unset HSA_OVERRIDE_GFX_VERSION
export LD_LIBRARY_PATH="/usr/local/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib/llvm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${TTLLM_VENV:-ttllm/.venv}/bin/python" ttllm/verify_rocm.py
GPU_MAX_HW_QUEUES=1 "${TTLLM_VENV:-ttllm/.venv}/bin/python" ttllm/verify_coexist.py
```

最初の検証はモデル不要で、製品版、音声 API、GPU 演算、CT2 デバイス認識と
実ロード先を確認する。後者はキャッシュ済みモデルと
`~/nemo-rocm-verify/audio/16k_pad/*.wav` を使って両 STT の転写を確認する。

llama.cpp は既存バイナリの `ldd` と `--list-devices`、実推論を先に確認する。
再ビルドする場合は最終配置先で configure/build する。ビルド後のディレクトリ名変更は
共有ライブラリの RUNPATH を壊す可能性がある。別ビルドを使う場合は移動せず、
`LLAMA_BIN=/path/to/build-rocm10/bin/llama-server ./start_all.sh` で選択できる。

最後に `./start_all.sh` で起動し、`http://localhost:8001/health` の STT 状態と、
ブラウザからの発話・応答・音声合成を確認する。

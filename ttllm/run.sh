#!/usr/bin/env bash
set -euo pipefail

# モデルはこのマシンに事前キャッシュ済み。既定でオフラインにして、
# load_model() 起動時に HuggingFace へリビジョン確認へ行って
# ネットワークが詰まると warmup ごとハングするのを防ぐ。
# 新しいモデルを取りに行きたいときは HF_HUB_OFFLINE=0 で起動する。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# 万一オンラインで起動した場合でも、メタデータ確認で無限に待たない。
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-10}"

# ROCm env for NeMo / whisperX on AMD Ryzen AI Max+ 395 (gfx1151).
# HSA_OVERRIDE_GFX_VERSION は設定しない (gfx1151 ネイティブビルドなので壊れる)。
unset HSA_OVERRIDE_GFX_VERSION
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
# /usr/local/lib はローカルビルドの ctranslate2 (whisperX 用) が参照する。
export LD_LIBRARY_PATH="/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

# STT バックエンド。auto は NeMo を使い、失敗したら whisperX に落ちる。
export STT_BACKEND="${STT_BACKEND:-auto}"
export STT_FALLBACK="${STT_FALLBACK:-whisperx}"

# 初音(最初の発話)を早めるため、最初の一文を短く言い切らせる。
# 文境界 [。！？\n] が早く出るほど three-vrm が早く VOICEVOX へ渡せる。
# export SYSTEM_PROMPT="${SYSTEM_PROMPT:-あなたはオリジナルキャラです。名前はコテコ。一人称は「コテコ」、元気いっぱいの明るい女の子として、「〜だよ！」「〜だね！」のような弾んだ口調で、親しみやすく簡潔に話してください。返答は必ず短い一文から始めること。最初の一文は15文字以内の相づち・結論・呼びかけにして、すぐ「。」で言い切る。詳しい説明はそのあとの文に分けて続ける。}"

# NeMo と whisperX を同居させた共用 venv (docs/STT移植_PHASE0.md)。
VENV="${TTLLM_VENV:-/home/$USER/AIassistant/ttllm/.venv}"
HOST="${BRIDGE_HOST:-0.0.0.0}"
PORT="${BRIDGE_PORT:-8001}"

cd "$(dirname "$(readlink -f "$0")")"
exec "$VENV/bin/python" -m uvicorn server:app --host "$HOST" --port "$PORT" "$@"

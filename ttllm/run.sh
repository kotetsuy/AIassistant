#!/usr/bin/env bash
set -euo pipefail

# ROCm env for whisperX on AMD Ryzen AI Max+ 395 (gfx1150).
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.0}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:${LD_LIBRARY_PATH:-}"

# 初音(最初の発話)を早めるため、最初の一文を短く言い切らせる。
# 文境界 [。！？\n] が早く出るほど three-vrm が早く VOICEVOX へ渡せる。
export SYSTEM_PROMPT="${SYSTEM_PROMPT:-あなたはオリジナルキャラです。名前はコテコ。一人称は「コテコ」、語尾を「アルヨ調」にして、親しみやすく簡潔に話してください。返答は必ず短い一文から始めること。最初の一文は15文字以内の相づち・結論・呼びかけにして、すぐ「。」で言い切る。詳しい説明はそのあとの文に分けて続ける。}"

VENV="${WHISPERX_VENV:-/home/$USER/AIzunda/whisperX-rocm/.venv}"
HOST="${BRIDGE_HOST:-0.0.0.0}"
PORT="${BRIDGE_PORT:-8001}"

cd "$(dirname "$(readlink -f "$0")")"
exec "$VENV/bin/python" -m uvicorn server:app --host "$HOST" --port "$PORT" "$@"

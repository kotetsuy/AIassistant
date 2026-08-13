#!/usr/bin/env bash
set -euo pipefail

# Build the shared venv that hosts BOTH STT backends (NeMo Speech + whisperX)
# plus the ttllm bridge and the three-vrm server.
#
# Why one venv: STT_BACKEND=auto falls back from NeMo to whisperX at runtime,
# which only works if both are importable in the same process. See
# ../NeMo-STT移植_仕様書.md §4.4 and ../docs/STT移植_PHASE0.md.

AIASSISTANT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
VENV="${TTLLM_VENV:-$AIASSISTANT/ttllm/.venv}"
CT2_SRC="${CTRANSLATE2_SRC:-/home/$USER/whisperx/ctranslate2-rocm/python}"

command -v uv >/dev/null 2>&1 || { echo "uv not found; see https://docs.astral.sh/uv/" >&2; exit 1; }

# Requirement: everything under AIassistant references NeMo Speech through the
# in-tree symlink, never ~/Speech directly.
[[ -e "$AIASSISTANT/Speech" ]] || {
    echo "missing symlink: $AIASSISTANT/Speech" >&2
    echo "create it with: ln -s ../Speech $AIASSISTANT/Speech" >&2
    exit 1
}
[[ -e "$AIASSISTANT/whisperX-rocm" ]] || {
    echo "missing symlink: $AIASSISTANT/whisperX-rocm" >&2
    exit 1
}

branch="$(git -C "$AIASSISTANT/Speech" branch --show-current 2>/dev/null || echo '?')"
if [[ "$branch" != "rocm-inference" ]]; then
    echo "WARNING: Speech is on branch '$branch', expected 'rocm-inference'." >&2
    echo "         The ROCm fix lives only on that branch; NeMo will fail to load." >&2
fi

uv venv --python 3.12 "$VENV"

# torch is pinned to 2.8.x because whisperX requires <2.9 (pyannote uses
# torchaudio.info / AudioMetaData, removed in torchaudio 2.9). NeMo only needs
# >=2.6, and Phase 0 measured it as no slower there than on 2.9.1.
# --index-strategy unsafe-best-match is required: torch+rocm depends on
# rocm[libraries], whose sdist exists only on the AMD index.
VIRTUAL_ENV="$VENV" uv pip install \
    --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow \
    "torch==2.8.0+rocm7.12.0" "torchaudio==2.8.0a0+rocm7.12.0" \
    -e "$AIASSISTANT/whisperX-rocm" \
    -e "$AIASSISTANT/Speech[asr]" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "python-multipart>=0.0.9" \
    "httpx>=0.27" \
    "pydantic>=2" \
    "aiohttp>=3.9"

# The resolver above picks up the PyPI ctranslate2, which is a CUDA build.
# Overwrite it with the locally built ROCm one.
if [[ -d "$CT2_SRC" ]]; then
    CTRANSLATE2_ROOT=/usr/local VIRTUAL_ENV="$VENV" \
        uv pip install --reinstall --no-deps pybind11 "$CT2_SRC"
else
    echo "WARNING: ctranslate2-rocm source not found at $CT2_SRC" >&2
    echo "         whisperX fallback will not work until it is built and installed." >&2
fi

# NeMo's from_pretrained() hits HuggingFace, but run.sh starts with
# HF_HUB_OFFLINE=1. Cache the model now or the server cannot start.
echo "Caching the NeMo ASR model..."
"$VENV/bin/hf" download nvidia/nemotron-3.5-asr-streaming-0.6b \
    nemotron-3.5-asr-streaming-0.6b.nemo >/dev/null

echo
echo "Shared venv ready: $VENV"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
for p in ("torch", "ctranslate2", "whisperx", "nemo-toolkit"):
    try:
        print(f"  {p:14s} {md.version(p)}")
    except Exception:
        print(f"  {p:14s} MISSING")
PY

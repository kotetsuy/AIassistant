#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [[ ! -x ".venv/bin/python" ]]; then
    echo ".venv が見つかりません。先に ./install.sh を実行してください。" >&2
    exit 1
fi

# A local Ubuntu PortAudio package can be unpacked here without sudo.
export LD_LIBRARY_PATH="$(pwd)/../.local/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec .venv/bin/python vtt.py "$@"

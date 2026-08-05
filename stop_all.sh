#!/usr/bin/env bash
# AIzunda パイプライン停止スクリプト。
#
#   ./stop_all.sh              → tmux セッション + VOICEVOX コンテナを停止
#   ./stop_all.sh --keep-voicevox → VOICEVOX は動かしたまま残す
#
# Chrome は閉じない (ユーザの操作を奪わない)。必要なら手で閉じる。

set -euo pipefail

SESSION="aiassistant"
VOICEVOX_CONTAINER="voicevox_engine"
KEEP_VOICEVOX=0

for arg in "$@"; do
    case "$arg" in
        --keep-voicevox) KEEP_VOICEVOX=1 ;;
        -h|--help)
            sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log()  { printf '\033[1;34m[stop]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[stop]\033[0m %s\n' "$*" >&2; }

# ---- 1. tmux セッションを落とす --------------------------------------

if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "tmux セッション ${SESSION} を終了します"
    tmux kill-session -t "$SESSION"
else
    log "tmux セッション ${SESSION} は起動していません"
fi

# ---- 2. 取りこぼしプロセスを止める -----------------------------------
# tmux のウィンドウを kill-session で閉じるとその中のプロセスも
# 親が消えて SIGHUP で落ちるが、SIGHUP を握り潰すものもあるので保険。
#
# start_all.sh は各サービスを cd してから相対パスで起動するので、cmdline に
# ディレクトリ名が出てこない (実際は "python -m uvicorn server:app ..." や
# "python server.py")。パス込みのパターンで pgrep しても永久に空振りするため、
# LISTEN ポートから pid を引き、cmdline で本人確認してから kill する。
# vtt はポートを持たないので cmdline パターンで拾う。

command -v ss >/dev/null || warn "ss が無いのでポートからの探索はスキップします"

# listening_pids <port> — 指定ポートで LISTEN している pid を列挙
listening_pids() {
    command -v ss >/dev/null || return 0
    ss -tlnpH "sport = :$1" 2>/dev/null |
        sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u
}

# stop_pids <label> <pid...> — SIGTERM して、落ちなければ SIGKILL
stop_pids() {
    local label="$1"; shift
    local pids=("$@") alive=() p
    (( ${#pids[@]} )) || return 0

    log "残存プロセスを停止: ${label} (pid=$(IFS=,; echo "${pids[*]}"))"
    kill "${pids[@]}" 2>/dev/null || true

    for _ in 1 2 3 4 5 6; do
        sleep 0.5
        alive=()
        for p in "${pids[@]}"; do
            kill -0 "$p" 2>/dev/null && alive+=("$p")
        done
        (( ${#alive[@]} )) || return 0
        pids=("${alive[@]}")
    done

    warn "  SIGTERM で落ちないので SIGKILL します (pid=$(IFS=,; echo "${pids[*]}"))"
    kill -9 "${pids[@]}" 2>/dev/null || true
}

# "<port>|<ラベル>|<cmdline に期待する正規表現>"
PORT_TARGETS=(
    "8080|llama-server|llama-server"
    "8001|ttllm|uvicorn"
    "8000|three-vrm|server\.py"
)

for target in "${PORT_TARGETS[@]}"; do
    IFS='|' read -r port label cmd_re <<<"$target"
    targets=()
    for pid in $(listening_pids "$port"); do
        [[ -r "/proc/$pid/cmdline" ]] || continue
        # 自分のプロセスで、かつ期待する cmdline のものだけ止める。
        # 無関係なプロセスがポートを使っていても巻き込まない。
        [[ "$(stat -c %U "/proc/$pid" 2>/dev/null)" == "$USER" ]] || continue
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
        if [[ "$cmd" =~ $cmd_re ]]; then
            targets+=("$pid")
        else
            warn ":${port} は想定外のプロセスが使用中なので触りません (pid=${pid}: ${cmd% })"
        fi
    done
    stop_pids "${label} (:${port})" ${targets[@]+"${targets[@]}"}
done

# vtt はポートを持たないので cmdline で拾う (自分のプロセスのみ)。
mapfile -t vtt_pids < <(pgrep -u "$USER" -f 'python[0-9.]* vtt\.py' || true)
stop_pids "vtt" ${vtt_pids[@]+"${vtt_pids[@]}"}

# ---- 3. VOICEVOX docker --------------------------------------------

if (( KEEP_VOICEVOX == 0 )); then
    if docker ps --format '{{.Names}}' | grep -qx "$VOICEVOX_CONTAINER"; then
        log "VOICEVOX コンテナ (${VOICEVOX_CONTAINER}) を停止します"
        docker stop "$VOICEVOX_CONTAINER" >/dev/null
    else
        log "VOICEVOX コンテナは既に停止しています"
    fi
else
    log "VOICEVOX は残します (--keep-voicevox)"
fi

log "停止完了"

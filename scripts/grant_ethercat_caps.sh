#!/bin/bash
# Give this repo's venv interpreter what the in-process EtherCAT master needs: cap_net_raw (a raw
# socket on the NIC) and cap_sys_nice (real-time priority for its loop).
#
#   scripts/grant_ethercat_caps.sh            # grant; re-run whenever `uv sync` recreates the venv
#   scripts/grant_ethercat_caps.sh --remove   # take them away again
#
# Only this venv gets them. .venv/bin/python is normally a symlink to a shared interpreter and
# setcap follows symlinks, so capping it in place would hand every program using that interpreter
# raw sockets -- it is replaced by the venv's own copy first. The other interpreter names become
# hard links to that copy: an interpreter holding capabilities runs in secure mode, and started
# through a symlink it cannot find its standard library, which breaks `uv sync` (uv runs python3).
#
# Run it as the venv's owner, not as root: only setcap needs root, and it asks for sudo itself. As
# root, the interpreter copy and anything it writes would be root-owned inside a user's venv.
set -euo pipefail
shopt -s nullglob

if [ "$(id -u)" = "0" ]; then
    echo "run this as the venv's owner, not as root; it uses sudo for setcap itself" >&2
    exit 1
fi

BIN="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin"
if [ ! -e "$BIN/python" ]; then
    echo "no venv at $BIN -- run \`uv sync\` first" >&2
    exit 1
fi
cd "$BIN"

if [ "${1:-}" = "--remove" ]; then
    sudo setcap -r python 2>/dev/null || true
    echo "capabilities removed from $BIN/python"
    exit 0
fi

if [ -L python ]; then
    cp --remove-destination "$(readlink -f python)" python
fi
for n in python3 python3.[0-9]*; do
    if [ -L "$n" ] || { [ -e "$n" ] && ! [ "$n" -ef python ]; }; then
        ln -f python "$n"
    fi
done
sudo setcap cap_net_raw,cap_sys_nice+ep python

getcap python
./python3 -I -B -c "import sys; print('venv interpreter starts:', sys.prefix)"

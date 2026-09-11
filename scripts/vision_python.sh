#!/usr/bin/env bash
# Vision interpreter: .venv python plus the system GL/X11/glib libraries
# that the full opencv build (pulled in by ultralytics) needs at load.
# The base exports mirror /tmp/run_python.sh; the EXTRA_* entries were
# derived with `ldd .venv/lib/python3.12/site-packages/cv2/cv2*.so` on
# this machine. Missing directories are skipped harmlessly.
# Post-v1: resolve proper opencv-headless packaging instead (see issues).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
BASE_LIBS=(
  "/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib"
  "/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib"
  "/nix/store/zks9mfsn4rqr6z9g6pcj2xqzcsplj0nb-zlib-1.3.2/lib"
)
EXTRA_LIBS=(
  "/nix/store/msrf7lbwkxmzw9igyxjbhz9qc8097yvc-glib-2.88.3/lib"
  "/nix/store/j5ml0bwadkmxc6185ilixlw85ywagxgv-libxcb-1.17.0/lib"
  "/nix/store/wwckb31fcbwj479g7qwcb3b7cv6416pf-libglvnd-1.7.0/lib"
)
LIBS=()
for d in "${EXTRA_LIBS[@]}" "${BASE_LIBS[@]}"; do
  [ -d "$d" ] && LIBS+=("$d")
done
export NIX_LD=/nix/store/n51dhmdbik1kfrsm62j5knavmigwrl1a-glibc-2.42-84/lib/ld-linux-x86-64.so.2
export LD_LIBRARY_PATH="$(IFS=:; echo "${LIBS[*]}")"
export LD_PRELOAD=/nix/store/wi6kaycsa2479qrxv9xy6yg3q5ggfs6j-gcc-15.3.0-libgcc/lib/libgcc_s.so.1
exec "$ROOT/.venv/bin/python" "$@"

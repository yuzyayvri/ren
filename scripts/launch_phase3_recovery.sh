#!/nix/store/0550j0i8bmzxbcnzrg1g51zigj7y12ih-bash-interactive-5.3p9/bin/bash
set -euo pipefail
ROOT=/home/yuzy/ren
RUN="$ROOT/artifacts/phase3_detector_v2/recovery"
mkdir -p "$RUN"
exec 9>"$RUN/recovery.lock"
flock -n 9 || exit 73
atomic(){ local t="$1"; shift; local x="$t.tmp.$$"; printf '%s\n' "$*" > "$x"; mv -f "$x" "$t"; }
atomic "$RUN/pid.marker" "pid=$$"
atomic "$RUN/start.marker" "timestamp=$(date --iso-8601=seconds)" "mode=single-recovery"
"$ROOT/scripts/run_phase3_python.sh" scripts/phase3_recovery_bindings.py
test ! -e "$RUN/recovery_authorization.json" -a ! -e "$RUN/recovery_attempt.json" -a ! -e "$RUN/payload.npz" || exit 74
status=0
trap 'status=$?; if test "$status" -eq 0; then atomic "$RUN/success.marker" "timestamp=$(date --iso-8601=seconds)" "exit_code=0"; else atomic "$RUN/failure.marker" "timestamp=$(date --iso-8601=seconds)" "exit_code=$status"; fi; rm -f "$RUN/pid.marker"' EXIT
cd "$ROOT"
exec "$ROOT/scripts/run_phase3_python.sh" -u -m scripts.phase3_recovery_real >>"$RUN/recovery.log" 2>&1

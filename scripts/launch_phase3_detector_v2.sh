#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash
set -euo pipefail
ROOT=/home/yuzy/ren
RUN="$ROOT/artifacts/phase3_detector_v2"
mkdir -p "$RUN"
LOCK="$RUN/job.lock"; PID="$RUN/pid.marker"; START="$RUN/start.marker"; SUCCESS="$RUN/success.marker"; FAILURE="$RUN/failure.marker"; LOG="$RUN/train.log"
exec 9>"$LOCK"; flock -n 9 || exit 73
free_gb() { df -P "$ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}'; }
test "$(free_gb)" -ge 10 || { echo "preflight free space below 10 GiB" >&2; exit 75; }
if test -s "$PID"; then old=$(sed -n 's/^pid=//p' "$PID"); test -z "$old" || ! kill -0 "$old" 2>/dev/null || { echo "matching job exists" >&2; exit 74; }; fi
rm -f "$SUCCESS" "$FAILURE"
atomic() { local target="$1"; shift; local tmp="${target}.tmp.$$"; printf '%s\n' "$*" > "$tmp"; mv -f "$tmp" "$target"; }
atomic "$PID" "pid=$$" "job=phase3-detector-yolo11n-20260909-v2"
atomic "$START" "timestamp=$(date --iso-8601=seconds)" "free_gib=$(free_gb)" "role=train_only" "model_sha256=43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c" "ultralytics=8.4.141"
watchdog() { while kill -0 "$child" 2>/dev/null; do test "$(free_gb)" -ge 5 || { echo "FAILSAFE: free space below 5 GiB" >> "$LOG"; kill -TERM "$child"; return; }; sleep 10; done; }
status=0
trap 'status=$?; kill "$watch" 2>/dev/null || true; if test "$status" -eq 0; then atomic "$SUCCESS" "timestamp=$(date --iso-8601=seconds)" "exit_code=0"; else atomic "$FAILURE" "timestamp=$(date --iso-8601=seconds)" "exit_code=$status"; fi; rm -f "$PID"; exit "$status"' EXIT
cd "$ROOT"
scripts/run_phase3_python.sh -u scripts/run_phase3_detector_train.py --project artifacts/phase3_detector_v2 >>"$LOG" 2>&1 & child=$!
watchdog & watch=$!
wait "$child"

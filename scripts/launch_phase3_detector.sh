#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash
set -euo pipefail
ROOT="/home/yuzy/ren"
RUN="$ROOT/artifacts/phase3_detector_v1"
LOCK="$RUN/job.lock"
PID="$RUN/pid.marker"
START="$RUN/start.marker"
SUCCESS="$RUN/success.marker"
FAILURE="$RUN/failure.marker"
LOG="$RUN/train.log"
mkdir -p "$RUN"
exec 9>"$LOCK"
flock -n 9 || { echo "already running" >&2; exit 73; }
if test -s "$PID"; then
  old=$(sed -n 's/^pid=//p' "$PID")
  if test -n "$old" && kill -0 "$old" 2>/dev/null; then echo "live matching job exists: $old" >&2; exit 74; fi
fi
rm -f "$SUCCESS" "$FAILURE"
atomic() { local target="$1"; shift; local tmp="${target}.tmp.$$"; printf '%s\n' "$*" > "$tmp"; mv -f "$tmp" "$target"; }
atomic "$PID" "pid=$$" "job=phase3-detector-yolo11n-20260909"
atomic "$START" "timestamp=$(date --iso-8601=seconds)" "command=scripts/run_phase3_python.sh -u scripts/run_phase3_detector_train.py --project artifacts/phase3_detector_v1" "model_sha256=43d8a7c86acc77282ddf4966d6526e091e5b064c140ed4a6e4c0b68ecbc3784c" "ultralytics=8.4.141" "role=train_only"
status=0
trap 'status=$?; if test "$status" -eq 0; then atomic "$SUCCESS" "timestamp=$(date --iso-8601=seconds)" "exit_code=0"; else atomic "$FAILURE" "timestamp=$(date --iso-8601=seconds)" "exit_code=$status"; fi; rm -f "$PID"; exit "$status"' EXIT
cd "$ROOT"
exec scripts/run_phase3_python.sh -u scripts/run_phase3_detector_train.py --project artifacts/phase3_detector_v1 >>"$LOG" 2>&1

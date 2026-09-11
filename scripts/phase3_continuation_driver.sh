#!/nix/store/0550j0i8bmzxbcnzrg1g51zigj7y12ih-bash-interactive-5.3p9/bin/bash
set -euo pipefail
ROOT=/home/yuzy/ren
RUN="$ROOT/artifacts/phase3_detector_v2"
DRIVER="$ROOT/artifacts/phase3_continuation_v1"
mkdir -p "$DRIVER"
exec 8>"$DRIVER/continuation.lock"
flock -n 8 || exit 73
atomic(){ local t="$1"; shift; local x="$t.tmp.$$"; printf '%s\n' "$*" > "$x"; mv -f "$x" "$t"; }
atomic "$DRIVER/start.marker" "timestamp=$(date --iso-8601=seconds)" "waits_for=$RUN/success.marker" "role=post-validation-only"
while :; do
  test -f "$RUN/failure.marker" && { atomic "$DRIVER/failure.marker" "timestamp=$(date --iso-8601=seconds)" "reason=detector training failed"; exit 1; }
  test -f "$RUN/success.marker" && break
  sleep 30
done
# Deliberately stop at the immutable handoff boundary. Downstream stages must
# be invoked only after an explicit validation/checkpoint/threshold freeze
# driver is present; never expose test data from this waiting unit.
atomic "$DRIVER/failure.marker" "timestamp=$(date --iso-8601=seconds)" "reason=downstream continuation driver not yet implemented; test exposure prevented"
exit 2

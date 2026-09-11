#!/nix/store/0550j0i8bmzxbcnzrg1g51zigj7y12ih-bash-interactive-5.3p9/bin/bash
set -euo pipefail
R=/home/yuzy/ren/artifacts/phase3_detector_v2/recovery_v1
mkdir -p "$R"
exec 9>"$R/recovery.lock"
flock -n 9 || exit 73
exec /home/yuzy/ren/scripts/run_phase3_python.sh -u scripts/phase3_recovery_real.py

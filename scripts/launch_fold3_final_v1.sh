#!/nix/store/90nk33c4fkyg4x4dfk5cykqiryf2nlqq-bash-interactive-5.3p15/bin/bash
set -u
RUN=/home/yuzy/ren/artifacts/phase2_final_protocol_fold3_final_v1
LOCK=/tmp/phase2_fold3_final_evaluation.lock
LOG="$RUN/prediction.log"
PID="$RUN/pid.marker"
START="$RUN/start.marker"
SUCCESS="$RUN/success.marker"
FAILURE="$RUN/failure.marker"
CMD=(/tmp/run_python.sh -u /home/yuzy/ren/scripts/run_final_protocol.py predict
  --run "$RUN"
  --manifest /home/yuzy/ren/artifacts/phase2_validity_filter_repair_v1/final_candidate_manifest.json
  --config /home/yuzy/ren/artifacts/phase2_validity_filter_repair_v1/final_execution_config.json
  --images "/home/yuzy/ren/data/tissue/fold3/Fold 3/images/fold3/images.npy"
  --embeddings /home/yuzy/ren/embeddings/path-foundation/fold3/embeddings.npy
  --segmentation /home/yuzy/ren/artifacts/phase2_development_run_v10_segmentation_only/candidate_detection.pt
  --validity /home/yuzy/ren/artifacts/phase2_validity_filter_repair_v1/fit.npz
  --standardizer /home/yuzy/ren/artifacts/phase2_path_foundation_context_classifier_v1/standardizer.npz
  --classifier /home/yuzy/ren/artifacts/phase2_path_foundation_context_classifier_v1/epoch_30.pt
  --role final_evaluation)
exec 9>"$LOCK"
if ! flock -n 9; then exit 73; fi
if [[ -s "$PID" ]]; then
  old=$(<"$PID")
  if kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then exit 74; fi
fi
tmp="$PID.tmp.$$"; printf '%s\n' "$$" >"$tmp" && mv -f "$tmp" "$PID"
tmp="$START.tmp.$$"; { date --iso-8601=seconds; printf 'command='; printf '%q ' "${CMD[@]}"; printf '\nlock=%s\n' "$LOCK"; } >"$tmp" && mv -f "$tmp" "$START"
rm -f "$SUCCESS" "$FAILURE"
set +e
"${CMD[@]}" >>"$LOG" 2>&1
rc=$?
set -e
now=$(date --iso-8601=seconds)
if (( rc == 0 )); then marker="$SUCCESS"; else marker="$FAILURE"; fi
tmp="$marker.tmp.$$"; printf 'timestamp=%s\nexit_code=%s\n' "$now" "$rc" >"$tmp" && mv -f "$tmp" "$marker"
rm -f "$PID"
exit "$rc"

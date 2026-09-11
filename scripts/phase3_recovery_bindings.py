import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root = Path('artifacts/phase3_detector_v2/recovery')
    root.mkdir(parents=True, exist_ok=True)
    model = 'runs/detect/artifacts/phase3_detector_v2/yolo11n_from_scratch/weights/best.pt'
    bindings = {'checkpoint_sha256': digest(model), 'freeze_sha256': digest('artifacts/phase3_detector_v2/detector_freeze.json'), 'exposure_sha256': digest('artifacts/phase3_detector_v2/test_exposed.marker'), 'report_sha256': digest('artifacts/phase3_detector_v2/detector_test_report.json'), 'evaluator': 'scripts/phase3_ap50.py', 'executor': 'scripts/phase3_recovery_executor.py', 'config': 'artifacts/phase3_protocol_v1/recovery_ap50_config.json', 'threshold': .65, 'capture_conf': .001, 'nms_iou': .70, 'max_det': 300, 'ultralytics': '8.4.141', 'seed': 20260909}
    tmp = root / 'bindings.tmp'; tmp.write_text(json.dumps(bindings, indent=2, sort_keys=True) + '\n'); tmp.replace(root / 'bindings.json')

if __name__ == '__main__': main()

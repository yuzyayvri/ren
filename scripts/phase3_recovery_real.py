"""Production adapter for the single authorized recovery capture and score."""
import json
import sys
from pathlib import Path

from scripts.phase3_recovery_executor import capture_production, score_committed

ROOT = Path('/home/yuzy/ren')
RUN = ROOT / 'artifacts/phase3_detector_v2/recovery'
MODEL = ROOT / 'runs/detect/artifacts/phase3_detector_v2/yolo11n_from_scratch/weights/best.pt'
MANIFEST = ROOT / 'artifacts/phase3_protocol_v1/source_manifest.json'

def main():
    if '--dry-run' in sys.argv:
        # Import-time validation only; never read protected test data in dry-run.
        print('phase3 recovery adapter ready')
        return
    bindings = json.loads((RUN / 'bindings.json').read_text())
    capture_production(RUN, MODEL, MANIFEST, bindings)
    data, _ = __import__('scripts.phase3_prediction_cache', fromlist=['read']).read(RUN)
    rows = []
    source = json.loads(MANIFEST.read_text())['txl_pbc']['splits']['test']
    for i, row in enumerate(source):
        from PIL import Image
        w, h = Image.open(row['image']).size
        truth=[]
        for line in (ROOT/'data/blood/txl-pbc/TXL-PBC/labels/test'/f"{Path(row['image']).stem}.txt").read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split()); truth.append((int(c),((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)))
        lo, hi = int(data['offsets'][i]), int(data['offsets'][i+1]); pred=[(int(c),tuple(b),float(s)) for b,c,s in zip(data['boxes'][lo:hi],data['classes'][lo:hi],data['confs'][lo:hi])]
        rows.append((row['id'], truth, pred))
    originals=json.loads((ROOT/'artifacts/phase3_detector_v2/detector_test_report.json').read_text())['test']
    score_committed(RUN, rows, originals, bindings)

if __name__ == '__main__': main()

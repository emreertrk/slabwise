import csv
import random
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import Pipeline

N = 40
SEED = 1


def main():
    p = Pipeline()
    pos = {k: i for i, k in enumerate(p.ids)}
    labels = []
    with open(ROOT / 'outputs' / 'jyesr_labels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['tcgdex_id'] in pos:
                labels.append(r)
    random.seed(SEED)
    sample = random.sample(labels, N)
    tmp = ROOT / 'outputs' / 'demo' / 'tmp_scan.jpg'
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shards = {}
    total = top1 = top5 = failed = 0
    for r in sample:
        sh = r['shard']
        if sh not in shards:
            shards[sh] = tarfile.open(ROOT / 'data' / 'jyesr' / ('data_shard_%s.tar' % sh))
        try:
            raw = shards[sh].extractfile('%s_front.jpg' % r['cert_id']).read()
        except KeyError:
            continue
        cv2.imwrite(str(tmp), cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))
        res = p.analyze(tmp)
        if 'hata' in res:
            failed += 1
            continue
        ids = [c['id'] for c in res['adaylar']]
        total += 1
        top1 += ids[0] == r['tcgdex_id']
        top5 += r['tcgdex_id'] in ids
    print('orneklem %d, tespit basarisiz %d' % (total, failed))
    print('top1 %d/%d = %.3f' % (top1, total, top1 / max(total, 1)))
    print('top5 %d/%d = %.3f' % (top5, total, top5 / max(total, 1)))


if __name__ == '__main__':
    main()

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import Pipeline
from src.common import config
from src.detection.infer import find_corners, warp_card

CACHE = config.OUTPUTS_DIR / 'eval_scans_cache.json'
JAPANESE = config.EVAL_SCANS_DIR / 'japonca.txt'
KEEP = 300
MODES = [('OCR yok', False, False), ('yalnizca ad', True, False), ('ad + numara', True, True)]


def main():
    files = sorted(config.EVAL_SCANS_DIR.glob('*.jpg'))
    if not files:
        raise SystemExit('outputs/eval_scans bos')
    japanese = set(JAPANESE.read_text(encoding='utf-8').split()) if JAPANESE.exists() else set()
    p = Pipeline()
    stats = {}
    cache = {}
    failed = 0
    for n, path in enumerate(files, 1):
        group, cert, true_id = path.stem.split('_', 2)
        lang = 'jp' if cert in japanese else 'en'
        img = cv2.imread(str(path))
        quad = find_corners(img, p.detector, p.dev)
        if quad is None:
            failed += 1
            continue
        card = warp_card(img, quad)
        sims = p.X @ p.embed(card)
        name_text = p.read_name(card)
        number_text = p.read_number(card)
        order = np.argsort(-sims)
        true_rank = next((int(r) for r, i in enumerate(order) if p.ids[i] == true_id), -1)
        cache[path.name] = {
            'true_id': true_id,
            'true_rank': true_rank,
            'name_text': name_text,
            'number_text': number_text,
            'top': [[p.ids[i], round(float(sims[i]), 4)] for i in order[:KEEP]],
        }
        for label, use_name, use_number in MODES:
            ranked = p.rank(sims, name_text if use_name else '', number_text if use_number else '')
            ids = [p.ids[i] for i, _, _ in ranked]
            for key in (group, lang, 'tum'):
                s = stats.setdefault((key, label), [0, 0, 0])
                s[0] += 1
                s[1] += ids[0] == true_id
                s[2] += true_id in ids
        if n % 10 == 0:
            print('%d/%d' % (n, len(files)), flush=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
    print('tespit basarisiz:', failed)
    print('ayrinti dosyasi:', CACHE)
    names = {'a': 'eski 40 ornek', 'b': 'yeni 60 ornek', 'tum': 'tum 100 ornek',
             'en': 'Ingilizce kartlar', 'jp': 'Japonca kartlar (indekste yok, etiketleri guvenilmez)'}
    for key in ('a', 'b', 'tum', 'en', 'jp'):
        print()
        print(names[key])
        for label, _, _ in MODES:
            total, top1, top5 = stats.get((key, label), [0, 0, 0])
            if total:
                print('  %-12s top1 %3d/%d = %.3f | top5 %3d/%d = %.3f' % (label, top1, total, top1 / total, top5, total, top5 / total))


if __name__ == '__main__':
    main()

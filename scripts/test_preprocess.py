import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import Pipeline
from src.detection.infer import find_corners, warp_card

CASES = [
    ('Palkia', str(ROOT / 'data/roboflow_clean/images/s1_pro2__253_jpg.rf.f35e00f2ec21cad8fd808e897036401b_25180d.jpg'), 'xyp-XY75'),
    ('Misty', str(ROOT / 'data/test_photos/90-misty\'s-psyduck.jpg'), 'gym2-90'),
    ('Charizard', str(ROOT / 'data/test_photos/maxresdefault.jpg'), 'A1-280'),
    ('Heracross', str(ROOT / 'data/test_photos/Stephychu_Illo_Header__23_.jpg'), 'sv01-002'),
]


def glare_fix(card):
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(card, mask, 5, cv2.INPAINT_TELEA)


def smooth(card):
    return cv2.bilateralFilter(card, 9, 75, 75)


VARIANTS = {
    'ham': lambda c: c,
    'duzgun': smooth,
    'parlama': glare_fix,
    'ikisi': lambda c: smooth(glare_fix(c)),
}


def rank_of(p, card, expected):
    q = p.embed(card)
    sims = p.X @ q
    order = np.argsort(-sims)
    top_ids = [p.ids[i] for i in order[:5]]
    rank = top_ids.index(expected) + 1 if expected in top_ids else None
    return rank, p.ids[order[0]], float(sims[order[0]])


def main():
    p = Pipeline()
    header = 'kart'.ljust(12) + ''.join(v.ljust(28) for v in VARIANTS)
    print(header)
    totals = {v: 0 for v in VARIANTS}
    for name, path, expected in CASES:
        img = cv2.imread(path)
        quad = find_corners(img, p.detector, p.dev)
        card = warp_card(img, quad)
        line = name.ljust(12)
        for v, fn in VARIANTS.items():
            rank, top1, s = rank_of(p, fn(card), expected)
            if rank is not None:
                totals[v] += 1
            line += ('sira %s, ilk %s %.2f' % (rank, top1, s)).ljust(28)
        print(line)
    print('top5 toplam:', totals)


if __name__ == '__main__':
    main()

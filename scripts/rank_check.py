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


def main():
    p = Pipeline()
    for name, path, expected in CASES:
        img = cv2.imread(path)
        quad = find_corners(img, p.detector, p.dev)
        card = warp_card(img, quad)
        sims = p.X @ p.embed(card)
        order = np.argsort(-sims)
        ids = [p.ids[i] for i in order]
        rank = ids.index(expected) + 1
        print(name, '| beklenen', expected, '| sira', rank, '| benzerlik %.3f' % float(sims[ids.index(expected)]))


if __name__ == '__main__':
    main()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.pipeline import Pipeline

CASES = [
    ('Palkia', str(ROOT / 'data/roboflow_clean/images/s1_pro2__253_jpg.rf.f35e00f2ec21cad8fd808e897036401b_25180d.jpg'), 'xyp-XY75'),
    ('Misty\'s Psyduck', str(ROOT / 'data/test_photos/90-misty\'s-psyduck.jpg'), 'gym2-90'),
    ('Charizard ex', str(ROOT / 'data/test_photos/maxresdefault.jpg'), 'A1-280'),
    ('Heracross', str(ROOT / 'data/test_photos/Stephychu_Illo_Header__23_.jpg'), 'sv01-002'),
]
GRADE = 8


def main():
    p = Pipeline()
    top1 = top5 = priced = 0
    for name, path, expected in CASES:
        r = p.analyze(path, grade=GRADE)
        if 'hata' in r:
            print(name, '| hata:', r['hata'])
            continue
        ids = [c['id'] for c in r['adaylar']]
        ok1 = ids[0] == expected
        ok5 = expected in ids
        top1 += ok1
        top5 += ok5
        f = r.get('fiyat')
        if f:
            priced += 1
            fiyat = '%.2f EUR (%.2f-%.2f) [%s]' % (f['estimate'], f['range_low'], f['range_high'], f['kaynak'])
        else:
            fiyat = 'yok'
        print(name, '| beklenen', expected, '| ilk', ids[0], 'top1', ok1, '| top5', ok5, '| fiyat', fiyat)
    n = len(CASES)
    print('---')
    print('top1 %d/%d, top5 %d/%d, fiyat uretilen %d/%d' % (top1, n, top5, n, priced, n))


if __name__ == '__main__':
    main()

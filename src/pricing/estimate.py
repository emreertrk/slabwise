import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import config

TABLE = config.MULTIPLIER_TABLE
PREMIUM_FLOOR = config.PREMIUM_FLOOR
PREMIUM_NOTE = 'Yuksek degerli kartlar icin kesin deger uzman degerlendirmesi gerektirir; bu aralik genel bir tahmindir.'
GRADE_TO_COND = {10: 'NM', 9: 'LP', 8: 'LP', 7: 'MP', 6: 'MP', 5: 'HP', 4: 'HP', 3: 'DMG', 2: 'DMG', 1: 'DMG'}
TIERS = [(0.0, 2.0, 'T1_<2'), (2.0, 5.0, 'T2_2-5'), (5.0, 15.0, 'T3_5-15'), (15.0, 100.0, 'T4_15-100'), (100.0, float('inf'), 'T5_100+')]

_cache = None


def load_table(path=TABLE):
    table = {}
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            table[(r['tier'], r['condition'])] = {
                'median': float(r['multiplier']),
                'p25': float(r['p25']),
                'p75': float(r['p75']),
                'n': int(r['n']),
            }
    return table


def get_table():
    global _cache
    if _cache is None:
        _cache = load_table()
    return _cache


def tier_for(nm_price):
    if nm_price <= 0:
        raise ValueError('fiyat pozitif olmali')
    for lo, hi, name in TIERS:
        if lo <= nm_price < hi:
            return name
    raise ValueError('kademe bulunamadi')


def estimate(nm_price, grade):
    if not 1 <= int(grade) <= 10:
        raise ValueError('not 1 ile 10 arasinda olmali')
    grade = int(grade)
    cond = GRADE_TO_COND[grade]
    tier = tier_for(nm_price)
    if cond == 'NM':
        m = {'median': 1.0, 'p25': 1.0, 'p75': 1.0, 'n': 0}
    else:
        m = get_table()[(tier, cond)]
    return {
        'reference_price': round(nm_price, 2),
        'grade': grade,
        'condition': cond,
        'tier': tier,
        'range_low': round(nm_price * m['p25'], 2),
        'estimate': round(nm_price * m['median'], 2),
        'range_high': round(nm_price * m['p75'], 2),
        'sample_size': m['n'],
        'note': PREMIUM_NOTE if nm_price >= PREMIUM_FLOOR else None,
    }

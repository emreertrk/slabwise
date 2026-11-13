import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pricing.estimate import estimate, tier_for


def test_tiers():
    assert tier_for(1.5) == 'T1_<2'
    assert tier_for(2) == 'T2_2-5'
    assert tier_for(15) == 'T4_15-100'
    assert tier_for(100) == 'T5_100+'


def test_nm_has_no_damage():
    r = estimate(50, 10)
    assert r['condition'] == 'NM'
    assert r['range_low'] == r['estimate'] == r['range_high'] == 50


def test_range_ordered():
    for g in range(1, 10):
        r = estimate(20, g)
        assert r['range_low'] <= r['estimate'] <= r['range_high'], g


def test_monotone_in_grade():
    prev = None
    for g in range(10, 0, -1):
        cur = estimate(20, g)['estimate']
        if prev is not None:
            assert cur <= prev + 1e-9, g
        prev = cur


def test_premium_note():
    assert estimate(150, 9)['note'] is not None
    assert estimate(50, 9)['note'] is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
    for p, g in [(1.5, 3), (50, 9), (150, 10), (150, 7)]:
        print(estimate(p, g))

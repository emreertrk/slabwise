import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_price_model import COLS, LEGACY_COLS, build_maps, encode, features, interval_bin

CARD = {
    'name': 'Charizard ex', 'localId': '199', 'rarity': 'Special illustration rare', 'category': 'Pokemon',
    'set': {'id': 'sv03.5', 'cardCount': {'official': 165, 'total': 207}}, 'illustrator': 'miki kudo',
    'dexId': [6], 'hp': 330, 'types': ['Fire'], 'stage': 'Stage2', 'retreat': 2,
    'attacks': [{'damage': '60+'}, {'damage': 200}], 'weaknesses': [{'type': 'Water'}],
    'variants': {'holo': True}, 'variants_detailed': [{'type': 'holo'}],
}


def test_number_features():
    f = features(CARD)
    assert f['num'] == 199 and f['official'] == 165
    assert f['over'] == 1
    assert abs(f['num_ratio'] - 199 / 165) < 1e-9
    assert f['max_damage'] == 200 and f['is_ex'] == 1


def test_encode_lengths():
    f = features(CARD)
    maps = build_maps([(f, 100.0), (features({**CARD, 'localId': '006', 'illustrator': None}), 2.0)])
    assert len(encode(f, maps)) == len(COLS)
    legacy = {c: maps[c] for c in ('set', 'rarity', 'stage', 'category', 'type')}
    assert len(encode(f, legacy)) == len(LEGACY_COLS)


def test_unknown_values_are_nan():
    f = features(CARD)
    maps = build_maps([(f, 100.0)])
    other = features({'name': 'X', 'localId': 'TG01', 'set': {'id': 'yok'}, 'illustrator': 'baska'})
    row = encode(other, maps)
    assert row[COLS.index('set')] != row[COLS.index('set')]
    assert row[COLS.index('illustrator_te')] != row[COLS.index('illustrator_te')]
    assert row[COLS.index('illustrator_n')] == 0.0
    assert other['lid_alpha'] == 1 and other['num'] == 1


def test_interval_bins():
    edges = [2.0, 5.0, 15.0, 100.0]
    assert [interval_bin(p, edges) for p in (0.5, 2.0, 14.9, 99.0, 250.0)] == [0, 1, 2, 3, 4]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)

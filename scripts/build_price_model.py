import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import config

CARDS = config.CARDS
SETS = config.SETS
OUT_MODEL = config.PRICE_MODEL
OUT_METRICS = config.PRICE_METRICS
CAT_COLS = ['set', 'rarity', 'stage', 'category', 'type']
LEGACY_NUM_COLS = ['hp', 'retreat', 'n_attacks', 'max_damage', 'dex', 'n_weak', 'name_len',
                   'holo', 'reverse', 'first_ed', 'normal', 'promo', 'is_ex', 'is_v', 'is_gx', 'n_variants']
NEW_NUM_COLS = ['set_order', 'num', 'official', 'num_ratio', 'over', 'lid_alpha', 'has_ability', 'n_resist',
                'evolves', 'n_dex', 'cameo', 'has_desc', 'amp', 'n_stamp', 'n_foil']
NUM_COLS = LEGACY_NUM_COLS + NEW_NUM_COLS
TE_KEYS = ['illustrator', 'dex']
TE_COLS = [k + '_te' for k in TE_KEYS]
FREQ_COLS = [k + '_n' for k in TE_KEYS]
LEGACY_COLS = CAT_COLS + LEGACY_NUM_COLS
COLS = CAT_COLS + NUM_COLS + TE_COLS + FREQ_COLS
SEED = 42
TE_SMOOTH = 5
FOLDS = 5
INTERVAL_LEVEL = 0.8
INTERVAL_EDGES = [2.0, 5.0, 15.0, 100.0]

_set_order = None


def set_order():
    global _set_order
    if _set_order is None:
        try:
            _set_order = {s['id']: i for i, s in enumerate(json.loads(SETS.read_text(encoding='utf-8')))}
        except OSError:
            _set_order = {}
    return _set_order


def ref_price(card):
    for v in card.get('variants_detailed', []) or []:
        cm = (v.get('pricing') or {}).get('cardmarket') or {}
        for key in ('trend', 'trend-holo', 'avg'):
            if cm.get(key) is not None and float(cm[key]) > 0:
                return float(cm[key])
    return None


def max_damage(card):
    best = 0
    for a in card.get('attacks') or []:
        digits = ''.join(ch for ch in str(a.get('damage') or '') if ch.isdigit())
        if digits:
            best = max(best, int(digits))
    return best


def features(card):
    name = card.get('name') or ''
    var = card.get('variants') or {}
    detailed = card.get('variants_detailed') or []
    types = card.get('types') or []
    dex = card.get('dexId') or []
    cset = card.get('set') or {}
    local_id = str(card.get('localId') or '')
    digits = re.search(r'\d+', local_id)
    num = int(digits.group()) if digits else None
    official = (cset.get('cardCount') or {}).get('official')
    return {
        'set': cset.get('id'),
        'rarity': card.get('rarity'),
        'stage': card.get('stage'),
        'category': card.get('category'),
        'type': types[0] if types else None,
        'hp': card.get('hp'),
        'retreat': card.get('retreat'),
        'n_attacks': len(card.get('attacks') or []),
        'max_damage': max_damage(card),
        'dex': dex[0] if dex else None,
        'n_weak': len(card.get('weaknesses') or []),
        'name_len': len(name),
        'holo': int(bool(var.get('holo'))),
        'reverse': int(bool(var.get('reverse'))),
        'first_ed': int(bool(var.get('firstEdition'))),
        'normal': int(bool(var.get('normal'))),
        'promo': int(bool(var.get('wPromo'))),
        'is_ex': int(name.lower().endswith(' ex')),
        'is_v': int(name.endswith(' V') or 'VMAX' in name or 'VSTAR' in name),
        'is_gx': int(name.endswith(' GX')),
        'n_variants': len(detailed),
        'set_order': set_order().get(cset.get('id')),
        'num': num,
        'official': official,
        'num_ratio': num / official if num and official else None,
        'over': int(bool(num and official and num > official)),
        'lid_alpha': int(not local_id.isdigit()),
        'has_ability': int(bool(card.get('abilities'))),
        'n_resist': len(card.get('resistances') or []),
        'evolves': int(bool(card.get('evolveFrom'))),
        'n_dex': len(dex),
        'cameo': len(card.get('cameoDexIds') or []),
        'has_desc': int(bool(card.get('description'))),
        'amp': int('&' in name),
        'n_stamp': sum(len(v.get('stamp') or []) for v in detailed),
        'n_foil': sum(1 for v in detailed if v.get('foil')),
        'illustrator': card.get('illustrator'),
    }


def load_rows():
    rows = []
    with open(CARDS, encoding='utf-8') as fh:
        for line in fh:
            card = json.loads(line)
            price = ref_price(card)
            if price is None:
                continue
            rows.append((features(card), price))
    return rows


def target_means(keys, y, prior):
    acc = defaultdict(lambda: [0.0, 0])
    for k, v in zip(keys, y):
        if k is not None:
            acc[str(k)][0] += v
            acc[str(k)][1] += 1
    return {k: (s + prior * TE_SMOOTH) / (n + TE_SMOOTH) for k, (s, n) in acc.items()}


def build_maps(rows):
    maps = {'_cols': COLS, 'te': {}, 'freq': {}}
    for col in CAT_COLS:
        values = sorted({str(f[col]) for f, _ in rows if f[col] is not None})
        maps[col] = {v: i + 1 for i, v in enumerate(values)}
    y = np.log(np.array([p for _, p in rows]))
    prior = float(np.mean(y))
    for key in TE_KEYS:
        keys = [f.get(key) for f, _ in rows]
        maps['te'][key] = target_means(keys, y, prior)
        maps['freq'][key] = dict(Counter(str(k) for k in keys if k is not None))
    return maps


def encode(feats, maps):
    out = []
    for col in maps.get('_cols', LEGACY_COLS):
        if col in CAT_COLS:
            val = feats.get(col)
            out.append(float(maps[col].get(str(val), np.nan)) if val is not None else np.nan)
        elif col in TE_COLS:
            val = feats.get(col[:-3])
            out.append(float(maps['te'][col[:-3]].get(str(val), np.nan)) if val is not None else np.nan)
        elif col in FREQ_COLS:
            val = feats.get(col[:-2])
            out.append(float(maps['freq'][col[:-2]].get(str(val), 0)) if val is not None else 0.0)
        else:
            val = feats.get(col)
            out.append(float(val) if val is not None else np.nan)
    return out


def fold_ids(n, k, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    ids = np.zeros(n, int)
    for pos, i in enumerate(order):
        ids[i] = pos % k
    return ids


def fit_model(rows):
    maps = build_maps(rows)
    X = np.array([encode(f, maps) for f, _ in rows])
    y = np.log(np.array([p for _, p in rows]))
    prior = float(np.mean(y))
    folds = fold_ids(len(rows), FOLDS, SEED)
    for key in TE_KEYS:
        col = COLS.index(key + '_te')
        keys = [f.get(key) for f, _ in rows]
        for k in range(FOLDS):
            inside = folds != k
            means = target_means([keys[i] for i in np.where(inside)[0]], y[inside], prior)
            for i in np.where(~inside)[0]:
                X[i, col] = means.get(str(keys[i]), np.nan) if keys[i] is not None else np.nan
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        categorical_features=[COLS.index(c) for c in CAT_COLS], random_state=SEED,
    )
    model.fit(X, y)
    return model, maps


def predict_log(model, maps, rows):
    return model.predict(np.array([encode(f, maps) for f, _ in rows]))


def interval_bin(price, edges):
    return int(np.searchsorted(edges, price, side='right'))


def build_interval(rows):
    y = np.log(np.array([p for _, p in rows]))
    folds = fold_ids(len(rows), FOLDS, SEED + 1)
    oof = np.zeros(len(rows))
    for k in range(FOLDS):
        inside = [rows[i] for i in np.where(folds != k)[0]]
        outside = np.where(folds == k)[0]
        model, maps = fit_model(inside)
        oof[outside] = predict_log(model, maps, [rows[i] for i in outside])
    resid = y - oof
    bins = np.array([interval_bin(p, INTERVAL_EDGES) for p in np.exp(oof)])
    alpha = (1 - INTERVAL_LEVEL) / 2
    q_low, q_high = [], []
    for b in range(len(INTERVAL_EDGES) + 1):
        r = resid[bins == b] if np.sum(bins == b) >= 30 else resid
        q_low.append(float(np.quantile(r, alpha)))
        q_high.append(float(np.quantile(r, 1 - alpha)))
    return {'level': INTERVAL_LEVEL, 'edges': INTERVAL_EDGES, 'q_low': q_low, 'q_high': q_high}


def interval_for(interval, price):
    b = interval_bin(price, interval['edges'])
    return interval['q_low'][b], interval['q_high'][b]


def median_abs_pct(pred, true):
    return float(np.median(np.abs(pred - true) / true))


def within_2x(pred, true):
    return float(np.mean((pred / true >= 0.5) & (pred / true <= 2)))


def main():
    rows = load_rows()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.8)
    train, val = rows[:cut], rows[cut:]
    model, maps = fit_model(train)
    interval = build_interval(train)
    y_va = np.log(np.array([p for _, p in val]))
    pred_va = predict_log(model, maps, val)
    true_va = np.exp(y_va)
    pred_eur = np.exp(pred_va)

    set_median = {}
    for f, p in train:
        set_median.setdefault(f['set'], []).append(p)
    global_median = float(np.median([p for _, p in train]))
    base_eur = np.array([float(np.median(set_median[f['set']])) if f['set'] in set_median else global_median for f, _ in val])

    resid = y_va - pred_va
    q25, q75 = float(np.quantile(resid, 0.25)), float(np.quantile(resid, 0.75))
    bounds = np.array([interval_for(interval, p) for p in pred_eur])
    covered = (resid >= bounds[:, 0]) & (resid <= bounds[:, 1])
    low = true_va < 100

    metrics = {
        'r2_log': round(float(r2_score(y_va, pred_va)), 4),
        'r2_eur': round(float(r2_score(true_va, pred_eur)), 4),
        'egitim_kart': len(train),
        'dogrulama_kart': len(val),
        'ozellik_sayisi': len(COLS),
        'model_mae_eur': round(float(np.mean(np.abs(pred_eur - true_va))), 3),
        'model_medyan_mutlak_yuzde_hata': round(median_abs_pct(pred_eur, true_va), 3),
        'model_2x_icinde_orani': round(within_2x(pred_eur, true_va), 3),
        'taban_set_medyani_mae_eur': round(float(np.mean(np.abs(base_eur - true_va))), 3),
        'taban_set_medyani_medyan_mutlak_yuzde_hata': round(median_abs_pct(base_eur, true_va), 3),
        'taban_set_medyani_2x_icinde_orani': round(within_2x(base_eur, true_va), 3),
        'alt100_kart': int(low.sum()),
        'alt100_medyan_mutlak_yuzde_hata': round(median_abs_pct(pred_eur[low], true_va[low]), 3),
        'alt100_2x_icinde_orani': round(within_2x(pred_eur[low], true_va[low]), 3),
        'ust100_kart': int((~low).sum()),
        'ust100_medyan_mutlak_yuzde_hata': round(median_abs_pct(pred_eur[~low], true_va[~low]), 3),
        'aralik_hedef_kapsama': INTERVAL_LEVEL,
        'aralik_gercek_kapsama': round(float(np.mean(covered)), 3),
        'aralik_gercek_kapsama_alt100': round(float(np.mean(covered[low])), 3),
        'aralik_log_q25': round(q25, 4),
        'aralik_log_q75': round(q75, 4),
    }

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'maps': maps, 'cat_cols': CAT_COLS, 'cols': COLS,
                 'resid_q': (q25, q75), 'interval': interval}, OUT_MODEL)
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print('aralik:', json.dumps(interval))
    print('model:', OUT_MODEL)


if __name__ == '__main__':
    main()

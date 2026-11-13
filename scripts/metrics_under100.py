import random

import joblib
import numpy as np
from sklearn.metrics import r2_score

from scripts.build_price_model import SEED, encode, load_rows, ROOT

LIMIT = 100


def main():
    bundle = joblib.load(ROOT / 'outputs' / 'models' / 'price_model.joblib')
    rows = load_rows()
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * 0.8)
    val = rows[cut:]
    X = np.array([encode(f, bundle['maps']) for f, _ in val])
    true_all = np.array([p for _, p in val])
    pred_all = np.exp(bundle['model'].predict(X))
    base_all = []
    train = rows[:cut]
    set_med = {}
    for f, p in train:
        set_med.setdefault(f['set'], []).append(p)
    g = float(np.median([p for _, p in train]))
    for f, _ in val:
        base_all.append(float(np.median(set_med[f['set']])) if f['set'] in set_med else g)
    base_all = np.array(base_all)

    m = true_all < LIMIT
    t, p, b = true_all[m], pred_all[m], base_all[m]
    ape = np.abs(p - t) / t
    ratio = p / t
    bape = np.abs(b - t) / t
    bratio = b / t
    n = len(t)
    print('100 EUR alti kart sayisi:', n, 'dogrulama toplami:', len(true_all), '(%.1f%%)' % (100 * n / len(true_all)))
    print('R2 (log):', round(float(r2_score(np.log(t), np.log(p))), 4))
    print('R2 (EUR):', round(float(r2_score(t, p)), 4))
    print('MAE (EUR):', round(float(np.mean(np.abs(p - t))), 3))
    print('RMSE (EUR):', round(float(np.sqrt(np.mean((p - t) ** 2))), 3))
    print('Medyan yuzde hata:', round(float(np.median(ape)), 3))
    print('Ortalama yuzde hata:', round(float(np.mean(ape)), 3))
    print('+-10%% icinde: %.3f' % np.mean(ape <= 0.10))
    print('+-25%% icinde: %.3f' % np.mean(ape <= 0.25))
    print('+-50%% icinde: %.3f' % np.mean(ape <= 0.50))
    print('2x icinde: %.3f' % np.mean((ratio >= 0.5) & (ratio <= 2)))
    print('Taban (set medyani) R2 (log):', round(float(r2_score(np.log(t), np.log(b))), 4))
    print('Taban medyan yuzde hata:', round(float(np.median(bape)), 3))
    print('Taban 2x icinde: %.3f' % np.mean((bratio >= 0.5) & (bratio <= 2)))
    q25, q75 = bundle['resid_q']
    lo = p * np.exp(q25)
    hi = p * np.exp(q75)
    print('Aralik (q25-q75) icinde gercek deger orani: %.3f' % np.mean((t >= lo) & (t <= hi)))


if __name__ == '__main__':
    main()

import random

import joblib
import numpy as np

from scripts.build_price_model import SEED, encode, load_rows, ROOT


def main():
    bundle = joblib.load(ROOT / 'outputs' / 'models' / 'price_model.joblib')
    rows = load_rows()
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * 0.8)
    val = rows[cut:]
    X = np.array([encode(f, bundle['maps']) for f, _ in val])
    true = np.array([p for _, p in val])
    pred = np.exp(bundle['model'].predict(X))
    m = true >= 100
    t, p = true[m], pred[m]
    ape = np.abs(p - t) / t
    print('100 EUR ustu kart sayisi:', int(m.sum()))
    print('MAPE (ortalama yuzde hata): %.3f' % np.mean(ape))
    print('Medyan yuzde hata: %.3f' % np.median(ape))
    print('MAE (EUR): %.2f' % np.mean(np.abs(p - t)))
    print('Ortalama yuzde hata >100 icin kucuk orneklem dengesiz: en buyuk hata %.3f' % np.max(ape))


if __name__ == '__main__':
    main()

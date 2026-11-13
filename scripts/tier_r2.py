import random

import joblib
import numpy as np
from sklearn.metrics import r2_score

from scripts.build_price_model import SEED, encode, load_rows, ROOT

BINS = [(0, 2, 'T1 <2'), (2, 5, 'T2 2-5'), (5, 15, 'T3 5-15'), (15, 100, 'T4 15-100'), (100, 1e9, 'T5 100+')]


def main():
    bundle = joblib.load(ROOT / 'outputs' / 'models' / 'price_model.joblib')
    rows = load_rows()
    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * 0.8)
    val = rows[cut:]
    X = np.array([encode(f, bundle['maps']) for f, _ in val])
    true = np.array([p for _, p in val])
    pred = np.exp(bundle['model'].predict(X))
    total_err = np.sum(np.abs(pred - true))
    print('kademe'.ljust(12), 'n'.rjust(5), 'R2_eur'.rjust(9), 'MAE'.rjust(8), 'medyanYuzde'.rjust(12), 'hataPayi'.rjust(9))
    for lo, hi, label in BINS:
        m = (true >= lo) & (true < hi)
        n = int(m.sum())
        if n < 2:
            print(label.ljust(12), str(n).rjust(5))
            continue
        r2 = r2_score(true[m], pred[m])
        mae = np.mean(np.abs(pred[m] - true[m]))
        mape = np.median(np.abs(pred[m] - true[m]) / true[m])
        share = np.sum(np.abs(pred[m] - true[m])) / total_err
        print(label.ljust(12), str(n).rjust(5), ('%.3f' % r2).rjust(9), ('%.2f' % mae).rjust(8), ('%.3f' % mape).rjust(12), ('%.1f%%' % (100 * share)).rjust(9))


if __name__ == '__main__':
    main()

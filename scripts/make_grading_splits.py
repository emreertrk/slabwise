import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict

src = 'outputs/index_all.csv'
dst = 'outputs/splits/grading_split.csv'
seed = sys.argv[1] if len(sys.argv) > 1 else 'capstone-v1'


def era(year):
    try:
        y = int(year)
    except ValueError:
        return 'unknown'
    if y < 2000:
        return '<2000'
    if y < 2010:
        return '2000-09'
    if y < 2020:
        return '2010-19'
    if y < 2023:
        return '2020-22'
    return '2023+'


def norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def lang(brand):
    return 'jp' if 'JAPANESE' in brand.upper() else 'en'


rows = [r for r in csv.DictReader(open(src, encoding='utf-8')) if r['grade']]
for r in rows:
    r['era'] = era(r['year'])
    r['lang'] = lang(r['brand'])
    r['key'] = '|'.join([norm(r['brand']), norm(r['card_number']), norm(r['subject'])])


def bucket(key):
    h = int(hashlib.sha1((seed + key).encode()).hexdigest(), 16) % 10000 / 10000
    return 'train' if h < 0.8 else 'val' if h < 0.9 else 'test'


low_keys = sorted(
    {r['key'] for r in rows if float(r['grade']) <= 4},
    key=lambda k: hashlib.sha1((seed + 'low' + k).encode()).hexdigest(),
)
low_split = {}
for i, k in enumerate(low_keys):
    q = i / len(low_keys)
    low_split[k] = 'train' if q < 0.8 else 'val' if q < 0.9 else 'test'

for r in rows:
    r['split'] = low_split.get(r['key']) or bucket(r['key'])

with open(dst, 'w', newline='', encoding='utf-8') as f:
    cols = ['shard', 'cert_id', 'has_back', 'grade', 'era', 'lang', 'key', 'split']
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)

print('kart', len(rows), 'benzersiz anahtar', len({r['key'] for r in rows}))
print('bolme', dict(Counter(r['split'] for r in rows)))
for s in ('train', 'val', 'test'):
    g = Counter(int(float(r['grade'])) for r in rows if r['split'] == s)
    print(s, 'not dagilimi', dict(sorted(g.items())))
for s in ('train', 'val', 'test'):
    print(s, 'donem', dict(Counter(r['era'] for r in rows if r['split'] == s)))

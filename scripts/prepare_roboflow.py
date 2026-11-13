import csv
import glob
import hashlib
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else 'data/roboflow'
out = sys.argv[2] if len(sys.argv) > 2 else 'data/roboflow_clean'
test_dir = 'pokemon-cards-merged.v4-card-obb.yolo26'
train_dir = 'PokemonCardDetectCard.v1i.yolo26'
ham_limit = 8

os.makedirs(os.path.join(out, 'images'), exist_ok=True)
os.makedirs(os.path.join(out, 'masks'), exist_ok=True)


def ahash(path):
    im = Image.open(path).convert('L').resize((16, 16))
    px = np.asarray(im, dtype=np.float32).ravel()
    bits = (px > px.mean()).astype(np.uint8)
    return int(''.join(map(str, bits)), 2)


def parse(label_path):
    polys, other = [], 0
    for line in open(label_path).read().splitlines():
        v = line.split()
        if not v:
            continue
        if len(v) >= 9 and (len(v) - 1) % 2 == 0:
            xy = np.array(v[1:], dtype=np.float32).reshape(-1, 2)
            polys.append(xy)
        else:
            other += 1
    return polys, other


def corners(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        a = cv2.approxPolyDP(c, eps * peri, True)
        if len(a) == 4:
            return a.reshape(-1, 2)
    return None


def collect(d):
    items = []
    for lp in sorted(glob.glob(os.path.join(root, d, '*', 'labels', '*.txt'))):
        sp = lp.split(os.sep)[-3]
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = None
        for ext in ('.jpg', '.jpeg', '.png'):
            c = os.path.join(root, d, sp, 'images', stem + ext)
            if os.path.exists(c):
                ip = c
                break
        if ip:
            items.append((d, sp, stem, ip, lp))
    return items


test_items = collect(test_dir)
train_items = collect(train_dir)
rows = []
drop = {'bos': 0, 'kutu_etiketi': 0, 'test_kopyasi': 0}

test_hashes = []
for d, sp, stem, ip, lp in test_items:
    test_hashes.append(ahash(ip))


def process(item, role):
    d, sp, stem, ip, lp = item
    polys, other = parse(lp)
    if not polys and other == 0:
        drop['bos'] += 1
        return
    if other:
        drop['kutu_etiketi'] += 1
        return
    h = ahash(ip)
    if role != 'test' and any(bin(h ^ t).count('1') <= ham_limit for t in test_hashes):
        drop['test_kopyasi'] += 1
        return
    im = cv2.imread(ip)
    H, W = im.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for xy in polys:
        pts = np.round(xy * [W, H]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    base = stem.split('.rf.')[0]
    name = f"{'s1' if role == 'test' else 's3'}_{stem[:60]}_{hashlib.md5(stem.encode()).hexdigest()[:6]}"
    shutil.copy(ip, os.path.join(out, 'images', name + '.jpg') if ip.endswith('.jpg') else os.path.join(out, 'images', name + os.path.splitext(ip)[1]))
    cv2.imwrite(os.path.join(out, 'masks', name + '.png'), mask)
    q = corners(mask) if len(polys) == 1 else None
    row = {'name': name, 'ext': os.path.splitext(ip)[1], 'source': d, 'orig_split': sp, 'group': base, 'role': role, 'n_cards': len(polys), 'width': W, 'height': H, 'ahash': h}
    for i in range(4):
        row[f'cx{i}'] = '' if q is None else int(q[i][0])
        row[f'cy{i}'] = '' if q is None else int(q[i][1])
    rows.append(row)


for it in test_items:
    process(it, 'test')

for it in train_items:
    process(it, 'train')

for r in rows:
    if r['role'] == 'train':
        h = int(hashlib.sha1(('rf' + r['group']).encode()).hexdigest(), 16) % 100
        r['role'] = 'val' if h < 15 else 'train'

cols = list(rows[0].keys())
with open(os.path.join(out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

from collections import Counter
print('atilan', drop)
print('rol', dict(Counter(r['role'] for r in rows)))
print('kart sayisi', dict(Counter((r['role'], min(r['n_cards'], 3)) for r in rows)))
print('4 kose cikarilan', sum(1 for r in rows if r['cx0'] != ''), 'toplam', len(rows))
print('benzersiz grup', len({r['group'] for r in rows if r['role'] != 'test'}), 'train grup', len({r['group'] for r in rows if r['role'] == 'train'}), 'val grup', len({r['group'] for r in rows if r['role'] == 'val'}))

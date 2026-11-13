import argparse
import csv
import glob
import io
import os
import random
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--src', required=True)
p.add_argument('--split_csv', required=True)
p.add_argument('--out', required=True)
p.add_argument('--width', type=int, default=512)
p.add_argument('--height', type=int, default=716)
p.add_argument('--cap_high', type=int, default=400)
p.add_argument('--margin', type=float, default=0.025)
p.add_argument('--seed', type=int, default=7)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--limit', type=int, default=0)
args = p.parse_args()

rows = list(csv.DictReader(open(args.split_csv, encoding='utf-8')))
rows = [r for r in rows if r['has_back'] == 'True' and r['era'] != 'unknown']
rng = random.Random(args.seed)

pools = defaultdict(list)
keep = []
for r in rows:
    g = int(float(r['grade']))
    if r['split'] == 'train' and g >= 9:
        pools[(g, r['era'])].append(r)
    else:
        keep.append(r)
for k, v in pools.items():
    rng.shuffle(v)
    keep.extend(v[: args.cap_high])
if args.limit:
    rng.shuffle(keep)
    keep = keep[: args.limit]

wanted_all = {r['cert_id']: r for r in keep}
os.makedirs(os.path.join(args.out, 'images'), exist_ok=True)


def save(data, cert, side):
    img = Image.open(io.BytesIO(data)).convert('RGB')
    mx, my = int(img.width * args.margin), int(img.height * args.margin)
    img = img.crop((mx, my, img.width - mx, img.height - my)).resize((args.width, args.height), Image.BICUBIC)
    img.save(os.path.join(args.out, 'images', f'{cert}_{side}.jpg'), quality=92)


def split_name(name):
    cert, _, side = os.path.basename(name).rpartition('_')
    return cert, side.split('.')[0]


def work_tar(path):
    done = []
    with tarfile.open(path) as t:
        for m in t:
            if not m.isfile():
                continue
            cert, side = split_name(m.name)
            if cert in wanted_all and side in ('front', 'back'):
                save(t.extractfile(m).read(), cert, side)
                done.append((cert, side))
    return path, done


def work_files(paths):
    done = []
    for path in paths:
        cert, side = split_name(path)
        with open(path, 'rb') as f:
            save(f.read(), cert, side)
        done.append((cert, side))
    return 'files', done


tars = sorted(glob.glob(os.path.join(args.src, '**', 'data_shard_*.tar'), recursive=True))
jobs = []
with ProcessPoolExecutor(args.workers) as ex:
    if tars:
        results = list(ex.map(work_tar, tars))
    else:
        files = []
        for d, _, fs in os.walk(args.src):
            for fn in fs:
                if fn.endswith(('_front.jpg', '_back.jpg')):
                    cert, side = split_name(fn)
                    if cert in wanted_all:
                        files.append(os.path.join(d, fn))
        chunks = [files[i::args.workers * 8] for i in range(args.workers * 8)]
        results = list(ex.map(work_files, chunks))

present = defaultdict(set)
for shard, done in results:
    for cert, side in done:
        present[cert].add(side)
    print('shard', shard, len(done), flush=True)

cols = ['cert_id', 'grade', 'era', 'lang', 'split', 'key']
with open(os.path.join(args.out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    n = 0
    for r in keep:
        if present[r['cert_id']] == {'front', 'back'}:
            w.writerow(r)
            n += 1
print('manifest satir', n)

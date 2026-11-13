import csv
import os
import re
import sys
import tarfile
from collections import Counter

root = sys.argv[1] if len(sys.argv) > 1 else "."
base = os.path.join(root, 'data', 'jyesr')
shards = sorted(f for f in os.listdir(base) if f.startswith('data_shard_') and f.endswith('.tar'))


def parse_grade(g):
    if not g or '(' in g or 'AUTHENTIC' in g.upper():
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*$', g.strip())
    return float(m.group(1)) if m else None


meta = {}
with open(os.path.join(base, 'metadata.csv'), newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        meta[r['cert_id']] = r

rows = []
for shard in shards:
    names = set()
    with tarfile.open(os.path.join(base, shard)) as t:
        for m in t:
            names.add(m.name)
    certs = sorted({n.split('_')[0] for n in names if n.endswith('_front.jpg')})
    for c in certs:
        r = meta.get(c)
        g = parse_grade(r['grade']) if r else None
        rows.append({
            'shard': shard.replace('data_shard_', '').replace('.tar', ''),
            'cert_id': c,
            'has_back': f'{c}_back.jpg' in names,
            'grade': '' if g is None else g,
            'grade_raw': r['grade'] if r else '',
            'name': r['card_name'] if r else '',
            'year': r['year'] if r else '',
            'brand': r['brand'] if r else '',
            'subject': r['subject'] if r else '',
            'card_number': r['card_number'] if r else '',
        })
    print(shard, len(certs), flush=True)

out = os.path.join(root, 'outputs', 'index_all.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

usable = [r for r in rows if r['grade'] != '' and r['has_back']]
print('toplam kart', len(rows), 'kullanilabilir', len(usable))
print('metadata disi', sum(1 for r in rows if not r['grade_raw']))
print('not dagilimi', sorted(Counter(int(r['grade']) for r in usable).items()))

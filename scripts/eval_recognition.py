import argparse
import csv
import json
import random
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'facebook/dinov2-base'


def load_index():
    X = np.load(ROOT / 'outputs' / 'index' / 'embeddings.npy') if (ROOT / 'outputs' / 'index' / 'embeddings.npy').exists() else np.load(ROOT / 'arsiv' / 'kaggle_ciktilari' / 'embed_out' / 'index' / 'embeddings.npy')
    ids = json.load(open(ROOT / 'outputs' / 'index' / 'ids.json') if (ROOT / 'outputs' / 'index' / 'ids.json').exists() else open(ROOT / 'arsiv' / 'kaggle_ciktilari' / 'embed_out' / 'index' / 'ids.json'))
    return X, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    X, ids = load_index()
    pos = {k: i for i, k in enumerate(ids)}

    labels = []
    with open(ROOT / 'outputs' / 'jyesr_labels.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['tcgdex_id'] in pos:
                labels.append(r)
    random.seed(args.seed)
    sample = random.sample(labels, min(args.n, len(labels)))

    shards = {}
    imgs = []
    for r in sample:
        sh = r['shard']
        if sh not in shards:
            shards[sh] = tarfile.open(ROOT / 'data' / 'jyesr' / f'data_shard_{sh}.tar')
        try:
            raw = shards[sh].extractfile(f"{r['cert_id']}_front.jpg").read()
        except KeyError:
            imgs.append(None)
            continue
        imgs.append(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR))

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    hit1 = hit5 = total = 0
    for r, im in zip(sample, imgs):
        if im is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            out = model(**proc(images=pil, return_tensors='pt'))
            v = out.pooler_output[0].numpy()
        v = v / np.linalg.norm(v)
        sims = X @ v
        top = np.argsort(-sims)[:5]
        top_ids = [ids[i] for i in top]
        total += 1
        if top_ids[0] == r['tcgdex_id']:
            hit1 += 1
        if r['tcgdex_id'] in top_ids:
            hit5 += 1
    print(json.dumps({'etiketli_havuz': len(labels), 'orneklem': total, 'top1': round(hit1 / max(total, 1), 3), 'top5': round(hit5 / max(total, 1), 3)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

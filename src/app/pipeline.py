import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_price_model import encode, features
from src.common import config
from src.detection.infer import find_corners, load_model, warp_card
from src.pricing.estimate import estimate
from src.recognition.textmatch import name_match, norm, number_level


def load_cards():
    meta = {}
    with open(config.CARDS, encoding='utf-8') as fh:
        for line in fh:
            c = json.loads(line)
            ref = None
            for v in c.get('variants_detailed', []):
                cm = (v.get('pricing') or {}).get('cardmarket') or {}
                for key in ('trend', 'trend-holo', 'avg'):
                    if cm.get(key) is not None:
                        ref = float(cm[key])
                        break
                if ref is not None:
                    break
            cset = c.get('set') or {}
            meta[c['id']] = {
                'name': c.get('name'),
                'set': cset.get('name'),
                'number': c.get('localId'),
                'official': (cset.get('cardCount') or {}).get('official'),
                'reference_price_eur': ref,
                'raw': c,
            }
    return meta


def load_reader():
    if os.environ.get('NO_OCR'):
        return None
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception:
        return None


def load_price_model():
    if not config.PRICE_MODEL.exists():
        return None
    return joblib.load(config.PRICE_MODEL)


def model_estimate(bundle, raw, grade):
    x = np.array([encode(features(raw), bundle['maps'])])
    base = float(np.exp(bundle['model'].predict(x)[0]))
    interval = bundle.get('interval')
    if interval:
        b = int(np.searchsorted(interval['edges'], base, side='right'))
        q_low, q_high = interval['q_low'][b], interval['q_high'][b]
        level = interval['level']
    else:
        q_low, q_high = bundle['resid_q']
        level = 0.5
    est = estimate(base, grade)
    est['range_low'] = round(est['range_low'] * math.exp(q_low), 2)
    est['range_high'] = round(est['range_high'] * math.exp(q_high), 2)
    est['range_level'] = level
    est['reference_price'] = round(base, 2)
    if base >= config.PREMIUM_FLOOR:
        est['uyari'] = 'Tahmin guvenilir degil: 100 EUR ustu kartlarda model dogrulugu dusuk'
    return est


class Pipeline:
    def __init__(self):
        self.detector, self.dev = load_model()
        self.proc = AutoImageProcessor.from_pretrained(config.EMBED_MODEL)
        self.embedder = AutoModel.from_pretrained(config.EMBED_MODEL).eval().to(self.dev)
        self.X = np.load(config.INDEX_DIR / 'embeddings.npy')
        self.ids = json.load(open(config.INDEX_DIR / 'ids.json', encoding='utf-8'))
        self.meta = load_cards()
        self.model = load_price_model()
        self.reader = load_reader()

    def read_band(self, card, band):
        if self.reader is None:
            return ''
        y0, y1, x0, x1 = band
        crop = cv2.resize(card[y0:y1, x0:x1], None, fx=config.OCR_SCALE, fy=config.OCR_SCALE, interpolation=cv2.INTER_CUBIC)
        return ' '.join(self.reader.readtext(crop, detail=0))

    def read_name(self, card):
        return norm(self.read_band(card, config.NAME_BAND))

    def read_number(self, card):
        return self.read_band(card, config.NUMBER_BAND)

    def embed(self, card_bgr):
        pil = Image.fromarray(cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=pil, return_tensors='pt').to(self.dev)
        with torch.no_grad():
            v = self.embedder(**inp).pooler_output[0].float().cpu().numpy()
        return v / np.linalg.norm(v)

    def rank(self, sims, name_text='', number_text='', k=config.TOP_K, pool_size=config.POOL_SIZE):
        pool = np.argsort(-sims)[:pool_size]
        scored = []
        for i in pool:
            m = self.meta.get(self.ids[i], {})
            named = name_match(m.get('name'), name_text)
            level = number_level(m.get('number'), m.get('official'), number_text)
            score = float(sims[i]) + config.NAME_BONUS * named + config.NUMBER_BONUS[level]
            scored.append((-score, int(i), named, level))
        scored.sort()
        return [(i, named, level) for _, i, named, level in scored[:k]]

    def analyze(self, img_path, grade=None, choice=0, k=config.TOP_K):
        img = cv2.imread(str(img_path))
        if img is None:
            return {'hata': 'gorsel okunamadi'}
        quad = find_corners(img, self.detector, self.dev)
        if quad is None:
            return {'hata': 'kart bulunamadi'}
        card = warp_card(img, quad)
        sims = self.X @ self.embed(card)
        name_text = self.read_name(card)
        number_text = self.read_number(card)
        candidates = []
        for i, named, level in self.rank(sims, name_text, number_text, k=k):
            cid = self.ids[i]
            m = self.meta.get(cid, {})
            candidates.append({
                'id': cid,
                'name': m.get('name'),
                'set': m.get('set'),
                'number': m.get('number'),
                'benzerlik': round(float(sims[i]), 3),
                'referans_fiyat_eur': m.get('reference_price_eur'),
                'ad_eslesti': bool(named),
                'numara_eslesti': level > 0,
            })
        result = {
            'kose_noktalari': np.round(quad, 1).tolist(),
            'adaylar': candidates,
            'kirpilmis': card,
            'okunan_ad': name_text,
            'okunan_numara': number_text,
        }
        if candidates and grade is not None:
            result['fiyat'] = self.price_for(candidates[choice]['id'], grade)
        return result

    def price_for(self, cid, grade):
        m = self.meta.get(cid, {})
        ref = m.get('reference_price_eur')
        if ref is not None and ref > 0:
            f = estimate(ref, grade)
            f['para_birimi'] = 'EUR'
            f['kaynak'] = 'kart'
            return f
        if self.model is not None and cid in self.meta:
            f = model_estimate(self.model, m['raw'], grade)
            f['para_birimi'] = 'EUR'
            f['kaynak'] = 'model'
            return f
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--grade', type=int, default=None)
    ap.add_argument('--choice', type=int, default=0)
    args = ap.parse_args()
    p = Pipeline()
    r = p.analyze(args.image, grade=args.grade, choice=args.choice)
    if 'hata' in r:
        raise SystemExit(r['hata'])
    out = config.DEMO_DIR / 'pipeline_card.jpg'
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), r.pop('kirpilmis'))
    print('kose noktalari:', r['kose_noktalari'])
    print('okunan ad (OCR):', r.get('okunan_ad') or '-')
    print('okunan alt bant (OCR):', r.get('okunan_numara') or '-')
    for i, c in enumerate(r['adaylar'], 1):
        tags = ('ad' if c['ad_eslesti'] else '-') + '/' + ('no' if c['numara_eslesti'] else '-')
        print(i, c['name'], '|', c['set'], '|', c['number'], '| benzerlik', c['benzerlik'], '| ref EUR', c['referans_fiyat_eur'], '| eslesme', tags)
    f = r.get('fiyat')
    if f:
        print()
        print('--- Fiyat (derece %s, %s) ---' % (f['grade'], f['condition']))
        print('Referans fiyat : %.2f EUR' % f['reference_price'])
        print('Tahmini fiyat  : %.2f EUR' % f['estimate'])
        print('Aralik         : %.2f - %.2f EUR' % (f['range_low'], f['range_high']))
        if f['kaynak'] == 'model':
            print('Kaynak         : model tahmini (kartin referans fiyati yok)')
            if f.get('uyari'):
                print('UYARI          : ' + f['uyari'])
        else:
            print('Kaynak         : kartin kendi referans fiyati')
    elif 'fiyat' in r:
        print()
        print('Fiyat hesaplanamadi.')
    print('kirpilmis kart:', out)


if __name__ == '__main__':
    main()

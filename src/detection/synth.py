import argparse
import csv
import glob
import json
import os
import random

import cv2
import numpy as np

CARD_W, CARD_H = 600, 800


def load_cards(tcgdex_dir, limit):
    paths = sorted(glob.glob(os.path.join(tcgdex_dir, 'images', '*.webp')))
    random.shuffle(paths)
    return paths[:limit] if limit else paths


def load_backgrounds(bg_dirs):
    out = []
    for d in bg_dirs:
        for ext in ('jpg', 'jpeg', 'png'):
            out += glob.glob(os.path.join(d, '**', f'*.{ext}'), recursive=True)
    return out


def random_background(paths, size):
    h, w = size
    if paths and random.random() < 0.9:
        im = cv2.imread(random.choice(paths))
        if im is not None:
            ih, iw = im.shape[:2]
            s = max(h / ih, w / iw) * random.uniform(1.0, 1.6)
            im = cv2.resize(im, (int(iw * s) + 1, int(ih * s) + 1))
            y = random.randint(0, im.shape[0] - h)
            x = random.randint(0, im.shape[1] - w)
            return im[y:y + h, x:x + w].copy()
    base = np.full((h, w, 3), [random.randint(20, 235) for _ in range(3)], np.uint8)
    noise = np.random.randint(-12, 12, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def random_quad(size, card_aspect):
    h, w = size
    scale = random.uniform(0.35, 0.9)
    ch = h * scale
    cw = ch * card_aspect
    if cw > w * 0.95:
        cw = w * 0.95
        ch = cw / card_aspect
    cx = random.uniform(cw / 2, w - cw / 2)
    cy = random.uniform(ch / 2, h - ch / 2)
    pts = np.array([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]], np.float32)
    a = np.deg2rad(random.uniform(-35, 35) if random.random() < 0.8 else random.uniform(-180, 180))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    pts = pts @ R.T
    tilt = random.uniform(0.0, 0.18)
    pts[:, 0] *= 1 + tilt * np.array([-1, 1, 1, -1]) * random.choice([-1, 1]) * pts[:, 1] / max(ch, 1)
    pts += np.array([cx, cy], np.float32)
    pts += np.random.uniform(-0.02, 0.02, pts.shape).astype(np.float32) * [cw, ch]
    return pts.astype(np.float32)


def add_glare(card, strength):
    h, w = card.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = random.uniform(0, np.pi)
    d = (xx * np.cos(ang) + yy * np.sin(ang)) / max(h, w)
    center = random.uniform(0.2, 0.8)
    band = np.exp(-((d - center) ** 2) / random.uniform(0.002, 0.02))
    glare = band[..., None] * strength * 255
    return np.clip(card.astype(np.float32) + glare, 0, 255).astype(np.uint8)


def compose(card_path, bg_paths, size):
    h, w = size
    card = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
    if card is None:
        return None
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)
    alpha = None
    if card.shape[2] == 4:
        alpha = card[..., 3]
        card = card[..., :3]
    card = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    alpha = np.full((CARD_H, CARD_W), 255, np.uint8) if alpha is None else cv2.resize(alpha, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        card = add_glare(card, random.uniform(0.15, 0.6))
    if random.random() < 0.3:
        card = cv2.copyMakeBorder(card, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(235, 235, 235))
        alpha = cv2.copyMakeBorder(alpha, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    ch, cw = card.shape[:2]
    bg = random_background(bg_paths, (h, w))
    quad = random_quad((h, w), cw / ch)
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(card, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(alpha, M, (w, h), flags=cv2.INTER_LINEAR)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    shadow = cv2.GaussianBlur(mask, (0, 0), random.uniform(6, 20))
    off = (random.randint(-12, 12), random.randint(6, 22))
    shadow = np.roll(shadow, off, axis=(1, 0)).astype(np.float32) / 255 * random.uniform(0.2, 0.6)
    img = bg.astype(np.float32) * (1 - shadow[..., None])
    soft = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32)[..., None] / 255
    img = img * (1 - soft) + warped.astype(np.float32) * soft
    img = np.clip(img * random.uniform(0.7, 1.2) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < 0.5:
        img = np.clip(img.astype(np.int16) + np.random.randint(-8, 8, img.shape), 0, 255).astype(np.uint8)
    q = (cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 95))
    ok, enc = cv2.imencode('.jpg', img, q)
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, mask, quad


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tcgdex', required=True)
    p.add_argument('--bg', nargs='*', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--seed', type=int, default=11)
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(os.path.join(a.out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'masks'), exist_ok=True)
    cards = load_cards(a.tcgdex, 0)
    bgs = load_backgrounds(a.bg)
    rows = []
    i = 0
    while len(rows) < a.n:
        r = compose(cards[i % len(cards)], bgs, (a.height, a.width))
        i += 1
        if r is None:
            continue
        img, mask, quad = r
        if mask.sum() / 255 < 0.08 * a.height * a.width:
            continue
        name = f'syn_{len(rows):06d}'
        cv2.imwrite(os.path.join(a.out, 'images', name + '.jpg'), img)
        cv2.imwrite(os.path.join(a.out, 'masks', name + '.png'), mask)
        row = {'name': name, 'card': os.path.basename(cards[(i - 1) % len(cards)])}
        for k in range(4):
            row[f'cx{k}'] = round(float(quad[k][0]), 1)
            row[f'cy{k}'] = round(float(quad[k][1]), 1)
        rows.append(row)
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('uretilen', len(rows), 'arka plan', len(bgs))


if __name__ == '__main__':
    main()

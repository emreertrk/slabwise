import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import config
from src.detection.train import MEAN, SIZE, STD, UNet

MODEL = config.DETECTOR_MODEL
CARD_W, CARD_H = config.CARD_W, config.CARD_H


def load_model(path=MODEL):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = UNet(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval().to(dev), dev


def order_points(pts):
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def predict_prob(model, dev, img_bgr):
    im = cv2.resize(img_bgr[:, :, ::-1], (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = (im / 255.0 - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()


def find_corners(img_bgr, model, dev):
    h, w = img_bgr.shape[:2]
    prob = predict_prob(model, dev, img_bgr)
    mask = ((prob > 0.5) * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(ap) == 4:
        quad = ap.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    quad = quad * np.array([w / SIZE, h / SIZE], np.float32)
    return order_points(quad)


def warp_card(img_bgr, quad, out_w=CARD_W, out_h=CARD_H):
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', default=str(config.DEMO_DIR / 'card_warp.jpg'))
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit('gorsel okunamadi')
    model, dev = load_model()
    quad = find_corners(img, model, dev)
    if quad is None:
        raise SystemExit('kart bulunamadi')
    print('kose noktalari (TL, TR, BR, BL):', np.round(quad, 1).tolist())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, warp_card(img, quad))
    print('kaydedildi:', args.out)


if __name__ == '__main__':
    main()

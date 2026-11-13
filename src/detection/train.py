import argparse
import csv
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_items(root, split):
    items = []
    with open(os.path.join(root, 'manifest.csv'), encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            role = r.get('role') or 'train'
            if split is not None and role != split:
                continue
            name = r['name']
            ext = r.get('ext') or '.jpg'
            img = os.path.join(root, 'images', name + ext)
            msk = os.path.join(root, 'masks', name + '.png')
            if os.path.exists(img) and os.path.exists(msk):
                items.append((img, msk))
    return items


class CardMasks(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        im = cv2.imread(ip)[:, :, ::-1]
        mk = cv2.imread(mp, 0)
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        mk = cv2.resize(mk, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                im = im[:, ::-1]
                mk = mk[:, ::-1]
            im = np.clip(im * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
        x = (im / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        y = torch.from_numpy(np.ascontiguousarray((mk > 127).astype(np.float32))).unsqueeze(0)
        return x, y


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def up(a, ref):
    return F.interpolate(a, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d4 = block(512 + 256, 256)
        self.d3 = block(256 + 128, 128)
        self.d2 = block(128 + 64, 64)
        self.d1 = block(64 + 64, 64)
        self.d0 = block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = self.l0(x)
        x1 = self.l1(self.pool(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        y = self.d4(torch.cat([up(x4, x3), x3], 1))
        y = self.d3(torch.cat([up(y, x2), x2], 1))
        y = self.d2(torch.cat([up(y, x1), x1], 1))
        y = self.d1(torch.cat([up(y, x0), x0], 1))
        y = self.d0(up(y, x))
        return self.out(y)


def build():
    return UNet()


def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    inter = (p * y).sum(dim=(1, 2, 3))
    total = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (total + 1)
    return bce + dice.mean()


def iou(model, loader, dev):
    model.eval()
    vals = []
    with torch.no_grad():
        for x, y in loader:
            p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
            inter = (p * y).sum(dim=(1, 2, 3))
            union = ((p + y) > 0).float().sum(dim=(1, 2, 3))
            vals += (inter / union.clamp(min=1)).tolist()
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True)
    ap.add_argument('--real', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count() if dev == 'cuda' else 0
    train_items = read_items(args.synth, None) + read_items(args.real, 'train')
    val_items = read_items(args.real, 'val')
    test_items = read_items(args.real, 'test')
    print('egitim', len(train_items), 'dogrulama', len(val_items), 'test', len(test_items), 'gpu', n_gpu, flush=True)

    tl = DataLoader(CardMasks(train_items, True), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(CardMasks(val_items, False), batch_size=args.batch, num_workers=args.workers)
    sl = DataLoader(CardMasks(test_items, False), batch_size=args.batch, num_workers=args.workers)

    base = build().to(dev)
    model = nn.DataParallel(base) if n_gpu > 1 else base
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))

    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = iou(model, vl, dev)
        rec = {'epoch': ep + 1, 'loss': float(np.mean(losses)), 'val_iou': val, 'sec': round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if val > best:
            best = val
            torch.save(base.state_dict(), os.path.join(args.out, 'best.pt'))

    base.load_state_dict(torch.load(os.path.join(args.out, 'best.pt'), map_location=dev))
    test = iou(model, sl, dev)
    result = {'best_val_iou': best, 'test_iou': test, 'n_train': len(train_items), 'n_val': len(val_items), 'n_test': len(test_items), 'gpu': n_gpu}
    json.dump({'log': log, 'result': result}, open(os.path.join(args.out, 'log.json'), 'w'), indent=2)
    print('SONUC', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

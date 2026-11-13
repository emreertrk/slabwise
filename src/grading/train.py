import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.grading.data import CardDataset, balanced_sampler, load_manifest
from src.grading.metrics import report
from src.grading.model import TwoViewGrader, grade_targets, logits_to_grade

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--backbone', default='resnet34')
p.add_argument('--epochs', type=int, default=12)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--height', type=int, default=716)
p.add_argument('--width', type=int, default=512)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--no_pretrained', action='store_true')
args = p.parse_args()

torch.manual_seed(7)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
df = load_manifest(args.data)
tr, va = df[df.split == 'train'], df[df.split == 'val']
if args.limit:
    tr, va = tr.sample(min(args.limit, len(tr)), random_state=7), va.sample(min(args.limit, len(va)), random_state=7)
size = (args.height, args.width)
tl = DataLoader(CardDataset(args.data, tr, True, size), batch_size=args.batch, sampler=balanced_sampler(tr), num_workers=args.workers, drop_last=True)
vl = DataLoader(CardDataset(args.data, va, False, size), batch_size=args.batch, num_workers=args.workers)

model = TwoViewGrader(args.backbone, pretrained=not args.no_pretrained).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl))
scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
os.makedirs(args.out, exist_ok=True)
best = -1.0
log = []


def evaluate():
    model.eval()
    ys, ps, es = [], [], []
    with torch.no_grad():
        for f, b, g, e in vl:
            with torch.autocast(dev, enabled=dev == 'cuda'):
                lg = model(f.to(dev), b.to(dev))
            ps += logits_to_grade(lg).cpu().tolist()
            ys += g.tolist()
            es += list(e)
    return report(np.array(ys), np.array(ps), es)


for ep in range(args.epochs):
    model.train()
    t0 = time.time()
    total = 0.0
    for f, b, g, _ in tl:
        f, b, g = f.to(dev), b.to(dev), g.to(dev)
        with torch.autocast(dev, enabled=dev == 'cuda'):
            lg = model(f, b)
        loss = F.binary_cross_entropy_with_logits(lg.float(), grade_targets(g))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
    m = evaluate()
    m['epoch'] = ep + 1
    m['train_loss'] = total / len(tl)
    m['seconds'] = round(time.time() - t0)
    log.append(m)
    print(json.dumps({k: v for k, v in m.items() if k != 'per_era'}), flush=True)
    score = m['qwk'] + m['within1'] - m['balanced_mae'] / 5
    if score > best:
        best = score
        torch.save(model.state_dict(), os.path.join(args.out, 'best.pt'))
    json.dump(log, open(os.path.join(args.out, 'log.json'), 'w'), indent=1)

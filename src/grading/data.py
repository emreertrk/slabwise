import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_transform(train, size):
    h, w = size
    if train:
        return T.Compose([
            T.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.97, 1.03)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(MEAN, STD)])


class CardDataset(Dataset):
    def __init__(self, root, df, train, size):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.tf = make_transform(train, size)

    def __len__(self):
        return len(self.df)

    def _load(self, cert, side):
        path = os.path.join(self.root, 'images', f'{cert}_{side}.jpg')
        return self.tf(Image.open(path).convert('RGB'))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cert = str(r['cert_id'])
        return self._load(cert, 'front'), self._load(cert, 'back'), int(round(float(r['grade']))), r['era']


def load_manifest(root):
    df = pd.read_csv(os.path.join(root, 'manifest.csv'), dtype={'cert_id': str})
    df['grade'] = df['grade'].astype(float).round().astype(int)
    return df


def balanced_sampler(df, num_samples=None):
    cell = df.groupby(['grade', 'era']).size()
    w = df.apply(lambda r: 1.0 / cell[(r['grade'], r['era'])], axis=1).values
    grade_total = df.groupby('grade')['grade'].transform('size').values
    w = w * np.sqrt(grade_total)
    w = w / w.sum()
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples or len(df), replacement=True)

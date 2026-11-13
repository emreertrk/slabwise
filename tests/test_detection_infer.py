import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.infer import CARD_H, CARD_W, order_points, warp_card


def test_order_points_shuffled():
    pts = np.array([[100, 300], [20, 20], [300, 320], [290, 15]], np.float32)
    tl, tr, br, bl = order_points(pts)
    assert tl.tolist() == [20, 20]
    assert tr.tolist() == [290, 15]
    assert br.tolist() == [300, 320]
    assert bl.tolist() == [100, 300]


def test_warp_shape():
    img = np.zeros((400, 300, 3), np.uint8)
    quad = np.array([[50, 40], [250, 40], [250, 360], [50, 360]], np.float32)
    out = warp_card(img, quad)
    assert out.shape == (CARD_H, CARD_W, 3)


if __name__ == '__main__':
    test_order_points_shuffled()
    test_warp_shape()
    print('ok')

"""
Current idea is to slide a tile (some ratio of the original image size)
around possible target images and compare the line drawings of the current img
and target img.

Using hausdorff distance would only work with binary line drawings, but i think i want to
have somesort of weighting, whereby prominent edges being similar to prominent edges
is more valuable(similar)

"""

import os
from PIL import Image
import cv2 as cv

tile_ratio = 0.5
stride = 10


def find_next_img(cur_img_tile, remaining_set_paths):
    img_scores = {}
    h, w = cur_img_tile

    for fp in remaining_set_paths:
        test_im = cv.imread(fp, cv.IMREAD_COLOR_BGR)
        m_h, m_w = test_im.size
        for i in range((m_h - h) // stride):
            for j in range((m_w - w) // stride):
                test_im_tile = test_im[
                    i * stride : i * stride + h, j * stride : j * stride + w
                ]

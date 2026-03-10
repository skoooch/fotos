from functools import reduce
from hashlib import file_digest
import os
from PIL import Image
import cv2 as cv


def create_reduced_dir(in_folder, out_folder, scale=5):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename)
        image = Image.open(in_fp)
        size = image.size
        image = image.resize((size[0] // scale, size[1] // scale))
        image.save(out_fp, quality=20, optimize=True)


def get_sobel_img(filepath, out_fp=None):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S

    src = cv.imread(filepath, cv.IMREAD_COLOR)
    src = cv.GaussianBlur(src, (3, 3), 0)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(
        gray,
        ddepth,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv.BORDER_DEFAULT,
    )

    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(
        gray,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv.BORDER_DEFAULT,
    )

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = ((cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) / 255) ** 2) * 255
    if out_fp == None:
        return grad
    else:
        cv.imwrite(out_fp, grad)


def get_canny_img(filepath, out_fp=None):
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 300, 300)
    print(edges.shape)
    exit()
    if out_fp == None:
        return edges
    else:
        cv.imwrite(out_fp, edges)


def create_sobel_folder(in_folder):
    out_folder = in_folder + "_sobel"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename)
        get_sobel_img(in_fp, out_fp)


def create_canny_folder(in_folder):
    out_folder = in_folder + "_canny"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename)
        get_canny_img(in_fp, out_fp)


if __name__ == "__main__":
    # reduce_scale("cusco-salkantay", "cusco_salkantay_small")
    # create_sobel_folder("cusco_salkantay_small")
    create_canny_folder("cusco_salkantay_small")

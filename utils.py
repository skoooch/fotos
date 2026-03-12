import os
from PIL import Image
import cv2 as cv
import numpy as np

import svgwrite


def create_reduced_dir(in_folder, out_folder, scale=2.5):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename)
        image = Image.open(in_fp)
        size = image.size
        image = image.resize((int(size[0] // scale), int(size[1] // scale)))
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


def get_structured_edge(edge_detector, filepath, out_fp=None):
    image = cv.imread(filepath)
    # keep a copy of the original image
    orig_image = image.copy()
    # convert to RGB image and convert to float32
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # type: ignore
    image = image.astype(np.float32) / 255.0
    # grayscale and blurring for canny edge detection
    gray = cv.cvtColor(orig_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    edges = edge_detector.detectEdges(image)
    orientation_map = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orientation_map)
    if out_fp == None:
        return edges
    else:
        cv.imwrite(out_fp, edges * 255.0)


def vectorize_contours(img, output_svg_path, epsilon_factor=0.005, threshold=0.1):
    """Vectorize edge image using OpenCV contour detection."""
    # Handle float32 structured edge output (range 0-1)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    # Ensure single channel
    if len(img_uint8.shape) == 3:
        img_uint8 = cv.cvtColor(img_uint8, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(img_uint8, int(threshold * 255), 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    h, w = img_uint8.shape[:2]
    dwg = svgwrite.Drawing(output_svg_path, size=(w, h))

    for contour in contours:
        if len(contour) < 3:
            continue
        # Approximate contour with fewer points (Douglas-Peucker)
        epsilon = epsilon_factor * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        dwg.add(dwg.polyline(points, stroke="black", fill="none", stroke_width=1))

    dwg.save()
    return contours


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


def create_structured_edge_folder(in_folder):
    edge_detector = cv.ximgproc.createStructuredEdgeDetection("model.yml/model.yml")
    out_folder = in_folder + "_structured"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename)
        get_structured_edge(edge_detector, in_fp, out_fp)


def create_vectorized_edge_folder(in_folder):
    edge_detector = cv.ximgproc.createStructuredEdgeDetection("model.yml/model.yml")
    out_folder = in_folder + "_vec"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for filename in os.listdir(in_folder):
        in_fp = os.path.join(in_folder, filename)
        out_fp = os.path.join(out_folder, filename.replace("JPG", "svg"))
        edges = get_structured_edge(edge_detector, in_fp)
        contours = vectorize_contours(
            edges,
            out_fp,
        )


if __name__ == "__main__":
    # create_reduced_dir("cusco_salkantay", "cusco_salkantay_med", scale=2.5)
    # create_sobel_folder("cusco_salkantay_small")
    create_vectorized_edge_folder("cusco_salkantay_med")

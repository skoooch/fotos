import os
from PIL import Image
import cv2 as cv
import numpy as np
from MLVcode.load_mat import load_mat
import svgwrite
from MLVcode.computeOrientation import computeOrientation
from MLVcode.computeLength import computeLength
from MLVcode.computeCurvature import computeCurvature
from MLVcode.drawLinedrawingProperty import drawLinedrawingProperty


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


def compute_contour_info(fp, contour_reduce_percent=0):
    vecLD_arr = load_mat(fp)

    for fn, vec in zip(
        vecLD_arr["allVecLDs"]["filename"], vecLD_arr["allVecLDs"]["vecLD"]
    ):

        # computeCurvature(vec)
        computeLength(vec)
        if contour_reduce_percent:
            delete_contours(vec, contour_reduce_percent)

        drawLinedrawingProperty(vec, "length", fn=fn)

        # drawLinedrawingProperty(vec, "curvature", fn=fn)
    return vecLD_arr


def delete_contours(vecLD, contour_reduce_percent):
    """
    Remove the shortest contours from vecLD, keeping only contours
    whose length is at or above the given percentile.

    Args:
        vecLD: The vectorized line drawing data structure.
        contour_reduce_percent: Percentile threshold (0-100). Contours with
            length below this percentile are removed.

    Returns:
        The modified vecLD with short contours removed.
    """
    contour_lengths = vecLD["contourLengths"].flatten()

    if len(contour_lengths) == 0:
        return vecLD

    length_threshold = np.percentile(contour_lengths, contour_reduce_percent)

    # Find indices of contours to keep (at or above the percentile)
    keep_indices = [
        i for i, length in enumerate(contour_lengths) if length >= length_threshold
    ]

    # Update all contour-related fields
    vecLD["contours"] = [vecLD["contours"][i] for i in keep_indices]
    vecLD["contourLengths"] = vecLD["contourLengths"][keep_indices]
    vecLD["lengths"] = [[vecLD["lengths"][0][i] for i in keep_indices]]
    vecLD["numContours"] = len(keep_indices)


def compute_distance(vec1, vec2, tile_width, tile1_x, tile1_y, tile2_x, tile2_y):
    """Compute hausdorff distance between line drawings of vec1 and vec2, within the tile location.

    Args:
        vec1: First vectorized line drawing data structure (with contours).
        vec2: Second vectorized line drawing data structure (with contours).
        tile_width: Width (and height) of the square tile region.
        tile_x: X coordinate of the top-left corner of the tile.
        tile_y: Y coordinate of the top-left corner of the tile.

    Returns:
        float: The Hausdorff distance between the two line drawings within the tile.
    """

    def get_points_in_tile(vecLD, tile_x, tile_y, tile_width):
        """Extract all segment endpoint pixels within the tile from a vecLD."""
        points = []
        for c in range(int(vecLD["numContours"])):
            contour = vecLD["contours"][c]
            if len(contour.shape) == 1:
                contour = contour[None, :]
            for seg in contour:
                x1, y1, x2, y2 = seg[0], seg[1], seg[2], seg[3]
                # Check if either endpoint falls within the tile
                if (
                    tile_x <= x1 < tile_x + tile_width
                    and tile_y <= y1 < tile_y + tile_width
                ):
                    points.append([x1, y1])
                if (
                    tile_x <= x2 < tile_x + tile_width
                    and tile_y <= y2 < tile_y + tile_width
                ):
                    points.append([x2, y2])
        return np.array(points) if points else None

    points1 = get_points_in_tile(vec1, tile1_x, tile1_y, tile_width)
    points2 = get_points_in_tile(vec2, tile2_x, tile2_y, tile_width)

    # If either set has no points in the tile, return infinity
    if points1 is None or points2 is None:
        return float("inf")

    # Compute directed Hausdorff: max of d(A->B) and d(B->A)
    def directed_hausdorff(A, B):
        """For each point in A, find min distance to any point in B, then take the max."""
        # A: (N, 2), B: (M, 2)
        # Compute pairwise distances using broadcasting
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  # (N, M, 2)
        dists = np.sqrt(np.sum(diff**2, axis=2))  # (N, M)
        min_dists = np.min(dists, axis=1)  # (N,)
        return np.max(min_dists) ** 2

    d_ab = directed_hausdorff(points1, points2)
    d_ba = directed_hausdorff(points2, points1)

    return max(d_ab, d_ba)


def test_distances(vec1, vec2):
    tile_ratio = 0.5
    stride = 10
    w = vec1["imsize"][0] * tile_ratio
    m_h, m_w = vec1["imsize"][0], vec1["imsize"][1]
    distances = {}
    for i_1 in range(0, int(m_h - w), stride):
        for j_1 in range(0, int(m_w - w), stride):
            for i_2 in range(0, int(m_h - w), stride):
                for j_2 in range(0, int(m_w - w), stride):
                    distances[(i_1, j_1, i_2, j_2)] = compute_distance(
                        vec1, vec2, w, i_1, j_1, i_2, i_2
                    )
    max_key = max(distances, key=lambda k: distances[k])
    print("Max distance key:", max_key)


if __name__ == "__main__":
    # create_reduced_dir("cusco_salkantay", "cusco_salkantay_med", scale=2.5)
    # create_sobel_folder("cusco_salkantay_small")
    vecLD_arr = compute_contour_info("smallVecLDs.mat", 95)
    test_distances(
        vecLD_arr["allVecLDs"]["vecLD"][0], vecLD_arr["allVecLDs"]["vecLD"][1]
    )
    # create_vectorized_edge_folder("cusco_salkantay_med")

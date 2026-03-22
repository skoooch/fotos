"""
Sequence images by finding smooth contour-based transitions.

Strategy:
  1. Compute weighted edge maps (not binary) preserving edge strength.
  2. Cheap global descriptors to build a K-nearest-neighbor shortlist.
  3. Expensive tile matching on ALL tile-pair combinations for shortlisted pairs,
     testing multiple tile sizes (ratios from T_R_MIN to T_R_MAX).
  4. Build a sparse cost graph with tile-position-aware TSP solver.
  5. Store tile positions for video generation.

The key insight: when image B sits between A and C in the sequence, the tile
position chosen for B must work well with BOTH neighbors simultaneously.
We precompute costs for every (tile_pos_A, tile_pos_B) combination per
shortlisted pair, then the greedy/2-opt solvers track per-image tile state.
"""

import os
import sys
import pickle
from math import inf
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import cv2 as cv
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist

import torch
import open_clip
from PIL import Image


# ── Config ──────────────────────────────────────────────────────────────────
TILE_RATIO = 0.75
T_R_MAX = 1
T_R_MIN = 0.6
T_R_STRIDE = 0.1
STRIDE = 50  # coarser stride for expanded tile-pair search
EDGE_BLUR_KSIZE = 3
EDGE_GAMMA = 1.5  # >1 suppresses weak edges, amplifies strong ones
MIN_EDGE_LENGTH = 30  # minimum connected-component size in pixels to keep
EDGE_THRESHOLD_LOW = 0.6  # discard edge pixels below this strength
NUM_2OPT_ITERATIONS = 50
K_NEIGHBORS = 30  # only tile-match the K most promising neighbors per image
TARGET_SHORT_EDGE = 512
EDGE_THRESHOLD = 0.1
MIN_EDGE_DENSITY = 0.01
REFINE_STRIDE = 5  # fine-grained stride for tile refinement
REFINE_RADIUS = 25  # search radius (pixels) around current tile position
REFINE_ITERATIONS = 4  # number of forward+backward sweeps
DISTANCE_METRIC = "embedding"  # "edge_descript", "embedding", or "combined"
EMBEDDING_WEIGHT = 0.5  # weight for embedding distance in combined mode
EDGE_METHOD = "hed"  # "hed", "scharr"

# ── Derived: list of tile ratios to test ─────────────────────────────────────
TILE_RATIOS = []
_r = T_R_MIN
while _r <= T_R_MAX + 1e-9:
    TILE_RATIOS.append(round(_r, 6))
    _r += T_R_STRIDE
TILE_RATIOS = sorted(set(TILE_RATIOS))

# Common comparison size: the tile pixel size produced by the smallest ratio
# on a given image. We resize all tiles to this common size before Chamfer.
# This is computed per-pair as: int(min(h_a, w_a, h_b, w_b) * T_R_MIN)

# ── HED Model ───────────────────────────────────────────────────────────────

_hed_net = None
_hed_crop_registered = False
HED_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
)
HED_CAFFEMODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
HED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _download_file(url, dest):
    """Download a file with progress indication."""
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"    Saved to {dest}")


def _get_hed_model():
    """Lazily load the HED network (once per process)."""
    global _hed_net, _hed_crop_registered
    if _hed_net is not None:
        return _hed_net

    prototxt_path = os.path.join(HED_MODEL_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(HED_MODEL_DIR, "hed_pretrained_bsds.caffemodel")

    # Download if not present
    if not os.path.exists(prototxt_path):
        _download_file(HED_PROTOTXT_URL, prototxt_path)

    if not os.path.exists(caffemodel_path):
        _download_file(HED_CAFFEMODEL_URL, caffemodel_path)

    # CropLayer is needed by HED — register it only once
    if not _hed_crop_registered:

        class CropLayer:
            def __init__(self, params, blobs):
                self.startX = 0
                self.startY = 0

            def getMemoryShapes(self, inputs):
                (inputShape, targetShape) = (inputs[0], inputs[1])
                batchSize, numChannels = inputShape[0], inputShape[1]
                height, width = targetShape[2], targetShape[3]
                self.startY = (inputShape[2] - targetShape[2]) // 2
                self.startX = (inputShape[3] - targetShape[3]) // 2
                return [[batchSize, numChannels, height, width]]

            def forward(self, inputs):
                return [
                    inputs[0][
                        :,
                        :,
                        self.startY : self.startY + inputs[1].shape[2],
                        self.startX : self.startX + inputs[1].shape[3],
                    ]
                ]

        cv.dnn_registerLayer("Crop", CropLayer)
        _hed_crop_registered = True

    print("  Loading HED contour detection model...")
    _hed_net = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Try CUDA backend, fall back to CPU
    use_cuda = False
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            _hed_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            _hed_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
            use_cuda = True
            print("    HED using CUDA backend")
        except Exception:
            use_cuda = False

    if not use_cuda:
        _hed_net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        _hed_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print("    HED using CPU backend")

    print("  HED model loaded.")
    return _hed_net


def _hed_edge_map(img_bgr, target_short_edge=TARGET_SHORT_EDGE):
    """
    Run HED on a BGR image and return a float32 edge map in [0, 1].
    HED produces soft contour probabilities — high values at semantically
    meaningful boundaries (object outlines, figure/ground, horizons).
    """
    h, w = img_bgr.shape[:2]
    scale = target_short_edge / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv.resize(img_bgr, (new_w, new_h), interpolation=cv.INTER_AREA)

    net = _get_hed_model()

    # HED expects mean-subtracted input
    blob = cv.dnn.blobFromImage(
        img_resized,
        scalefactor=1.0,
        size=(new_w, new_h),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()

    # Output is (1, 1, H, W), values in ~[0, 1]
    edge_map = out[0, 0]
    edge_map = np.clip(edge_map, 0, 1).astype(np.float32)

    return edge_map


# ── CLIP Embedding Cache ────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _load_clip_model():
    """Lazily load the CLIP model (once per process)."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_device

    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading CLIP model on {_clip_device}...")
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip_model = _clip_model.to(_clip_device)
    _clip_model.eval()
    print(f"  CLIP model loaded.")
    return _clip_model, _clip_preprocess, _clip_device


def _unload_clip_model():
    """Free CLIP model memory before heavy multiprocessing."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        del _clip_model
        _clip_model = None
        _clip_preprocess = None
        _clip_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
        print("  CLIP model unloaded to free memory.")


# ── Edge Extraction ─────────────────────────────────────────────────────────


def _scharr_edge_map(image_path, target_short_edge=TARGET_SHORT_EDGE):
    """
    Fallback Scharr-based edge map with NMS (original method).
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    h, w = img.shape
    scale = target_short_edge / min(h, w)
    img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    img = cv.GaussianBlur(img, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)

    grad_x = cv.Scharr(img, cv.CV_64F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_64F, 0, 1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Vectorized NMS
    angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi % 180.0
    mag_pad = np.pad(magnitude, 1, mode="constant", constant_values=0)
    h_m, w_m = magnitude.shape

    direction = np.zeros_like(angle, dtype=np.int32)
    direction[(angle >= 22.5) & (angle < 67.5)] = 1
    direction[(angle >= 67.5) & (angle < 112.5)] = 2
    direction[(angle >= 112.5) & (angle < 157.5)] = 3

    r_idx = np.arange(1, h_m + 1)[:, None]
    c_idx = np.arange(1, w_m + 1)[None, :]

    n1 = np.empty((4, h_m, w_m), dtype=np.float64)
    n2 = np.empty((4, h_m, w_m), dtype=np.float64)
    n1[0] = mag_pad[r_idx, c_idx - 1]
    n2[0] = mag_pad[r_idx, c_idx + 1]
    n1[1] = mag_pad[r_idx + 1, c_idx - 1]
    n2[1] = mag_pad[r_idx - 1, c_idx + 1]
    n1[2] = mag_pad[r_idx - 1, c_idx]
    n2[2] = mag_pad[r_idx + 1, c_idx]
    n1[3] = mag_pad[r_idx - 1, c_idx - 1]
    n2[3] = mag_pad[r_idx + 1, c_idx + 1]

    rows = np.arange(h_m)[:, None]
    cols = np.arange(w_m)[None, :]
    neighbor1 = n1[direction, rows, cols]
    neighbor2 = n2[direction, rows, cols]
    magnitude = magnitude * ((magnitude >= neighbor1) & (magnitude >= neighbor2))

    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max

    return magnitude.astype(np.float32)


def get_weighted_edges(image_path, target_short_edge=TARGET_SHORT_EDGE):
    """
    Produce a float32 edge-strength map in [0, 1].

    If EDGE_METHOD == "hed", uses the HED deep contour detector which
    natively produces semantically meaningful boundaries (object outlines,
    figure/ground, horizons) and suppresses texture.

    Post-processing:
      1. Threshold out low-confidence pixels
      2. Connected-component filtering to remove small fragments
      3. Gamma correction to further separate strong from weak
    """
    if EDGE_METHOD == "hed":
        img_bgr = cv.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        magnitude = _hed_edge_map(img_bgr, target_short_edge)
    else:
        magnitude = _scharr_edge_map(image_path, target_short_edge)

    # ── Threshold out low-confidence edge pixels ──
    magnitude[magnitude < EDGE_THRESHOLD_LOW] = 0.0

    # ── Connected-component length filtering ──
    edge_binary = (magnitude > 0).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(edge_binary, connectivity=8)

    component_sizes = np.bincount(labels.ravel())
    small_labels = np.where(component_sizes < MIN_EDGE_LENGTH)[0]
    keep_mask = np.ones(num_labels, dtype=bool)
    keep_mask[0] = False  # background
    keep_mask[small_labels] = False
    magnitude = magnitude * keep_mask[labels]

    # ── Re-normalize after filtering ──
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max

    # ── Gamma correction ──
    magnitude = np.power(magnitude, EDGE_GAMMA)

    return magnitude.astype(np.float32)


# ── Cheap Global Descriptor ────────────────────────────────────────────────


def compute_image_features(filenames, image_folder):
    """
    Compute CLIP embeddings for all images.

    Args:
        filenames: list of image filenames
        image_folder: path to folder containing the images

    Returns:
        np.ndarray of shape (n, embedding_dim), L2-normalized
    """
    model, preprocess, device = _load_clip_model()

    embeddings = []
    batch_size = 32
    print(f"  Computing CLIP embeddings for {len(filenames)} images...")

    for batch_start in range(0, len(filenames), batch_size):
        batch_fns = filenames[batch_start : batch_start + batch_size]
        batch_tensors = []
        for fn in batch_fns:
            path = os.path.join(image_folder, fn)
            try:
                pil_img = Image.open(path).convert("RGB")
                tensor = preprocess(pil_img)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"    CLIP skip {fn}: {e}")
                # Use a zero vector as fallback
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=device if device != "cpu" else "cpu"
        ):
            features = model.encode_image(batch)
            features = features.float()
            features /= features.norm(dim=-1, keepdim=True)

        embeddings.append(features.cpu().numpy())

        done = min(batch_start + batch_size, len(filenames))
        if done % 100 == 0 or done == len(filenames):
            print(f"    {done}/{len(filenames)} embeddings computed")

    embeddings = np.vstack(embeddings)
    print(f"  CLIP embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def compute_edge_descriptor(edge_map, num_spatial_bins=4, num_orient_bins=8):
    """
    Compute a lightweight descriptor from the edge map:
      - Divide image into spatial grid
      - In each cell, build a histogram of edge orientations weighted by magnitude
      - Concatenate into a single vector

    This is essentially a simplified HOG on the edge-strength map.
    Fast to compute, fast to compare (cosine distance on short vectors).
    """
    h, w = edge_map.shape

    blurred = cv.GaussianBlur((edge_map * 255).astype(np.uint8), (3, 3), 0).astype(
        np.float32
    )
    gx = cv.Scharr(blurred, cv.CV_32F, 1, 0)
    gy = cv.Scharr(blurred, cv.CV_32F, 0, 1)
    orientation = np.arctan2(gy, gx)  # [-pi, pi]
    orientation = (orientation + np.pi) / (2 * np.pi)  # [0, 1]

    descriptor = []
    cell_h = h // num_spatial_bins
    cell_w = w // num_spatial_bins

    for si in range(num_spatial_bins):
        for sj in range(num_spatial_bins):
            y0 = si * cell_h
            x0 = sj * cell_w
            y1 = y0 + cell_h
            x1 = x0 + cell_w

            mag_cell = edge_map[y0:y1, x0:x1]
            ori_cell = orientation[y0:y1, x0:x1]

            bins = np.linspace(0, 1, num_orient_bins + 1)
            hist, _ = np.histogram(
                ori_cell.ravel(), bins=bins, weights=mag_cell.ravel()
            )
            descriptor.extend(hist)

    descriptor = np.array(descriptor, dtype=np.float32)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm
    return descriptor


# ── Tile Position Utilities ─────────────────────────────────────────────────

# A tile position is now a tuple (y, x, tile_size) to support variable sizes.


def _tile_positions(h, w, tile_size, stride=STRIDE):
    """Return list of (y, x) tile top-left positions for an image at a given tile_size."""
    positions = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            positions.append((y, x))
    if not positions:
        positions.append((0, 0))
    return positions


def _tile_positions_multi(h, w, tile_ratios=TILE_RATIOS, stride=STRIDE):
    """
    Return list of (y, x, tile_size) positions for ALL tile ratios.
    """
    positions = []
    for ratio in tile_ratios:
        tile_size = int(min(h, w) * ratio)
        if tile_size < 16 or tile_size > h or tile_size > w:
            continue
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                positions.append((y, x, tile_size))
        # Ensure at least one position for this ratio
        if not any(p[2] == tile_size for p in positions):
            positions.append((0, 0, tile_size))
    if not positions:
        # Absolute fallback
        ts = int(min(h, w) * TILE_RATIOS[0])
        ts = max(16, min(ts, h, w))
        positions.append(((h - ts) // 2, (w - ts) // 2, ts))
    return positions


def _precompute_tile_data_multi(
    edge_map, positions_with_size, common_size, threshold=EDGE_THRESHOLD
):
    """
    Precompute mask, weights, distance transform for each tile position,
    resizing all tiles to `common_size` for comparable Chamfer distances.

    positions_with_size: list of (y, x, tile_size)
    common_size: int — all tiles are resized to (common_size, common_size)

    Returns list of (y, x, tile_size, mask, weights, sum_w, dt) for valid tiles.
    """
    tiles = []
    for y, x, ts in positions_with_size:
        tile = edge_map[y : y + ts, x : x + ts]
        if tile.shape[0] != ts or tile.shape[1] != ts:
            continue
        # Resize to common comparison size
        if ts != common_size:
            tile = cv.resize(
                tile, (common_size, common_size), interpolation=cv.INTER_AREA
            )
        mask = tile > threshold
        if mask.mean() < MIN_EDGE_DENSITY:
            continue
        weights = tile[mask]
        sum_w = weights.sum()
        if sum_w == 0:
            continue
        dt = distance_transform_edt(~mask)
        tiles.append((y, x, ts, mask, weights, sum_w, dt))
    return tiles


def get_tile_positions_for_image(edge_map, tile_ratios=TILE_RATIOS, stride=STRIDE):
    """
    Return the list of valid tile positions for a single image across all
    tile ratios.

    Returns:
        valid: list of (y, x, tile_size) tuples
    """
    h, w = edge_map.shape
    all_positions = _tile_positions_multi(h, w, tile_ratios, stride)

    valid = []
    for y, x, ts in all_positions:
        tile = edge_map[y : y + ts, x : x + ts]
        if tile.shape[0] != ts or tile.shape[1] != ts:
            continue
        mask = tile > EDGE_THRESHOLD
        if mask.mean() >= MIN_EDGE_DENSITY and tile[mask].sum() > 0:
            valid.append((y, x, ts))

    if not valid:
        # Fallback to center at smallest ratio
        ts = int(min(h, w) * tile_ratios[0])
        ts = max(16, min(ts, h, w))
        valid.append(((h - ts) // 2, (w - ts) // 2, ts))

    return valid


# ── All Tile-Pair Cost Computation ──────────────────────────────────────────


def find_all_tile_pair_costs(edge_a, edge_b, tile_ratios=TILE_RATIOS, stride=STRIDE):
    """
    Compute Chamfer cost for ALL valid tile position combinations between
    two images, across ALL tile ratios.

    Tiles of different sizes are resized to a common comparison size
    (the smallest tile size across both images) before computing Chamfer distance.

    Returns:
        costs: dict mapping ((ay, ax, a_ts), (by, bx, b_ts)) -> cost
        best_cost: float (minimum cost across all pairs)
        best_pos_a: (y, x, tile_size) of best tile in image a
        best_pos_b: (y, x, tile_size) of best tile in image b
    """
    h_a, w_a = edge_a.shape
    h_b, w_b = edge_b.shape

    # Common comparison size: smallest tile across all ratios and both images
    min_dim = min(h_a, w_a, h_b, w_b)
    common_size = int(min_dim * min(tile_ratios))
    if common_size < 16:
        common_size = 16

    positions_a = _tile_positions_multi(h_a, w_a, tile_ratios, stride)
    positions_b = _tile_positions_multi(h_b, w_b, tile_ratios, stride)

    tiles_a = _precompute_tile_data_multi(edge_a, positions_a, common_size)
    tiles_b = _precompute_tile_data_multi(edge_b, positions_b, common_size)

    if not tiles_a or not tiles_b:
        return {}, inf, None, None

    costs = {}
    best_cost = inf
    best_pos_a = (tiles_a[0][0], tiles_a[0][1], tiles_a[0][2])
    best_pos_b = (tiles_b[0][0], tiles_b[0][1], tiles_b[0][2])

    for ay, ax, a_ts, mask_a, weights_a, sum_wa, dt_a in tiles_a:
        for by, bx, b_ts, mask_b, weights_b, sum_wb, dt_b in tiles_b:
            d_ab = np.sum(weights_a * dt_b[mask_a]) / sum_wa
            d_ba = np.sum(weights_b * dt_a[mask_b]) / sum_wb
            cost = (d_ab + d_ba) / 2.0
            costs[((ay, ax, a_ts), (by, bx, b_ts))] = cost
            if cost < best_cost:
                best_cost = cost
                best_pos_a = (ay, ax, a_ts)
                best_pos_b = (by, bx, b_ts)

    return costs, best_cost, best_pos_a, best_pos_b


def _compute_all_pairs(args):
    """Worker for parallel all-tile-pair matching."""
    i, j, edge_i, edge_j = args
    costs, best_cost, pos_a, pos_b = find_all_tile_pair_costs(edge_i, edge_j)
    return i, j, costs, best_cost, pos_a, pos_b


# ── Pairwise Cost Matrix (Sparse, KNN-based) ───────────────────────────────


def build_edge_maps(image_folder, extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp")):
    """Load all images from a folder and compute weighted edge maps."""
    edge_maps = {}
    for fn in sorted(os.listdir(image_folder)):
        if os.path.splitext(fn)[1].lower() in extensions:
            path = os.path.join(image_folder, fn)
            try:
                edge_maps[fn] = get_weighted_edges(path)
                print(f"  edges: {fn}  shape={edge_maps[fn].shape}")
            except Exception as e:
                print(f"  SKIP {fn}: {e}")
    return edge_maps


def build_knn_shortlist(
    edge_maps, k=K_NEIGHBORS, distance_metric=DISTANCE_METRIC, image_folder=None
):
    """
    Compute global descriptors for all images and find the
    K nearest neighbors for each image.

    Returns:
        filenames: list[str]
        symmetric_pairs: set of (i, j) tuples with i < j
        neighbors: dict mapping index -> list of neighbor indices
        dist_matrix: np.ndarray coarse distance matrix
    """
    filenames = list(edge_maps.keys())
    n = len(filenames)

    if distance_metric == "edge_descript":
        print(f"  Computing edge descriptors for {n} images...")
        descriptors = np.array(
            [compute_edge_descriptor(edge_maps[fn]) for fn in filenames]
        )
        print(f"  Computing cosine distance matrix...")
        dist_matrix = cdist(descriptors, descriptors, metric="cosine")

    elif distance_metric == "embedding":
        if image_folder is None:
            raise ValueError("image_folder required for embedding distance metric")
        embeddings = compute_image_features(filenames, image_folder)
        print(f"  Computing cosine distance matrix from CLIP embeddings...")
        dist_matrix = cdist(embeddings, embeddings, metric="cosine")

    elif distance_metric == "combined":
        if image_folder is None:
            raise ValueError("image_folder required for combined distance metric")
        print(f"  Computing edge descriptors for {n} images...")
        edge_descs = np.array(
            [compute_edge_descriptor(edge_maps[fn]) for fn in filenames]
        )
        edge_dist = cdist(edge_descs, edge_descs, metric="cosine")

        embeddings = compute_image_features(filenames, image_folder)
        embed_dist = cdist(embeddings, embeddings, metric="cosine")

        edge_max = edge_dist.max()
        embed_max = embed_dist.max()
        if edge_max > 0:
            edge_dist_norm = edge_dist / edge_max
        else:
            edge_dist_norm = edge_dist
        if embed_max > 0:
            embed_dist_norm = embed_dist / embed_max
        else:
            embed_dist_norm = embed_dist

        w = EMBEDDING_WEIGHT
        dist_matrix = (1 - w) * edge_dist_norm + w * embed_dist_norm
        print(f"  Combined distance matrix: {1-w:.0%} edge + {w:.0%} semantic")

    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    # For each image, find K nearest (excluding self)
    neighbors = {}
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = inf
        nearest = np.argsort(dists)[:k]
        neighbors[i] = nearest.tolist()

    # Make symmetric
    symmetric_pairs = set()
    for i, nbrs in neighbors.items():
        for j in nbrs:
            symmetric_pairs.add((min(i, j), max(i, j)))

    print(
        f"  KNN shortlist: {len(symmetric_pairs)} pairs "
        f"(vs {n*(n-1)//2} exhaustive, "
        f"{100*len(symmetric_pairs)/(n*(n-1)//2):.1f}%)"
    )

    return filenames, symmetric_pairs, neighbors, dist_matrix


def build_sparse_cost_data(
    edge_maps, max_workers=None, k=K_NEIGHBORS, image_folder=None
):
    """
    1. Build KNN shortlist using cheap descriptors.
    2. Run expensive ALL tile-pair matching on shortlisted pairs (multi-ratio).
    3. Compute fallback costs for non-shortlisted pairs.

    Returns:
        filenames: list of filenames
        all_pair_costs: dict (i,j) -> {((ay,ax,a_ts),(by,bx,b_ts)): cost, ...}
                        where i < j, keyed as canonical (min,max) pairs
        best_costs: dict (i,j) -> float, best cost per pair (for fallback matrix)
        tile_positions_per_image: dict idx -> list of (y,x,tile_size) valid positions
        fallback_cost_matrix: n×n array with coarse costs for non-shortlisted
        coarse_dist: n×n raw coarse distance matrix
    """
    filenames, shortlisted_pairs, neighbors, coarse_dist = build_knn_shortlist(
        edge_maps, k=k, image_folder=image_folder
    )
    _unload_clip_model()
    n = len(filenames)

    all_pair_costs = {}  # (i,j) -> {((ay,ax,a_ts),(by,bx,b_ts)): cost}
    best_costs = {}  # (i,j) -> float

    # Precompute tile positions per image (multi-ratio)
    tile_positions_per_image = {}
    for idx, fn in enumerate(filenames):
        positions = get_tile_positions_for_image(edge_maps[fn])
        tile_positions_per_image[idx] = positions

    # Prepare tasks for shortlisted pairs
    tasks = [
        (i, j, edge_maps[filenames[i]], edge_maps[filenames[j]])
        for i, j in shortlisted_pairs
    ]

    total = len(tasks)
    print(
        f"  All-tile-pair matching {total} shortlisted pairs (of {n*(n-1)//2} total)..."
    )
    print(f"  Tile ratios: {TILE_RATIOS}")
    print(
        f"  Avg tile positions per image: {np.mean([len(v) for v in tile_positions_per_image.values()]):.1f}"
    )

    done = 0
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_all_pairs, t): t for t in tasks}
        for future in as_completed(futures):
            i, j, costs, best_cost, pos_a, pos_b = future.result()
            key = (min(i, j), max(i, j))
            all_pair_costs[key] = costs
            best_costs[key] = best_cost
            done += 1
            if done % 10 == 0 or done == total:
                with open("tile_matching.log", "a") as log:
                    log.write(f"{done}/{total}\n")
                if done % 100 == 0 or done == total:
                    print(f"    {done}/{total} pairs computed")

    # Build fallback cost matrix for non-shortlisted pairs
    best_cost_matrix = np.full((n, n), inf)
    for (i, j), cost in best_costs.items():
        best_cost_matrix[i, j] = cost
        best_cost_matrix[j, i] = cost

    tile_costs = best_cost_matrix[best_cost_matrix < inf]
    if len(tile_costs) > 0:
        coarse_scale = np.median(tile_costs) / (
            np.median(coarse_dist[coarse_dist > 0]) + 1e-9
        )
        penalty = np.percentile(tile_costs, 90) if len(tile_costs) > 0 else 100
    else:
        coarse_scale = 1.0
        penalty = 100

    fallback_cost_matrix = np.full((n, n), inf)
    for i in range(n):
        for j in range(i + 1, n):
            if best_cost_matrix[i, j] < inf:
                fallback_cost_matrix[i, j] = best_cost_matrix[i, j]
                fallback_cost_matrix[j, i] = best_cost_matrix[i, j]
            else:
                fallback = coarse_dist[i, j] * coarse_scale + penalty
                fallback_cost_matrix[i, j] = fallback
                fallback_cost_matrix[j, i] = fallback

    print(f"  Tile-pair cost tables built for {len(all_pair_costs)} pairs.")
    total_entries = sum(len(v) for v in all_pair_costs.values())
    print(f"  Total tile-pair entries: {total_entries}")

    return (
        filenames,
        all_pair_costs,
        best_costs,
        tile_positions_per_image,
        fallback_cost_matrix,
        coarse_dist,
    )


# ── Tile-Aware Cost Lookup ──────────────────────────────────────────────────

# Tile positions are now (y, x, tile_size) tuples throughout.


def lookup_cost(i, j, pos_i, pos_j, all_pair_costs, fallback_cost_matrix):
    """
    Look up the cost of transitioning from image i (at tile pos_i) to
    image j (at tile pos_j).

    pos_i and pos_j are (y, x, tile_size) tuples or None.

    If the pair was shortlisted, we have exact tile-pair costs.
    Otherwise, fall back to coarse cost.
    """
    key = (min(i, j), max(i, j))
    pair_costs = all_pair_costs.get(key)

    if pair_costs is None:
        return fallback_cost_matrix[i, j]

    if pos_i is None or pos_j is None:
        # No tile position locked yet — return the best possible cost for this pair
        if pair_costs:
            return min(pair_costs.values())
        return fallback_cost_matrix[i, j]

    # Orient positions correctly: pair_costs is keyed as (pos_smaller_idx, pos_larger_idx)
    if i < j:
        tile_key = (pos_i, pos_j)
    else:
        tile_key = (pos_j, pos_i)

    cost = pair_costs.get(tile_key)
    if cost is not None:
        return cost

    # Position not in precomputed set (e.g. from refinement) — find nearest
    # available position for each image
    return fallback_cost_matrix[i, j]


def best_cost_for_arrival(
    i, j, pos_i, all_pair_costs, fallback_cost_matrix, tile_positions_j
):
    """
    Given that image i is locked at pos_i (y, x, tile_size), find the best
    tile position for image j and the corresponding cost.

    Returns:
        best_cost: float
        best_pos_j: (y, x, tile_size) or None
    """
    key = (min(i, j), max(i, j))
    pair_costs = all_pair_costs.get(key)

    if pair_costs is None or pos_i is None:
        return fallback_cost_matrix[i, j], None

    best_cost = inf
    best_pos_j = None

    for pos_j in tile_positions_j:
        if i < j:
            tile_key = (pos_i, pos_j)
        else:
            tile_key = (pos_j, pos_i)

        cost = pair_costs.get(tile_key)
        if cost is not None and cost < best_cost:
            best_cost = cost
            best_pos_j = pos_j

    if best_pos_j is None:
        return fallback_cost_matrix[i, j], None

    return best_cost, best_pos_j


def best_cost_any_pos(i, j, all_pair_costs, fallback_cost_matrix):
    """
    Find the best possible cost between i and j over ALL tile position
    combinations. Used when neither image has a locked position.

    Returns:
        best_cost: float
        best_pos_i: (y, x, tile_size) or None
        best_pos_j: (y, x, tile_size) or None
    """
    key = (min(i, j), max(i, j))
    pair_costs = all_pair_costs.get(key)

    if pair_costs is None or not pair_costs:
        return fallback_cost_matrix[i, j], None, None

    best_tile_key = min(pair_costs, key=pair_costs.get)
    best_cost = pair_costs[best_tile_key]
    pos_smaller, pos_larger = best_tile_key

    if i < j:
        return best_cost, pos_smaller, pos_larger
    else:
        return best_cost, pos_larger, pos_smaller


# ── TSP Solver: Tile-Aware Greedy + 2-opt ──────────────────────────────────


def greedy_nearest_neighbor_tileaware(
    n,
    all_pair_costs,
    fallback_cost_matrix,
    tile_positions_per_image,
):
    """
    Nearest-neighbor heuristic that tracks each image's current tile position.
    Tile positions are (y, x, tile_size) tuples.

    When we move from image `current` to `next`:
      - If `current` already has a locked tile position, we find the best
        position for `next` given that constraint.
      - If `current` has no locked position (it's the start), we pick the
        pair (pos_current, pos_next) that minimizes cost.

    Returns:
        best_path: list of image indices
        best_total: float total cost
        best_positions: dict image_idx -> (ty, tx, tile_size)
    """
    best_path = None
    best_total = inf
    best_positions = None

    rng = np.random.default_rng(42)
    num_starts = min(50, n) if n > 200 else n
    start_nodes = rng.choice(n, size=num_starts, replace=False) if n > 200 else range(n)

    for si, start in enumerate(start_nodes):
        start = int(start)
        visited = {start}
        path = [start]
        total = 0.0
        current = start
        cur_positions = {}  # image_idx -> (ty, tx, tile_size)

        for _ in range(n - 1):
            best_nxt = None
            best_nxt_cost = inf
            best_nxt_pos_cur = None
            best_nxt_pos_nxt = None

            cur_pos = cur_positions.get(current)

            for j in range(n):
                if j in visited:
                    continue

                if cur_pos is not None:
                    # Current image has locked position — find best pos for j
                    cost, pos_j = best_cost_for_arrival(
                        current,
                        j,
                        cur_pos,
                        all_pair_costs,
                        fallback_cost_matrix,
                        tile_positions_per_image.get(j, []),
                    )
                    if cost < best_nxt_cost:
                        best_nxt_cost = cost
                        best_nxt = j
                        best_nxt_pos_cur = cur_pos
                        best_nxt_pos_nxt = pos_j
                else:
                    # Current has no locked position — find best pair
                    cost, pos_c, pos_j = best_cost_any_pos(
                        current, j, all_pair_costs, fallback_cost_matrix
                    )
                    if cost < best_nxt_cost:
                        best_nxt_cost = cost
                        best_nxt = j
                        best_nxt_pos_cur = pos_c
                        best_nxt_pos_nxt = pos_j

            if best_nxt is None:
                break

            # Lock in tile positions
            if best_nxt_pos_cur is not None:
                cur_positions[current] = best_nxt_pos_cur
            if best_nxt_pos_nxt is not None:
                cur_positions[best_nxt] = best_nxt_pos_nxt

            total += best_nxt_cost
            path.append(best_nxt)
            visited.add(best_nxt)
            current = best_nxt

        if total < best_total:
            best_total = total
            best_path = path[:]
            best_positions = dict(cur_positions)

        if (si + 1) % 10 == 0 or (si + 1) == num_starts:
            print(
                f"    Greedy start {si+1}/{num_starts}, best so far: {best_total:.2f}"
            )

    return best_path, best_total, best_positions


def _optimize_positions_for_path(
    path,
    positions,
    all_pair_costs,
    fallback_cost_matrix,
    tile_positions_per_image,
):
    """
    Given a fixed path ordering, optimize tile positions for each image
    to minimize total adjacent cost. Does a forward+backward sweep.

    Modifies `positions` in-place and returns the total cost.
    """
    n = len(path)

    for sweep in range(2):  # forward then backward
        if sweep == 0:
            order = range(n)
        else:
            order = range(n - 1, -1, -1)

        for k in order:
            idx = path[k]
            candidates = tile_positions_per_image.get(idx, [])
            if not candidates:
                continue

            # Compute cost for each candidate position considering neighbors
            best_pos = positions.get(idx)
            best_cost = inf

            # Get current neighbor costs
            prev_idx = path[k - 1] if k > 0 else None
            next_idx = path[k + 1] if k < n - 1 else None

            for pos in candidates:
                cost = 0.0
                if prev_idx is not None:
                    prev_pos = positions.get(prev_idx)
                    cost += lookup_cost(
                        prev_idx,
                        idx,
                        prev_pos,
                        pos,
                        all_pair_costs,
                        fallback_cost_matrix,
                    )
                if next_idx is not None:
                    next_pos = positions.get(next_idx)
                    cost += lookup_cost(
                        idx,
                        next_idx,
                        pos,
                        next_pos,
                        all_pair_costs,
                        fallback_cost_matrix,
                    )
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

            if best_pos is not None:
                positions[idx] = best_pos

    # Compute total path cost
    total = 0.0
    for k in range(n - 1):
        i, j = path[k], path[k + 1]
        total += lookup_cost(
            i,
            j,
            positions.get(i),
            positions.get(j),
            all_pair_costs,
            fallback_cost_matrix,
        )
    return total


def _path_cost_tileaware(path, positions, all_pair_costs, fallback_cost_matrix):
    """Compute total cost of a path with current tile positions."""
    total = 0.0
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        total += lookup_cost(
            i,
            j,
            positions.get(i),
            positions.get(j),
            all_pair_costs,
            fallback_cost_matrix,
        )
    return total


def two_opt_tileaware(
    path,
    positions,
    all_pair_costs,
    fallback_cost_matrix,
    tile_positions_per_image,
    max_iterations=NUM_2OPT_ITERATIONS,
):
    """
    Improve a path by repeatedly reversing sub-segments, with tile-position
    awareness.

    After each accepted swap, we re-optimize tile positions at the affected
    boundary nodes to get an accurate cost comparison.
    """
    best = path[:]
    best_positions = dict(positions)
    best_cost = _path_cost_tileaware(
        best, best_positions, all_pair_costs, fallback_cost_matrix
    )

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Current edges: (i-1, i) and (j, j+1 if exists)
                a, b = best[i - 1], best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else None

                old_cost = lookup_cost(
                    a,
                    b,
                    best_positions.get(a),
                    best_positions.get(b),
                    all_pair_costs,
                    fallback_cost_matrix,
                )
                new_cost = lookup_cost(
                    a,
                    c,
                    best_positions.get(a),
                    best_positions.get(c),
                    all_pair_costs,
                    fallback_cost_matrix,
                )

                if d is not None:
                    old_cost += lookup_cost(
                        c,
                        d,
                        best_positions.get(c),
                        best_positions.get(d),
                        all_pair_costs,
                        fallback_cost_matrix,
                    )
                    new_cost += lookup_cost(
                        b,
                        d,
                        best_positions.get(b),
                        best_positions.get(d),
                        all_pair_costs,
                        fallback_cost_matrix,
                    )

                if new_cost < old_cost:
                    # Accept the reversal
                    candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                    candidate_positions = dict(best_positions)

                    # Re-optimize tile positions at the boundary nodes
                    boundary_indices = set()
                    if i - 1 >= 0:
                        boundary_indices.add(i - 1)
                    boundary_indices.add(i)
                    boundary_indices.add(j)
                    if j + 1 < len(candidate):
                        boundary_indices.add(j + 1)

                    for k in boundary_indices:
                        idx = candidate[k]
                        candidates_pos = tile_positions_per_image.get(idx, [])
                        if not candidates_pos:
                            continue

                        prev_idx = candidate[k - 1] if k > 0 else None
                        next_idx = candidate[k + 1] if k < len(candidate) - 1 else None

                        best_pos = candidate_positions.get(idx)
                        best_local_cost = inf

                        for pos in candidates_pos:
                            cost = 0.0
                            if prev_idx is not None:
                                cost += lookup_cost(
                                    prev_idx,
                                    idx,
                                    candidate_positions.get(prev_idx),
                                    pos,
                                    all_pair_costs,
                                    fallback_cost_matrix,
                                )
                            if next_idx is not None:
                                cost += lookup_cost(
                                    idx,
                                    next_idx,
                                    pos,
                                    candidate_positions.get(next_idx),
                                    all_pair_costs,
                                    fallback_cost_matrix,
                                )
                            if cost < best_local_cost:
                                best_local_cost = cost
                                best_pos = pos

                        if best_pos is not None:
                            candidate_positions[idx] = best_pos

                    # Recompute full cost for accuracy
                    candidate_cost = _path_cost_tileaware(
                        candidate,
                        candidate_positions,
                        all_pair_costs,
                        fallback_cost_matrix,
                    )

                    if candidate_cost < best_cost:
                        best = candidate
                        best_positions = candidate_positions
                        best_cost = candidate_cost
                        improved = True

        print(f"  2-opt iteration {iteration}: cost = {best_cost:.2f}")

    return best, best_cost, best_positions


# ── Tile Position Refinement ────────────────────────────────────────────────


def _chamfer_cost_common(tile_a, tile_b, common_size, threshold=EDGE_THRESHOLD):
    """
    Compute symmetric weighted Chamfer distance between two edge tiles,
    resizing both to common_size first.
    Both inputs should be float32 edge-strength maps (possibly different sizes).
    """
    if tile_a.shape[0] != common_size or tile_a.shape[1] != common_size:
        tile_a = cv.resize(
            tile_a, (common_size, common_size), interpolation=cv.INTER_AREA
        )
    if tile_b.shape[0] != common_size or tile_b.shape[1] != common_size:
        tile_b = cv.resize(
            tile_b, (common_size, common_size), interpolation=cv.INTER_AREA
        )

    mask_a = tile_a > threshold
    mask_b = tile_b > threshold

    w_a = tile_a[mask_a]
    w_b = tile_b[mask_b]
    sum_wa = w_a.sum()
    sum_wb = w_b.sum()

    if sum_wa == 0 or sum_wb == 0:
        return inf

    dt_a = distance_transform_edt(~mask_a)
    dt_b = distance_transform_edt(~mask_b)

    d_ab = np.sum(w_a * dt_b[mask_a]) / sum_wa
    d_ba = np.sum(w_b * dt_a[mask_b]) / sum_wb
    return (d_ab + d_ba) / 2.0


def refine_tile_positions(
    path,
    filenames,
    edge_maps,
    tile_positions,
    tile_ratios=TILE_RATIOS,
    refine_stride=REFINE_STRIDE,
    refine_radius=REFINE_RADIUS,
    num_iterations=REFINE_ITERATIONS,
):
    """
    After sequencing, refine each image's tile position to minimize the
    sum of Chamfer distances to its predecessor and successor.

    This uses a much finer stride than the initial tile matching, since
    we only search a small neighborhood around the current position and
    only consider adjacent pairs. Searches across all tile ratios.

    Args:
        path: sequence of image indices
        filenames: list of filenames
        edge_maps: dict fn -> edge map
        tile_positions: dict image_idx -> (ty, tx, tile_size) from the tile-aware solver

    Returns:
        refined_positions: list of (tile_y, tile_x, tile_size) per sequence index
    """
    n = len(path)

    # Compute common comparison size across all images in the path
    min_dim = inf
    for idx in path:
        fn = filenames[idx]
        em = edge_maps[fn]
        h, w = em.shape
        min_dim = min(min_dim, h, w)
    common_size = int(min_dim * min(tile_ratios))
    common_size = max(16, common_size)

    # Initialize positions from tile-aware solver results
    positions = []  # (ty, tx, ts) for each index in path order
    for k, idx in enumerate(path):
        fn = filenames[idx]
        em = edge_maps[fn]
        h, w = em.shape

        pos = tile_positions.get(idx)
        if pos is not None:
            ty, tx, ts = pos
        else:
            # Default to center at middle ratio
            mid_ratio = tile_ratios[len(tile_ratios) // 2]
            ts = int(min(h, w) * mid_ratio)
            ts = max(16, min(ts, h, w))
            ty, tx = (h - ts) // 2, (w - ts) // 2

        # Clamp
        ty = max(0, min(ty, h - ts))
        tx = max(0, min(tx, w - ts))
        positions.append((ty, tx, ts))

    def get_tile(seq_idx):
        """Extract the current tile for a given sequence position."""
        idx = path[seq_idx]
        fn = filenames[idx]
        em = edge_maps[fn]
        ty, tx, ts = positions[seq_idx]
        return em[ty : ty + ts, tx : tx + ts]

    def compute_total_cost():
        """Compute total adjacent Chamfer cost for the current positions."""
        total = 0.0
        for k in range(n - 1):
            t_a = get_tile(k)
            t_b = get_tile(k + 1)
            if t_a.size > 0 and t_b.size > 0:
                total += _chamfer_cost_common(t_a, t_b, common_size)
        return total

    initial_cost = compute_total_cost()
    print(f"    Refinement initial cost: {initial_cost:.2f}")

    for iteration in range(num_iterations):
        changed = False

        sweep_order = list(range(1, n - 1)) + list(range(n - 2, 0, -1))
        sweep_order = [0] + sweep_order + [n - 1]

        for k in sweep_order:
            with open("tile_matching.log", "a") as log:
                log.write(f"refine {iteration}-{k}\n")
            idx = path[k]
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            cur_ty, cur_tx, cur_ts = positions[k]

            # Determine which neighbors to consider
            neighbors = []
            if k > 0:
                neighbors.append(k - 1)
            if k < n - 1:
                neighbors.append(k + 1)

            if not neighbors:
                continue

            neighbor_tiles = [get_tile(nk) for nk in neighbors]

            best_cost = inf
            best_pos = (cur_ty, cur_tx, cur_ts)

            # Evaluate current position first
            cur_tile = em[cur_ty : cur_ty + cur_ts, cur_tx : cur_tx + cur_ts]
            if cur_tile.shape[0] == cur_ts and cur_tile.shape[1] == cur_ts:
                cur_cost = 0.0
                valid = True
                for nt in neighbor_tiles:
                    if nt.size > 0:
                        cur_cost += _chamfer_cost_common(cur_tile, nt, common_size)
                    else:
                        valid = False
                        break
                if valid:
                    best_cost = cur_cost
                    best_pos = (cur_ty, cur_tx, cur_ts)

            # Search the neighborhood across all tile ratios
            for ratio in tile_ratios:
                ts = int(min(h, w) * ratio)
                if ts < 16 or ts > h or ts > w:
                    continue

                # Scale refine_radius relative to the tile size ratio
                # so we search proportionally
                if cur_ts > 0:
                    scale_factor = ts / cur_ts
                else:
                    scale_factor = 1.0
                scaled_radius = max(1, int(refine_radius * scale_factor))

                # Center of search: scale current position to this tile size
                center_ty = max(0, min(int(cur_ty * scale_factor), h - ts))
                center_tx = max(0, min(int(cur_tx * scale_factor), w - ts))

                y_min = max(0, center_ty - scaled_radius)
                y_max = min(h - ts, center_ty + scaled_radius)
                x_min = max(0, center_tx - scaled_radius)
                x_max = min(w - ts, center_tx + scaled_radius)

                if y_max < y_min or x_max < x_min:
                    continue

                for ty in range(y_min, y_max + 1, refine_stride):
                    for tx in range(x_min, x_max + 1, refine_stride):
                        if ty == cur_ty and tx == cur_tx and ts == cur_ts:
                            continue

                        candidate = em[ty : ty + ts, tx : tx + ts]
                        if candidate.shape[0] != ts or candidate.shape[1] != ts:
                            continue

                        cost = 0.0
                        skip = False
                        for nt in neighbor_tiles:
                            if nt.size == 0:
                                skip = True
                                break
                            c = _chamfer_cost_common(candidate, nt, common_size)
                            cost += c
                            if cost >= best_cost:
                                skip = True
                                break
                        if skip:
                            continue

                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (ty, tx, ts)

            if best_pos != (cur_ty, cur_tx, cur_ts):
                positions[k] = best_pos
                changed = True

        new_cost = compute_total_cost()
        print(f"    Refinement iteration {iteration + 1}: cost = {new_cost:.2f}")

        if not changed:
            print(f"    Converged after {iteration + 1} iterations.")
            break

    final_cost = compute_total_cost()
    improvement = initial_cost - final_cost
    print(
        f"    Refinement done: {initial_cost:.2f} → {final_cost:.2f} "
        f"(improved by {improvement:.2f}, {100*improvement/(initial_cost + 1e-9):.1f}%)"
    )

    return positions


# ── Main Sequencing Pipeline ────────────────────────────────────────────────


def sequence(
    image_folder,
    output_file="sequence_order_v3.txt",
    cache_file="cost_matrix_cache",
    max_workers=None,
    cache_prefix="",
):
    cache_file = cache_file + cache_prefix + ".pkl"
    print("Step 1/5: Building edge maps...")
    edge_maps = build_edge_maps(image_folder)
    n = len(edge_maps)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    print(f"  Tile ratios: {TILE_RATIOS}")

    if os.path.exists(cache_file):
        print("Step 2/5: Loading cached cost data...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        filenames = cached["filenames"]
        all_pair_costs = cached["all_pair_costs"]
        tile_positions_per_image = cached["tile_positions_per_image"]
        fallback_cost_matrix = cached["fallback_cost_matrix"]

        if set(filenames) != set(edge_maps.keys()):
            print("  Cache stale — recomputing.")
            (
                filenames,
                all_pair_costs,
                best_costs,
                tile_positions_per_image,
                fallback_cost_matrix,
                coarse_dist,
            ) = build_sparse_cost_data(
                edge_maps, max_workers=max_workers, image_folder=image_folder
            )
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "filenames": filenames,
                        "all_pair_costs": all_pair_costs,
                        "tile_positions_per_image": tile_positions_per_image,
                        "fallback_cost_matrix": fallback_cost_matrix,
                    },
                    f,
                )
    else:
        print("Step 2/5: Computing sparse pairwise costs (KNN + all tile pairs)...")
        (
            filenames,
            all_pair_costs,
            best_costs,
            tile_positions_per_image,
            fallback_cost_matrix,
            coarse_dist,
        ) = build_sparse_cost_data(
            edge_maps, max_workers=max_workers, image_folder=image_folder
        )
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "filenames": filenames,
                    "all_pair_costs": all_pair_costs,
                    "tile_positions_per_image": tile_positions_per_image,
                    "fallback_cost_matrix": fallback_cost_matrix,
                },
                f,
            )
    print()

    print("Step 3/5: Tile-aware greedy nearest-neighbor path...")
    path, greedy_cost, tile_positions = greedy_nearest_neighbor_tileaware(
        n, all_pair_costs, fallback_cost_matrix, tile_positions_per_image
    )
    print(f"  Greedy cost: {greedy_cost:.2f}\n")

    print("Step 4/5: Tile-aware 2-opt refinement...")
    path, final_cost, tile_positions = two_opt_tileaware(
        path,
        tile_positions,
        all_pair_costs,
        fallback_cost_matrix,
        tile_positions_per_image,
    )
    print(f"  2-opt cost:  {final_cost:.2f}\n")

    # One final full position optimization pass on the settled path
    print("  Final position optimization on settled path...")
    final_cost = _optimize_positions_for_path(
        path,
        tile_positions,
        all_pair_costs,
        fallback_cost_matrix,
        tile_positions_per_image,
    )
    print(f"  Optimized cost: {final_cost:.2f}\n")

    print("Step 5/5: Fine-grained tile position refinement...")
    refined_positions = refine_tile_positions(
        path, filenames, edge_maps, tile_positions
    )
    print()

    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            ty, tx, ts = refined_positions[k]
            f.write(f"{fn},{ty},{tx},{ts},{h},{w}\n")

    print(f"Sequence written to {output_file}")
    print("Order:")
    for idx in path:
        print(f"  {filenames[idx]}")


# ── Video Generation ────────────────────────────────────────────────────────


def generate_vid(
    sequence_file,
    foto_folder,
    output_path="output_v4.mp4",
    fps=15,
):
    entries = []
    with open(sequence_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            fn = parts[0]
            tile_y, tile_x = int(parts[1]), int(parts[2])
            tile_size_ld = int(parts[3])
            ld_h, ld_w = int(parts[4]), int(parts[5])
            entries.append((fn, tile_y, tile_x, tile_size_ld, ld_h, ld_w))

    if not entries:
        print("No entries in sequence file.")
        return

    frames = []
    for fn, tile_y, tile_x, tile_size_ld, ld_h, ld_w in entries:
        base = os.path.splitext(fn)[0]
        foto_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".bmp"):
            candidate = os.path.join(foto_folder, base + ext)
            if os.path.exists(candidate):
                foto_path = candidate
                break
        if foto_path is None:
            candidate = os.path.join(foto_folder, fn)
            if os.path.exists(candidate):
                foto_path = candidate

        if foto_path is None:
            print(f"  SKIP: no photo for {fn}")
            continue

        img = cv.imread(foto_path)
        if img is None:
            continue

        actual_h, actual_w = img.shape[:2]

        scale = min(actual_h, actual_w) / min(ld_h, ld_w)

        ts_actual = int(tile_size_ld * scale)
        ts_actual = min(ts_actual, actual_h, actual_w)

        sy = max(0, min(int(tile_y * scale), actual_h - ts_actual))
        sx = max(0, min(int(tile_x * scale), actual_w - ts_actual))

        crop = img[sy : sy + ts_actual, sx : sx + ts_actual]
        if crop.size == 0:
            continue
        frames.append(crop)

    if not frames:
        print("No frames.")
        return

    # Use the smallest frame size as output size for consistency
    out_h = min(f.shape[0] for f in frames)
    out_w = min(f.shape[1] for f in frames)
    # Make square based on the minimum dimension
    out_size = min(out_h, out_w)

    writer = cv.VideoWriter(
        output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (out_size, out_size)
    )
    for frame in frames:
        writer.write(cv.resize(frame, (out_size, out_size)))
    writer.release()
    print(f"Video saved: {output_path} ({len(frames)} frames @ {fps}fps)")


def preview_edges(
    image_folder, num_samples=6, extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp")
):
    """
    Display a random sample of images alongside their edge maps for visual
    inspection. Useful for tuning EDGE_GAMMA, EDGE_TOPK_PERCENTILE, etc.

    Args:
        image_folder: path to folder containing images
        num_samples: how many images to show (default 6)
    """
    import matplotlib.pyplot as plt

    all_files = sorted(
        fn
        for fn in os.listdir(image_folder)
        if os.path.splitext(fn)[1].lower() in extensions
    )

    if not all_files:
        print("No images found.")
        return

    rng = np.random.default_rng()
    num_samples = min(num_samples, len(all_files))
    chosen = rng.choice(all_files, size=num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, fn in enumerate(chosen):
        path = os.path.join(image_folder, fn)

        # Original image
        img_bgr = cv.imread(path)
        if img_bgr is None:
            print(f"  Cannot read {fn}, skipping.")
            continue
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # Edge map
        edge_map = get_weighted_edges(path)

        # Count stats for the title
        nonzero_pct = 100.0 * np.count_nonzero(edge_map) / edge_map.size
        edge_max = edge_map.max()
        edge_mean = edge_map[edge_map > 0].mean() if np.any(edge_map > 0) else 0.0

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(fn, fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(edge_map, cmap="hot", vmin=0, vmax=max(edge_max, 1e-6))
        axes[row, 1].set_title(
            f"Edges — {nonzero_pct:.1f}% nonzero, max={edge_max:.3f}, mean={edge_mean:.3f}\n"
            f"(method={EDGE_METHOD}, gamma={EDGE_GAMMA}, min_len={MIN_EDGE_LENGTH}, thresh={EDGE_THRESHOLD_LOW})",
            fontsize=9,
        )
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.suptitle(
        f"Edge Preview — {num_samples} random images from {image_folder}",
        fontsize=13,
        y=1.01,
    )
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        foto_folder = sys.argv[2] if len(sys.argv) > 2 else "."
        generate_vid("sequence_order_v3.txt", foto_folder)
    elif len(sys.argv) > 1 and sys.argv[1] == "preview":
        image_folder = sys.argv[2] if len(sys.argv) > 2 else "."
        num = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        preview_edges(image_folder, num_samples=num)
    else:
        image_folder = sys.argv[1] if len(sys.argv) > 1 else "."
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
        d = sys.argv[3] if len(sys.argv) > 3 else ""
        sequence(image_folder, max_workers=workers, cache_prefix=d)

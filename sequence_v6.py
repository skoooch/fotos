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
import json
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
T_R_MIN = 0.7
T_R_STRIDE = 0.1
STRIDE = 50  # coarser stride for expanded tile-pair search
EDGE_BLUR_KSIZE = 3
EDGE_GAMMA = 1.5  # >1 suppresses weak edges, amplifies strong ones
MIN_EDGE_LENGTH = 30  # minimum connected-component size in pixels to keep
EDGE_THRESHOLD_LOW = 0.6  # discard edge pixels below this strength
NUM_2OPT_ITERATIONS = 50
K_NEIGHBORS = 35  # only tile-match the K most promising neighbors per image
TARGET_SHORT_EDGE = 512
EDGE_THRESHOLD = 0.1
MIN_EDGE_DENSITY = 0.01
REFINE_STRIDE = 5  # fine-grained stride for tile refinement
REFINE_RADIUS = 25  # search radius (pixels) around current tile position
REFINE_ITERATIONS = 2  # number of forward+backward sweeps
DISTANCE_METRIC = "embedding"  # "edge_descript", "embedding", or "combined"
EMBEDDING_WEIGHT = 0.5  # weight for embedding distance in combined mode
EDGE_METHOD = "hed"  # "hed", "scharr"
NUM_LOOKBACK = 4
LOOKBACK_EXP = 0.7
FACE_ALIGN_WEIGHT = 0.3  # how much face alignment reduces cost (fraction of Chamfer)

# ── Multi-signal cost weights ───────────────────────────────────────────────
CHAMFER_WEIGHT = 0.50  # edge structure continuity
CLIP_WEIGHT = 0.30  # semantic continuity
COLOR_WEIGHT = 0.20  # tonal/palette continuity

PREPROCESSED_SEQS = []
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

    Returns:
        magnitude: float32 edge map
        scale: float, the scale factor from original image to edge map coords
    """
    if EDGE_METHOD == "hed":
        img_bgr = cv.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        h, w = img_bgr.shape[:2]
        scale = target_short_edge / min(h, w)
        magnitude = _hed_edge_map(img_bgr, target_short_edge)
    else:
        magnitude = _scharr_edge_map(image_path, target_short_edge)
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        h, w = img.shape
        scale = target_short_edge / min(h, w)

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

    return magnitude.astype(np.float32), scale


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


def compute_color_histograms(
    filenames, image_folder, target_short_edge=TARGET_SHORT_EDGE
):
    """
    Compute per-image color histograms in LAB space for tonal similarity.

    LAB is perceptually uniform, so histogram distance correlates with
    how different two images *look* in terms of color/tone.

    Returns:
        np.ndarray of shape (n, num_bins * 3), L2-normalized
    """
    num_bins = 32
    histograms = []
    print(f"  Computing color histograms for {len(filenames)} images...")

    for fn in filenames:
        path = os.path.join(image_folder, fn)
        try:
            img = cv.imread(path)
            if img is None:
                histograms.append(np.zeros(num_bins * 3, dtype=np.float32))
                continue

            h, w = img.shape[:2]
            scale = target_short_edge / min(h, w)
            img = cv.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA
            )

            lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

            hist = []
            # L channel: [0, 256]
            h_l = cv.calcHist([lab], [0], None, [num_bins], [0, 256]).flatten()
            # A channel: [0, 256]
            h_a = cv.calcHist([lab], [1], None, [num_bins], [0, 256]).flatten()
            # B channel: [0, 256]
            h_b = cv.calcHist([lab], [2], None, [num_bins], [0, 256]).flatten()

            combined = np.concatenate([h_l, h_a, h_b]).astype(np.float32)
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined /= norm
            histograms.append(combined)
        except Exception as e:
            print(f"    Color hist skip {fn}: {e}")
            histograms.append(np.zeros(num_bins * 3, dtype=np.float32))

    return np.array(histograms, dtype=np.float32)


def compute_tile_color_histogram(img_bgr_tile, num_bins=32):
    """
    Compute a normalized LAB color histogram for a single BGR tile crop.
    Returns a float32 vector of length num_bins * 3.
    """
    if img_bgr_tile is None or img_bgr_tile.size == 0:
        return np.zeros(num_bins * 3, dtype=np.float32)

    lab = cv.cvtColor(img_bgr_tile, cv.COLOR_BGR2LAB)
    h_l = cv.calcHist([lab], [0], None, [num_bins], [0, 256]).flatten()
    h_a = cv.calcHist([lab], [1], None, [num_bins], [0, 256]).flatten()
    h_b = cv.calcHist([lab], [2], None, [num_bins], [0, 256]).flatten()

    combined = np.concatenate([h_l, h_a, h_b]).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    return combined


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
def _face_alignment_bonus(ay, ax, a_ts, by, bx, b_ts, bboxes_a, bboxes_b):
    """
    Compute a cost reduction for face alignment between two tiles.

    For each face in image A's tile, find the best-matching face in image B's tile.
    Measure how closely the face centers align (in normalized tile coordinates).

    Returns:
        bonus: float >= 0. Higher = better alignment = more cost reduction.
               0 if no faces in either tile.
    """

    def faces_in_tile(bboxes, ty, tx, ts):
        results = []
        for bbox in bboxes:
            fx, fy, fw, fh, conf = (
                bbox["x"],
                bbox["y"],
                bbox["w"],
                bbox["h"],
                bbox["confidence"],
            )
            fcx = fx + fw / 2.0
            fcy = fy + fh / 2.0
            if ty <= fcy <= ty + ts and tx <= fcx <= tx + ts:
                cx_norm = (fcx - tx) / ts
                cy_norm = (fcy - ty) / ts
                size_norm = (fw * fh) / (ts * ts)
                results.append((cx_norm, cy_norm, size_norm, conf))
        return results

    faces_a = faces_in_tile(bboxes_a, ay, ax, a_ts)
    faces_b = faces_in_tile(bboxes_b, by, bx, b_ts)

    total_align = 0.0
    matched = 0

    for cxa, cya, sa, conf_a in faces_a:
        best_dist = inf
        best_size_match = 0.0
        best_conf_weight = 0.0
        for cxb, cyb, sb, conf_b in faces_b:
            # Raw positional distance (confidence applied to the BONUS, not the distance)
            dist = np.sqrt((cxa - cxb) ** 2 + (cya - cyb) ** 2)
            if dist < best_dist:
                best_dist = dist
                if sa > 0 and sb > 0:
                    ratio = min(sa, sb) / max(sa, sb)
                else:
                    ratio = 0.0
                best_size_match = ratio
                # Higher confidence = this face matters more = bigger bonus when aligned
                best_conf_weight = conf_a * conf_b

        if best_dist < 1.0:
            position_score = max(0.0, 1.0 - best_dist)
            align_score = position_score * (0.7 + 0.3 * best_size_match)
            # Confidence amplifies the bonus: confident faces produce stronger alignment signal
            align_score *= best_conf_weight
            total_align += align_score
            matched += 1

    if matched > 0:
        total_align /= matched

    return total_align * FACE_ALIGN_WEIGHT


# Scale face bboxes from original image coords to edge map coords
def scale_bboxes(bboxes, scale):
    """Scale face bboxes from original image coords to edge map coords."""
    if not bboxes:
        return []
    return [
        {
            "x": b["x"] * scale,
            "y": b["y"] * scale,
            "w": b["w"] * scale,
            "h": b["h"] * scale,
            "confidence": b.get("confidence", 1.0),
        }
        for b in bboxes
    ]


def find_all_tile_pair_costs(
    edge_a,
    edge_b,
    bboxes_i,
    bboxes_j,
    scale_i,
    scale_j,
    clip_dist_ij=0.0,
    img_path_a=None,
    img_path_b=None,
    tile_ratios=TILE_RATIOS,
    stride=STRIDE,
):
    """
    Compute multi-signal cost for ALL valid tile position combinations.

    Cost = CHAMFER_WEIGHT * chamfer (edge continuity)
         + CLIP_WEIGHT * clip_dist (semantic continuity, same for all tiles of this pair)
         + COLOR_WEIGHT * color_dist (tonal continuity, per tile pair)
         - face alignment bonus (multiplicative reduction)
    """
    h_a, w_a = edge_a.shape
    h_b, w_b = edge_b.shape

    min_dim = min(h_a, w_a, h_b, w_b)
    common_size = int(min_dim * min(tile_ratios))
    if common_size < 16:
        common_size = 16

    scaled_a = scale_bboxes(bboxes_i, scale_i)
    scaled_b = scale_bboxes(bboxes_j, scale_j)
    has_faces = bool(scaled_a) and bool(scaled_b)

    positions_a = _tile_positions_multi(h_a, w_a, tile_ratios, stride)
    positions_b = _tile_positions_multi(h_b, w_b, tile_ratios, stride)

    tiles_a = _precompute_tile_data_multi(edge_a, positions_a, common_size)
    tiles_b = _precompute_tile_data_multi(edge_b, positions_b, common_size)

    if not tiles_a or not tiles_b:
        return {}, inf, None, None

    # Load full images for color histogram computation (once per pair)
    img_bgr_a = cv.imread(img_path_a) if img_path_a else None
    img_bgr_b = cv.imread(img_path_b) if img_path_b else None

    # Precompute color histograms for all tile positions
    color_hists_a = {}
    color_hists_b = {}

    if img_bgr_a is not None and COLOR_WEIGHT > 0:
        actual_h_a, actual_w_a = img_bgr_a.shape[:2]
        inv_scale_a = 1.0 / scale_i
        for ay, ax, a_ts, *_ in tiles_a:
            key = (ay, ax, a_ts)
            # Map tile coords from edge-map space to original image space
            oa_ts = int(a_ts * inv_scale_a)
            oa_y = int(ay * inv_scale_a)
            oa_x = int(ax * inv_scale_a)
            oa_ts = max(16, min(oa_ts, actual_h_a, actual_w_a))
            oa_y = max(0, min(oa_y, actual_h_a - oa_ts))
            oa_x = max(0, min(oa_x, actual_w_a - oa_ts))
            crop = img_bgr_a[oa_y : oa_y + oa_ts, oa_x : oa_x + oa_ts]
            # Resize to consistent size for histogram comparison
            crop = cv.resize(crop, (128, 128), interpolation=cv.INTER_AREA)
            color_hists_a[key] = compute_tile_color_histogram(crop)

    if img_bgr_b is not None and COLOR_WEIGHT > 0:
        actual_h_b, actual_w_b = img_bgr_b.shape[:2]
        inv_scale_b = 1.0 / scale_j
        for by, bx, b_ts, *_ in tiles_b:
            key = (by, bx, b_ts)
            ob_ts = int(b_ts * inv_scale_b)
            ob_y = int(by * inv_scale_b)
            ob_x = int(bx * inv_scale_b)
            ob_ts = max(16, min(ob_ts, actual_h_b, actual_w_b))
            ob_y = max(0, min(ob_y, actual_h_b - ob_ts))
            ob_x = max(0, min(ob_x, actual_w_b - ob_ts))
            crop = img_bgr_b[ob_y : ob_y + ob_ts, ob_x : ob_x + ob_ts]
            crop = cv.resize(crop, (128, 128), interpolation=cv.INTER_AREA)
            color_hists_b[key] = compute_tile_color_histogram(crop)

    costs = {}
    best_cost = inf
    best_pos_a = (tiles_a[0][0], tiles_a[0][1], tiles_a[0][2])
    best_pos_b = (tiles_b[0][0], tiles_b[0][1], tiles_b[0][2])

    for ay, ax, a_ts, mask_a, weights_a, sum_wa, dt_a in tiles_a:
        hist_a = color_hists_a.get((ay, ax, a_ts))

        for by, bx, b_ts, mask_b, weights_b, sum_wb, dt_b in tiles_b:
            # Signal 1: Chamfer distance (edge structure continuity)
            d_ab = np.sum(weights_a * dt_b[mask_a]) / sum_wa
            d_ba = np.sum(weights_b * dt_a[mask_b]) / sum_wb
            chamfer = (d_ab + d_ba) / 2.0

            # Signal 2: CLIP distance (semantic continuity — same for all tiles)
            # Already normalized to [0, 2] (cosine distance)
            clip_term = clip_dist_ij

            # Signal 3: Color histogram distance (tonal continuity — per tile)
            color_dist = 0.0
            hist_b = color_hists_b.get((by, bx, b_ts))
            if hist_a is not None and hist_b is not None:
                # Cosine distance between normalized histograms: [0, 2]
                dot = np.dot(hist_a, hist_b)
                color_dist = 1.0 - dot  # [0, 2], typically [0, 1]

            # Combine signals (normalize chamfer to similar scale as others)
            # Chamfer is in pixels (~0-50 range), CLIP/color are in [0, 2]
            # We normalize chamfer by dividing by common_size to get ~[0, 1]
            chamfer_norm = chamfer / max(common_size, 1)

            cost = (
                CHAMFER_WEIGHT * chamfer_norm
                + CLIP_WEIGHT * clip_term
                + COLOR_WEIGHT * color_dist
            )

            # Face alignment bonus (multiplicative reduction on combined cost)
            if has_faces:
                bonus = _face_alignment_bonus(
                    ay, ax, a_ts, by, bx, b_ts, scaled_a, scaled_b
                )
                cost *= 1.0 - min(bonus, 0.9)

            costs[((ay, ax, a_ts), (by, bx, b_ts))] = cost
            if cost < best_cost:
                best_cost = cost
                best_pos_a = (ay, ax, a_ts)
                best_pos_b = (by, bx, b_ts)

    return costs, best_cost, best_pos_a, best_pos_b


def _compute_all_pairs(args):
    """Worker for parallel all-tile-pair matching."""
    (
        i,
        j,
        edge_i,
        edge_j,
        bboxes_i,
        bboxes_j,
        scale_i,
        scale_j,
        clip_dist_ij,
        path_i,
        path_j,
    ) = args
    costs, best_cost, pos_a, pos_b = find_all_tile_pair_costs(
        edge_i,
        edge_j,
        bboxes_i,
        bboxes_j,
        scale_i,
        scale_j,
        clip_dist_ij=clip_dist_ij,
        img_path_a=path_i,
        img_path_b=path_j,
    )
    return i, j, costs, best_cost, pos_a, pos_b


# ── Pairwise Cost Matrix (Sparse, KNN-based) ───────────────────────────────


def build_edge_maps(image_folder, extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp")):
    """Load all images from a folder and compute weighted edge maps."""
    edge_maps = {}  # fn -> edge_map (float32)
    edge_scales = {}  # fn -> scale factor (float)
    for fn in sorted(os.listdir(image_folder)):
        if os.path.splitext(fn)[1].lower() in extensions:
            path = os.path.join(image_folder, fn)
            try:
                em, sc = get_weighted_edges(path)
                edge_maps[fn] = em
                edge_scales[fn] = sc
                print(f"  edges: {fn}  shape={em.shape}  scale={sc:.4f}")
            except Exception as e:
                print(f"  SKIP {fn}: {e}")
        elif "seq" in fn and os.path.isdir(os.path.join(image_folder, fn)):
            seq_dir_path = os.path.join(image_folder, fn)
            seq_files = os.listdir(seq_dir_path)
            first_last = []
            seq_path = os.path.join(seq_dir_path, "sequence.txt")
            if not os.path.exists(seq_path):
                # Auto-generate sequence file for the sub-folder
                with open(seq_path, "w") as sf:
                    for file in sorted(seq_files):
                        if os.path.splitext(file)[1].lower() in extensions:
                            file_path = os.path.join(seq_dir_path, file)
                            try:
                                with Image.open(file_path) as img:
                                    width, height = img.size
                                    ts = min(width, height)
                                    if width > height:
                                        sf.write(
                                            f"{file},{0},{(width - ts) // 2},{ts},{height},{width}\n"
                                        )
                                    else:
                                        sf.write(
                                            f"{file},{(height - ts) // 2},{0},{ts},{height},{width}\n"
                                        )
                            except Exception as e:
                                print(f"  SKIP sub-seq file {file}: {e}")
            with open(seq_path, "r") as sf:
                cleaned = [line for line in sf if line.strip()]
            if len(cleaned) >= 2:
                for line in [cleaned[0], cleaned[-1]]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    sq_fn = parts[0]
                    sq_file_path = os.path.join(seq_dir_path, sq_fn)
                    if not os.path.exists(sq_file_path):
                        print(f"  SKIP sub-seq image (not found): {sq_file_path}")
                        continue
                    try:
                        sq_em, sq_sc = get_weighted_edges(sq_file_path)
                        # Store with folder-qualified name to avoid collisions
                        qualified_fn = os.path.join(fn, sq_fn)
                        edge_maps[qualified_fn] = sq_em
                        edge_scales[qualified_fn] = sq_sc
                        tile_y, tile_x = int(parts[1]), int(parts[2])
                        tile_size_ld = int(parts[3])
                        ld_h, ld_w = int(parts[4]), int(parts[5])
                        first_last.append(
                            (qualified_fn, tile_y, tile_x, tile_size_ld, ld_h, ld_w)
                        )
                        print(
                            f"  sub-seq edge: {qualified_fn}  shape={sq_em.shape}  "
                            f"tile=({tile_y},{tile_x},{tile_size_ld})"
                        )
                    except Exception as e:
                        print(f"  SKIP sub-seq {sq_fn}: {e}")
            if len(first_last) == 2:
                PREPROCESSED_SEQS.append((fn, tuple(first_last)))
                print(
                    f"  Registered pre-sequenced group: {fn} "
                    f"({first_last[0][0]} → {first_last[1][0]})"
                )
            elif len(first_last) == 1:
                # Single-image sub-sequence — just treat as a regular image
                print(f"  Sub-sequence {fn} has only 1 image, treating as regular.")
            else:
                print(f"  Sub-sequence {fn} has no valid images, skipping.")
    return edge_maps, edge_scales


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
        clip_embeddings: np.ndarray of shape (n, embed_dim) or None
    """
    filenames = list(edge_maps.keys())
    n = len(filenames)
    clip_embeddings = None
    if distance_metric == "embedding":
        if image_folder is None:
            raise ValueError("image_folder required for embedding distance metric")
        clip_embeddings = compute_image_features(filenames, image_folder)
        print(f"  Computing cosine distance matrix from CLIP embeddings...")
        dist_matrix = cdist(clip_embeddings, clip_embeddings, metric="cosine")
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    # If we didn't compute CLIP yet (edge_descript mode), do it now for cost function
    if clip_embeddings is None and image_folder is not None:
        print(f"  Computing CLIP embeddings for multi-signal cost...")
        clip_embeddings = compute_image_features(filenames, image_folder)

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

    return filenames, symmetric_pairs, neighbors, dist_matrix, clip_embeddings


def build_sparse_cost_data(
    edge_maps, edge_scales, max_workers=None, k=K_NEIGHBORS, image_folder=None
):
    """
    1. Build KNN shortlist using cheap descriptors.
    2. Run expensive ALL tile-pair matching on shortlisted pairs (multi-ratio).
    3. Compute fallback costs for non-shortlisted pairs.
    """
    filenames, shortlisted_pairs, neighbors, coarse_dist, clip_embeddings = (
        build_knn_shortlist(edge_maps, k=k, image_folder=image_folder)
    )

    # Build index mapping for preprocessed sequences
    fn_to_idx = {fn: i for i, fn in enumerate(filenames)}
    forced_edges = []  # list of (first_idx, last_idx, first_pos, last_pos)
    forced_edge_set = set()  # set of (first_idx, last_idx) for quick lookup

    for seq_name, (first_info, last_info) in PREPROCESSED_SEQS:
        first_fn, f_ty, f_tx, f_ts, f_h, f_w = first_info
        last_fn, l_ty, l_tx, l_ts, l_h, l_w = last_info

        first_idx = fn_to_idx.get(first_fn)
        last_idx = fn_to_idx.get(last_fn)

        if first_idx is None or last_idx is None:
            print(
                f"  WARNING: sub-seq {seq_name} endpoints not in filenames, skipping. "
                f"first={first_fn} ({first_idx}), last={last_fn} ({last_idx})"
            )
            continue

        # Convert tile positions from sequence-file space (ld_h, ld_w) to edge-map space
        # The edge map is at TARGET_SHORT_EDGE scale; the sequence file stores at (ld_h, ld_w)
        em_first = edge_maps[first_fn]
        em_last = edge_maps[last_fn]
        em_fh, em_fw = em_first.shape
        em_lh, em_lw = em_last.shape

        # Scale from sequence-file coords to edge-map coords
        f_scale = em_fh / f_h if f_h > 0 else 1.0
        l_scale = em_lh / l_h if l_h > 0 else 1.0

        first_pos = (
            int(f_ty * f_scale),
            int(f_tx * f_scale),
            int(f_ts * f_scale),
        )
        last_pos = (
            int(l_ty * l_scale),
            int(l_tx * l_scale),
            int(l_ts * l_scale),
        )

        # Clamp to valid range
        first_pos = (
            max(0, min(first_pos[0], em_fh - first_pos[2])),
            max(0, min(first_pos[1], em_fw - first_pos[2])),
            max(16, min(first_pos[2], em_fh, em_fw)),
        )
        last_pos = (
            max(0, min(last_pos[0], em_lh - last_pos[2])),
            max(0, min(last_pos[1], em_lw - last_pos[2])),
            max(16, min(last_pos[2], em_lh, em_lw)),
        )

        forced_edges.append((first_idx, last_idx, first_pos, last_pos))
        forced_edge_set.add((first_idx, last_idx))

        # Ensure this pair is shortlisted for tile-pair cost computation
        pair_key = (min(first_idx, last_idx), max(first_idx, last_idx))
        shortlisted_pairs.add(pair_key)

        # Also ensure neighbors of first/last include each other
        if last_idx not in neighbors.get(first_idx, []):
            neighbors.setdefault(first_idx, []).append(last_idx)
        if first_idx not in neighbors.get(last_idx, []):
            neighbors.setdefault(last_idx, []).append(first_idx)

        print(
            f"  Forced edge: {first_fn}[{first_idx}] → {last_fn}[{last_idx}] "
            f"(seq: {seq_name})"
        )

    # Precompute pairwise CLIP distances for shortlisted pairs
    clip_dist_cache = {}
    if clip_embeddings is not None:
        for i, j in shortlisted_pairs:
            # Cosine distance: 1 - dot(a, b) for L2-normalized vectors
            clip_dist_cache[(i, j)] = float(
                1.0 - np.dot(clip_embeddings[i], clip_embeddings[j])
            )

    _unload_clip_model()
    n = len(filenames)
    with open("face_bboxes.json") as f:
        face_bboxes = json.load(f)

    all_pair_costs = {}
    best_costs = {}

    tile_positions_per_image = {}
    for idx, fn in enumerate(filenames):
        positions = get_tile_positions_for_image(edge_maps[fn])
        tile_positions_per_image[idx] = positions

    # For forced-edge endpoints, ensure their locked tile position is in the
    # valid positions list (so the solver can find it)
    for first_idx, last_idx, first_pos, last_pos in forced_edges:
        if first_pos not in tile_positions_per_image.get(first_idx, []):
            tile_positions_per_image.setdefault(first_idx, []).append(first_pos)
        if last_pos not in tile_positions_per_image.get(last_idx, []):
            tile_positions_per_image.setdefault(last_idx, []).append(last_pos)

    # Resolve image_folder for sub-sequence images
    def _resolve_image_path(fn):
        """Resolve a filename to its full path, handling sub-folder qualified names."""
        if image_folder is None:
            return None
        # For qualified names like "seq_folder/image.jpg"
        full_path = os.path.join(image_folder, fn)
        if os.path.exists(full_path):
            return full_path
        # Try without qualification
        base = os.path.basename(fn)
        fallback = os.path.join(image_folder, base)
        if os.path.exists(fallback):
            return fallback
        return full_path  # return anyway, let downstream handle missing

    tasks = [
        (
            i,
            j,
            edge_maps[filenames[i]],
            edge_maps[filenames[j]],
            face_bboxes.get(filenames[i], []),
            face_bboxes.get(filenames[j], []),
            edge_scales[filenames[i]],
            edge_scales[filenames[j]],
            clip_dist_cache.get((i, j), 0.5),
            _resolve_image_path(filenames[i]),
            _resolve_image_path(filenames[j]),
        )
        for i, j in shortlisted_pairs
    ]

    total = len(tasks)
    print(
        f"  All-tile-pair matching {total} shortlisted pairs (of {n*(n-1)//2} total)..."
    )
    print(f"  Tile ratios: {TILE_RATIOS}")
    print(
        f"  Cost weights: chamfer={CHAMFER_WEIGHT}, clip={CLIP_WEIGHT}, color={COLOR_WEIGHT}"
    )
    print(
        f"  Avg tile positions per image: {np.mean([len(v) for v in tile_positions_per_image.values()]):.1f}"
    )
    if forced_edges:
        print(f"  Forced edges (pre-sequenced): {len(forced_edges)}")

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

    # ...existing code (fallback cost matrix building)...
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
        forced_edges,
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
    i, j, pos_i, all_pair_costs, fallback_cost_matrix, tile_positions_j, pos_tails
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

        base_cost = pair_costs.get(tile_key)
        if base_cost is None:
            continue

        tail_cost = 0
        for tail_i, pos_tail_tuple in enumerate(pos_tails):
            if pos_tail_tuple is not None:
                (tail, pos_tail) = pos_tail_tuple
                tail_key = (min(tail, j), max(tail, j))
                tail_pair_costs = all_pair_costs.get(tail_key)
                if tail_pair_costs is not None:
                    if tail < j:
                        tile_key_tail = (pos_tail, pos_j)
                    else:
                        tile_key_tail = (pos_j, pos_tail)
                    tail_cost_add = tail_pair_costs.get(tile_key_tail)
                    tail_cost += (tail_cost_add if tail_cost_add is not None else fallback_cost_matrix[tail, j]) * (LOOKBACK_EXP**(tail_i + 1))
                else:
                    tail_cost += fallback_cost_matrix[tail, j] * (LOOKBACK_EXP**(tail_i + 1))
        
        cost = base_cost + tail_cost
        if cost < best_cost:
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
    forced_edges=None,
):
    """
    Nearest-neighbor heuristic that tracks each image's current tile position.
    Tile positions are (y, x, tile_size) tuples.

    Forced edges: when we visit a forced-edge start node (first_idx), the
    next node MUST be the corresponding last_idx, using the locked tile
    positions from the sub-sequence file.

    Returns:
        best_path: list of image indices
        best_total: float total cost
        best_positions: dict image_idx -> (ty, tx, tile_size)
    """
    if forced_edges is None:
        forced_edges = []

    # Build lookup: first_idx -> (last_idx, first_pos, last_pos)
    forced_map = {}
    # Also build reverse: last_idx -> first_idx (so we never start from a last node
    # without its first node)
    forced_last_to_first = {}
    locked_positions = {}  # idx -> pos (locked tile positions for forced endpoints)

    for first_idx, last_idx, first_pos, last_pos in forced_edges:
        forced_map[first_idx] = (last_idx, first_pos, last_pos)
        forced_last_to_first[last_idx] = first_idx
        locked_positions[first_idx] = first_pos
        locked_positions[last_idx] = last_pos

    best_path = None
    best_total = inf
    best_positions = None

    rng = np.random.default_rng(42)
    num_starts = min(50, n) if n > 200 else n
    start_nodes = rng.choice(n, size=num_starts, replace=False) if n > 200 else range(n)

    for si, start in enumerate(start_nodes):
        start = int(start)

        # Don't start from a forced-edge "last" node — we'd need its "first" before it
        if start in forced_last_to_first:
            # Start from its first instead
            start = forced_last_to_first[start]

        visited = {start}
        path = [start]
        total = 0.0
        current = start
        cur_positions = {}  # image_idx -> (ty, tx, tile_size)
        pos_tails = [None for i in range(NUM_LOOKBACK)]

        # Apply locked position for start if it's a forced endpoint
        if start in locked_positions:
            cur_positions[start] = locked_positions[start]

        # If start is a forced-edge first, immediately add the last
        if start in forced_map:
            last_idx, first_pos, last_pos = forced_map[start]
            cur_positions[start] = first_pos
            cur_positions[last_idx] = last_pos
            # Cost of this forced edge
            cost = lookup_cost(
                start,
                last_idx,
                first_pos,
                last_pos,
                all_pair_costs,
                fallback_cost_matrix,
            )
            total += cost
            path.append(last_idx)
            visited.add(last_idx)
            pos_tails[0] = (start, first_pos)
            current = last_idx

        for _ in range(n - len(visited)):
            best_nxt = None
            best_nxt_cost = inf
            best_nxt_pos_cur = None
            best_nxt_pos_nxt = None

            cur_pos = cur_positions.get(current)

            for j in range(n):
                if j in visited:
                    continue

                # Skip forced-edge "last" nodes — they can only be reached
                # via their "first" node
                if j in forced_last_to_first and forced_last_to_first[j] not in visited:
                    continue

                if cur_pos is not None:
                    # If j has a locked position (forced endpoint), use it
                    if j in locked_positions:
                        pos_j = locked_positions[j]
                        cost = lookup_cost(
                            current,
                            j,
                            cur_pos,
                            pos_j,
                            all_pair_costs,
                            fallback_cost_matrix,
                        )
                        if cost < best_nxt_cost:
                            best_nxt_cost = cost
                            best_nxt = j
                            best_nxt_pos_cur = cur_pos
                            best_nxt_pos_nxt = pos_j
                    else:
                        cost, pos_j = best_cost_for_arrival(
                            current,
                            j,
                            cur_pos,
                            all_pair_costs,
                            fallback_cost_matrix,
                            tile_positions_per_image.get(j, []),
                            pos_tails,
                        )
                        if cost < best_nxt_cost:
                            best_nxt_cost = cost
                            best_nxt = j
                            best_nxt_pos_cur = cur_pos
                            best_nxt_pos_nxt = pos_j
                else:
                    if j in locked_positions:
                        pos_j = locked_positions[j]
                        # Find best pos for current given j is locked
                        best_pair_cost = inf
                        best_pair_pos_c = None
                        for pos_c in tile_positions_per_image.get(current, []):
                            c = lookup_cost(
                                current,
                                j,
                                pos_c,
                                pos_j,
                                all_pair_costs,
                                fallback_cost_matrix,
                            )
                            if c < best_pair_cost:
                                best_pair_cost = c
                                best_pair_pos_c = pos_c
                        if best_pair_cost < best_nxt_cost:
                            best_nxt_cost = best_pair_cost
                            best_nxt = j
                            best_nxt_pos_cur = best_pair_pos_c
                            best_nxt_pos_nxt = pos_j
                    else:
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
            for lookback_i in range(NUM_LOOKBACK - 1, 0, -1):
                pos_tails[lookback_i] = pos_tails[lookback_i - 1]
            pos_tails[0] = (current, best_nxt_pos_cur)
            total += best_nxt_cost
            path.append(best_nxt)
            visited.add(best_nxt)
            current = best_nxt

            # If we just arrived at a forced-edge first node, immediately
            # traverse to its last node
            if current in forced_map and forced_map[current][0] not in visited:
                last_idx, first_pos, last_pos = forced_map[current]
                cur_positions[current] = first_pos
                cur_positions[last_idx] = last_pos
                cost = lookup_cost(
                    current,
                    last_idx,
                    first_pos,
                    last_pos,
                    all_pair_costs,
                    fallback_cost_matrix,
                )
                total += cost
                for lookback_i in range(NUM_LOOKBACK - 1, 0, -1):
                    pos_tails[lookback_i] = pos_tails[lookback_i - 1]
                pos_tails[0] = (current, first_pos)
                path.append(last_idx)
                visited.add(last_idx)
                current = last_idx

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
    path, positions, all_pair_costs, fallback_cost_matrix, tile_positions_per_image
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
    forced_edges=None,
):
    """
    Improve a path by repeatedly reversing sub-segments, with tile-position
    awareness and NUM_LOOKBACK cost evaluation.

    Forced edges are never broken: if a reversal would separate a forced
    first→last pair or reverse their order, it is skipped.
    """
    if forced_edges is None:
        forced_edges = []

    # Build set of forced (first, last) pairs and locked positions
    forced_pairs = set()
    locked_positions = {}
    for first_idx, last_idx, first_pos, last_pos in forced_edges:
        forced_pairs.add((first_idx, last_idx))
        locked_positions[first_idx] = first_pos
        locked_positions[last_idx] = last_pos

    def _violates_forced_edges(candidate_path):
        """Check if a candidate path breaks any forced edge constraints."""
        for first_idx, last_idx in forced_pairs:
            try:
                fi = candidate_path.index(first_idx)
                li = candidate_path.index(last_idx)
            except ValueError:
                return True  # missing node
            if li != fi + 1:
                return True  # must be immediately adjacent, first before last
        return False

    best = path[:]
    best_positions = dict(positions)
    best_cost = _path_cost_tileaware(
        best, best_positions, all_pair_costs, fallback_cost_matrix
    )

    def _window_cost(p, pos, start, end):
        """Cost of edges in p[start..end]."""
        c = 0.0
        for k in range(start, end):
            if k < 0 or k + 1 >= len(p):
                continue
            c += lookup_cost(
                p[k],
                p[k + 1],
                pos.get(p[k]),
                pos.get(p[k + 1]),
                all_pair_costs,
                fallback_cost_matrix,
            )
        return c

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                # Evaluate the full affected window including lookback
                window_start = max(0, i - NUM_LOOKBACK)
                window_end = min(len(best) - 1, j + NUM_LOOKBACK)

                old_window_cost = _window_cost(
                    best, best_positions, window_start, window_end
                )

                # Build candidate with reversed segment
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]

                # Check forced edge constraints
                if forced_pairs and _violates_forced_edges(candidate):
                    continue

                candidate_positions = dict(best_positions)

                # Re-optimize tile positions at boundary + lookback nodes
                # but NOT for locked positions
                boundary_indices = set()
                for offset in range(NUM_LOOKBACK + 1):
                    if i - offset >= 0:
                        boundary_indices.add(i - offset)
                    if i + offset < len(candidate):
                        boundary_indices.add(i + offset)
                    if j - offset >= 0:
                        boundary_indices.add(j - offset)
                    if j + offset < len(candidate):
                        boundary_indices.add(j + offset)

                for k in sorted(boundary_indices):
                    idx = candidate[k]

                    # Don't change locked positions
                    if idx in locked_positions:
                        candidate_positions[idx] = locked_positions[idx]
                        continue

                    candidates_pos = tile_positions_per_image.get(idx, [])
                    if not candidates_pos:
                        continue

                    best_pos = candidate_positions.get(idx)
                    best_local_cost = inf

                    for pos in candidates_pos:
                        cost = 0.0
                        for offset in range(1, NUM_LOOKBACK + 1):
                            if k - offset >= 0:
                                nb = candidate[k - offset]
                                cost += lookup_cost(
                                    nb, idx,
                                    candidate_positions.get(nb), pos,
                                    all_pair_costs, fallback_cost_matrix,
                                ) * (LOOKBACK_EXP ** (offset - 1))
                            if k + offset < len(candidate):
                                nb = candidate[k + offset]
                                cost += lookup_cost(
                                    idx, nb,
                                    pos, candidate_positions.get(nb),
                                    all_pair_costs, fallback_cost_matrix,
                                ) * (LOOKBACK_EXP ** (offset - 1))

                        if cost < best_local_cost:
                            best_local_cost = cost
                            best_pos = pos

                    if best_pos is not None:
                        candidate_positions[idx] = best_pos

                new_window_cost = _window_cost(
                    candidate, candidate_positions, window_start, window_end
                )

                if new_window_cost < old_window_cost:
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
    edge_scales=None,
    face_bboxes=None,
    tile_ratios=TILE_RATIOS,
    refine_stride=REFINE_STRIDE,
    refine_radius=REFINE_RADIUS,
    num_iterations=REFINE_ITERATIONS,
    forced_edges=None,
):

    n = len(path)

    if face_bboxes is None:
        face_bboxes = {}
    if edge_scales is None:
        edge_scales = {}
    if forced_edges is None:
        forced_edges = []

    # Build set of locked image indices (from forced edges)
    locked_indices = set()
    for first_idx, last_idx, first_pos, last_pos in forced_edges:
        locked_indices.add(first_idx)
        locked_indices.add(last_idx)

    # Compute common comparison size across all images in the path
    min_dim = inf
    for idx in path:
        fn = filenames[idx]
        em = edge_maps[fn]
        h, w = em.shape
        min_dim = min(min_dim, h, w)
    common_size = int(min_dim * min(tile_ratios))
    common_size = max(16, common_size)

    # Precompute scaled face bboxes per sequence index using exact scale
    scaled_face_bboxes = []
    for k, idx in enumerate(path):
        fn = filenames[idx]
        bboxes = face_bboxes.get(fn, [])
        sc = edge_scales.get(fn, 1.0)
        scaled_face_bboxes.append(scale_bboxes(bboxes, sc))

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
            mid_ratio = tile_ratios[len(tile_ratios) // 2]
            ts = int(min(h, w) * mid_ratio)
            ts = max(16, min(ts, h, w))
            ty, tx = (h - ts) // 2, (w - ts) // 2

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

    def neighbor_cost(
        k, candidate_ty, candidate_tx, candidate_ts, neighbor_tiles, neighbor_indices
    ):
        """Compute total cost from a candidate position to all neighbors, including face bonus."""
        cost = 0.0
        for ni, nk in enumerate(neighbor_indices):
            nt = neighbor_tiles[ni]
            if nt.size == 0:
                return inf

            chamfer = _chamfer_cost_common(
                edge_maps[filenames[path[k]]][
                    candidate_ty : candidate_ty + candidate_ts,
                    candidate_tx : candidate_tx + candidate_ts,
                ],
                nt,
                common_size,
            )

            # Face alignment bonus
            nty, ntx, nts = positions[nk]
            bboxes_k = scaled_face_bboxes[k]
            bboxes_n = scaled_face_bboxes[nk]
            if bboxes_k and bboxes_n:
                bonus = _face_alignment_bonus(
                    candidate_ty,
                    candidate_tx,
                    candidate_ts,
                    nty,
                    ntx,
                    nts,
                    bboxes_k,
                    bboxes_n,
                )
                chamfer *= 1.0 - min(bonus, 0.9)

            # Weight by distance: immediate neighbors full weight, lookback decays
            dist = abs(k - nk)
            weight = 0.5 ** (dist - 1) if dist > 1 else 1.0
            cost += chamfer * weight

        return cost

    def compute_total_cost():
        """Compute total adjacent Chamfer cost with face bonuses and lookback."""
        total = 0.0
        for k in range(n - 1):
            t_a = get_tile(k)
            t_b = get_tile(k + 1)
            if t_a.size > 0 and t_b.size > 0:
                chamfer = _chamfer_cost_common(t_a, t_b, common_size)
                ay, ax, a_ts = positions[k]
                by, bx, b_ts = positions[k + 1]
                bboxes_a = scaled_face_bboxes[k]
                bboxes_b = scaled_face_bboxes[k + 1]
                if bboxes_a and bboxes_b:
                    bonus = _face_alignment_bonus(
                        ay, ax, a_ts, by, bx, b_ts, bboxes_a, bboxes_b
                    )
                    chamfer *= 1.0 - min(bonus, 0.9)
                total += chamfer
        return total

    initial_cost = compute_total_cost()
    print(f"    Refinement initial cost: {initial_cost:.2f}")
    if locked_indices:
        print(f"    Locked indices (from forced edges): {locked_indices}")

    for iteration in range(num_iterations):
        changed = False

        sweep_order = list(range(1, n - 1)) + list(range(n - 2, 0, -1))
        sweep_order = [0] + sweep_order + [n - 1]

        for k in sweep_order:
            idx = path[k]

            # Skip locked positions (forced edge endpoints)
            if idx in locked_indices:
                continue

            with open("tile_matching.log", "a") as log:
                log.write(f"refine {iteration}-{k}\n")
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            cur_ty, cur_tx, cur_ts = positions[k]

            # Collect neighbors: immediate + lookback
            neighbor_indices = []
            for offset in range(1, NUM_LOOKBACK + 1):
                if k - offset >= 0:
                    neighbor_indices.append(k - offset)
                if k + offset < n:
                    neighbor_indices.append(k + offset)

            if not neighbor_indices:
                continue

            neighbor_tiles = [get_tile(nk) for nk in neighbor_indices]

            best_cost = inf
            best_pos = (cur_ty, cur_tx, cur_ts)

            # Evaluate current position
            cur_tile = em[cur_ty : cur_ty + cur_ts, cur_tx : cur_tx + cur_ts]
            if cur_tile.shape[0] == cur_ts and cur_tile.shape[1] == cur_ts:
                cur_cost = neighbor_cost(
                    k, cur_ty, cur_tx, cur_ts, neighbor_tiles, neighbor_indices
                )
                if cur_cost < inf:
                    best_cost = cur_cost
                    best_pos = (cur_ty, cur_tx, cur_ts)

            # Search the neighborhood across all tile ratios
            for ratio in tile_ratios:
                ts = int(min(h, w) * ratio)
                if ts < 16 or ts > h or ts > w:
                    continue

                if cur_ts > 0:
                    scale_factor = ts / cur_ts
                else:
                    scale_factor = 1.0
                scaled_radius = max(1, int(refine_radius * scale_factor))

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

                        cost = neighbor_cost(
                            k, ty, tx, ts, neighbor_tiles, neighbor_indices
                        )
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
    output_file="sequence_order_v3_4lookback.txt",
    cache_file="cost_matrix_cache",
    max_workers=None,
    cache_prefix="",
):
    cache_file = cache_file + cache_prefix + ".pkl"
    print("Step 1/5: Building edge maps...")
    edge_maps, edge_scales = build_edge_maps(image_folder)
    n = len(edge_maps)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    print(f"  Tile ratios: {TILE_RATIOS}")
    if PREPROCESSED_SEQS:
        print(f"  Pre-sequenced groups: {len(PREPROCESSED_SEQS)}")
        for seq_name, (first_info, last_info) in PREPROCESSED_SEQS:
            print(f"    {seq_name}: {first_info[0]} → {last_info[0]}")

    forced_edges = []

    if os.path.exists(cache_file) and cache_file != "new":
        print("Step 2/5: Loading cached cost data...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        filenames = cached["filenames"]
        all_pair_costs = cached["all_pair_costs"]
        tile_positions_per_image = cached["tile_positions_per_image"]
        fallback_cost_matrix = cached["fallback_cost_matrix"]
        forced_edges = cached.get("forced_edges", [])

        if set(filenames) != set(edge_maps.keys()):
            print("  Cache stale — recomputing.")
            (
                filenames,
                all_pair_costs,
                best_costs,
                tile_positions_per_image,
                fallback_cost_matrix,
                coarse_dist,
                forced_edges,
            ) = build_sparse_cost_data(
                edge_maps,
                edge_scales,
                max_workers=max_workers,
                image_folder=image_folder,
            )
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "filenames": filenames,
                        "all_pair_costs": all_pair_costs,
                        "tile_positions_per_image": tile_positions_per_image,
                        "fallback_cost_matrix": fallback_cost_matrix,
                        "forced_edges": forced_edges,
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
            forced_edges,
        ) = build_sparse_cost_data(
            edge_maps, edge_scales, max_workers=max_workers, image_folder=image_folder
        )
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "filenames": filenames,
                    "all_pair_costs": all_pair_costs,
                    "tile_positions_per_image": tile_positions_per_image,
                    "fallback_cost_matrix": fallback_cost_matrix,
                    "forced_edges": forced_edges,
                },
                f,
            )
    print()

    print("Step 3/5: Tile-aware greedy nearest-neighbor path...")
    path, greedy_cost, tile_positions = greedy_nearest_neighbor_tileaware(
        n,
        all_pair_costs,
        fallback_cost_matrix,
        tile_positions_per_image,
        forced_edges=forced_edges,
    )
    print(f"  Greedy cost: {greedy_cost:.2f}\n")

    print("Step 4/5: Tile-aware 2-opt refinement...")
    path, final_cost, tile_positions = two_opt_tileaware(
        path,
        tile_positions,
        all_pair_costs,
        fallback_cost_matrix,
        tile_positions_per_image,
        forced_edges=forced_edges,
    )
    print(f"  2-opt cost:  {final_cost:.2f}\n")

    # One final full position optimization pass on the settled path
    # but respect locked positions
    print("  Final position optimization on settled path...")
    locked_positions = {}
    for first_idx, last_idx, first_pos, last_pos in forced_edges:
        locked_positions[first_idx] = first_pos
        locked_positions[last_idx] = last_pos
    # Apply locked positions before optimization
    for idx, pos in locked_positions.items():
        tile_positions[idx] = pos
    final_cost = _optimize_positions_for_path(
        path,
        tile_positions,
        all_pair_costs,
        fallback_cost_matrix,
        tile_positions_per_image,
    )
    # Re-apply locked positions (in case optimizer changed them)
    for idx, pos in locked_positions.items():
        tile_positions[idx] = pos
    print(f"  Optimized cost: {final_cost:.2f}\n")

    print("Step 5/5: Fine-grained tile position refinement...")
    with open("face_bboxes.json") as f:
        face_bboxes_refine = json.load(f)
    refined_positions = refine_tile_positions(
        path,
        filenames,
        edge_maps,
        tile_positions,
        edge_scales=edge_scales,
        face_bboxes=face_bboxes_refine,
        forced_edges=forced_edges,
    )
    print()

    # ── Expand forced edges into full sub-sequences in the output ────────
    # Build a map from (first_fn, last_fn) -> seq_folder_name
    seq_lookup = {}
    for seq_name, (first_info, last_info) in PREPROCESSED_SEQS:
        seq_lookup[(first_info[0], last_info[0])] = seq_name

    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            ty, tx, ts = refined_positions[k]

            # Check if this is the start of a forced edge (sub-sequence)
            if k + 1 < len(path):
                next_fn = filenames[path[k + 1]]
                seq_name = seq_lookup.get((fn, next_fn))
                if seq_name is not None:
                    # Write the full sub-sequence from its sequence file
                    seq_path = os.path.join(image_folder, seq_name, "sequence.txt")
                    if os.path.exists(seq_path):
                        with open(seq_path, "r") as sf:
                            seq_lines = [l.strip() for l in sf if l.strip()]
                        for sl in seq_lines:
                            parts = sl.split(",")
                            sq_fn = parts[0]
                            sq_ty, sq_tx = int(parts[1]), int(parts[2])
                            sq_ts = int(parts[3])
                            sq_h, sq_w = int(parts[4]), int(parts[5])
                            # Write with sub-folder path so generate_vid can find it
                            f.write(
                                f"{os.path.join(seq_name, sq_fn)},"
                                f"{sq_ty},{sq_tx},{sq_ts},{sq_h},{sq_w}\n"
                            )
                        continue  # Skip writing first_fn separately (it's in the sub-seq)

            # Check if this is the last node of a forced edge — already written
            # as part of the sub-sequence above, so skip
            if k > 0:
                prev_fn = filenames[path[k - 1]]
                if (prev_fn, fn) in seq_lookup:
                    continue  # Already written as part of the sub-sequence

            f.write(f"{fn},{ty},{tx},{ts},{h},{w}\n")

    print(f"Sequence written to {output_file}")
    print("Order:")
    for idx in path:
        print(f"  {filenames[idx]}")


# ── Video Generation ────────────────────────────────────────────────────────


def generate_vid(
    sequence_file,
    foto_folder,
    output_path="output_v6.mp4",
    fps=11,
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
    out_h = max(f.shape[0] for f in frames)
    out_w = max(f.shape[1] for f in frames)
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
        edge_map, _ = get_weighted_edges(path)

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

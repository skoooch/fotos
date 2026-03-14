"""
Sequence images by finding smooth contour-based transitions.

Strategy:
  1. Compute weighted edge maps (not binary) preserving edge strength.
  2. For every image pair, find the best overlapping tile using
     weighted Chamfer distance.
  3. Build a full pairwise cost matrix.
  4. Solve sequencing with greedy nearest-neighbor + 2-opt refinement.
  5. Store tile positions for video generation.
"""

import os
import sys
import pickle
from math import inf
from itertools import combinations

import numpy as np
import cv2 as cv
from scipy.ndimage import distance_transform_edt


# ── Config ──────────────────────────────────────────────────────────────────
TILE_RATIO = 0.45  # smaller tiles = more selective, cleaner overlaps
STRIDE = 40
EDGE_BLUR_KSIZE = 3
NUM_2OPT_ITERATIONS = 50  # how many passes of 2-opt improvement


# ── Edge Extraction ─────────────────────────────────────────────────────────


def get_weighted_edges(image_path, target_short_edge=512):
    """
    Produce a float32 edge-strength map in [0, 1].

    Uses Scharr gradients (more rotational symmetry than Sobel) and
    keeps the magnitude as a continuous weight rather than binarizing.
    """
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Resize so the short edge = target_short_edge (keeps computation sane)
    h, w = img.shape
    scale = target_short_edge / min(h, w)
    img = cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

    img = cv.GaussianBlur(img, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)

    grad_x = cv.Scharr(img, cv.CV_64F, 1, 0)
    grad_y = cv.Scharr(img, cv.CV_64F, 0, 1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize to [0, 1]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max

    return magnitude.astype(np.float32)


# ── Weighted Chamfer Distance ───────────────────────────────────────────────


def weighted_chamfer_distance(tile_a, tile_b, threshold=0.1):
    """
    Compute a weighted Chamfer-like distance between two edge-strength tiles.

    For each pixel with edge weight > threshold in tile_a, find its nearest
    edge pixel in tile_b (via distance transform) and weight the distance
    by the edge strength.  Average over both directions.

    Returns a float (lower = more similar contours).
    """
    mask_a = (tile_a > threshold).astype(np.float32)
    mask_b = (tile_b > threshold).astype(np.float32)

    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return inf  # nothing to compare

    # Distance transform: distance of every pixel to nearest edge in the *other* tile
    dt_b = distance_transform_edt(1 - mask_b)
    dt_a = distance_transform_edt(1 - mask_a)

    # A → B: for each edge pixel in A, look up distance to nearest B edge,
    #         weighted by A's edge strength
    weights_a = tile_a[mask_a > 0]
    dists_a_to_b = dt_b[mask_a > 0]
    d_ab = np.sum(weights_a * dists_a_to_b) / np.sum(weights_a)

    # B → A
    weights_b = tile_b[mask_b > 0]
    dists_b_to_a = dt_a[mask_b > 0]
    d_ba = np.sum(weights_b * dists_b_to_a) / np.sum(weights_b)

    return (d_ab + d_ba) / 2.0


# ── Best Tile Pair ──────────────────────────────────────────────────────────


def find_best_tile_pair(edge_a, edge_b, tile_ratio=TILE_RATIO, stride=STRIDE):
    """
    Slide tiles over both images and return the best matching pair of
    tile positions along with the Chamfer distance.

    Returns:
        (cost, (ay, ax), (by, bx), tile_size)
    """
    h_a, w_a = edge_a.shape
    h_b, w_b = edge_b.shape
    tile_size = int(min(h_a, w_a, h_b, w_b) * tile_ratio)

    if tile_size < 16:
        return inf, (0, 0), (0, 0), tile_size

    # Precompute distance transforms once per image
    dt_a = distance_transform_edt((edge_a <= 0.1).astype(np.float32))
    dt_b = distance_transform_edt((edge_b <= 0.1).astype(np.float32))

    # Collect candidate tile positions for each image
    def tile_positions(h, w):
        positions = []
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                positions.append((y, x))
        if not positions:
            positions.append((0, 0))
        return positions

    positions_a = tile_positions(h_a, w_a)
    positions_b = tile_positions(h_b, w_b)

    # Pre-filter: skip tiles with very few edges (< 1% edge pixels)
    min_edge_density = 0.01

    def has_edges(edge_map, y, x):
        tile = edge_map[y : y + tile_size, x : x + tile_size]
        return (tile > 0.1).mean() >= min_edge_density

    positions_a = [(y, x) for y, x in positions_a if has_edges(edge_a, y, x)]
    positions_b = [(y, x) for y, x in positions_b if has_edges(edge_b, y, x)]

    if not positions_a or not positions_b:
        return inf, (0, 0), (0, 0), tile_size

    best_cost = inf
    best_a = positions_a[0]
    best_b = positions_b[0]

    for ay, ax in positions_a:
        tile_a = edge_a[ay : ay + tile_size, ax : ax + tile_size]
        mask_a = (tile_a > 0.1).astype(np.float32)
        weights_a = tile_a[mask_a > 0]
        sum_weights_a = np.sum(weights_a)
        if sum_weights_a == 0:
            continue

        for by, bx in positions_b:
            tile_b = edge_b[by : by + tile_size, bx : bx + tile_size]
            mask_b = (tile_b > 0.1).astype(np.float32)

            if mask_b.sum() == 0:
                continue

            # A → B direction
            dt_b_tile = dt_b[by : by + tile_size, bx : bx + tile_size]
            # We need dt of tile_b, but we have global dt_b which is the
            # distance in the *full* image.  For tile-local comparison we
            # need a local distance transform.
            local_dt_b = distance_transform_edt(1 - mask_b)
            dists_a_to_b = local_dt_b[mask_a > 0]
            d_ab = np.sum(weights_a * dists_a_to_b) / sum_weights_a

            # Early exit — if one direction already exceeds best, skip
            if d_ab >= best_cost:
                continue

            # B → A direction
            weights_b = tile_b[mask_b > 0]
            sum_weights_b = np.sum(weights_b)
            if sum_weights_b == 0:
                continue

            local_dt_a = distance_transform_edt(1 - mask_a)
            dists_b_to_a = local_dt_a[mask_b > 0]
            d_ba = np.sum(weights_b * dists_b_to_a) / sum_weights_b

            cost = (d_ab + d_ba) / 2.0
            if cost < best_cost:
                best_cost = cost
                best_a = (ay, ax)
                best_b = (by, bx)

    return best_cost, best_a, best_b, tile_size


# ── Pairwise Cost Matrix ───────────────────────────────────────────────────


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


def build_cost_matrix(edge_maps):
    """
    Compute pairwise best-tile Chamfer cost for all image pairs.

    Returns:
        filenames: list[str]
        cost_matrix: np.ndarray of shape (n, n)
        tile_info: dict mapping (i, j) -> ((ay, ax), (by, bx), tile_size)
    """
    filenames = list(edge_maps.keys())
    n = len(filenames)
    cost_matrix = np.full((n, n), inf)
    tile_info = {}

    total_pairs = n * (n - 1) // 2
    done = 0

    for i, j in combinations(range(n), 2):
        cost, pos_a, pos_b, ts = find_best_tile_pair(
            edge_maps[filenames[i]], edge_maps[filenames[j]]
        )
        cost_matrix[i, j] = cost
        cost_matrix[j, i] = cost
        tile_info[(i, j)] = (pos_a, pos_b, ts)
        tile_info[(j, i)] = (pos_b, pos_a, ts)
        done += 1
        if done % 10 == 0 or done == total_pairs:
            print(f"  pairs: {done}/{total_pairs}")

    return filenames, cost_matrix, tile_info


# ── TSP Solver: Greedy + 2-opt ─────────────────────────────────────────────


def greedy_nearest_neighbor(cost_matrix):
    """Build an initial path using nearest-neighbor heuristic, starting from
    the node that yields the best total greedy cost (try all starts)."""
    n = cost_matrix.shape[0]
    best_path = None
    best_total = inf

    for start in range(n):
        visited = {start}
        path = [start]
        total = 0.0
        current = start
        for _ in range(n - 1):
            row = cost_matrix[current].copy()
            row[list(visited)] = inf
            nxt = int(np.argmin(row))
            total += row[nxt]
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        if total < best_total:
            best_total = total
            best_path = path[:]

    return best_path, best_total


def two_opt(path, cost_matrix, max_iterations=NUM_2OPT_ITERATIONS):
    """Improve a path by repeatedly reversing sub-segments."""

    def path_cost(p):
        return sum(cost_matrix[p[k], p[k + 1]] for k in range(len(p) - 1))

    best = path[:]
    best_cost = path_cost(best)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                c = path_cost(candidate)
                if c < best_cost:
                    best = candidate
                    best_cost = c
                    improved = True
        print(f"  2-opt iteration {iteration}: cost = {best_cost:.2f}")

    return best, best_cost


# ── Main Sequencing Pipeline ────────────────────────────────────────────────


def sequence(
    image_folder,
    output_file="sequence_order_v2.txt",
    cache_file="cost_matrix_cache.pkl",
):
    print("Step 1/4: Building edge maps...")
    edge_maps = build_edge_maps(image_folder)
    n = len(edge_maps)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    # Cache the expensive pairwise computation
    if os.path.exists(cache_file):
        print("Step 2/4: Loading cached cost matrix...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        filenames = cached["filenames"]
        cost_matrix = cached["cost_matrix"]
        tile_info = cached["tile_info"]
        # Validate cache matches current images
        if set(filenames) != set(edge_maps.keys()):
            print("  Cache stale — recomputing.")
            filenames, cost_matrix, tile_info = build_cost_matrix(edge_maps)
            with open(cache_file, "wb") as f:
                pickle.dump(
                    {
                        "filenames": filenames,
                        "cost_matrix": cost_matrix,
                        "tile_info": tile_info,
                    },
                    f,
                )
    else:
        print("Step 2/4: Computing pairwise tile costs...")
        filenames, cost_matrix, tile_info = build_cost_matrix(edge_maps)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "filenames": filenames,
                    "cost_matrix": cost_matrix,
                    "tile_info": tile_info,
                },
                f,
            )
    print()

    print("Step 3/4: Greedy nearest-neighbor path...")
    path, greedy_cost = greedy_nearest_neighbor(cost_matrix)
    print(f"  Greedy cost: {greedy_cost:.2f}\n")

    print("Step 4/4: 2-opt refinement...")
    path, final_cost = two_opt(path, cost_matrix)
    print(f"  Final cost:  {final_cost:.2f}\n")

    # Write output: for each consecutive pair, store the tile of the *incoming* image
    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            if k == 0:
                # First image — use centre tile
                ts = int(min(h, w) * TILE_RATIO)
                ty, tx = (h - ts) // 2, (w - ts) // 2
            else:
                prev_idx = path[k - 1]
                key = (prev_idx, idx)
                _, (ty, tx), ts = tile_info.get(
                    key, ((0, 0), (0, 0), int(min(h, w) * TILE_RATIO))
                )
                # tile_info stores (pos_a, pos_b, tile_size); pos_b is the incoming image's tile
                info = tile_info.get(key)
                if info:
                    (_, _), (ty, tx), ts = info[0], info[1], info[2]
                    # Unpack correctly: tile_info[(i,j)] = (pos_i, pos_j, tile_size)
                    ty, tx = info[1]
                    ts = info[2]
            f.write(f"{fn},{ty},{tx},{h},{w}\n")

    print(f"Sequence written to {output_file}")
    print("Order:")
    for idx in path:
        print(f"  {filenames[idx]}")


# ── Video Generation (reused from v1 with minor tweaks) ────────────────────


def generate_vid(
    sequence_file,
    foto_folder,
    output_path="output_v2.mp4",
    fps=8,
    tile_ratio=TILE_RATIO,
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
            ld_h, ld_w = int(parts[3]), int(parts[4])
            entries.append((fn, tile_y, tile_x, ld_h, ld_w))

    if not entries:
        print("No entries in sequence file.")
        return

    frames = []
    for fn, tile_y, tile_x, ld_h, ld_w in entries:
        base = os.path.splitext(fn)[0]
        foto_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".bmp"):
            candidate = os.path.join(foto_folder, base + ext)
            if os.path.exists(candidate):
                foto_path = candidate
                break
        # Also try the filename as-is (in case it's already a photo filename)
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
        scale_y = actual_h / ld_h
        scale_x = actual_w / ld_w
        ts_ld = int(min(ld_h, ld_w) * tile_ratio)

        sy = max(0, min(int(tile_y * scale_y), actual_h - int(ts_ld * scale_y)))
        sx = max(0, min(int(tile_x * scale_x), actual_w - int(ts_ld * scale_x)))
        sh = min(int(ts_ld * scale_y), actual_h - sy)
        sw = min(int(ts_ld * scale_x), actual_w - sx)

        crop = img[sy : sy + sh, sx : sx + sw]
        if crop.size == 0:
            continue
        frames.append(crop)

    if not frames:
        print("No frames.")
        return

    out_h, out_w = frames[0].shape[:2]
    writer = cv.VideoWriter(
        output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    for frame in frames:
        writer.write(cv.resize(frame, (out_w, out_h)))
    writer.release()
    print(f"Video saved: {output_path} ({len(frames)} frames @ {fps}fps)")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        foto_folder = sys.argv[2] if len(sys.argv) > 2 else "."
        generate_vid("sequence_order_v2.txt", foto_folder)
    else:
        image_folder = sys.argv[1] if len(sys.argv) > 1 else "."
        sequence(image_folder)

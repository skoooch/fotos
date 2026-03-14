"""
Sequence images by finding smooth contour-based transitions.

Strategy:
  1. Compute weighted edge maps (not binary) preserving edge strength.
  2. Cheap global descriptors to build a K-nearest-neighbor shortlist.
  3. Expensive tile matching ONLY on shortlisted pairs.
  4. Build a sparse cost graph, solve with greedy + 2-opt.
  5. Store tile positions for video generation.
"""

import os
import sys
import pickle
from math import inf
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2 as cv
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist


# ── Config ──────────────────────────────────────────────────────────────────
TILE_RATIO = 0.45
STRIDE = 40
EDGE_BLUR_KSIZE = 3
NUM_2OPT_ITERATIONS = 50
K_NEIGHBORS = 15  # only tile-match the K most promising neighbors per image
TARGET_SHORT_EDGE = 512
EDGE_THRESHOLD = 0.1
MIN_EDGE_DENSITY = 0.01


# ── Edge Extraction ─────────────────────────────────────────────────────────


def get_weighted_edges(image_path, target_short_edge=TARGET_SHORT_EDGE):
    """
    Produce a float32 edge-strength map in [0, 1].
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

    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max

    return magnitude.astype(np.float32)


# ── Cheap Global Descriptor ────────────────────────────────────────────────


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

    # Recompute gradients for orientation (we only stored magnitude)
    # Use a small blur to stabilize orientation estimates
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

            # Weighted orientation histogram
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


# ── Tile Matching (unchanged core logic) ────────────────────────────────────


def find_best_tile_pair(edge_a, edge_b, tile_ratio=TILE_RATIO, stride=STRIDE):
    """
    Slide tiles over both images and return the best matching pair of
    tile positions along with the Chamfer distance.
    """
    h_a, w_a = edge_a.shape
    h_b, w_b = edge_b.shape
    tile_size = int(min(h_a, w_a, h_b, w_b) * tile_ratio)

    if tile_size < 16:
        return inf, (0, 0), (0, 0), tile_size

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

    threshold = EDGE_THRESHOLD

    def has_edges(edge_map, y, x):
        tile = edge_map[y : y + tile_size, x : x + tile_size]
        return (tile > threshold).mean() >= MIN_EDGE_DENSITY

    positions_a = [(y, x) for y, x in positions_a if has_edges(edge_a, y, x)]
    positions_b = [(y, x) for y, x in positions_b if has_edges(edge_b, y, x)]

    if not positions_a or not positions_b:
        return inf, (0, 0), (0, 0), tile_size

    tiles_a = []
    for ay, ax in positions_a:
        tile = edge_a[ay : ay + tile_size, ax : ax + tile_size]
        mask = tile > threshold
        weights = tile[mask]
        sum_w = weights.sum()
        if sum_w == 0:
            continue
        dt = distance_transform_edt(~mask)
        tiles_a.append((ay, ax, mask, weights, sum_w, dt))

    tiles_b = []
    for by, bx in positions_b:
        tile = edge_b[by : by + tile_size, bx : bx + tile_size]
        mask = tile > threshold
        weights = tile[mask]
        sum_w = weights.sum()
        if sum_w == 0:
            continue
        dt = distance_transform_edt(~mask)
        tiles_b.append((by, bx, mask, weights, sum_w, dt))

    if not tiles_a or not tiles_b:
        return inf, (0, 0), (0, 0), tile_size

    best_cost = inf
    best_a = (tiles_a[0][0], tiles_a[0][1])
    best_b = (tiles_b[0][0], tiles_b[0][1])

    for ay, ax, mask_a, weights_a, sum_wa, dt_a in tiles_a:
        for by, bx, mask_b, weights_b, sum_wb, dt_b in tiles_b:
            d_ab = np.sum(weights_a * dt_b[mask_a]) / sum_wa
            if d_ab >= best_cost:
                continue
            d_ba = np.sum(weights_b * dt_a[mask_b]) / sum_wb
            cost = (d_ab + d_ba) / 2.0
            if cost < best_cost:
                best_cost = cost
                best_a = (ay, ax)
                best_b = (by, bx)

    return best_cost, best_a, best_b, tile_size


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


def build_knn_shortlist(edge_maps, k=K_NEIGHBORS):
    """
    Compute cheap global descriptors for all images and find the
    K nearest neighbors for each image using cosine distance.

    Returns:
        filenames: list[str]
        neighbors: dict mapping index -> list of neighbor indices
        descriptors: np.ndarray of shape (n, descriptor_dim)
    """
    filenames = list(edge_maps.keys())
    n = len(filenames)

    print(f"  Computing descriptors for {n} images...")
    descriptors = np.array([compute_edge_descriptor(edge_maps[fn]) for fn in filenames])

    print(f"  Computing cosine distance matrix...")
    # cosine distance: fast even for n=2000 (2000x128 matrix)
    dist_matrix = cdist(descriptors, descriptors, metric="cosine")

    # For each image, find K nearest (excluding self)
    neighbors = {}
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = inf  # exclude self
        nearest = np.argsort(dists)[:k]
        neighbors[i] = nearest.tolist()

    # Make symmetric: if i is neighbor of j, also consider j neighbor of i
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


def _compute_pair(args):
    """Worker for parallel tile matching."""
    i, j, edge_i, edge_j = args
    cost, pos_a, pos_b, ts = find_best_tile_pair(edge_i, edge_j)
    return i, j, cost, pos_a, pos_b, ts


def build_sparse_cost_matrix(edge_maps, max_workers=None, k=K_NEIGHBORS):
    """
    1. Build KNN shortlist using cheap descriptors.
    2. Run expensive tile matching only on shortlisted pairs.
    3. Fill non-shortlisted pairs with the cheap cosine distance (scaled).

    Returns:
        filenames, cost_matrix, tile_info
    """
    filenames, shortlisted_pairs, neighbors, coarse_dist = build_knn_shortlist(
        edge_maps, k=k
    )
    n = len(filenames)
    cost_matrix = np.full((n, n), inf)
    tile_info = {}

    # Prepare tasks for shortlisted pairs only
    tasks = [
        (i, j, edge_maps[filenames[i]], edge_maps[filenames[j]])
        for i, j in shortlisted_pairs
    ]

    total = len(tasks)
    print(f"  Tile-matching {total} shortlisted pairs (of {n*(n-1)//2} total)...")

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_pair, t): t for t in tasks}
        for future in as_completed(futures):
            i, j, cost, pos_a, pos_b, ts = future.result()
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost
            tile_info[(i, j)] = (pos_a, pos_b, ts)
            tile_info[(j, i)] = (pos_b, pos_a, ts)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"    {done}/{total}")

    # Fill remaining entries with scaled coarse distance as fallback
    # This ensures the TSP solver can still traverse non-shortlisted edges
    # but strongly prefers the accurately-measured ones
    tile_costs = cost_matrix[cost_matrix < inf]
    if len(tile_costs) > 0:
        # Scale coarse distances to be in a similar range as tile costs
        # but with a penalty so they're less attractive
        coarse_scale = np.median(tile_costs) / (
            np.median(coarse_dist[coarse_dist > 0]) + 1e-9
        )
        penalty = np.percentile(tile_costs, 90) if len(tile_costs) > 0 else 100
    else:
        coarse_scale = 1.0
        penalty = 100

    for i in range(n):
        for j in range(i + 1, n):
            if cost_matrix[i, j] == inf:
                fallback = coarse_dist[i, j] * coarse_scale + penalty
                cost_matrix[i, j] = fallback
                cost_matrix[j, i] = fallback

    return filenames, cost_matrix, tile_info


# ── TSP Solver: Greedy + 2-opt ─────────────────────────────────────────────


def greedy_nearest_neighbor(cost_matrix):
    """Nearest-neighbor heuristic, trying all start nodes."""
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
                # Incremental cost check instead of full recompute
                # Cost of current edges: (i-1,i) + (j,j+1)
                # Cost of swapped edges: (i-1,j) + (i,j+1)
                a, b = best[i - 1], best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else None

                old_cost = cost_matrix[a, b]
                new_cost = cost_matrix[a, c]
                if d is not None:
                    old_cost += cost_matrix[c, d]
                    new_cost += cost_matrix[b, d]

                if new_cost < old_cost:
                    best = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                    best_cost = path_cost(best)  # recompute for accuracy
                    improved = True

        print(f"  2-opt iteration {iteration}: cost = {best_cost:.2f}")

    return best, best_cost


# ── Main Sequencing Pipeline ────────────────────────────────────────────────


def sequence(
    image_folder,
    output_file="sequence_order_v3.txt",
    cache_file="cost_matrix_cache.pkl",
    max_workers=None,
):
    print("Step 1/4: Building edge maps...")
    edge_maps = build_edge_maps(image_folder)
    n = len(edge_maps)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    if os.path.exists(cache_file):
        print("Step 2/4: Loading cached cost matrix...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        filenames = cached["filenames"]
        cost_matrix = cached["cost_matrix"]
        tile_info = cached["tile_info"]
        if set(filenames) != set(edge_maps.keys()):
            print("  Cache stale — recomputing.")
            filenames, cost_matrix, tile_info = build_sparse_cost_matrix(
                edge_maps, max_workers=max_workers
            )
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
        print("Step 2/4: Computing sparse pairwise costs (KNN + tile matching)...")
        filenames, cost_matrix, tile_info = build_sparse_cost_matrix(
            edge_maps, max_workers=max_workers
        )
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
    # For n=2000, trying all starts is O(n^2) per start × n starts = O(n^3).
    # Use a limited number of random starts instead.
    if n > 200:
        # Sample starts instead of trying all
        rng = np.random.default_rng(42)
        start_indices = rng.choice(n, size=min(50, n), replace=False)
        best_path = None
        best_total = inf
        for start in start_indices:
            visited = {int(start)}
            path = [int(start)]
            total = 0.0
            current = int(start)
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
        path, greedy_cost = best_path, best_total
    else:
        path, greedy_cost = greedy_nearest_neighbor(cost_matrix)
    print(f"  Greedy cost: {greedy_cost:.2f}\n")

    print("Step 4/4: 2-opt refinement...")
    path, final_cost = two_opt(path, cost_matrix)
    print(f"  Final cost:  {final_cost:.2f}\n")

    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            if k == 0:
                ts = int(min(h, w) * TILE_RATIO)
                ty, tx = (h - ts) // 2, (w - ts) // 2
            else:
                prev_idx = path[k - 1]
                key = (prev_idx, idx)
                info = tile_info.get(key)
                if info:
                    ty, tx = info[1]
                    ts = info[2]
                else:
                    ts = int(min(h, w) * TILE_RATIO)
                    ty, tx = (h - ts) // 2, (w - ts) // 2
            f.write(f"{fn},{ty},{tx},{h},{w}\n")

    print(f"Sequence written to {output_file}")
    print("Order:")
    for idx in path:
        print(f"  {filenames[idx]}")


# ── Video Generation ────────────────────────────────────────────────────────


def generate_vid(
    sequence_file,
    foto_folder,
    output_path="output_v3.mp4",
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
        generate_vid("sequence_order_v3.txt", foto_folder)
    else:
        image_folder = sys.argv[1] if len(sys.argv) > 1 else "."
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
        sequence(image_folder, max_workers=workers)

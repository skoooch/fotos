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
STRIDE = 25
EDGE_BLUR_KSIZE = 3
NUM_2OPT_ITERATIONS = 50
K_NEIGHBORS = 30  # only tile-match the K most promising neighbors per image
TARGET_SHORT_EDGE = 512
EDGE_THRESHOLD = 0.1
MIN_EDGE_DENSITY = 0.01
REFINE_STRIDE = 8  # fine-grained stride for tile refinement
REFINE_RADIUS = 25  # search radius (pixels) around current tile position
REFINE_ITERATIONS = 10  # number of forward+backward sweeps
DISTANCE_METRIC = "embedding"  # "edge_descript", "embedding", or "combined"
EMBEDDING_WEIGHT = 0.5  # weight for embedding distance in combined mode

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


def build_knn_shortlist(
    edge_maps, k=K_NEIGHBORS, distance_metric=DISTANCE_METRIC, image_folder=None
):
    """
    Compute global descriptors for all images and find the
    K nearest neighbors for each image.

    distance_metric options:
      - "edge_descript": HOG-like edge descriptor (cosine distance)
      - "embedding": CLIP semantic embedding (cosine distance)
      - "combined": weighted combination of both

    Returns:
        filenames: list[str]
        neighbors: dict mapping index -> list of neighbor indices
        descriptors: np.ndarray of shape (n, descriptor_dim)
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
        # Edge descriptors
        print(f"  Computing edge descriptors for {n} images...")
        edge_descs = np.array(
            [compute_edge_descriptor(edge_maps[fn]) for fn in filenames]
        )
        edge_dist = cdist(edge_descs, edge_descs, metric="cosine")

        # CLIP embeddings
        embeddings = compute_image_features(filenames, image_folder)
        embed_dist = cdist(embeddings, embeddings, metric="cosine")

        # Normalize both to [0, 1] range before combining
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


def build_sparse_cost_matrix(
    edge_maps, max_workers=None, k=K_NEIGHBORS, image_folder=None
):
    """
    1. Build KNN shortlist using cheap descriptors.
    2. Run expensive tile matching only on shortlisted pairs.
    3. Fill non-shortlisted pairs with the cheap cosine distance (scaled).

    Returns:
        filenames, cost_matrix, tile_info
    """
    filenames, shortlisted_pairs, neighbors, coarse_dist = build_knn_shortlist(
        edge_maps, k=k, image_folder=image_folder
    )
    _unload_clip_model()
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
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, 8)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid
    # fork() memory duplication. NumPy releases the GIL so threads
    # still get good parallelism for the heavy array operations.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_pair, t): t for t in tasks}
        for future in as_completed(futures):
            i, j, cost, pos_a, pos_b, ts = future.result()
            cost_matrix[i, j] = cost
            cost_matrix[j, i] = cost
            tile_info[(i, j)] = (pos_a, pos_b, ts)
            tile_info[(j, i)] = (pos_b, pos_a, ts)
            done += 1
            if done % 10 == 0 or done == total:
                with open("tile_matching.log", "a") as log:
                    log.write(f"{done}/{total}\n")

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


# ── Tile Position Refinement ────────────────────────────────────────────────


def _chamfer_cost(tile_a, tile_b, threshold=EDGE_THRESHOLD):
    """
    Compute symmetric weighted Chamfer distance between two edge tiles.
    Both inputs should be same-sized float32 edge-strength maps.
    """
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
    tile_info,
    tile_ratio=TILE_RATIO,
    refine_stride=REFINE_STRIDE,
    refine_radius=REFINE_RADIUS,
    num_iterations=REFINE_ITERATIONS,
):
    """
    After sequencing, refine each image's tile position to minimize the
    sum of Chamfer distances to its predecessor and successor.

    This uses a much finer stride than the initial tile matching, since
    we only search a small neighborhood around the current position and
    only consider adjacent pairs.

    Strategy:
      - For each image i in the sequence, its tile position affects:
        cost(i-1, i) and cost(i, i+1).
      - We search a small window around the current tile position and
        pick the position that minimizes the sum of both adjacent costs.
      - Sweep forward and backward multiple times until convergence.

    Returns:
        refined_positions: list of (tile_y, tile_x, tile_size) per sequence index
    """
    n = len(path)

    # Initialize positions from tile_info
    positions = []  # (ty, tx, ts) for each index in path order
    for k, idx in enumerate(path):
        fn = filenames[idx]
        em = edge_maps[fn]
        h, w = em.shape
        if k == 0:
            ts = int(min(h, w) * tile_ratio)
            ty, tx = (h - ts) // 2, (w - ts) // 2
        else:
            prev_idx = path[k - 1]
            key = (prev_idx, idx)
            info = tile_info.get(key)
            if info:
                ty, tx = info[1]
                ts = info[2]
            else:
                ts = int(min(h, w) * tile_ratio)
                ty, tx = (h - ts) // 2, (w - ts) // 2
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
            if t_a.shape == t_b.shape and t_a.size > 0:
                total += _chamfer_cost(t_a, t_b)
        return total

    initial_cost = compute_total_cost()
    print(f"    Refinement initial cost: {initial_cost:.2f}")

    for iteration in range(num_iterations):
        changed = False

        # Forward sweep: refine positions[1], positions[2], ..., positions[n-2]
        # Backward sweep: refine positions[n-2], ..., positions[1]
        sweep_order = list(range(1, n - 1)) + list(range(n - 2, 0, -1))
        # Also include first and last with partial constraints
        sweep_order = [0] + sweep_order + [n - 1]

        for k in sweep_order:
            with open("tile_matching.log", "a") as log:
                log.write(f"{iteration - k}\n")
            idx = path[k]
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            cur_ty, cur_tx, ts = positions[k]

            # Determine which neighbors to consider
            neighbors = []
            if k > 0:
                neighbors.append(k - 1)
            if k < n - 1:
                neighbors.append(k + 1)

            if not neighbors:
                continue

            # Get neighbor tiles (fixed during this step)
            neighbor_tiles = []
            for nk in neighbors:
                nt = get_tile(nk)
                neighbor_tiles.append(nt)

            # Search window around current position
            y_min = max(0, cur_ty - refine_radius)
            y_max = min(h - ts, cur_ty + refine_radius)
            x_min = max(0, cur_tx - refine_radius)
            x_max = min(w - ts, cur_tx + refine_radius)

            if y_max < y_min or x_max < x_min:
                continue

            best_cost = inf
            best_pos = (cur_ty, cur_tx)

            # Evaluate current position first
            cur_tile = em[cur_ty : cur_ty + ts, cur_tx : cur_tx + ts]
            if cur_tile.shape[0] == ts and cur_tile.shape[1] == ts:
                cur_cost = 0.0
                valid = True
                for nt in neighbor_tiles:
                    if nt.shape == cur_tile.shape:
                        cur_cost += _chamfer_cost(cur_tile, nt)
                    else:
                        valid = False
                        break
                if valid:
                    best_cost = cur_cost
                    best_pos = (cur_ty, cur_tx)

            # Search the neighborhood
            for ty in range(y_min, y_max + 1, refine_stride):
                for tx in range(x_min, x_max + 1, refine_stride):
                    if ty == cur_ty and tx == cur_tx:
                        continue  # already evaluated

                    candidate = em[ty : ty + ts, tx : tx + ts]
                    if candidate.shape[0] != ts or candidate.shape[1] != ts:
                        continue

                    cost = 0.0
                    skip = False
                    for nt in neighbor_tiles:
                        if nt.shape != candidate.shape:
                            skip = True
                            break
                        c = _chamfer_cost(candidate, nt)
                        cost += c
                        if cost >= best_cost:
                            skip = True
                            break
                    if skip:
                        continue

                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (ty, tx)

            if best_pos != (cur_ty, cur_tx):
                positions[k] = (best_pos[0], best_pos[1], ts)
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
        f"(improved by {improvement:.2f}, {100*improvement/initial_cost:.1f}%)"
    )

    return positions


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
                edge_maps, max_workers=max_workers, image_folder=image_folder
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
            edge_maps, max_workers=max_workers, image_folder=image_folder
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

    print("Step 5: Fine-grained tile position refinement...")
    refined_positions = refine_tile_positions(path, filenames, edge_maps, tile_info)
    print()

    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            ty, tx, ts = refined_positions[k]
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
    fps=15,
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

        # Use a uniform scale so rotated images don't get stretched.
        # The edge map was produced by scaling the short edge to TARGET_SHORT_EDGE,
        # so recover that same scale factor uniformly.
        scale = min(actual_h, actual_w) / min(ld_h, ld_w)

        ts_ld = int(min(ld_h, ld_w) * tile_ratio)
        ts_actual = int(ts_ld * scale)

        # Clamp tile size to image dimensions (square crop)
        ts_actual = min(ts_actual, actual_h, actual_w)

        # Clamp position so the full square tile fits within the image
        sy = max(0, min(int(tile_y * scale), actual_h - ts_actual))
        sx = max(0, min(int(tile_x * scale), actual_w - ts_actual))

        crop = img[sy : sy + ts_actual, sx : sx + ts_actual]
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

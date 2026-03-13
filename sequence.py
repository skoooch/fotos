"""
Current idea is to slide a tile (some ratio of the original image size)
around possible target images and compare the line drawings of the current img
and target img.

Using hausdorff distance would only work with binary line drawings, but i think i want to
have somesort of weighting, whereby prominent edges being similar to prominent edges
is more valuable(similar)

"""

from math import inf
from mimetypes import init
import os
from PIL import Image
import cv2 as cv
from utils import *

tile_ratio = 0.7
stride = 50


def find_next_img(binary_vecLDs, curr_tile, remaining_set_paths):
    img_scores = {}
    best_distance = inf
    best_key = None
    curr_fn, i_1, j_1, m_h, m_w = curr_tile
    img1 = binary_vecLDs[curr_fn]
    w = int(min(m_w, m_h) * tile_ratio)
    tile1_edges = img1[i_1 : i_1 + w, j_1 : j_1 + w]
    for (
        fp
    ) in (
        remaining_set_paths
    ):  # eventually change this to just store the binary files first

        temp_best_distance = inf
        temp_best_key = None
        img2 = binary_vecLDs[fp]
        m_h, m_w = img2.shape
        dist1 = distance_transform_edt(1 - img1)
        dist2 = distance_transform_edt(1 - img2)
        print(f"Comparing {curr_fn} to {fp}")
        for i_2 in range(0, int(m_h - w), stride):
            for j_2 in range(0, int(m_w - w), stride):
                tile2_edges = img2[i_2 : i_2 + w, j_2 : j_2 + w]
                if not tile2_edges.any():
                    continue  # skip empty tiles

                # d(A->B): for each edge pixel in tile1,
                # find max distance to nearest edge in tile2
                # Use the precomputed distance transform of img2
                tile2_dist = dist2[i_2 : i_2 + w, j_2 : j_2 + w]
                d_ab = np.max(tile2_dist[tile1_edges == 1]) if tile1_edges.any() else 0

                tile1_dist = dist1[i_1 : i_1 + w, j_1 : j_1 + w]
                d_ba = np.max(tile1_dist[tile2_edges == 1]) if tile2_edges.any() else 0
                dist = max(d_ab, d_ba)
                if dist < temp_best_distance:
                    temp_best_distance = dist
                    temp_best_key = (fp, i_2, j_2, m_h, m_w)
        if temp_best_distance < best_distance:
            best_distance = temp_best_distance
            best_key = temp_best_key
    return best_key


def sequence():
    vecLDs = get_vecLDs("vecLDs_processed.pickle")
    binary_vecLDs = get_binary_by_filename(vecLDs)

    remaining_set_paths = set(binary_vecLDs.keys())
    order = []
    # Random start
    init_fn = remaining_set_paths.pop()
    prev_tile = (
        init_fn,
        200,
        200,
        binary_vecLDs[init_fn].shape[0],
        binary_vecLDs[init_fn].shape[1],
    )
    while len(remaining_set_paths) > 0:
        found_tile = find_next_img(binary_vecLDs, prev_tile, remaining_set_paths)
        prev_tile = found_tile
        order.append(found_tile)
        print(
            f"Added image {found_tile[0]} for tile position ({found_tile[1], found_tile[2]})"
        )
        remaining_set_paths.remove(found_tile[0])
        with open("sequence_order.txt", "w") as f:
            for item in order:
                f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]}\n")
    print(order)


def generate_vid(
    sequence_file, foto_folder, output_path="output.mp4", fps=8, tile_ratio=0.7
):
    """
    Create video using fotos from foto_folder, based on the sequence given in sequence_file.

    The sequence file is a CSV where each line is:
        filename, tile_y, tile_x, img_height, img_width

    The tile coordinates are relative to the line drawing image size (img_height x img_width),
    so they are scaled proportionally to the actual photo dimensions.

    Args:
        sequence_file: Path to the sequence CSV file.
        foto_folder: Path to the folder containing the original photos.
        output_path: Path for the output video file.
        fps: Frames per second for the output video.
        tile_ratio: Ratio used for the tile size (must match what was used in sequencing).
    """
    # Parse the sequence file
    entries = []
    with open(sequence_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            svg_fn = parts[0]
            tile_y = int(parts[1])
            tile_x = int(parts[2])
            ld_h = int(parts[3])
            ld_w = int(parts[4])
            entries.append((svg_fn, tile_y, tile_x, ld_h, ld_w))

    if not entries:
        print("No entries found in sequence file.")
        return

    # Determine a consistent output frame size from the first entry
    # Use the tile size in the actual photo resolution
    # We'll collect all frames first to determine consistent dimensions
    frames = []
    for svg_fn, tile_y, tile_x, ld_h, ld_w in entries:
        # Convert .svg filename to the original photo extension
        base_name = os.path.splitext(svg_fn)[0]

        # Try common image extensions
        foto_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".bmp"]:
            candidate = os.path.join(foto_folder, base_name + ext)
            if os.path.exists(candidate):
                foto_path = candidate
                break

        if foto_path is None:
            print(f"Warning: Could not find photo for {svg_fn}, skipping.")
            continue

        img = cv.imread(foto_path)
        if img is None:
            print(f"Warning: Could not read {foto_path}, skipping.")
            continue

        actual_h, actual_w = img.shape[:2]

        # Scale tile coordinates from line-drawing space to photo space
        scale_y = actual_h / ld_h
        scale_x = actual_w / ld_w

        tile_w_ld = int(min(ld_w, ld_h) * tile_ratio)

        scaled_tile_y = int(tile_y * scale_y)
        scaled_tile_x = int(tile_x * scale_x)
        scaled_tile_h = int(tile_w_ld * scale_y)
        scaled_tile_w = int(tile_w_ld * scale_x)

        # Clamp to image bounds
        scaled_tile_y = min(scaled_tile_y, actual_h - scaled_tile_h)
        scaled_tile_x = min(scaled_tile_x, actual_w - scaled_tile_w)
        scaled_tile_y = max(scaled_tile_y, 0)
        scaled_tile_x = max(scaled_tile_x, 0)
        scaled_tile_h = min(scaled_tile_h, actual_h - scaled_tile_y)
        scaled_tile_w = min(scaled_tile_w, actual_w - scaled_tile_x)

        crop = img[
            scaled_tile_y : scaled_tile_y + scaled_tile_h,
            scaled_tile_x : scaled_tile_x + scaled_tile_w,
        ]
        frames.append(crop)
        print(
            f"Added frame from {base_name}: tile ({scaled_tile_y},{scaled_tile_x}) size ({scaled_tile_h}x{scaled_tile_w})"
        )

    if not frames:
        print("No frames were generated.")
        return

    # Use a consistent output resolution (resize all frames to match the first)
    out_h, out_w = frames[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for frame in frames:
        resized = cv.resize(frame, (out_w, out_h))
        writer.write(resized)

    writer.release()
    print(f"Video saved to {output_path} ({len(frames)} frames, {fps} fps)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "video":
        foto_folder = sys.argv[2] if len(sys.argv) > 2 else "fotos"
        generate_vid("sequence_order.txt", foto_folder)
    else:
        sequence()

"""
Sequence images using a video transformer for next-frame prediction.

Strategy:
  1. Compute CLIP embeddings for all images to build a KNN shortlist.
  2. Greedily build a sequence:
     a. Start with 1 image (best connected by CLIP similarity).
     b. For each step, consider the K=30 most similar unvisited images.
     c. For each candidate image, evaluate multiple tile positions/sizes.
     d. Use a video transformer to score how well each candidate tile
        continues the existing sequence (lower perplexity = better fit).
     e. Pick the best (image, tile) and append to the sequence.
  3. Output sequence file compatible with generate_vid().

Requires ~32GB VRAM for the video model.
"""

import os
import sys
import gc
import time
from math import inf

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from PIL import Image

import open_clip
from scipy.spatial.distance import cdist

# ── Config ──────────────────────────────────────────────────────────────────
T_R_MIN = 0.6
T_R_MAX = 1.0
T_R_STRIDE = 0.1
STRIDE = 50
TARGET_SHORT_EDGE = 512
K_NEIGHBORS = 30
MAX_CONTEXT_FRAMES = 12  # max prior frames to feed as context to the video model
TILE_CANDIDATES_PER_IMAGE = 20  # max tile positions to evaluate per candidate image
FRAME_SIZE = 480  # video model input resolution
BATCH_EVAL_SIZE = 4  # how many candidates to evaluate in parallel

# Tile ratios
TILE_RATIOS = []
_r = T_R_MIN
while _r <= T_R_MAX + 1e-9:
    TILE_RATIOS.append(round(_r, 6))
    _r += T_R_STRIDE
TILE_RATIOS = sorted(set(TILE_RATIOS))


# ── CLIP Embeddings ─────────────────────────────────────────────────────────


def compute_clip_embeddings(filenames, image_folder):
    """Compute L2-normalized CLIP embeddings for all images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test CUDA actually works before committing
    if device == "cuda":
        try:
            test = torch.randn(1, 3, 16, 16, device=device)
            _ = F.conv2d(test, torch.randn(1, 3, 3, 3, device=device), padding=1)
            del test, _
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  CUDA test failed ({e}), falling back to CPU for CLIP")
            device = "cpu"

    print(f"  Loading CLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

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
                batch_tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad(), torch.amp.autocast(
            device_type="cuda" if device == "cuda" else "cpu"
        ):
            features = model.encode_image(batch)
            features = features.float()
            features /= features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu().numpy())

        done = min(batch_start + batch_size, len(filenames))
        if done % 100 == 0 or done == len(filenames):
            print(f"    {done}/{len(filenames)} embeddings computed")

    # Free CLIP model
    del model
    torch.cuda.empty_cache()
    gc.collect()

    embeddings = np.vstack(embeddings).astype(np.float32)
    print(f"  CLIP embeddings shape: {embeddings.shape}")
    return embeddings


# ── Image Loading & Tile Extraction ─────────────────────────────────────────


def load_image_dimensions(
    image_folder, extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp")
):
    """Load filenames and their dimensions (at the working scale)."""
    image_info = {}  # fn -> (scaled_h, scaled_w)
    for fn in sorted(os.listdir(image_folder)):
        if os.path.splitext(fn)[1].lower() in extensions:
            path = os.path.join(image_folder, fn)
            try:
                img = cv.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = TARGET_SHORT_EDGE / min(h, w)
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                image_info[fn] = (scaled_h, scaled_w)
                print(
                    f"  loaded: {fn}  original=({h},{w})  scaled=({scaled_h},{scaled_w})"
                )
            except Exception as e:
                print(f"  SKIP {fn}: {e}")
    return image_info


def get_valid_tile_positions(
    scaled_h, scaled_w, tile_ratios=TILE_RATIOS, stride=STRIDE
):
    """
    Return list of (y, x, tile_size) valid tile positions for an image,
    based only on geometry (no edge map needed).
    """
    positions = []
    for ratio in tile_ratios:
        ts = int(min(scaled_h, scaled_w) * ratio)
        if ts < 16 or ts > scaled_h or ts > scaled_w:
            continue
        for y in range(0, scaled_h - ts + 1, stride):
            for x in range(0, scaled_w - ts + 1, stride):
                positions.append((y, x, ts))
    if not positions:
        ts = int(min(scaled_h, scaled_w) * tile_ratios[0])
        ts = max(16, min(ts, scaled_h, scaled_w))
        positions.append(((scaled_h - ts) // 2, (scaled_w - ts) // 2, ts))
    return positions


def extract_tile_rgb(image_path, ty, tx, ts, output_size=FRAME_SIZE):
    """Extract a tile from the full-resolution image and resize to output_size."""
    img = cv.imread(image_path)
    if img is None:
        return None
    actual_h, actual_w = img.shape[:2]

    # Tile coords are in TARGET_SHORT_EDGE scale; map back to original
    scale = TARGET_SHORT_EDGE / min(actual_h, actual_w)
    inv_scale = 1.0 / scale

    # Map tile coordinates to original image space
    orig_ts = int(ts * inv_scale)
    orig_ty = int(ty * inv_scale)
    orig_tx = int(tx * inv_scale)

    # Clamp to image bounds
    orig_ts = max(16, min(orig_ts, actual_h, actual_w))
    orig_ty = max(0, min(orig_ty, actual_h - orig_ts))
    orig_tx = max(0, min(orig_tx, actual_w - orig_ts))

    crop = img[orig_ty : orig_ty + orig_ts, orig_tx : orig_tx + orig_ts]
    if crop.size == 0:
        return None

    crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    crop_resized = cv.resize(
        crop_rgb, (output_size, output_size), interpolation=cv.INTER_AREA
    )
    return crop_resized


def sample_tile_positions(all_positions, max_candidates=TILE_CANDIDATES_PER_IMAGE):
    """
    Sub-sample tile positions if there are too many. Prefers diversity
    by stratified sampling across tile sizes.
    """
    if len(all_positions) <= max_candidates:
        return all_positions

    # Group by tile size
    by_size = {}
    for pos in all_positions:
        ts = pos[2]
        by_size.setdefault(ts, []).append(pos)

    # Allocate roughly equal budget per tile size
    per_size = max(1, max_candidates // len(by_size))
    sampled = []
    rng = np.random.default_rng(42)
    for ts, positions in sorted(by_size.items()):
        if len(positions) <= per_size:
            sampled.extend(positions)
        else:
            idxs = rng.choice(len(positions), size=per_size, replace=False)
            sampled.extend([positions[i] for i in idxs])

    # If still over budget, randomly trim
    if len(sampled) > max_candidates:
        idxs = rng.choice(len(sampled), size=max_candidates, replace=False)
        sampled = [sampled[i] for i in idxs]

    return sampled


# ── Video Transformer Scorer ───────────────────────────────────────────────


class VideoTransformerScorer:
    """
    Uses a video generation/prediction model to score how well a candidate
    frame continues an existing sequence.

    Uses CogVideoX-2b (~20-24GB VRAM with fp16) to compute a pseudo-likelihood.

    The approach:
      - Encode the context frames + candidate as a video clip.
      - Measure reconstruction loss in latent space.
      - Lower reconstruction loss = better continuation.
    """

    def __init__(self, device="cuda", model_name="THUDM/CogVideoX-2b"):
        self.device = device
        self.model_name = model_name
        self.pipe = None
        self.vae = None
        self.transformer = None
        self.scheduler = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        print(f"  Loading video transformer: {self.model_name}...")
        print(f"  This may take a minute and ~20GB VRAM...")

        try:
            from diffusers import CogVideoXPipeline

            self.pipe = CogVideoXPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            )

            # Extract components we need
            self.vae = self.pipe.vae.to(self.device)
            self.vae.eval()
            self.vae.enable_slicing()

            self.transformer = self.pipe.transformer.to(self.device)
            self.transformer.eval()

            self.scheduler = self.pipe.scheduler

            self.text_encoder = self.pipe.text_encoder.to(self.device)
            self.text_encoder.eval()
            self.tokenizer = self.pipe.tokenizer

            self._loaded = True
            print(f"  Video transformer loaded successfully.")
            print(f"  VRAM used: ~{torch.cuda.memory_allocated() / 1e9:.1f}GB")

        except ImportError:
            print("  diffusers not available, falling back to VAE-only scoring.")
            self._load_vae_only()
        except Exception as e:
            print(f"  Failed to load CogVideoX: {e}")
            print("  Falling back to VAE-only scoring.")
            self._load_vae_only()

    def _load_vae_only(self):
        """Fallback: use video-aware VAE for scoring."""
        try:
            from diffusers.models import AutoencoderKLCogVideoX

            self.vae = AutoencoderKLCogVideoX.from_pretrained(
                self.model_name,
                subfolder="vae",
                torch_dtype=torch.float16,
            ).to(self.device)
            self.vae.eval()
            self.vae.enable_slicing()
            self.transformer = None
            self._loaded = True
            print("  VAE-only scorer loaded.")
        except Exception as e2:
            print(f"  VAE fallback also failed: {e2}")
            print("  Will use perceptual scoring instead.")
            self._loaded = True
            self.vae = None
            self.transformer = None

    def unload(self):
        """Free all model memory."""
        for attr in ("vae", "transformer", "text_encoder", "pipe"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()

    def _frames_to_tensor(self, frames, size=FRAME_SIZE):
        """
        Convert list of numpy RGB frames (H, W, 3) uint8 to
        video tensor (B=1, C, T, H, W) normalized to [-1, 1].
        """
        tensors = []
        for f in frames:
            if f.shape[0] != size or f.shape[1] != size:
                f = cv.resize(f, (size, size), interpolation=cv.INTER_AREA)
            t = torch.from_numpy(f.copy()).float().permute(2, 0, 1) / 127.5 - 1.0
            tensors.append(t)
        video = torch.stack(tensors, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        return video.to(self.device, dtype=torch.float16)

    def score_candidates(self, context_frames, candidate_frames):
        """
        Score how well each candidate frame continues the context sequence.

        Returns:
            scores: list of float, lower = better continuation
        """
        if not self._loaded:
            self.load()

        if self.vae is None and self.transformer is None:
            return self._score_perceptual(context_frames, candidate_frames)

        if self.transformer is not None:
            return self._score_with_transformer(context_frames, candidate_frames)
        else:
            return self._score_with_vae(context_frames, candidate_frames)

    @torch.no_grad()
    def _score_with_transformer(self, context_frames, candidate_frames):
        """Score using the full transformer via single-step denoising."""
        scores = []

        # Null text embedding for unconditional scoring
        null_tokens = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        with torch.no_grad():
            null_embeds = self.text_encoder(null_tokens)[0].half()

        for i in range(0, len(candidate_frames), BATCH_EVAL_SIZE):
            batch = candidate_frames[i : i + BATCH_EVAL_SIZE]

            for cand_frame in batch:
                full_seq = context_frames[-MAX_CONTEXT_FRAMES:] + [cand_frame]
                full_tensor = self._frames_to_tensor(full_seq)

                try:
                    full_latent = self.vae.encode(full_tensor).latent_dist.mean

                    noise = torch.randn_like(full_latent) * 0.1
                    noisy_latent = full_latent + noise

                    self.scheduler.set_timesteps(50)
                    t = self.scheduler.timesteps[45]
                    timesteps = torch.tensor([t], device=self.device).long()

                    model_output = self.transformer(
                        hidden_states=noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=null_embeds,
                    )
                    predicted = (
                        model_output.sample
                        if hasattr(model_output, "sample")
                        else model_output[0]
                    )

                    recon_error = F.mse_loss(
                        predicted[:, :, -1:], full_latent[:, :, -1:]
                    ).item()
                    scores.append(recon_error)
                except Exception as e:
                    print(f"    Transformer scoring failed: {e}, using VAE fallback")
                    scores.append(self._score_single_vae(context_frames, cand_frame))

        return scores

    @torch.no_grad()
    def _score_with_vae(self, context_frames, candidate_frames):
        """VAE-only scoring via temporal reconstruction error."""
        return [self._score_single_vae(context_frames, cf) for cf in candidate_frames]

    @torch.no_grad()
    def _score_single_vae(self, context_frames, cand_frame):
        """Score a single candidate using VAE reconstruction."""
        full_seq = context_frames[-MAX_CONTEXT_FRAMES:] + [cand_frame]
        while len(full_seq) < 2:
            full_seq = [full_seq[0]] + full_seq

        full_tensor = self._frames_to_tensor(full_seq)

        try:
            latent = self.vae.encode(full_tensor).latent_dist.mean
            recon = self.vae.decode(latent).sample

            orig_last = full_tensor[:, :, -1:]
            recon_last = recon[:, :, -1:]
            error = F.mse_loss(recon_last, orig_last).item()

            if recon.shape[2] >= 2:
                temporal_diff = F.mse_loss(
                    recon[:, :, -1:] - recon[:, :, -2:-1],
                    full_tensor[:, :, -1:] - full_tensor[:, :, -2:-1],
                ).item()
                error = 0.7 * error + 0.3 * temporal_diff

            return error
        except Exception as e:
            print(f"    VAE scoring error: {e}")
            return self._score_perceptual_single(context_frames, cand_frame)

    def _score_perceptual(self, context_frames, candidate_frames):
        """Perceptual fallback using pixel-space temporal smoothness."""
        return [
            self._score_perceptual_single(context_frames, cf) for cf in candidate_frames
        ]

    def _score_perceptual_single(self, context_frames, cand_frame):
        """
        Simple perceptual scoring: color histogram continuity +
        edge structure similarity + motion smoothness.
        """
        if not context_frames:
            return 0.0

        last_frame = context_frames[-1]
        size = 256
        a = cv.resize(last_frame, (size, size))
        b = cv.resize(cand_frame, (size, size))

        # Color histogram distance
        hist_dist = 0.0
        for ch in range(3):
            h_a = cv.calcHist([a], [ch], None, [64], [0, 256])
            h_b = cv.calcHist([b], [ch], None, [64], [0, 256])
            cv.normalize(h_a, h_a)
            cv.normalize(h_b, h_b)
            hist_dist += cv.compareHist(h_a, h_b, cv.HISTCMP_BHATTACHARYYA)
        hist_dist /= 3.0

        # Edge structure similarity
        gray_a = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
        gray_b = cv.cvtColor(b, cv.COLOR_RGB2GRAY)
        edges_a = cv.Canny(gray_a, 50, 150).astype(np.float32)
        edges_b = cv.Canny(gray_b, 50, 150).astype(np.float32)
        if edges_a.max() > 0:
            edges_a /= edges_a.max()
        if edges_b.max() > 0:
            edges_b /= edges_b.max()
        edge_dist = np.mean((edges_a - edges_b) ** 2)

        # Pixel MSE
        pixel_dist = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) / (
            255.0**2
        )

        # Motion smoothness penalty
        motion_penalty = 0.0
        if len(context_frames) >= 2:
            prev_frame = context_frames[-2]
            p = cv.resize(prev_frame, (size, size)).astype(np.float32)
            a_f = a.astype(np.float32)
            b_f = b.astype(np.float32)
            vel_prev = a_f - p
            vel_next = b_f - a_f
            accel = np.mean((vel_next - vel_prev) ** 2) / (255.0**2)
            motion_penalty = accel * 0.5

        return (
            0.3 * hist_dist + 0.3 * edge_dist + 0.2 * pixel_dist + 0.2 * motion_penalty
        )


# ── Main Sequencing Pipeline ────────────────────────────────────────────────


def find_start_image(dist_matrix):
    """Find the image most connected to others (lowest average distance)."""
    avg_dist = dist_matrix.mean(axis=1)
    return int(np.argmin(avg_dist))


def sequence_with_video_transformer(
    image_folder,
    output_file="sequence_order_vt.txt",
    max_context=MAX_CONTEXT_FRAMES,
    k_neighbors=K_NEIGHBORS,
):
    """
    Greedily build an image sequence using a video transformer to score
    candidate next frames.
    """
    print("=" * 70)
    print("Video Transformer Guided Image Sequencing")
    print("=" * 70)

    # Step 1: Load image dimensions
    print("\nStep 1/4: Loading image dimensions...")
    image_info = load_image_dimensions(image_folder)
    filenames = list(image_info.keys())
    n = len(filenames)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    # Step 2: CLIP embeddings and KNN
    print("Step 2/4: Computing CLIP embeddings and KNN graph...")
    embeddings = compute_clip_embeddings(filenames, image_folder)
    dist_matrix = cdist(embeddings, embeddings, metric="cosine")
    print(f"  Distance matrix: {dist_matrix.shape}\n")

    knn = {}
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = inf
        nearest = np.argsort(dists)[:k_neighbors]
        knn[i] = nearest.tolist()

    # Precompute valid tile positions for each image (geometry only)
    print("Step 3/4: Computing valid tile positions...")
    tile_positions_per_image = {}
    for idx, fn in enumerate(filenames):
        scaled_h, scaled_w = image_info[fn]
        positions = get_valid_tile_positions(scaled_h, scaled_w)
        tile_positions_per_image[idx] = positions
        if (idx + 1) % 50 == 0 or idx + 1 == n:
            print(
                f"    {idx + 1}/{n} images processed "
                f"(avg {np.mean([len(v) for v in tile_positions_per_image.values()]):.0f} positions/image)"
            )
    print()

    # Step 4: Load video transformer and run greedy sequencing
    print("Step 4/4: Loading video transformer and building sequence...")
    scorer = VideoTransformerScorer(device="cuda")
    scorer.load()
    print()

    print("Greedy sequence construction...")
    start_idx = find_start_image(dist_matrix)
    print(f"  Starting with image {start_idx}: {filenames[start_idx]}")

    visited = {start_idx}
    path = [start_idx]

    # Default tile for start image: center crop, largest ratio
    scaled_h, scaled_w = image_info[filenames[start_idx]]
    start_ts = int(min(scaled_h, scaled_w) * TILE_RATIOS[-1])
    start_ts = max(16, min(start_ts, scaled_h, scaled_w))
    start_ty = (scaled_h - start_ts) // 2
    start_tx = (scaled_w - start_ts) // 2
    tile_choices = [(start_ty, start_tx, start_ts)]

    start_frame = extract_tile_rgb(
        os.path.join(image_folder, filenames[start_idx]),
        start_ty,
        start_tx,
        start_ts,
    )
    if start_frame is None:
        img = cv.imread(os.path.join(image_folder, filenames[start_idx]))
        start_frame = cv.cvtColor(
            cv.resize(img, (FRAME_SIZE, FRAME_SIZE)), cv.COLOR_BGR2RGB
        )

    context_frames = [start_frame]

    for step in range(n - 1):
        current_idx = path[-1]
        t_start = time.time()

        # Get K nearest unvisited neighbors
        candidates_idx = []
        for j in knn[current_idx]:
            if j not in visited:
                candidates_idx.append(j)
            if len(candidates_idx) >= k_neighbors:
                break

        if len(candidates_idx) < k_neighbors:
            dists = dist_matrix[current_idx].copy()
            for v in visited:
                dists[v] = inf
            remaining_sorted = np.argsort(dists)
            for j in remaining_sorted:
                j = int(j)
                if j not in visited and j not in candidates_idx:
                    candidates_idx.append(j)
                if len(candidates_idx) >= k_neighbors:
                    break

        if not candidates_idx:
            print(f"  No more candidates at step {step + 1}!")
            break

        # For each candidate, sample tile positions and extract frames
        all_candidate_info = []  # (image_idx, ty, tx, ts, frame)
        for cand_idx in candidates_idx:
            fn = filenames[cand_idx]
            positions = tile_positions_per_image.get(cand_idx, [])
            sampled_positions = sample_tile_positions(
                positions, TILE_CANDIDATES_PER_IMAGE
            )

            for ty, tx, ts in sampled_positions:
                frame = extract_tile_rgb(os.path.join(image_folder, fn), ty, tx, ts)
                if frame is not None:
                    all_candidate_info.append((cand_idx, ty, tx, ts, frame))

        if not all_candidate_info:
            best_idx = candidates_idx[0]
            fn = filenames[best_idx]
            scaled_h, scaled_w = image_info[fn]
            ts = int(min(scaled_h, scaled_w) * TILE_RATIOS[-1])
            ts = max(16, min(ts, scaled_h, scaled_w))
            ty, tx = (scaled_h - ts) // 2, (scaled_w - ts) // 2
            frame = extract_tile_rgb(os.path.join(image_folder, fn), ty, tx, ts)
            if frame is None:
                img = cv.imread(os.path.join(image_folder, fn))
                frame = cv.cvtColor(
                    cv.resize(img, (FRAME_SIZE, FRAME_SIZE)), cv.COLOR_BGR2RGB
                )
            all_candidate_info = [(best_idx, ty, tx, ts, frame)]

        # Score all candidates
        context = context_frames[-max_context:]
        candidate_frames_list = [info[4] for info in all_candidate_info]

        print(
            f"  Step {step + 1}/{n - 1}: scoring {len(candidate_frames_list)} candidates "
            f"from {len(candidates_idx)} images...",
            end="",
            flush=True,
        )

        scores = scorer.score_candidates(context, candidate_frames_list)

        best_score_idx = int(np.argmin(scores))
        best_img_idx, best_ty, best_tx, best_ts, best_frame = all_candidate_info[
            best_score_idx
        ]
        best_score = scores[best_score_idx]

        path.append(best_img_idx)
        visited.add(best_img_idx)
        tile_choices.append((best_ty, best_tx, best_ts))
        context_frames.append(best_frame)

        if len(context_frames) > max_context * 2:
            context_frames = context_frames[-(max_context + 2) :]

        elapsed = time.time() - t_start
        print(
            f" → {filenames[best_img_idx]} "
            f"(tile={best_ts}px @ ({best_ty},{best_tx}), "
            f"score={best_score:.6f}, {elapsed:.1f}s)"
        )

        if (step + 1) % 20 == 0:
            _write_sequence(
                output_file + ".checkpoint",
                path,
                filenames,
                image_info,
                tile_choices,
            )
            print(f"    Checkpoint saved ({step + 1}/{n - 1})")

    scorer.unload()

    _write_sequence(output_file, path, filenames, image_info, tile_choices)

    print(f"\nSequence written to {output_file}")
    print(f"Sequenced {len(path)}/{n} images.")
    print("\nOrder:")
    for idx in path:
        print(f"  {filenames[idx]}")


def _write_sequence(output_file, path, filenames, image_info, tile_choices):
    """Write the sequence file: filename,ty,tx,ts,scaled_h,scaled_w"""
    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            scaled_h, scaled_w = image_info[fn]
            ty, tx, ts = tile_choices[k]
            f.write(f"{fn},{ty},{tx},{ts},{scaled_h},{scaled_w}\n")


# ── Video Generation ────────────────────────────────────────────────────────


def generate_vid(
    sequence_file,
    foto_folder,
    output_path="output_vt.mp4",
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

    out_h = max(f.shape[0] for f in frames)
    out_w = max(f.shape[1] for f in frames)
    out_size = min(out_h, out_w)

    writer = cv.VideoWriter(
        output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (out_size, out_size)
    )
    for frame in frames:
        writer.write(cv.resize(frame, (out_size, out_size)))
    writer.release()
    print(f"Video saved: {output_path} ({len(frames)} frames @ {fps}fps)")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        foto_folder = sys.argv[2] if len(sys.argv) > 2 else "."
        seq_file = sys.argv[3] if len(sys.argv) > 3 else "sequence_order_vt.txt"
        generate_vid(seq_file, foto_folder)
    else:
        image_folder = sys.argv[1] if len(sys.argv) > 1 else "."
        sequence_with_video_transformer(image_folder)

"""
Sequence images using a video transformer for next-frame prediction.

Strategy:
  1. Compute CLIP embeddings for all images to build a KNN shortlist.
  2. Compute edge maps for tile extraction.
  3. Greedily build a sequence:
     a. Start with 1 image (best connected by CLIP similarity).
     b. For each step, consider the K=30 most similar unvisited images.
     c. For each candidate image, evaluate multiple tile positions/sizes.
     d. Use a video transformer to score how well each candidate tile
        continues the existing sequence (lower perplexity = better fit).
     e. Pick the best (image, tile) and append to the sequence.
  4. Output sequence file compatible with generate_vid().

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
EDGE_THRESHOLD = 0.1
MIN_EDGE_DENSITY = 0.01
EDGE_BLUR_KSIZE = 3
EDGE_GAMMA = 1.5
EDGE_THRESHOLD_LOW = 0.6
MIN_EDGE_LENGTH = 30
EDGE_METHOD = "hed"

# Tile ratios
TILE_RATIOS = []
_r = T_R_MIN
while _r <= T_R_MAX + 1e-9:
    TILE_RATIOS.append(round(_r, 6))
    _r += T_R_STRIDE
TILE_RATIOS = sorted(set(TILE_RATIOS))

# ── HED Model ───────────────────────────────────────────────────────────────

_hed_net = None
_hed_crop_registered = False
HED_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HED_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
)
HED_CAFFEMODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"


def _download_file(url, dest):
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"    Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"    Saved to {dest}")


def _get_hed_model():
    global _hed_net, _hed_crop_registered
    if _hed_net is not None:
        return _hed_net

    prototxt_path = os.path.join(HED_MODEL_DIR, "deploy.prototxt")
    caffemodel_path = os.path.join(HED_MODEL_DIR, "hed_pretrained_bsds.caffemodel")

    if not os.path.exists(prototxt_path):
        _download_file(HED_PROTOTXT_URL, prototxt_path)
    if not os.path.exists(caffemodel_path):
        _download_file(HED_CAFFEMODEL_URL, caffemodel_path)

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

    print("  Loading HED model...")
    _hed_net = cv.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            _hed_net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            _hed_net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    print("  HED model loaded.")
    return _hed_net


def _hed_edge_map(img_bgr, target_short_edge=TARGET_SHORT_EDGE):
    h, w = img_bgr.shape[:2]
    scale = target_short_edge / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv.resize(img_bgr, (new_w, new_h), interpolation=cv.INTER_AREA)
    net = _get_hed_model()
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
    edge_map = out[0, 0]
    return np.clip(edge_map, 0, 1).astype(np.float32)


def get_weighted_edges(image_path, target_short_edge=TARGET_SHORT_EDGE):
    if EDGE_METHOD == "hed":
        img_bgr = cv.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        magnitude = _hed_edge_map(img_bgr, target_short_edge)
    else:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {image_path}")
        h, w = img.shape
        scale = target_short_edge / min(h, w)
        img = cv.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA
        )
        img = cv.GaussianBlur(img, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)
        grad_x = cv.Scharr(img, cv.CV_64F, 1, 0)
        grad_y = cv.Scharr(img, cv.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude /= mag_max
        magnitude = magnitude.astype(np.float32)

    magnitude[magnitude < EDGE_THRESHOLD_LOW] = 0.0
    edge_binary = (magnitude > 0).astype(np.uint8)
    num_labels, labels = cv.connectedComponents(edge_binary, connectivity=8)
    component_sizes = np.bincount(labels.ravel())
    small_labels = np.where(component_sizes < MIN_EDGE_LENGTH)[0]
    keep_mask = np.ones(num_labels, dtype=bool)
    keep_mask[0] = False
    keep_mask[small_labels] = False
    magnitude = magnitude * keep_mask[labels]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max
    magnitude = np.power(magnitude, EDGE_GAMMA)
    return magnitude.astype(np.float32)


# ── CLIP Embeddings ─────────────────────────────────────────────────────────


def compute_clip_embeddings(filenames, image_folder):
    """Compute L2-normalized CLIP embeddings for all images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


# ── Tile Extraction ─────────────────────────────────────────────────────────


def get_valid_tile_positions(edge_map, tile_ratios=TILE_RATIOS, stride=STRIDE):
    """Return list of (y, x, tile_size) valid tile positions for an image."""
    h, w = edge_map.shape
    positions = []
    for ratio in tile_ratios:
        ts = int(min(h, w) * ratio)
        if ts < 16 or ts > h or ts > w:
            continue
        for y in range(0, h - ts + 1, stride):
            for x in range(0, w - ts + 1, stride):
                tile = edge_map[y : y + ts, x : x + ts]
                if tile.shape[0] != ts or tile.shape[1] != ts:
                    continue
                mask = tile > EDGE_THRESHOLD
                if mask.mean() >= MIN_EDGE_DENSITY and tile[mask].sum() > 0:
                    positions.append((y, x, ts))
    if not positions:
        ts = int(min(h, w) * tile_ratios[0])
        ts = max(16, min(ts, h, w))
        positions.append(((h - ts) // 2, (w - ts) // 2, ts))
    return positions


def extract_tile_rgb(image_path, ty, tx, ts, output_size=FRAME_SIZE):
    """Extract a tile from the full-resolution image and resize to output_size."""
    img = cv.imread(image_path)
    if img is None:
        return None
    actual_h, actual_w = img.shape[:2]

    # Edge maps are computed at TARGET_SHORT_EDGE scale; map back to original
    edge_scale = TARGET_SHORT_EDGE / min(actual_h, actual_w)
    inv_scale = 1.0 / edge_scale

    sy = max(0, min(int(ty * inv_scale), actual_h - int(ts * inv_scale)))
    sx = max(0, min(int(tx * inv_scale), actual_w - int(ts * inv_scale)))
    sts = max(16, min(int(ts * inv_scale), actual_h - sy, actual_w - sx))

    crop = img[sy : sy + sts, sx : sx + sts]
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

    We use CogVideoX-2b (a relatively compact video transformer that fits
    in ~20-24GB VRAM with fp16) to compute a pseudo-likelihood score.

    The approach:
      - Encode the context frames as a video clip.
      - For each candidate next frame, compute how well the model "predicts"
        it given the context by measuring reconstruction loss (MSE in latent
        space after encoding the full sequence including the candidate).
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
            from diffusers.models import AutoencoderKLCogVideoX

            # Load just the VAE for encoding frames into latent space
            # The VAE's reconstruction quality tells us about frame coherence
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

            # Free the full pipeline to save memory
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
        """Fallback: use a video-aware VAE or image VAE for scoring."""
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
            print("  Will use perceptual (LPIPS-like) scoring instead.")
            self._loaded = True
            self.vae = None
            self.transformer = None

    def unload(self):
        """Free all model memory."""
        if self.vae is not None:
            del self.vae
            self.vae = None
        if self.transformer is not None:
            del self.transformer
            self.transformer = None
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()

    def _frames_to_tensor(self, frames, size=FRAME_SIZE):
        """
        Convert list of numpy RGB frames (H, W, 3) uint8 to a batched
        video tensor of shape (B, C, T, H, W) normalized to [-1, 1].
        """
        tensors = []
        for f in frames:
            if f.shape[0] != size or f.shape[1] != size:
                f = cv.resize(f, (size, size), interpolation=cv.INTER_AREA)
            t = torch.from_numpy(f).float().permute(2, 0, 1) / 127.5 - 1.0
            tensors.append(t)
        # (T, C, H, W)
        video = torch.stack(tensors, dim=0)
        # (B=1, C, T, H, W)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
        # CogVideoX VAE expects (B, C, T, H, W)
        return video.to(self.device, dtype=torch.float16)

    def score_candidates(self, context_frames, candidate_frames):
        """
        Score how well each candidate frame continues the context sequence.

        Args:
            context_frames: list of numpy RGB frames (the sequence so far)
            candidate_frames: list of numpy RGB frames (one per candidate)

        Returns:
            scores: list of float, lower = better continuation
        """
        if not self._loaded:
            self.load()

        if self.vae is None and self.transformer is None:
            # Perceptual fallback: just use pixel-space temporal smoothness
            return self._score_perceptual(context_frames, candidate_frames)

        if self.transformer is not None:
            return self._score_with_transformer(context_frames, candidate_frames)
        else:
            return self._score_with_vae(context_frames, candidate_frames)

    @torch.no_grad()
    def _score_with_transformer(self, context_frames, candidate_frames):
        """
        Score using the full transformer. We encode context + candidate into
        latent space, run a single denoising step, and measure how close the
        model's prediction is to the actual latent — lower error means the
        model "expects" this frame given the context.
        """
        scores = []

        # Encode context frames once
        if len(context_frames) > 0:
            ctx_tensor = self._frames_to_tensor(context_frames)
            ctx_latent = self.vae.encode(ctx_tensor).latent_dist.mean
        else:
            ctx_latent = None

        # Get a null text embedding for unconditional scoring
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
            batch_scores = []

            for cand_frame in batch:
                # Build full sequence: context + candidate
                full_seq = context_frames[-MAX_CONTEXT_FRAMES:] + [cand_frame]
                full_tensor = self._frames_to_tensor(full_seq)

                # Encode to latent space
                full_latent = self.vae.encode(full_tensor).latent_dist.mean

                # Add a small amount of noise and try to denoise
                # The reconstruction quality indicates temporal coherence
                noise = torch.randn_like(full_latent) * 0.1
                noisy_latent = full_latent + noise

                # Use scheduler to set up a single step
                self.scheduler.set_timesteps(50)
                t = self.scheduler.timesteps[45]  # mild noise level
                timesteps = torch.tensor([t], device=self.device).long()

                # Forward through transformer
                try:
                    model_output = self.transformer(
                        hidden_states=noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=null_embeds,
                    )
                    if hasattr(model_output, "sample"):
                        predicted = model_output.sample
                    else:
                        predicted = model_output[0]

                    # Score: how well does the model reconstruct the sequence?
                    # Focus on the last frame (the candidate)
                    recon_error = F.mse_loss(
                        predicted[:, :, -1:], full_latent[:, :, -1:]
                    ).item()
                    batch_scores.append(recon_error)
                except Exception as e:
                    print(f"    Transformer scoring failed: {e}, using VAE fallback")
                    batch_scores.append(
                        self._score_single_vae(context_frames, cand_frame)
                    )

            scores.extend(batch_scores)

        return scores

    @torch.no_grad()
    def _score_with_vae(self, context_frames, candidate_frames):
        """
        VAE-only scoring: encode context + candidate as a video clip,
        decode, and measure reconstruction error. The VAE's temporal
        compression means temporally coherent sequences reconstruct better.
        """
        scores = []

        for cand_frame in candidate_frames:
            scores.append(self._score_single_vae(context_frames, cand_frame))

        return scores

    @torch.no_grad()
    def _score_single_vae(self, context_frames, cand_frame):
        """Score a single candidate using VAE reconstruction."""
        full_seq = context_frames[-MAX_CONTEXT_FRAMES:] + [cand_frame]

        # CogVideoX VAE needs at least a few frames
        # Pad if necessary
        while len(full_seq) < 2:
            full_seq = [full_seq[0]] + full_seq

        full_tensor = self._frames_to_tensor(full_seq)

        try:
            latent = self.vae.encode(full_tensor).latent_dist.mean
            recon = self.vae.decode(latent).sample

            # Reconstruction error on the last frame
            orig_last = full_tensor[:, :, -1:]
            recon_last = recon[:, :, -1:]
            error = F.mse_loss(recon_last, orig_last).item()

            # Also measure temporal consistency: difference between
            # reconstruction of last two frames
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
        Simple perceptual scoring: weighted combination of
        color histogram continuity + edge structure similarity.
        """
        if not context_frames:
            return 0.0

        last_frame = context_frames[-1]

        # Resize both to common size
        size = 256
        a = cv.resize(last_frame, (size, size))
        b = cv.resize(cand_frame, (size, size))

        # Color histogram distance (per channel)
        hist_dist = 0.0
        for ch in range(3):
            h_a = cv.calcHist([a], [ch], None, [64], [0, 256])
            h_b = cv.calcHist([b], [ch], None, [64], [0, 256])
            cv.normalize(h_a, h_a)
            cv.normalize(h_b, h_b)
            hist_dist += cv.compareHist(h_a, h_b, cv.HISTCMP_BHATTACHARYYA)
        hist_dist /= 3.0

        # Structural similarity via edge comparison
        gray_a = cv.cvtColor(a, cv.COLOR_RGB2GRAY)
        gray_b = cv.cvtColor(b, cv.COLOR_RGB2GRAY)
        edges_a = cv.Canny(gray_a, 50, 150).astype(np.float32)
        edges_b = cv.Canny(gray_b, 50, 150).astype(np.float32)
        if edges_a.max() > 0:
            edges_a /= edges_a.max()
        if edges_b.max() > 0:
            edges_b /= edges_b.max()
        edge_dist = np.mean((edges_a - edges_b) ** 2)

        # Pixel MSE (gentle weight)
        pixel_dist = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2) / (
            255.0**2
        )

        # Multi-frame smoothness bonus: if we have 2+ context frames, penalize
        # abrupt changes in the motion direction
        motion_penalty = 0.0
        if len(context_frames) >= 2:
            prev_frame = context_frames[-2]
            p = cv.resize(prev_frame, (size, size)).astype(np.float32)
            a_f = a.astype(np.float32)
            b_f = b.astype(np.float32)
            # "velocity" of previous transition
            vel_prev = a_f - p
            # "velocity" of candidate transition
            vel_next = b_f - a_f
            # Acceleration (change in velocity) — should be small for smooth motion
            accel = np.mean((vel_next - vel_prev) ** 2) / (255.0**2)
            motion_penalty = accel * 0.5

        score = (
            0.3 * hist_dist + 0.3 * edge_dist + 0.2 * pixel_dist + 0.2 * motion_penalty
        )
        return score


# ── Edge Map Building ───────────────────────────────────────────────────────


def build_edge_maps(image_folder, extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp")):
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

    Pipeline:
      1. Load images, compute edge maps and CLIP embeddings.
      2. Build KNN graph from CLIP embeddings.
      3. Load video transformer.
      4. Greedy loop:
         a. Get K nearest unvisited neighbors of current image.
         b. For each neighbor, sample tile positions.
         c. Extract tile crops at full resolution.
         d. Score each (neighbor, tile) combo with the video transformer.
         e. Pick the best, append to sequence.
      5. Write output.
    """
    print("=" * 70)
    print("Video Transformer Guided Image Sequencing")
    print("=" * 70)

    # Step 1: Edge maps
    print("\nStep 1/5: Building edge maps...")
    edge_maps = build_edge_maps(image_folder)
    filenames = list(edge_maps.keys())
    n = len(filenames)
    print(f"  Found {n} images.\n")

    if n < 2:
        print("Need at least 2 images.")
        return

    # Step 2: CLIP embeddings and KNN
    print("Step 2/5: Computing CLIP embeddings and KNN graph...")
    embeddings = compute_clip_embeddings(filenames, image_folder)
    dist_matrix = cdist(embeddings, embeddings, metric="cosine")
    print(f"  Distance matrix: {dist_matrix.shape}\n")

    # Precompute KNN for each image
    knn = {}
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = inf
        nearest = np.argsort(dists)[:k_neighbors]
        knn[i] = nearest.tolist()

    # Precompute valid tile positions for each image
    print("Step 3/5: Computing valid tile positions...")
    tile_positions_per_image = {}
    for idx, fn in enumerate(filenames):
        positions = get_valid_tile_positions(edge_maps[fn])
        tile_positions_per_image[idx] = positions
        if (idx + 1) % 50 == 0 or idx + 1 == n:
            print(
                f"    {idx + 1}/{n} images processed "
                f"(avg {np.mean([len(v) for v in tile_positions_per_image.values()]):.0f} positions/image)"
            )
    print()

    # Step 4: Load video transformer
    print("Step 4/5: Loading video transformer scorer...")
    scorer = VideoTransformerScorer(device="cuda")
    scorer.load()
    print()

    # Step 5: Greedy sequencing
    print("Step 5/5: Greedy sequence construction...")
    start_idx = find_start_image(dist_matrix)
    print(f"  Starting with image {start_idx}: {filenames[start_idx]}")

    # Initialize sequence
    visited = {start_idx}
    path = [start_idx]

    # Pick a default tile for the start image (center, largest ratio)
    em = edge_maps[filenames[start_idx]]
    h, w = em.shape
    start_ts = int(min(h, w) * TILE_RATIOS[-1])
    start_ts = max(16, min(start_ts, h, w))
    start_ty = (h - start_ts) // 2
    start_tx = (w - start_ts) // 2
    tile_choices = [(start_ty, start_tx, start_ts)]  # per path index

    # Extract the start frame
    start_frame = extract_tile_rgb(
        os.path.join(image_folder, filenames[start_idx]), start_ty, start_tx, start_ts
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
        # First check KNN
        for j in knn[current_idx]:
            if j not in visited:
                candidates_idx.append(j)
            if len(candidates_idx) >= k_neighbors:
                break

        # If KNN doesn't have enough, search more broadly
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

        # For each candidate image, sample tile positions and extract frames
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
            # Fallback: just pick the nearest unvisited
            best_idx = candidates_idx[0]
            fn = filenames[best_idx]
            em = edge_maps[fn]
            h, w = em.shape
            ts = int(min(h, w) * TILE_RATIOS[-1])
            ts = max(16, min(ts, h, w))
            ty, tx = (h - ts) // 2, (w - ts) // 2
            frame = extract_tile_rgb(os.path.join(image_folder, fn), ty, tx, ts)
            if frame is None:
                img = cv.imread(os.path.join(image_folder, fn))
                frame = cv.cvtColor(
                    cv.resize(img, (FRAME_SIZE, FRAME_SIZE)), cv.COLOR_BGR2RGB
                )
            all_candidate_info = [(best_idx, ty, tx, ts, frame)]

        # Score all candidates with the video transformer
        context = context_frames[-max_context:]
        candidate_frames_list = [info[4] for info in all_candidate_info]

        print(
            f"  Step {step + 1}/{n - 1}: scoring {len(candidate_frames_list)} candidates "
            f"from {len(candidates_idx)} images...",
            end="",
            flush=True,
        )

        scores = scorer.score_candidates(context, candidate_frames_list)

        # Find best
        best_score_idx = int(np.argmin(scores))
        best_img_idx, best_ty, best_tx, best_ts, best_frame = all_candidate_info[
            best_score_idx
        ]
        best_score = scores[best_score_idx]

        # Add to sequence
        path.append(best_img_idx)
        visited.add(best_img_idx)
        tile_choices.append((best_ty, best_tx, best_ts))
        context_frames.append(best_frame)

        # Keep context bounded
        if len(context_frames) > max_context * 2:
            context_frames = context_frames[-(max_context + 2) :]

        elapsed = time.time() - t_start
        print(f" → {filenames[best_img_idx]} (score={best_score:.6f}, {elapsed:.1f}s)")

        # Periodic checkpoint
        if (step + 1) % 20 == 0:
            _write_sequence(
                output_file + ".checkpoint", path, filenames, edge_maps, tile_choices
            )
            print(f"    Checkpoint saved ({step + 1}/{n - 1})")

    # Cleanup
    scorer.unload()

    # Write final output
    _write_sequence(output_file, path, filenames, edge_maps, tile_choices)

    print(f"\nSequence written to {output_file}")
    print(f"Sequenced {len(path)}/{n} images.")
    print("\nOrder:")
    for idx in path:
        print(f"  {filenames[idx]}")


def _write_sequence(output_file, path, filenames, edge_maps, tile_choices):
    """Write the sequence file in the same format as sequence_v6.py."""
    with open(output_file, "w") as f:
        for k, idx in enumerate(path):
            fn = filenames[idx]
            em = edge_maps[fn]
            h, w = em.shape
            ty, tx, ts = tile_choices[k]
            f.write(f"{fn},{ty},{tx},{ts},{h},{w}\n")


# ── Video Generation (reused from sequence_v6.py) ──────────────────────────


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

"""
Segment figure and ground using SegGPT (Segment Everything In Context).

Uses the SegGPT model from Meta to perform in-context segmentation,
separating figure (foreground) from ground (background) in images.
"""

import os
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import SegGptForImageSegmentation, SegGptImageProcessor


# ---- Constants ----
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
SEG_GPT_IMAGE_SIZE = 448


# ---- Argument Parsing ----


def get_args_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment figure and ground using SegGPT"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image to segment.",
    )
    parser.add_argument(
        "--prompt_image",
        type=str,
        default=None,
        help="Path to the prompt (example) image for in-context learning.",
    )
    parser.add_argument(
        "--prompt_mask",
        type=str,
        default=None,
        help="Path to the prompt mask corresponding to the prompt image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save segmentation results.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="seggpt_vit_large.pth",
        help="Path to the SegGPT model checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seggpt_vit_large_patch16_input896x448",
        help="Name of the SegGPT model architecture.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--seg_type",
        type=str,
        default="instance",
        choices=["instance", "semantic"],
        help="Type of segmentation: 'instance' or 'semantic'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=128.0,
        help="Threshold for binarizing the predicted mask (0-255).",
    )
    parser.add_argument(
        "--use_huggingface",
        action="store_true",
        help="Use HuggingFace Transformers SegGPT model instead of local checkpoint.",
    )
    return parser


# ---- Image Utilities ----


def load_image(image_path, target_size=SEG_GPT_IMAGE_SIZE):
    """Load an image, resize it, and return the PIL image, numpy array, and original size."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img = img.resize((target_size, target_size), Image.LANCZOS)
    img_np = np.array(img) / 255.0
    return img, img_np, original_size


def normalize_image(img_np):
    """Normalize an image array using ImageNet mean and std."""
    return (img_np - IMAGENET_MEAN) / IMAGENET_STD


def prepare_tensor(img_np, device):
    """Convert a (H, W, 3) numpy image to a (1, 3, H, W) tensor on the given device."""
    tensor = torch.from_numpy(img_np).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def ensure_3ch(mask_np):
    """Ensure a mask array has 3 channels."""
    if mask_np.ndim == 2:
        return np.stack([mask_np] * 3, axis=-1)
    return mask_np


# ---- Model Loading ----


def load_seggpt_model_local(model_name, ckpt_path, device):
    """Load SegGPT model from a local checkpoint."""
    try:
        import models_seggpt
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The 'models_seggpt' module is required for local checkpoint inference.\n"
            "Clone the SegGPT repo and ensure models_seggpt.py is on your PYTHONPATH,\n"
            "or use --use_huggingface instead."
        )

    model = getattr(models_seggpt, model_name)()
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded with message: {msg}")

    model.eval()
    model.to(device)
    return model


def load_seggpt_model_huggingface(device):
    """Load SegGPT model from HuggingFace Transformers."""
    model_id = "BAAI/seggpt-vit-large"
    processor = SegGptImageProcessor.from_pretrained(model_id)
    model = SegGptForImageSegmentation.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return model, processor


# ---- Inference ----


def inference_seggpt_local(model, device, input_img_np, prompt_img_np, prompt_mask_np):
    """
    Run SegGPT inference using a local model.

    Concatenates [prompt_image; input_image] vertically and
    [prompt_mask; blank_mask], then predicts the mask for the input image.
    """
    input_norm = normalize_image(input_img_np)
    prompt_norm = normalize_image(prompt_img_np)

    if prompt_mask_np.max() > 1.0:
        prompt_mask_np = prompt_mask_np / 255.0

    input_concat = np.concatenate([prompt_norm, input_norm], axis=0)

    prompt_mask_3ch = ensure_3ch(prompt_mask_np)
    blank_mask = np.zeros_like(prompt_mask_3ch)
    mask_concat = np.concatenate([prompt_mask_3ch, blank_mask], axis=0)

    input_tensor = prepare_tensor(input_concat, device)
    mask_tensor = prepare_tensor(mask_concat, device)
    valid = torch.ones(1, 1, dtype=torch.bool, device=device)

    with torch.no_grad():
        output = model(input_tensor, mask_tensor, valid)

    # Extract prediction for the input image (bottom half)
    output = output[:, :, output.shape[2] // 2 :, :]

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = output * IMAGENET_STD + IMAGENET_MEAN
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return output


def inference_seggpt_huggingface(
    model, processor, device, input_image, prompt_image, prompt_mask
):
    """Run SegGPT inference using the HuggingFace Transformers model."""
    inputs = processor(
        images=input_image,
        prompt_images=prompt_image,
        prompt_masks=prompt_mask,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(input_image.size[1], input_image.size[0])]
    )
    return result[0]


# ---- Post-processing ----


def create_binary_mask(prediction, threshold=128):
    """Convert a prediction to a binary (0/255) mask."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()

    gray = np.mean(prediction, axis=-1) if prediction.ndim == 3 else prediction
    return (gray > threshold).astype(np.uint8) * 255


def extract_figure_ground(input_image, binary_mask):
    """
    Separate an image into figure (foreground) and ground (background) using a binary mask.

    Returns RGBA images for figure and ground, plus an overlay visualization.
    """
    input_np = np.array(input_image)

    if binary_mask.shape[:2] != input_np.shape[:2]:
        mask_pil = Image.fromarray(binary_mask).resize(
            (input_np.shape[1], input_np.shape[0]), Image.NEAREST
        )
        binary_mask = np.array(mask_pil)

    mask_bool = binary_mask > 128

    # Figure: foreground with transparent background
    figure_rgba = np.zeros((*input_np.shape[:2], 4), dtype=np.uint8)
    figure_rgba[..., :3] = input_np
    figure_rgba[..., 3] = mask_bool.astype(np.uint8) * 255

    # Ground: background with transparent foreground
    ground_rgba = np.zeros((*input_np.shape[:2], 4), dtype=np.uint8)
    ground_rgba[..., :3] = input_np
    ground_rgba[..., 3] = (~mask_bool).astype(np.uint8) * 255

    # Red-tinted overlay visualization
    overlay = input_np.copy()
    overlay[mask_bool] = (
        overlay[mask_bool] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
    ).astype(np.uint8)

    return (
        Image.fromarray(figure_rgba),
        Image.fromarray(ground_rgba),
        Image.fromarray(overlay),
    )


# ---- Prompt Generation ----


def generate_default_prompt(input_image_path):
    """
    Generate a default prompt mask using a center-biased saliency heuristic
    when no prompt is provided.
    """
    img = Image.open(input_image_path).convert("RGB")
    img_resized = img.resize((SEG_GPT_IMAGE_SIZE, SEG_GPT_IMAGE_SIZE), Image.LANCZOS)
    img_np = np.array(img_resized).astype(np.float32)

    h, w = img_np.shape[:2]
    cy, cx = h / 2, w / 2

    # Center-distance prior
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    center_prior = 1.0 - (dist / np.sqrt(cx**2 + cy**2))

    # Color-difference saliency
    mean_color = img_np.mean(axis=(0, 1))
    color_diff = np.sqrt(np.sum((img_np - mean_color) ** 2, axis=-1))
    color_diff /= color_diff.max() + 1e-8

    # Combine and threshold
    saliency = center_prior * 0.5 + color_diff * 0.5
    mask = (saliency > np.percentile(saliency, 60)).astype(np.uint8) * 255

    # Convert to 3-channel RGB PIL image so HuggingFace processor handles it correctly
    mask_rgb = np.stack([mask, mask, mask], axis=-1)
    mask_pil = Image.fromarray(mask_rgb, mode="RGB")

    return img_resized, mask_pil


# ---- Saving Results ----


def save_results(output_dir, base_name, binary_mask_full, figure, ground, overlay):
    """Save all segmentation result images to disk."""
    paths = {
        "mask": os.path.join(output_dir, f"{base_name}_mask.png"),
        "figure": os.path.join(output_dir, f"{base_name}_figure.png"),
        "ground": os.path.join(output_dir, f"{base_name}_ground.png"),
        "overlay": os.path.join(output_dir, f"{base_name}_overlay.png"),
    }

    Image.fromarray(binary_mask_full).save(paths["mask"])
    figure.save(paths["figure"])
    ground.save(paths["ground"])
    overlay.save(paths["overlay"])

    print(f"\nResults saved to {output_dir}/")
    for label, path in paths.items():
        print(f"  {label:>12s}:  {path}")


# ---- Main ----


def main():
    args = get_args_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    has_prompt = args.prompt_image and args.prompt_mask

    if args.use_huggingface:
        print("Loading SegGPT model from HuggingFace...")
        model, processor = load_seggpt_model_huggingface(device)

        input_image = Image.open(args.input_image).convert("RGB")

        if has_prompt:
            prompt_image = Image.open(args.prompt_image).convert("RGB")
            prompt_mask = Image.open(args.prompt_mask).convert("RGB")
        else:
            print("No prompt provided. Using auto-generated saliency-based prompt.")
            prompt_image, prompt_mask = generate_default_prompt(args.input_image)

        print("Running SegGPT inference (HuggingFace)...")
        prediction = inference_seggpt_huggingface(
            model, processor, device, input_image, prompt_image, prompt_mask
        )

    else:
        if not os.path.exists(args.ckpt_path):
            print(
                f"Checkpoint not found at {args.ckpt_path}.\n"
                "Please download the SegGPT checkpoint or use --use_huggingface.\n"
                "Download: https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth"
            )
            return

        print(f"Loading SegGPT model from {args.ckpt_path}...")
        model = load_seggpt_model_local(args.model, args.ckpt_path, device)

        _, input_img_np, _ = load_image(args.input_image)

        if has_prompt:
            _, prompt_img_np, _ = load_image(args.prompt_image)
            prompt_mask_pil = Image.open(args.prompt_mask).convert("RGB")
            prompt_mask_pil = prompt_mask_pil.resize(
                (SEG_GPT_IMAGE_SIZE, SEG_GPT_IMAGE_SIZE), Image.NEAREST
            )
            prompt_mask_np = np.array(prompt_mask_pil) / 255.0
        else:
            print("No prompt provided. Using auto-generated saliency-based prompt.")
            prompt_pil, prompt_mask_gray = generate_default_prompt(args.input_image)
            prompt_img_np = np.array(prompt_pil) / 255.0
            prompt_mask_np = (
                np.array(
                    prompt_mask_gray.resize(
                        (SEG_GPT_IMAGE_SIZE, SEG_GPT_IMAGE_SIZE), Image.NEAREST
                    )
                )
                / 255.0
            )
            prompt_mask_np = ensure_3ch(prompt_mask_np)

        print("Running SegGPT inference (local)...")
        prediction = inference_seggpt_local(
            model, device, input_img_np, prompt_img_np, prompt_mask_np
        )

    # Post-process and save
    binary_mask = create_binary_mask(prediction, args.threshold)
    input_image_pil = Image.open(args.input_image).convert("RGB")
    mask_pil = Image.fromarray(binary_mask).resize(input_image_pil.size, Image.NEAREST)
    binary_mask_full = np.array(mask_pil)

    figure, ground, overlay = extract_figure_ground(input_image_pil, binary_mask_full)

    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    save_results(args.output_dir, base_name, binary_mask_full, figure, ground, overlay)
    print("Done!")


if __name__ == "__main__":
    main()

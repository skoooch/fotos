import sys
import os
import argparse
import json
import urllib.request
from PIL import Image
import numpy as np
import mediapipe as mp


# ── Model download ──────────────────────────────────────────────────────────

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")


def _ensure_model():
    """Download the face detection model if not already present."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading face detection model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


# ── Face detection ──────────────────────────────────────────────────────────


def detect_faces(
    sequence_out,
    foto_folder,
    reset=False,
    extensions=(".jpg", ".jpeg", ".png", ".tif", ".bmp"),
    confidence_threshold=0.5,
):
    """
    Detect facial bounding boxes for all images in foto_folder.
    Writes results as JSON: {filename: [{"x": .., "y": .., "w": .., "h": ..}, ...], ...}

    Args:
        sequence_out: Output file path for the face detection results.
        foto_folder: Folder containing the photos.
        reset: If False, skip images already present in sequence_out.
        confidence_threshold: Minimum detection confidence.
    """
    _ensure_model()

    # Load existing results if not resetting
    existing = {}
    if not reset and os.path.exists(sequence_out):
        with open(sequence_out, "r") as f:
            existing = json.load(f)

    results = dict(existing)

    # Set up the new Tasks API face detector
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=confidence_threshold,
    )

    with FaceDetector.create_from_options(options) as detector:
        files = sorted(
            fn
            for fn in os.listdir(foto_folder)
            if os.path.splitext(fn)[1].lower() in extensions
        )

        for i, fn in enumerate(files):
            if fn in results and not reset:
                continue

            path = os.path.join(foto_folder, fn)
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
            h, w = img_np.shape[:2]

            # Create a MediaPipe Image from the numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

            detection_result = detector.detect(mp_image)

            faces = []
            if detection_result.detections:
                for det in detection_result.detections:
                    bbox = det.bounding_box
                    faces.append(
                        {
                            "x": bbox.origin_x,
                            "y": bbox.origin_y,
                            "w": bbox.width,
                            "h": bbox.height,
                            "confidence": round(det.categories[0].score, 3),
                        }
                    )

            results[fn] = faces

            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print(f"  {i + 1}/{len(files)} processed")

    with open(sequence_out, "w") as f:
        json.dump(results, f, indent=2)

    total_faces = sum(len(v) for v in results.values())
    print(f"Face detection done: {total_faces} faces in {len(results)} images")
    print(f"Results saved to {sequence_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify facial bboxes of input images"
    )
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Compute for all images in foto_folder, vs just new ones",
    )
    parser.add_argument("file_out", help="Output JSON file for face bboxes")
    parser.add_argument("foto_folder", help="Folder containing photos")
    args = parser.parse_args()

    detect_faces(args.file_out, args.foto_folder, args.reset)

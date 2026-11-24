"""
Step 1 of Model B: turn hand photos into 21 hand keypoints each.

For every image, MediaPipe Hands finds 21 points on the hand (wrist,
knuckles, fingertips), each with (x, y, z). We save them as 63 numbers
per image, plus the letter label.

Setup (once):
    pip install mediapipe opencv-python numpy
    Download the hand model file (about 7.5 MB) into this folder:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Usage:
    # Kaggle ASL Alphabet training images, 300 per letter
    python extract_keypoints.py --data_dir asl_alphabet_train/asl_alphabet_train \
        --out kaggle_keypoints.npz --per_class 300

    # Your team's own recordings (same folder layout: one folder per letter)
    python extract_keypoints.py --data_dir our_recordings --out ours_keypoints.npz --per_class 0

Folder layout expected:
    data_dir/A/*.jpg, data_dir/B/*.jpg, ...   (one subfolder per letter)
Folders that aren't one of the 24 static letters (J, Z, del, nothing, space)
are skipped.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# 24 static letters: J and Z need motion, so they are left out
LETTERS = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def make_landmarker(model_path, min_conf=0.5):
    """Create MediaPipe's hand detector for single still images."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,                       # one signing hand per image
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def load_rgb(image_path):
    """Read an image as an RGB array (works with any characters in the path)."""
    data = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def enlarge_and_pad(rgb, scale=2.0, pad_frac=0.25):
    """Make the photo bigger and add a border, so small hands and hands
    touching the edge are easier for MediaPipe to find."""
    big = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad = int(pad_frac * max(big.shape[:2]))
    return cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def brighten(rgb):
    """Boost contrast in dark photos (CLAHE on the lightness channel only)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def detect(landmarker, rgb):
    """Run MediaPipe on an RGB array. Returns (21, 3) points in pixels, or None."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None                        # MediaPipe failed to find a hand
    hand = result.hand_landmarks[0]        # list of 21 landmarks
    h, w = rgb.shape[:2]
    # MediaPipe gives x, y as fractions of width/height. Convert to pixels so
    # a non-square image (webcam) isn't stretched. z uses the same scale as x.
    # (Enlarging/padding changes position and size, which normalize() removes.)
    return np.array([[p.x * w, p.y * h, p.z * w] for p in hand], dtype=np.float32)


def get_keypoints(landmarkers, image_path, retry=True):
    """Try to find the hand; if that fails, retry on easier versions of the photo.

    Returns (points, stage): stage says which attempt worked, or (None, None).
      1 original photo
      2 enlarged + padded
      3 enlarged + padded + brightened, lower confidence threshold
    """
    normal, lenient = landmarkers
    rgb = load_rgb(image_path)
    if rgb is None:
        return None, None
    points = detect(normal, rgb)
    if points is not None or not retry:
        return points, (1 if points is not None else None)
    big = enlarge_and_pad(rgb)
    points = detect(normal, big)
    if points is not None:
        return points, 2
    points = detect(lenient, brighten(big))
    if points is not None:
        return points, 3
    return None, None


def normalize(points):
    """Make keypoints independent of where the hand is and how big it is.

    - Center: subtract the wrist (point 0), so the wrist is at (0, 0, 0).
    - Scale: divide by the largest wrist-to-point distance, so every hand
      is about the same size.
    Returns 63 numbers (21 points x 3).
    """
    centered = points - points[0]
    scale = np.linalg.norm(centered[:, :2], axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="folder with one subfolder per letter")
    parser.add_argument("--out", required=True, help="output .npz file")
    parser.add_argument("--per_class", type=int, default=300,
                        help="max images per letter (0 = use all)")
    parser.add_argument("--model", default="hand_landmarker.task", help="MediaPipe model file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_retry", action="store_true",
                        help="only try each photo once (the original behaviour)")
    parser.add_argument("--retry_conf", type=float, default=0.3,
                        help="confidence threshold for the last retry")
    args = parser.parse_args()

    random.seed(args.seed)
    landmarkers = (make_landmarker(args.model, 0.5),
                   make_landmarker(args.model, args.retry_conf))
    stage_counts = {1: 0, 2: 0, 3: 0}
    data_dir = Path(args.data_dir)

    X, y, files = [], [], []
    failures = []                          # images where no hand was found
    report = []                            # per-letter counts

    for label, letter in enumerate(LETTERS):
        folder = data_dir / letter
        if not folder.is_dir():
            print(f"  {letter}: folder missing, skipped")
            continue
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if args.per_class > 0 and len(images) > args.per_class:
            images = random.sample(images, args.per_class)

        found = 0
        for path in images:
            points, stage = get_keypoints(landmarkers, path, retry=not args.no_retry)
            if points is None:
                failures.append(str(path))
                continue
            stage_counts[stage] += 1
            X.append(normalize(points))
            y.append(label)
            files.append(str(path))
            found += 1

        rate = 100 * (1 - found / max(len(images), 1))
        report.append((letter, len(images), found, rate))
        print(f"  {letter}: {found}/{len(images)} hands found ({rate:.1f}% failed)")

    if not X:
        print(f"\nNo hand keypoints were found. Check --data_dir: it must be the folder "
              f"that directly contains the letter folders A, B, C, ...\n"
              f"  Currently: {data_dir.resolve()}\n"
              f"  Tip (Windows): run  dir A /s /b /ad  to find where the A folder is.")
        raise SystemExit(1)

    X = np.stack(X).astype(np.float32)     # (N, 63)
    y = np.array(y, dtype=np.int64)        # (N,)  class numbers 0..23
    np.savez(args.out, X=X, y=y, files=np.array(files), classes=np.array(LETTERS))

    # Failure log: "how often MediaPipe fails to find a hand" is one of the project metrics
    log_path = Path(args.out).with_suffix(".failures.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "images", "hands_found", "failure_rate_%"])
        writer.writerows([(l, n, k, f"{r:.1f}") for l, n, k, r in report])

    total = sum(n for _, n, _, _ in report)
    print(f"\nSaved {len(y)} samples to {args.out}  (X shape {X.shape})")
    print(f"MediaPipe failed on {len(failures)}/{total} images "
          f"({100 * len(failures) / max(total, 1):.1f}%). Per-letter log: {log_path}")
    if not args.no_retry:
        print(f"Found on 1st try: {stage_counts[1]} | after enlarging: {stage_counts[2]} "
              f"| after brightening + lower threshold: {stage_counts[3]}")


if __name__ == "__main__":
    main()

"""
Quick check: are our recordings mirrored compared with the Kaggle photos?

Loads the trained Model B (model_b.pt) and tests it on our recordings twice:
  1. as recorded
  2. mirrored left-right (x coordinates flipped)
No retraining needed. If accuracy jumps when mirrored, the Kaggle hands and
our hands are mirror images of each other.

Usage:
    python diagnose_mirror.py --test ours_keypoints.npz
"""
import argparse

import numpy as np
import torch

from train_model_b import KeypointClassifier, predict


def mirror(X):
    """Flip every hand left-right: negate the x coordinate of all 21 points."""
    pts = X.reshape(-1, 21, 3).copy()
    pts[:, :, 0] *= -1
    return pts.reshape(-1, 63)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="ours_keypoints.npz")
    p.add_argument("--model", default="model_b.pt")
    args = p.parse_args()

    data = np.load(args.test)
    X, y, classes = data["X"].astype(np.float32), data["y"], list(data["classes"])
    model = KeypointClassifier(len(classes))
    model.load_state_dict(torch.load(args.model))

    pred_raw = predict(model, X)
    pred_mir = predict(model, mirror(X))
    print(f"As recorded: {(pred_raw == y).mean():.3f}")
    print(f"Mirrored:    {(pred_mir == y).mean():.3f}\n")

    print("Letter  as-recorded  mirrored")
    for i, letter in enumerate(classes):
        m = y == i
        if m.any():
            print(f"  {letter}      {(pred_raw[m] == i).mean():5.2f}      {(pred_mir[m] == i).mean():5.2f}")


if __name__ == "__main__":
    main()

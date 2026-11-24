"""
Step 2 of Model B: train a small neural network on the 63 keypoint numbers.

Usage:
    python train_model_b.py --train kaggle_keypoints.npz
    python train_model_b.py --train kaggle_keypoints.npz --test ours_keypoints.npz --augment

--train   keypoints from the Kaggle images (split into train / val / test)
--test    optional: keypoints from YOUR recordings = the real generalization test
--augment optional: randomly rotate / rescale hands during training

Trains 3 times with different random seeds and reports mean +/- spread,
as the project plan asks. Saves results.json, a confusion matrix plot,
and the best model's weights.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class KeypointClassifier(nn.Module):
    """63 numbers in -> 24 letter scores out. Same idea as the TA's Discussion 3 Classifier."""

    def __init__(self, n_classes=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),   # [B,63]  -> [B,128]
            nn.ReLU(),
            nn.Dropout(0.2),      # randomly zero 20% of units in training (less overfitting)
            nn.Linear(128, 64),   # [B,128] -> [B,64]
            nn.ReLU(),
            nn.Linear(64, n_classes),  # [B,64] -> [B,24] raw scores (logits)
        )

    def forward(self, x):
        return self.net(x)


def augment(x, mirror=False):
    """Keypoint augmentation: small random rotation, size change and jitter.

    Like image augmentation, but on the 21 points. With mirror=True, half the
    hands are also flipped left-right, so the model learns both pointing
    directions (and left-handed signers).
    """
    B = x.shape[0]
    pts = x.view(B, 21, 3)
    if mirror:
        flip = torch.where(torch.rand(B) < 0.5, -1.0, 1.0)
        pts = torch.stack([pts[:, :, 0] * flip[:, None], pts[:, :, 1], pts[:, :, 2]], dim=2)
    angle = (torch.rand(B) * 2 - 1) * math.radians(15)          # +/- 15 degrees
    c, s = torch.cos(angle), torch.sin(angle)
    xy = pts[:, :, :2]
    rot_x = c[:, None] * xy[:, :, 0] - s[:, None] * xy[:, :, 1]
    rot_y = s[:, None] * xy[:, :, 0] + c[:, None] * xy[:, :, 1]
    pts = torch.stack([rot_x, rot_y, pts[:, :, 2]], dim=2)
    pts = pts * (1 + (torch.rand(B, 1, 1) * 2 - 1) * 0.1)       # +/- 10% size
    pts = pts + torch.randn_like(pts) * 0.01                     # tiny jitter
    return pts.view(B, 63)


def split_indices(y, seed, val_frac=0.15, test_frac=0.15):
    """Stratified split: every letter keeps the same train/val/test proportions."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_test, n_val = int(len(idx) * test_frac), int(len(idx) * val_frac)
        test += list(idx[:n_test])
        val += list(idx[n_test:n_test + n_val])
        train += list(idx[n_test + n_val:])
    return np.array(train), np.array(val), np.array(test)


@torch.no_grad()
def predict(model, X):
    model.eval()
    return model(torch.from_numpy(X)).argmax(dim=1).numpy()


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def train_one(X, y, tr, va, n_classes, seed, epochs, use_aug, use_mirror=False):
    torch.manual_seed(seed)
    model = KeypointClassifier(n_classes)                 # create ONCE, before the loop
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                        batch_size=64, shuffle=True)

    best_acc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if use_aug or use_mirror:
                xb = augment(xb, mirror=use_mirror)
            optimizer.zero_grad()          # 1 clear
            logits = model(xb)             # 2 guess
            loss = loss_fn(logits, yb)     # 3 grade
            loss.backward()                # 4 blame
            optimizer.step()               # 5 fix
        val_acc = (predict(model, X[va]) == y[va]).mean()
        if val_acc > best_acc:             # keep the epoch that did best on validation
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", default=None, help="your own recordings (.npz)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--mirror", action="store_true",
                   help="also mirror half the training hands (both pointing directions)")
    p.add_argument("--out", default="model_b")
    args = p.parse_args()

    data = np.load(args.train)
    X, y, classes = data["X"].astype(np.float32), data["y"].astype(np.int64), list(data["classes"])
    n = len(classes)
    # The split is fixed (seed 0) so all 3 runs are tested on the same images;
    # only the model's starting weights and batch order change between seeds.
    tr, va, te = split_indices(y, seed=0)
    print(f"{len(y)} samples, {n} letters | train {len(tr)}, val {len(va)}, test {len(te)}")
    print("Note: Kaggle images are near-duplicates of one signer, so this split is the "
          "'lab score'. Your own recordings (--test) are the real test.\n")

    ours = np.load(args.test) if args.test else None
    results, best_model, best_val = [], None, -1
    for seed in args.seeds:
        model, val_acc = train_one(X, y, tr, va, n, seed, args.epochs, args.augment, args.mirror)
        pred = predict(model, X[te])
        run = {"seed": seed, "val_acc": float(val_acc), "test_acc": float((pred == y[te]).mean())}
        if ours is not None:
            run["ours_acc"] = float((predict(model, ours["X"].astype(np.float32)) == ours["y"]).mean())
        results.append(run)
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in run.items()))
        if val_acc > best_val:
            best_val, best_model = val_acc, model

    def summary(key):
        vals = np.array([r[key] for r in results])
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    report = {"runs": results, "kaggle_test": summary("test_acc"), "augment": args.augment, "mirror": args.mirror}
    print(f"\nKaggle test accuracy: {report['kaggle_test']['mean']:.3f} "
          f"+/- {report['kaggle_test']['std']:.3f}")
    if ours is not None:
        report["our_recordings"] = summary("ours_acc")
        gap = report["kaggle_test"]["mean"] - report["our_recordings"]["mean"]
        report["generalization_gap"] = gap
        print(f"Our recordings accuracy: {report['our_recordings']['mean']:.3f} "
              f"+/- {report['our_recordings']['std']:.3f}  (gap {gap:.3f})")

    # Per-letter accuracy + confusion matrix from the best run
    eval_X, eval_y, name = ((ours["X"].astype(np.float32), ours["y"], "our recordings")
                            if ours is not None else (X[te], y[te], "Kaggle test"))
    cm = confusion_matrix(eval_y, predict(best_model, eval_X), n)
    per_letter = {classes[i]: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
                  for i in range(n)}
    report["per_letter_acc"] = per_letter
    report["confusion_matrix_on"] = name
    report["confusion_matrix"] = cm.tolist()
    worst = sorted((v, k) for k, v in per_letter.items() if v is not None)[:5]
    print("Hardest letters:", ", ".join(f"{k} {v:.2f}" for v, k in worst))

    torch.save(best_model.state_dict(), f"{args.out}.pt")
    with open(f"{args.out}_results.json", "w") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), classes)
        ax.set_yticks(range(n), classes)
        ax.set_xlabel("Predicted letter")
        ax.set_ylabel("True letter")
        ax.set_title(f"Model B confusion matrix ({name})")
        fig.colorbar(im, ax=ax, label="fraction of true letter")
        fig.tight_layout()
        fig.savefig(f"{args.out}_confusion.png", dpi=150)
        print(f"Saved {args.out}.pt, {args.out}_results.json, {args.out}_confusion.png")
    except ImportError:
        print(f"Saved {args.out}.pt and {args.out}_results.json (install matplotlib for the plot)")


if __name__ == "__main__":
    main()

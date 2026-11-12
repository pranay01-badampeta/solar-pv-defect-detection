"""Train a CNN to classify solar PV panel images by defect type.

Expects an image folder with one sub-folder per class, for example the Kaggle
"Solar Panel Images: Clean and Faulty" dataset:
    data/images/Bird-drop/  data/images/Clean/  data/images/Dusty/
    data/images/Electrical-damage/  data/images/Physical-Damage/  data/images/Snow-Covered/

Usage:
    python train_cnn.py --data data/images --arch cnn --epochs 30
    python train_cnn.py --data data/images --arch resnet18 --epochs 15   # transfer learning
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from model import build_model, eval_transform, train_transform

SEED = 42


def split_indices(n: int, val: float, test: float):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_test, n_val = int(n * test), int(n * val)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            p = logits.argmax(1)
            correct += (p == y).sum().item()
            count += len(y)
            preds.append(p.cpu())
            labels.append(y.cpu())
    return total_loss / count, correct / count, torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/images")
    ap.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    train_ds = ImageFolder(args.data, transform=train_transform)
    eval_ds = ImageFolder(args.data, transform=eval_transform)
    classes = train_ds.classes
    tr, va, te = split_indices(len(train_ds), val=0.15, test=0.15)
    loaders = {
        "train": DataLoader(Subset(train_ds, tr), batch_size=args.batch, shuffle=True, num_workers=2),
        "val": DataLoader(Subset(eval_ds, va), batch_size=args.batch, num_workers=2),
        "test": DataLoader(Subset(eval_ds, te), batch_size=args.batch, num_workers=2),
    }
    print(f"{len(train_ds)} images, {len(classes)} classes: {classes}")
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} on {device}")

    model = build_model(args.arch, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        vl, va_acc, _, _ = run_epoch(model, loaders["val"], loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va_acc})
        print(f"epoch {epoch:3d}  train loss {tl:.3f} acc {ta:.3f}  |  val loss {vl:.3f} acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"arch": args.arch, "classes": classes, "state_dict": model.state_dict()},
                       out / "solar_cnn.pt")

    ckpt = torch.load(out / "solar_cnn.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, test_acc, preds, labels = run_epoch(model, loaders["test"], loss_fn, device)
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    print(f"\nTest accuracy: {test_acc:.3f}\n")
    print(classification_report(labels, preds, target_names=classes, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=classes).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    (out / "metrics.json").write_text(json.dumps({"test_accuracy": test_acc, "report": report,
                                                  "history": history}, indent=2))
    print(f"Best model saved to {out}/solar_cnn.pt")


if __name__ == "__main__":
    main()

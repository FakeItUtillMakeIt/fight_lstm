#!/usr/bin/env python
"""
3-class fight detection: normal / fight / climb
Branch: 3cls_det
"""
import numpy as np
import torch
import torch.nn as nn
import os
import json
import argparse
from sklearn.model_selection import train_test_split

NUM_FRAMES = 10
FEATURE_DIM = 72
CLASS_NAMES = ["normal", "fight", "climb"]


class BiLSTM3Cls(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_units=128, dropout=0.3, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_units,
            batch_first=True, bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def _normalize_sequence(seq):
    seq = seq.copy()
    for d in range(seq.shape[1]):
        col = seq[:, d]
        std = col.std()
        if std > 1e-6:
            seq[:, d] = (col - col.mean()) / std
    return seq


def make_windowed_samples(features, n_samples, window=NUM_FRAMES):
    T = len(features)
    if T < window:
        features = np.tile(features, ((window // T) + 1, 1))[:window]
        T = window

    samples = []
    for i in range(n_samples):
        stride = max(1, np.random.randint(1, min(4, max(2, T // 20))))
        start = np.random.randint(0, max(1, T - window * stride))
        idx = np.arange(start, min(T, start + window * stride), stride)
        if len(idx) < window:
            idx = np.concatenate([idx, np.full(window - len(idx), idx[-1])])
        else:
            idx = idx[:window]

        seq = features[idx].copy()
        seq += np.random.randn(*seq.shape) * 0.005
        for point in range(18):
            if np.random.random() < 0.1:
                seq[:, point*2:(point+1)*2] = 0
                seq[:, 36+point*2:36+(point+1)*2] = 0
        if np.random.random() < 0.3:
            jitter = np.random.randn(*seq.shape) * 0.005
            jitter = np.convolve(jitter.flatten(), np.ones(3)/3, mode='same').reshape(seq.shape)
            seq += jitter

        seq = _normalize_sequence(seq)
        samples.append(seq)

    return np.array(samples)


def train_epoch(model, x, y, device, optimizer, criterion, batch_size=16):
    model.train()
    perm = torch.randperm(len(x))
    total_loss, correct = 0, 0
    for i in range(0, len(x), batch_size):
        idx = perm[i:i + batch_size]
        bx, by = x[idx].to(device), y[idx].to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(idx)
        correct += (pred.argmax(dim=1) == by).sum().item()
    return total_loss / len(x), correct / len(x)


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    pred = model(x.to(device))
    loss = nn.CrossEntropyLoss()(pred, y.to(device)).item()
    acc = (pred.argmax(dim=1) == y.to(device)).float().mean().item()
    return loss, acc, torch.softmax(pred, dim=1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cached_features")
    parser.add_argument("--normal-videos", nargs="+", required=True,
                        help="Normal/non-fight/non-climb videos")
    parser.add_argument("--fight-videos", nargs="+", required=True)
    parser.add_argument("--climb-videos", nargs="+", required=True,
                        help="Climbing fence videos")
    parser.add_argument("--samples-per-video", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="fight_3cls_model.pt")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Classes: {CLASS_NAMES}")

    all_data, all_labels = [], []

    # normal = class 0
    for name in args.normal_videos:
        fpath = os.path.join(args.cache_dir, f"{name}.npy")
        if not os.path.exists(fpath):
            print(f"  [SKIP] {name}.npy not found")
            continue
        feats = np.load(fpath)
        print(f"  normal -> {name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.zeros(len(samples), dtype=np.int64))

    # fight = class 1
    for name in args.fight_videos:
        fpath = os.path.join(args.cache_dir, f"{name}.npy")
        if not os.path.exists(fpath):
            print(f"  [SKIP] {name}.npy not found")
            continue
        feats = np.load(fpath)
        print(f"  fight   -> {name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.ones(len(samples), dtype=np.int64))

    # climb = class 2
    for name in args.climb_videos:
        fpath = os.path.join(args.cache_dir, f"{name}.npy")
        if not os.path.exists(fpath):
            print(f"  [SKIP] {name}.npy not found")
            continue
        feats = np.load(fpath)
        print(f"  climb   -> {name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.full(len(samples), 2, dtype=np.int64))

    if len(all_data) == 0:
        print("No data loaded!")
        return

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    for i, name in enumerate(CLASS_NAMES):
        n = (labels == i).sum()
        print(f"  {name}: {n} samples")

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    print(f"  x_train dtype={x_train.dtype}, y_train dtype={y_train.dtype}")

    model = BiLSTM3Cls(dropout=0.4).to(device)

    # Class weights for imbalance
    class_counts = [(labels == i).sum() for i in range(3)]
    total = sum(class_counts)
    weights = torch.tensor([total / max(1, c) for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, x_train, y_train, device, optimizer, criterion)
        val_loss, val_acc, val_preds = evaluate(model, x_test, y_test, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'num_classes': 3,
                'feature_dim': FEATURE_DIM,
                'num_frames': NUM_FRAMES,
            }, args.output)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Final evaluation
    checkpoint = torch.load(args.output, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, preds = evaluate(model, x_test, y_test, device)
    pred_labels = preds.argmax(axis=1)
    true_labels = y_test.numpy()

    print(f"\n--- Final (best val_acc={best_val_acc:.3f}) ---")
    for i, name in enumerate(CLASS_NAMES):
        mask = true_labels == i
        if mask.sum() > 0:
            p = preds[mask]
            print(f"  {name:10s}: mean={p[:, i].mean():.4f}, recall={(pred_labels[mask] == i).mean():.3f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"{'':12s} {'pred_normal':>12s} {'pred_fight':>12s} {'pred_climb':>12s}")
    for i, true_name in enumerate(CLASS_NAMES):
        mask = true_labels == i
        if mask.sum() > 0:
            row = [(pred_labels[mask] == j).mean() for j in range(3)]
            print(f"  {true_name:10s} {row[0]:12.3f} {row[1]:12.3f} {row[2]:12.3f}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

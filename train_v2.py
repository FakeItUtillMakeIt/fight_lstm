#!/usr/bin/env python
"""
Train fight detection Bi-LSTM on raw normalized keypoint features.
Uses 72-dim features (36 keypoints + 36 velocities) per frame.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import json
import argparse
from sklearn.model_selection import train_test_split

NUM_FRAMES = 10
FEATURE_DIM = 72  # 18 keypoints * 2 (x,y) + 18 keypoints * 2 (vx,vy)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_units=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_units,
            batch_first=True, bidirectional=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


def _normalize_sequence(seq):
    """
    Per-sequence standardization: zero-mean unit-variance per channel.
    This removes video-specific biases (camera position, person size).
    """
    seq = seq.copy()
    for d in range(seq.shape[1]):
        col = seq[:, d]
        std = col.std()
        if std > 1e-6:
            seq[:, d] = (col - col.mean()) / std
    return seq


def make_windowed_samples(features, n_samples, window=NUM_FRAMES):
    """Generate windowed samples with diverse augmentations."""
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

        # 1. Gaussian noise
        seq += np.random.randn(*seq.shape) * 0.005
        # 2. Random keypoint dropout (simulate occlusion)
        for point in range(18):
            if np.random.random() < 0.1:
                seq[:, point*2:(point+1)*2] = 0
                seq[:, 36+point*2:36+(point+1)*2] = 0
        # 3. Temporal jitter
        if np.random.random() < 0.3:
            jitter = np.random.randn(*seq.shape) * 0.005
            jitter = np.convolve(jitter.flatten(), np.ones(3)/3, mode='same').reshape(seq.shape)
            seq += jitter

        seq = _normalize_sequence(seq)
        samples.append(seq)

    return np.array(samples)


def load_cached_features(cache_dir):
    index_path = os.path.join(cache_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return {}


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
        correct += ((pred > 0.5).float() == by).sum().item()
    return total_loss / len(x), correct / len(x)


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    pred = model(x.to(device))
    loss = nn.BCELoss()(pred, y.to(device)).item()
    acc = ((pred > 0.5).float() == y.to(device)).float().mean().item()
    return loss, acc, pred.cpu().numpy().flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cached_features_v2")
    parser.add_argument("--fight-videos", nargs="+", required=True)
    parser.add_argument("--nonfight-videos", nargs="+", required=True)
    parser.add_argument("--samples-per-video", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", default="fight_lstm_model.pt")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, feat_dim={FEATURE_DIM}")

    index = load_cached_features(args.cache_dir)

    all_data, all_labels = [], []

    for name in args.fight_videos:
        fpath = index.get(name, {}).get("file") or os.path.join(args.cache_dir, f"{name}.npy")
        feats = np.load(fpath)
        print(f"  fight -> {name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.ones(len(samples)))

    for name in args.nonfight_videos:
        fpath = index.get(name, {}).get("file") or os.path.join(args.cache_dir, f"{name}.npy")
        feats = np.load(fpath)
        print(f"  non-fight -> {name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.zeros(len(samples)))

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\nTotal: {len(labels)} samples ({n_pos} fight, {n_neg} non-fight)")

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # class weights for balanced loss
    pos_weight = n_neg / max(1, n_pos)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = BiLSTMClassifier(dropout=0.4).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, x_train, y_train, device, optimizer, criterion)
        val_loss, val_acc, val_preds = evaluate(model, x_test, y_test, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:2d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Final evaluation with best model
    model.load_state_dict(torch.load(args.output, weights_only=True))
    _, _, preds = evaluate(model, x_test, y_test, device)
    test_labels = y_test.numpy().flatten()

    print("\n--- Final (best val_acc={:.3f}) ---".format(best_val_acc))
    for name, mask in [("Non-fight", test_labels == 0), ("Fight", test_labels == 1)]:
        if mask.sum() > 0:
            p = preds[mask]
            print(f"  {name}: mean={p.mean():.4f}, std={p.std():.4f}, range=[{p.min():.4f}, {p.max():.4f}]")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
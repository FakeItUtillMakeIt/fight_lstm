#!/usr/bin/env python
"""
Train fight detection Bi-LSTM from cached video features.
Supports multi-video training with proper data augmentation.
"""
import numpy as np
import torch
import torch.nn as nn
import os
import json
import argparse
from sklearn.model_selection import train_test_split

from config import NUM_FRAMES, FEATURE_DIM, NUM_PAIRS, NUM_ANGLES


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_units=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_units,
            batch_first=True, bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


def minmax_scale(data):
    N, F, D = data.shape
    scaled = data.copy()
    for i in range(N):
        for j in range(F):
            for k in range(NUM_PAIRS):
                s = k * NUM_ANGLES
                e = (k + 1) * NUM_ANGLES
                seg = scaled[i, j, s:e]
                mx = seg.max()
                mn = seg.min()
                if mx - mn > 0:
                    scaled[i, j, s:e] = (seg - mn) / (mx - mn)
    return scaled


def make_windowed_samples(features, n_samples, window=NUM_FRAMES, augment=True):
    T = len(features)
    if T < window:
        factor = (window // T) + 1
        features = np.tile(features, (factor, 1))[:window]
        T = window

    samples = []
    for _ in range(n_samples):
        start = np.random.randint(0, max(1, T - window))
        seq = features[start:start + window].copy()

        if augment:
            # small gaussian noise
            noise_std = np.random.uniform(0.001, 0.01)
            seq += np.random.randn(*seq.shape) * noise_std
            # occasionally time-reverse (some motions are symmetric)
            if np.random.random() < 0.1:
                seq = seq[::-1].copy()
            # occasionally drop frames (simulate detection failures)
            if np.random.random() < 0.2:
                mask = np.random.random(window) > 0.1
                seq[~mask] = 0

        samples.append(seq)
    return np.array(samples)


def load_cached_features(cache_dir):
    index_path = os.path.join(cache_dir, "index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return {}


def train_model(model, x_train, y_train, x_test, y_test, device, epochs=30, lr=0.001, weight_decay=1e-4):
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_size = min(16, len(x_train))
    best_val = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        total_loss, correct = 0, 0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            bx, by = x_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += ((pred > 0.5).float() == by).sum().item()

        train_acc = correct / len(x_train)

        model.eval()
        with torch.no_grad():
            tp = model(x_test.to(device))
            ty = y_test.to(device)
            val_loss = criterion(tp, ty).item()
            val_acc = ((tp > 0.5).float() == ty).float().mean().item()
            best_val = max(best_val, val_acc)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:2d}/{epochs} | loss: {total_loss/len(x_train):.6f} "
                  f"train: {train_acc:.4f} | val_loss: {val_loss:.6f} val: {val_acc:.4f}")

    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cached_features",
                        help="Directory with cached .npy features")
    parser.add_argument("--fight-videos", nargs="+", required=True,
                        help="Video names (without .mp4) labeled as fight")
    parser.add_argument("--nonfight-videos", nargs="+", required=True,
                        help="Video names (without .mp4) labeled as non-fight")
    parser.add_argument("--samples-per-video", type=int, default=40,
                        help="Training samples per video")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default="fight_lstm_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    index = load_cached_features(args.cache_dir)

    all_data = []
    all_labels = []

    for name in args.fight_videos:
        fpath = index.get(name, {}).get("file") or os.path.join(args.cache_dir, f"{name}.npy")
        feats = np.load(fpath)
        print(f"  fight/{name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.ones(len(samples)))

    for name in args.nonfight_videos:
        fpath = index.get(name, {}).get("file") or os.path.join(args.cache_dir, f"{name}.npy")
        feats = np.load(fpath)
        print(f"  nonfight/{name}: {feats.shape}")
        samples = make_windowed_samples(feats, args.samples_per_video)
        all_data.append(samples)
        all_labels.append(np.zeros(len(samples)))

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    data = minmax_scale(data)
    print(f"\nTotal: {data.shape} samples, {int(labels.sum())} fight, {len(labels) - int(labels.sum())} non-fight")

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = BiLSTMClassifier(dropout=0.3).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining...")
    train_model(model, x_train, y_train, x_test, y_test, device, epochs=args.epochs)

    print("\n--- Final Evaluation ---")
    model.eval()
    with torch.no_grad():
        tp = model(x_test.to(device))
        ty = y_test.to(device)
        test_acc = ((tp > 0.5).float() == ty).float().mean().item()
        test_vals = tp.cpu().numpy().flatten()
        test_labels = ty.cpu().numpy().flatten()
        for name, mask in [("Non-fight", test_labels == 0), ("Fight", test_labels == 1)]:
            if mask.sum() > 0:
                p = test_vals[mask]
                print(f"  {name}: mean={p.mean():.4f}, std={p.std():.4f}, "
                      f"range=[{p.min():.4f}, {p.max():.4f}]")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
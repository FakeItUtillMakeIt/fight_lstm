import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from config import NUM_FRAMES, FEATURE_DIM


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


def train_model(model, x_train, y_train, x_test, y_test, device, epochs=15, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_size = min(8, len(x_train))
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        total_loss = 0
        correct = 0
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i + batch_size]
            bx = x_train[idx].to(device)
            by = y_train[idx].to(device)

            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += ((pred > 0.5).float() == by).sum().item()

        train_acc = correct / len(x_train)
        avg_loss = total_loss / len(x_train)

        model.eval()
        with torch.no_grad():
            tx = x_test.to(device)
            ty = y_test.to(device)
            test_pred = model(tx)
            test_loss = criterion(test_pred, ty).item()
            test_acc = ((test_pred > 0.5).float() == ty).float().mean().item()

        print(f"Epoch {epoch+1:2d}/{epochs} | loss: {avg_loss:.6f} "
              f"train_acc: {train_acc:.4f} | val_loss: {test_loss:.6f} "
              f"val_acc: {test_acc:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to single fight video")
    parser.add_argument("--fight-videos", type=str, nargs="+", default=None,
                        help="Paths to fight videos")
    parser.add_argument("--nonfight-videos", type=str, nargs="+", default=None,
                        help="Paths to non-fight videos")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use purely synthetic data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.fight_videos:
        from real_data_generator import generate_dataset_from_videos
        print("Training on real video features")
        fight_list = args.fight_videos
        nf_list = args.nonfight_videos if args.nonfight_videos else []
        data, labels = generate_dataset_from_videos(
            fight_list, nf_list, n_samples=100
        )
    elif args.video:
        from real_data_generator import generate_dataset_from_video
        print(f"Training on single fight video + synthetic non-fight: {args.video}")
        data, labels = generate_dataset_from_video(
            args.video, n_fight=100, n_nonfight=100,
        )
    elif args.synthetic:
        from data_generator import generate_dataset
        print("Training on synthetic data")
        data, labels = generate_dataset(n_fight=50, n_nonfight=50)
    else:
        from real_data_generator import generate_dataset_from_video
        print("Training on single fight video (default)")
        data, labels = generate_dataset_from_video(
            "/home/sevnce/project/video/fight.mp4",
            n_fight=100, n_nonfight=100,
        )

    print(f"Data: {data.shape}, Labels: {labels.shape}")
    print(f"Fight: {int(labels.sum())}, Non-fight: {len(labels) - int(labels.sum())}")

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = BiLSTMClassifier().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    print("\nTraining Bi-LSTM model...")
    train_model(model, x_train, y_train, x_test, y_test, device, epochs=20)

    print("\n--- Evaluation ---")
    model.eval()
    with torch.no_grad():
        train_pred = model(x_train.to(device))
        train_acc = ((train_pred > 0.5).float() == y_train.to(device)).float().mean().item()
        test_pred = model(x_test.to(device))
        test_acc = ((test_pred > 0.5).float() == y_test.to(device)).float().mean().item()

        # show prediction distribution on test set
        test_vals = test_pred.cpu().numpy().flatten()
        test_labels = y_test.cpu().numpy().flatten()
        print(f"\nTest prediction stats:")
        for name, mask in [("Non-fight", test_labels == 0), ("Fight", test_labels == 1)]:
            if mask.sum() > 0:
                probs = test_vals[mask]
                print(f"  {name}: mean={probs.mean():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

    print(f"\nTrain Accuracy: {train_acc * 100:.2f}%")
    print(f"Test  Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), "fight_lstm_model.pt")
    print("Model saved to fight_lstm_model.pt")


if __name__ == "__main__":
    main()
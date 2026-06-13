import cv2
import numpy as np
import argparse
import collections
import torch
import torch.nn as nn

from pose_extractor import extract_keypoints, _get_pose_detector

NUM_FRAMES = 10
FEATURE_DIM = 72
NECK, RHIP, LHIP, RSHOULDER, LSHOULDER = 1, 8, 11, 2, 5

SKELETON_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [0, 14], [14, 16], [0, 15], [15, 17]
]


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_units=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units,
                            batch_first=True, bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


def normalize_skeleton(skeleton):
    sk = skeleton.copy()
    valid = (sk[:, 0] > 0) | (sk[:, 1] > 0)
    if valid.sum() == 0:
        return np.zeros((18, 2), dtype=np.float32)

    cx, cy, n = 0.0, 0.0, 0
    for i in [NECK, RHIP, LHIP, RSHOULDER, LSHOULDER]:
        if valid[i]:
            cx += sk[i, 0]; cy += sk[i, 1]; n += 1
    if n > 0:
        cx /= n; cy /= n
    sk[:, 0] -= cx; sk[:, 1] -= cy

    scale = 1.0
    if valid[NECK] and (valid[RHIP] or valid[LHIP]):
        hip_y = (sk[RHIP, 1] + sk[LHIP, 1]) / 2 if (valid[RHIP] and valid[LHIP]) \
                else (sk[RHIP, 1] if valid[RHIP] else sk[LHIP, 1])
        torso = abs(sk[NECK, 1] - hip_y)
        if torso > 10:
            scale = 1.0 / torso
    sk *= scale
    sk[~valid] = 0
    return sk


def compute_raw_features(skeleton):
    sk = normalize_skeleton(skeleton)
    return sk.flatten().astype(np.float32)


def std_normalize(seq):
    seq = seq.copy()
    for d in range(seq.shape[1]):
        std = seq[:, d].std()
        if std > 1e-6:
            seq[:, d] = (seq[:, d] - seq[:, d].mean()) / std
    return seq


def draw_skeleton(frame, skeleton, color=(0, 255, 0)):
    for i1, i2 in SKELETON_EDGES:
        x1, y1 = skeleton[i1]; x2, y2 = skeleton[i2]
        if x1 <= 0 and y1 <= 0: continue
        if x2 <= 0 and y2 <= 0: continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(frame, (int(x1), int(y1)), 3, color, -1)
        cv2.circle(frame, (int(x2), int(y2)), 3, color, -1)


def main():
    parser = argparse.ArgumentParser(description="Fight Detection Inference")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--model", type=str, default="fight_lstm_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.input and not args.webcam:
        print("Specify --input <video> or --webcam")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(dropout=0.0)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Model loaded, device={device}")

    source = 0 if args.webcam else args.input
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pose = _get_pose_detector()
    print(f"Video FPS: {fps:.1f}")

    raw_window = collections.deque(maxlen=NUM_FRAMES)
    vel_window = collections.deque(maxlen=NUM_FRAMES)
    scores_history = collections.deque(maxlen=15)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int(frame_idx * 1000 / fps)
        skeletons = extract_keypoints(frame, pose, ts_ms)
        skeleton = skeletons[0] if skeletons else np.zeros((18, 2))
        frame_idx += 1

        raw = compute_raw_features(skeleton)
        raw_window.append(raw)

        fight_prob = 0.0
        if len(raw_window) == NUM_FRAMES:
            raw_arr = np.array(raw_window, dtype=np.float32)
            vel_arr = np.zeros_like(raw_arr)
            vel_arr[1:] = raw_arr[1:] - raw_arr[:-1]
            seq = np.concatenate([raw_arr, vel_arr], axis=1)
            seq = std_normalize(seq)

            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                fight_prob = model(seq_t).item()
            scores_history.append(fight_prob)
        else:
            scores_history.append(0.0)

        avg_score = np.mean(scores_history) if scores_history else 0.0
        is_fight = avg_score > args.threshold

        color = (0, 0, 255) if is_fight else (0, 255, 0)
        label = f"FIGHT: {avg_score:.2f}" if is_fight else f"NORMAL: {avg_score:.2f}"
        if len(raw_window) < NUM_FRAMES:
            label = f"Warmup ({len(raw_window)}/{NUM_FRAMES})"

        COLORS = [
            (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (128, 0, 255),
        ]
        for pi, sk in enumerate(skeletons):
            c = COLORS[pi % len(COLORS)]
            draw_skeleton(frame, sk, color=c)

        n_people = len(skeletons)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        cv2.putText(frame, f"People: {n_people}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Fight Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import numpy as np
import cv2
from config import (
    POSE_PAIRS, NUM_FRAMES, FEATURE_DIM, NUM_ANGLES, ANGLE_TABLE, NUM_PAIRS
)
from pose_extractor import _get_pose_detector, extract_keypoints


def compute_angle_vector(skeleton):
    vec = np.zeros(FEATURE_DIM)
    for p_idx, (i1, i2) in enumerate(POSE_PAIRS):
        x1, y1 = skeleton[i1]
        x2, y2 = skeleton[i2]
        if (x1 <= 0 and y1 <= 0) or (x2 <= 0 and y2 <= 0):
            continue
        dx = x1 - x2
        dy = y1 - y2
        mag = np.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            continue
        dx /= mag
        dy /= mag
        best = 0
        best_dot = -2.0
        for a_idx, (ca, sa) in enumerate(ANGLE_TABLE):
            dot = dx * ca + dy * sa
            if dot > best_dot:
                best_dot = dot
                best = a_idx
        vec[p_idx * NUM_ANGLES + best] += 1
    return vec


def _minmax_scale(data):
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


def _extract_video_features(video_path, max_frames=200):
    """Extract sequential angle features from a real video."""
    cap = cv2.VideoCapture(video_path)
    detector = _get_pose_detector()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    all_features = []
    frame_idx = 0
    while len(all_features) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        ts = int(frame_idx * 1000 / fps)
        sks = extract_keypoints(frame, detector, ts)
        sk = sks[0] if sks else np.zeros((18, 2))
        vec = compute_angle_vector(sk)
        all_features.append(vec)
        frame_idx += 1
    cap.release()
    return np.array(all_features)


def _generate_nonfight_standing(n_samples, n_frames=NUM_FRAMES):
    """
    Generate non-fight samples simulating a person standing still
    with minor sway. Very low angular variation across frames.
    """
    samples = []
    for _ in range(n_samples):
        # base pose with small random variation
        base_vec = np.random.rand(FEATURE_DIM) * 0.5
        seq = []
        for t in range(n_frames):
            noise = np.random.randn(FEATURE_DIM) * 0.02
            sway = np.sin(t * 0.5 + np.random.rand(FEATURE_DIM) * 2 * np.pi) * 0.03
            vec = np.clip(base_vec + noise + sway, 0, None)
            seq.append(vec)
        samples.append(np.array(seq))
    return np.array(samples)


def _generate_nonfight_walking(n_samples, n_frames=NUM_FRAMES):
    """
    Generate non-fight samples simulating walking:
    smooth, rhythmic limb movement with low variance.
    """
    samples = []
    for _ in range(n_samples):
        base_vec = np.random.rand(FEATURE_DIM) * 0.4
        walk_freq = np.random.uniform(1.5, 3.0)
        seq = []
        for t in range(n_frames):
            rhythmic = np.sin(t * walk_freq + np.random.rand(FEATURE_DIM) * 0.3) * 0.08
            noise = np.random.randn(FEATURE_DIM) * 0.015
            vec = np.clip(base_vec + rhythmic + noise, 0, None)
            seq.append(vec)
        samples.append(np.array(seq))
    return np.array(samples)


def _generate_nonfight_talking(n_samples, n_frames=NUM_FRAMES):
    """
    Generate non-fight samples simulating talking/gesturing:
    moderate upper body variation, stable lower body.
    Upper body pairs: 0-6 (Neck-Nose, Neck-Shoulders, Shoulders-Elbows, Elbows-Wrists)
    """
    upper_pairs = set(range(7))
    samples = []
    for _ in range(n_samples):
        base_vec = np.random.rand(FEATURE_DIM) * 0.3
        seq = []
        for t in range(n_frames):
            vec = base_vec.copy()
            noise = np.random.randn(FEATURE_DIM) * 0.02
            # vary upper body across frames
            for p in range(NUM_PAIRS):
                s, e = p * NUM_ANGLES, (p + 1) * NUM_ANGLES
                if p in upper_pairs:
                    vary = np.sin(t * 2.0 + p) * 0.06
                    vec[s:e] = np.clip(vec[s:e] + vary, 0, None)
                else:
                    vary = np.sin(t * 0.3 + p) * 0.01
                    vec[s:e] = np.clip(vec[s:e] + vary, 0, None)
            vec += noise
            vec = np.clip(vec, 0, None)
            seq.append(vec)
        samples.append(np.array(seq))
    return np.array(samples)


def _generate_nonfight_samples(fight_features, n_samples=50):
    """Generate realistic non-fight samples with diverse motion patterns."""
    n_standing = n_samples // 3
    n_walking = n_samples // 3
    n_talking = n_samples - n_standing - n_walking

    parts = []
    if n_standing > 0:
        parts.append(_generate_nonfight_standing(n_standing))
    if n_walking > 0:
        parts.append(_generate_nonfight_walking(n_walking))
    if n_talking > 0:
        parts.append(_generate_nonfight_talking(n_talking))

    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _generate_fight_samples(fight_features, n_samples=50):
    """Generate fight samples by augmenting real fight features."""
    T, D = fight_features.shape
    if T < NUM_FRAMES:
        factor = (NUM_FRAMES // T) + 1
        fight_features = np.tile(fight_features, (factor, 1))[:NUM_FRAMES]

    samples = []
    for _ in range(n_samples):
        start = np.random.randint(0, max(1, T - NUM_FRAMES))
        seq = fight_features[start:start + NUM_FRAMES].copy()
        noise = np.random.randn(*seq.shape) * 0.01
        seq += noise
        samples.append(seq)
    return np.array(samples)


def generate_dataset_from_videos(fight_videos, nonfight_videos, n_samples=100):
    """Generate dataset from real fight and non-fight videos."""
    print(f"Extracting fight features from: {fight_videos}")
    fight_feats = []
    for v in fight_videos:
        ft = _extract_video_features(v, max_frames=300)
        fight_feats.append(ft)
    fight_all = np.concatenate(fight_feats, axis=0)
    print(f"  Total fight frames: {fight_all.shape[0]}")

    fight = _generate_fight_samples(fight_all, n_samples)

    if nonfight_videos:
        print(f"Extracting non-fight features from: {nonfight_videos}")
        nf_feats = []
        for v in nonfight_videos:
            ft = _extract_video_features(v, max_frames=300)
            nf_feats.append(ft)
        nf_all = np.concatenate(nf_feats, axis=0)
        print(f"  Total non-fight frames: {nf_all.shape[0]}")
        nonfight = _generate_fight_samples(nf_all, n_samples)  # use same windowed augmentation
    else:
        print("No non-fight videos, generating synthetic non-fight samples...")
        nonfight = _generate_nonfight_samples(fight_all, n_samples)

    data = np.concatenate([nonfight, fight], axis=0)
    labels = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    data = _minmax_scale(data)
    return data, labels


def generate_dataset_from_video(video_path, n_fight=100, n_nonfight=100):
    """Legacy: single fight video with synthetic non-fight."""
    print(f"Extracting features from {video_path} ...")
    ft = _extract_video_features(video_path)
    print(f"Extracted {len(ft)} frames, feature dim: {ft.shape[1]}")

    print(f"Generating {n_fight} fight + {n_nonfight} non-fight samples...")
    fight = _generate_fight_samples(ft, n_fight)
    nonfight = _generate_nonfight_samples(ft, n_nonfight)

    data = np.concatenate([nonfight, fight], axis=0)
    labels = np.concatenate([np.zeros(n_nonfight), np.ones(n_fight)])

    data = _minmax_scale(data)
    return data, labels


if __name__ == "__main__":
    data, labels = generate_dataset_from_video(
        "/home/sevnce/project/video/fight.mp4"
    )
    print(f"Data: {data.shape}, Labels: {labels.shape}")
    print(f"Fight: {int(labels.sum())}, Non-fight: {len(labels) - int(labels.sum())}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
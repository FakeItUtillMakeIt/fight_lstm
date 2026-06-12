import numpy as np
from config import (
    NUM_FRAMES, FEATURE_DIM, NUM_PAIRS, NUM_ANGLES,
    POSE_PAIRS, ANGLE_TABLE
)


def _generate_base_skeleton():
    """Generate a plausible base skeleton with COCO (18,2) coordinates."""
    sk = np.zeros((18, 2), dtype=np.float32)
    # rough normalized coords (0~1)
    sk[0]  = [0.50, 0.15]   # Nose
    sk[14] = [0.48, 0.12]   # REye
    sk[15] = [0.52, 0.12]   # LEye
    sk[16] = [0.46, 0.13]   # REar
    sk[17] = [0.54, 0.13]   # LEar
    sk[1]  = [0.50, 0.25]   # Neck
    sk[2]  = [0.38, 0.28]   # RShoulder
    sk[3]  = [0.28, 0.42]   # RElbow
    sk[4]  = [0.20, 0.55]   # RWrist
    sk[5]  = [0.62, 0.28]   # LShoulder
    sk[6]  = [0.72, 0.42]   # LElbow
    sk[7]  = [0.80, 0.55]   # LWrist
    sk[8]  = [0.42, 0.52]   # RHip
    sk[9]  = [0.42, 0.72]   # RKnee
    sk[10] = [0.42, 0.90]   # RAnkle
    sk[11] = [0.58, 0.52]   # LHip
    sk[12] = [0.58, 0.72]   # LKnee
    sk[13] = [0.58, 0.90]   # LAnkle
    return sk


def _skeleton_to_angle_vector(skeleton):
    """Compute 260-dim angle vector for a single skeleton."""
    vec = np.zeros(NUM_PAIRS * NUM_ANGLES)
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


def _generate_sequence(fight=False, n_frames=NUM_FRAMES, noise_std=0.02):
    """
    Generate a sequential feature matrix for one video.

    fight=True: high variance, rapid joint movements.
    fight=False: smooth, low variance random walk.
    """
    base = _generate_base_skeleton()
    frames = []

    if fight:
        jitter_std = 0.12
        walk_std = 0.08
    else:
        jitter_std = 0.03
        walk_std = 0.015

    current = base.copy()
    for _ in range(n_frames):
        skeleton = current.copy()
        # add jitter to limb joints (indices 2-13)
        jitter = np.random.randn(18, 2) * jitter_std * 0.3
        skeleton += jitter
        skeleton[:, 0] = np.clip(skeleton[:, 0], 0, 1)
        skeleton[:, 1] = np.clip(skeleton[:, 1], 0, 1)

        vec = _skeleton_to_angle_vector(skeleton)
        frames.append(vec)

        # random walk on limb joints
        walk = np.random.randn(18, 2) * walk_std
        current[2:14] += walk[2:14]
        current[:, 0] = np.clip(current[:, 0], 0, 1)
        current[:, 1] = np.clip(current[:, 1], 0, 1)

    return np.stack(frames, axis=0)


def generate_dataset(n_fight=50, n_nonfight=50):
    """
    Generate synthetic dataset.

    Returns:
        data: np.ndarray shape (n_fight + n_nonfight, NUM_FRAMES, FEATURE_DIM)
        labels: np.ndarray shape (n_fight + n_nonfight,)
    """
    np.random.seed(42)

    fight_data = np.stack([_generate_sequence(fight=True) for _ in range(n_fight)])
    nonfight_data = np.stack([_generate_sequence(fight=False) for _ in range(n_nonfight)])

    data = np.concatenate([nonfight_data, fight_data], axis=0)
    labels = np.concatenate([np.zeros(n_nonfight), np.ones(n_fight)])

    # min-max scale per angle group
    N, F, D = data.shape
    for i in range(N):
        for j in range(F):
            for k in range(NUM_PAIRS):
                s = k * NUM_ANGLES
                e = (k + 1) * NUM_ANGLES
                seg = data[i, j, s:e]
                mx = seg.max()
                mn = seg.min()
                if mx - mn > 0:
                    data[i, j, s:e] = (seg - mn) / (mx - mn)

    return data, labels


if __name__ == "__main__":
    data, labels = generate_dataset()
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Fight samples: {int(labels.sum())}")
    print(f"Non-fight samples: {len(labels) - int(labels.sum())}")
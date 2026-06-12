import numpy as np
import math
from config import (
    POSE_PAIRS, NUM_PAIRS, NUM_ANGLES, FEATURE_DIM, ANGLE_TABLE
)


def compute_angle_vector(skeleton):
    """
    Compute 260-dim angle feature vector from a single skeleton (18 keypoints).

    Args:
        skeleton: np.ndarray of shape (18, 2) - COCO keypoint (x, y) coordinates.
                  Points at (0,0) are treated as missing.

    Returns:
        np.ndarray of shape (260,) - angle histogram for 13 pairs x 20 bins.
    """
    angle_vector = np.zeros((NUM_PAIRS, NUM_ANGLES))

    for pair_idx, (idx1, idx2) in enumerate(POSE_PAIRS):
        x1, y1 = skeleton[idx1]
        x2, y2 = skeleton[idx2]

        if (x1 <= 0 and y1 <= 0) or (x2 <= 0 and y2 <= 0):
            continue

        dx = x1 - x2
        dy = y1 - y2
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            continue

        dx /= mag
        dy /= mag

        best_bin = 0
        best_dot = -2.0
        for a_idx, (cos_a, sin_a) in enumerate(ANGLE_TABLE):
            dot = dx * cos_a + dy * sin_a
            if dot > best_dot:
                best_dot = dot
                best_bin = a_idx

        angle_vector[pair_idx][best_bin] += 1

    return angle_vector.reshape(FEATURE_DIM)


def compute_frame_features(skeletons):
    """
    Compute features for all people in a single frame.

    Args:
        skeletons: list of np.ndarray, each shape (18, 2)

    Returns:
        np.ndarray of shape (260,) - aggregated angle histogram for this frame.
    """
    frame_features = np.zeros(FEATURE_DIM)
    for sk in skeletons:
        frame_features += compute_angle_vector(sk)
    return frame_features


def minmax_scale(data):
    """
    Min-max scale per angle group (each 20-bin block) across all samples.

    Args:
        data: np.ndarray of shape (N, NUM_FRAMES, FEATURE_DIM)

    Returns:
        np.ndarray of same shape, min-max scaled per angle group.
    """
    scaled = data.copy()
    N, F, D = scaled.shape
    for i in range(N):
        for j in range(F):
            for k in range(NUM_PAIRS):
                start = k * NUM_ANGLES
                end = (k + 1) * NUM_ANGLES
                seg = scaled[i, j, start:end]
                mx = seg.max()
                mn = seg.min()
                if mx - mn > 0:
                    scaled[i, j, start:end] = (seg - mn) / (mx - mn)
                else:
                    scaled[i, j, start:end] = 0
    return scaled
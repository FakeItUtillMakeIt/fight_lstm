#!/usr/bin/env python
"""
Feature extraction using raw normalized keypoint coordinates + velocities.
Much more informative than sparse angle histograms.
"""
import numpy as np
import cv2
import os
import json
import time

from pose_extractor import _get_pose_detector, extract_keypoints

# COCO keypoint indices for normalization
NECK = 1
RHIP = 8
LHIP = 11
RSHOULDER = 2
LSHOULDER = 5
NOSE = 0


def normalize_skeleton(skeleton, ref_length=1.0):
    """
    Center skeleton at hip-shoulder midpoint, scale to unit torso length.
    Returns normalized (18, 2) array with missing points at (0,0).
    """
    sk = skeleton.copy()
    valid_mask = (sk[:, 0] > 0) | (sk[:, 1] > 0)
    if valid_mask.sum() == 0:
        return np.zeros((18, 2), dtype=np.float32)

    center_x, center_y = 0, 0
    count = 0
    for idx in [NECK, RHIP, LHIP, RSHOULDER, LSHOULDER]:
        if valid_mask[idx]:
            center_x += sk[idx, 0]
            center_y += sk[idx, 1]
            count += 1
    if count > 0:
        center_x /= count
        center_y /= count

    sk[:, 0] -= center_x
    sk[:, 1] -= center_y

    scale = 1.0
    if valid_mask[NECK] and (valid_mask[RHIP] or valid_mask[LHIP]):
        if valid_mask[RHIP] and valid_mask[LHIP]:
            hip_y = (sk[RHIP, 1] + sk[LHIP, 1]) / 2
        elif valid_mask[RHIP]:
            hip_y = sk[RHIP, 1]
        else:
            hip_y = sk[LHIP, 1]
        torso = abs(sk[NECK, 1] - hip_y)
        if torso > 10:
            scale = ref_length / torso

    sk *= scale
    sk[~valid_mask] = 0
    return sk


def extract_raw_features(video_path, detector, max_frames=0, skip=1):
    """
    Extract raw normalized keypoint features + velocities from video.

    Returns: dict with 'keypoints' (N, 36) normalized coords,
             'velocities' (N, 36) frame-to-frame displacements
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    all_keypoints = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        ts = int(frame_idx * 1000 / fps)
        sks = extract_keypoints(frame, detector, ts)
        sk = sks[0] if sks else np.zeros((18, 2))
        sk = normalize_skeleton(sk)
        all_keypoints.append(sk.flatten().astype(np.float32))

        frame_idx += 1
        if max_frames > 0 and len(all_keypoints) >= max_frames:
            break

    cap.release()
    kps = np.array(all_keypoints)

    vels = np.zeros_like(kps)
    if len(kps) > 1:
        vels[1:] = kps[1:] - kps[:-1]

    return np.concatenate([kps, vels], axis=1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Video files to process")
    parser.add_argument("--output", default="cached_features_v2",
                        help="Output directory")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("Loading pose detector...")
    detector = _get_pose_detector()
    print(f"Processing {len(args.videos)} videos...")

    index = {}
    for vf in args.videos:
        name = os.path.splitext(os.path.basename(vf))[0]
        t0 = time.time()
        feats = extract_raw_features(vf, detector, args.max_frames, args.skip)
        elapsed = time.time() - t0

        out_path = os.path.join(args.output, f"{name}.npy")
        np.save(out_path, feats)
        index[name] = {"file": out_path, "frames": len(feats), "dim": feats.shape[1]}
        print(f"  {name}: {feats.shape} saved ({elapsed:.0f}s)")

    with open(os.path.join(args.output, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"Done. {len(index)} videos -> {args.output}/")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Extract and cache angle features from all videos for faster training.
Run once to extract features, then train from cache.
"""
import numpy as np
import cv2
import os
import json
import argparse
import time

from config import POSE_PAIRS, NUM_FRAMES, FEATURE_DIM, NUM_ANGLES, ANGLE_TABLE
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


def extract_video_features(video_path, detector, max_frames=0, skip=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {os.path.basename(video_path)}: {total} frames, fps={fps:.1f}")

    features = []
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
        vec = compute_angle_vector(sk)
        features.append(vec)

        frame_idx += 1
        if max_frames > 0 and len(features) >= max_frames:
            break

    cap.release()
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="/home/sevnce/project/video",
                        help="Directory containing videos")
    parser.add_argument("--output", default="cached_features",
                        help="Output directory for cached features")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Max frames per video")
    parser.add_argument("--skip", type=int, default=3,
                        help="Sample every N frames")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video files to process")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.videos:
        video_files = args.videos
    else:
        video_files = sorted(
            f for f in os.listdir(args.video_dir) if f.endswith(".mp4")
        )
        video_files = [os.path.join(args.video_dir, f) for f in video_files]

    print(f"Loading pose detector...")
    detector = _get_pose_detector()
    print(f"Processing {len(video_files)} videos...")

    index = {}
    for vf in video_files:
        name = os.path.splitext(os.path.basename(vf))[0]
        t0 = time.time()
        feats = extract_video_features(vf, detector, args.max_frames, args.skip)
        elapsed = time.time() - t0

        out_path = os.path.join(args.output, f"{name}.npy")
        np.save(out_path, feats)
        index[name] = {
            "file": out_path,
            "frames": len(feats),
            "time": f"{elapsed:.1f}s",
        }
        print(f"  -> saved {len(feats)} frames to {out_path} ({elapsed:.1f}s)")

    with open(os.path.join(args.output, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nDone. Index saved to {args.output}/index.json")


if __name__ == "__main__":
    main()
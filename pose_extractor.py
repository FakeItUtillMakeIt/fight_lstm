import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

from config import NUM_KEYPOINTS

# torchvision COCO-17 keypoints -> our COCO-18 format
# torchvision: 0:nose, 1:Leye, 2:Reye, 3:Lear, 4:Rear,
#  5:Lshoulder, 6:Rshoulder, 7:Lelbow, 8:Relbow, 9:Lwrist, 10:Rwrist,
# 11:Lhip, 12:Rhip, 13:Lknee, 14:Rknee, 15:Lankle, 16:Rankle
TV_TO_COCO = {
    0: 0,    # nose -> Nose
    2: 14,   # Reye -> REye
    1: 15,   # Leye -> LEye
    4: 16,   # Rear -> REar
    3: 17,   # Lear -> LEar
    6: 2,    # Rshoulder -> RShoulder
    5: 5,    # Lshoulder -> LShoulder
    8: 3,    # Relbow -> RElbow
    7: 6,    # Lelbow -> LElbow
    10: 4,   # Rwrist -> RWrist
    9: 7,    # Lwrist -> LWrist
    12: 8,   # Rhip -> RHip
    11: 11,  # Lhip -> LHip
    14: 9,   # Rknee -> RKnee
    13: 12,  # Lknee -> LKnee
    16: 10,  # Rankle -> RAnkle
    15: 13,  # Lankle -> LAnkle
}

_device = None
_model = None


def _get_device():
    global _device
    if _device is None:
        try:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.zeros(1).to(_device)
        except RuntimeError:
            _device = torch.device("cpu")
    return _device


def _get_pose_detector():
    global _model
    if _model is None:
        print("Loading KeypointRCNN model (first use)...")
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        _model = keypointrcnn_resnet50_fpn(weights=weights)
        _model.to(_get_device())
        _model.eval()
        print("Model loaded.")
    return _model


def extract_keypoints(frame, pose_detector=None, timestamp_ms=0):
    """
    Extract COCO-18 format keypoints from a single frame using KeypointRCNN.

    Args:
        frame: BGR image (H, W, 3)
        pose_detector: KeypointRCNN model instance
        timestamp_ms: unused, kept for API compatibility

    Returns:
        list of np.ndarray, each shape (18, 2)
    """
    if pose_detector is None:
        pose_detector = _get_pose_detector()

    device = _get_device()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = pose_detector(tensor)

    if len(outputs) == 0:
        return []

    h, w = frame.shape[:2]
    skeletons = []

    for i in range(len(outputs[0]["keypoints"])):
        kps = outputs[0]["keypoints"][i].cpu().numpy()
        score = float(outputs[0]["scores"][i].cpu())
        if score < 0.6:
            continue

        skeleton = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        valid = set()

        for tv_idx, coco_idx in TV_TO_COCO.items():
            x, y, conf = kps[tv_idx]
            if conf > 0.5:
                skeleton[coco_idx] = [x, y]
                valid.add(coco_idx)

        if 2 in valid and 5 in valid:
            skeleton[1] = (skeleton[2] + skeleton[5]) / 2.0
        elif 2 in valid:
            skeleton[1] = skeleton[2] + np.array([0, -30])
        elif 5 in valid:
            skeleton[1] = skeleton[5] + np.array([0, -30])

        skeletons.append(skeleton)

    return skeletons


def extract_keypoints_batch(frames, pose_detector=None, fps=10):
    """Extract keypoints from a list of frames."""
    return [extract_keypoints(f, pose_detector) for f in frames]
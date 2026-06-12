import math

# COCO keypoint indices (18 points)
# 0:Nose  1:Neck  2:RShoulder  3:RElbow  4:RWrist
# 5:LShoulder  6:LElbow  7:LWrist  8:RHip  9:RKnee
# 10:RAnkle  11:LHip  12:LKnee  13:LAnkle
# 14:REye  15:LEye  16:REar  17:LEar

# 13 body part pairs for angle calculation (matches original paper)
# [Neck,RShoulder] [Neck,LShoulder] [RShoulder,RElbow] [RElbow,RWrist]
# [LShoulder,LElbow] [LElbow,LWrist] [Neck,RHip] [RHip,RKnee]
# [RKnee,RAnkle] [Neck,LHip] [LHip,LKnee] [LKnee,LAnkle] [Neck,Nose]
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0]
]

NUM_PAIRS = len(POSE_PAIRS)         # 13
NUM_ANGLES = 20                     # 20 angle bins
FEATURE_DIM = NUM_PAIRS * NUM_ANGLES  # 260
NUM_FRAMES = 10                     # frames per video sample
NUM_KEYPOINTS = 18                  # COCO keypoints

# MediaPipe 33 landmarks -> COCO 18 keypoints
MEDIAPIPE_TO_COCO = {
    0: 0,    # Nose
    2: 14,   # REye
    5: 15,   # LEye
    7: 17,   # LEar
    8: 16,   # REar
    11: 5,   # LShoulder
    12: 2,   # RShoulder
    13: 6,   # LElbow
    14: 3,   # RElbow
    15: 7,   # LWrist
    16: 4,   # RWrist
    23: 11,  # LHip
    24: 8,   # RHip
    25: 12,  # LKnee
    26: 9,   # RKnee
    27: 13,  # LAnkle
    28: 10,  # RAnkle
}

# Precomputed angle table: 20 directions (0,18,36,...,342 degrees)
# cos/sin values used for vector quantization
ANGLE_TABLE = [
    (math.cos(math.radians(a)), math.sin(math.radians(a)))
    for a in range(0, 360, 18)
]
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
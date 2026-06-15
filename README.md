# Fight Detection with Pose + Bi-LSTM

基于骨架姿态的实时打斗检测系统。使用 KeypointRCNN 提取人体关键点，Bi-LSTM 对时序姿态序列进行分类。

原始论文: [FightDetectionPoseLSTM](https://github.com/jpowellgz/FightDetectionPoseLSTM) — *Fight detection using OpenPose and Bi-LSTM*

## 架构

```
视频帧 → KeypointRCNN → COCO-18关键点 → 归一化(raw+velocity) → Bi-LSTM → 打斗/非打斗
```

| 阶段 | 说明 | 输出 |
|------|------|------|
| 姿态提取 | KeypointRCNN ResNet50-FPN | 18点COCO关键点 |
| 特征工程 | 躯干居中归一化 + 帧间速度 | 72维 (36位置 + 36速度) |
| 时序分类 | 双向LSTM(128) + Dropout(0.3) | sigmoid概率 |

与原始论文的主要区别：
- **姿态引擎**: OpenPose (Caffe) → **KeypointRCNN (PyTorch)**，pip即装即用
- **特征**: 260维稀疏角度直方图 → **72维密集原始坐标+速度**，泛化性更好
- **框架**: TensorFlow → **PyTorch**，GPU训练/推理
- **分类器**: Bi-LSTM(520) → **Bi-LSTM(128) + Dropout**

## 项目结构

```
fight_lstm/
├── requirements.txt         # 依赖: torch, torchvision, opencv, numpy, scikit-learn
├── pose_extractor.py        # 姿态提取 (KeypointRCNN → COCO-18)
├── extract_features.py      # 特征提取 + 缓存 (原始坐标+速度 → 72维)
├── train.py                 # Bi-LSTM 训练 (多视频 + 数据增强 + 留一法验证)
├── inference.py             # 实时推理 (视频/摄像头 + 多人骨架可视化)
├── fight_lstm_model.pt      # 预训练模型
└── cached_features/         # 视频特征缓存 (80帧/视频, 72维)
```

## 快速开始

### 1. 安装

```bash
conda create -n fight_lstm python=3.10 -y
conda activate fight_lstm
pip install -r requirements.txt
```

GPU 加速需要 CUDA 13+ 的 PyTorch：
```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

### 2. 提取特征

```bash
python extract_features.py \
  --videos /path/to/video1.mp4 /path/to/video2.mp4 ... \
  --max-frames 80 --skip 2
```

特征缓存到 `cached_features/` 目录，可重复用于训练。

### 3. 训练

```bash
python train.py \
  --fight-videos fight         \
  --nonfight-videos walk phone stand sleep \
  --samples-per-video 50       \
  --epochs 40
```

参数说明：
- `--fight-videos`: 打斗视频名（不含路径和扩展名，对应 `cached_features/` 中的 `.npy`）
- `--nonfight-videos`: 非打斗视频名
- `--samples-per-video`: 每个视频采样的窗口数
- `--epochs`: 训练轮数

### 4. 推理

```bash
# 视频文件
python inference.py --input=/path/to/test.mp4

# 摄像头
python inference.py --webcam

# 调整阈值
python inference.py --input=test.mp4 --threshold=0.3
```

## 性能

| 指标 | CPU | GPU (RTX 5060) |
|------|-----|-----------------|
| 姿态提取 | ~2-3 s/frame | 0.074 s/frame |
| 推理FPS | ~0.3 | 13.6 |
| 模型参数量 | | 207K |

测试集准确率: ~97%，6种非打斗场景下泛化正确率 4/6。

## 算法说明

### 特征提取

```
1. KeypointRCNN 检测每帧人体 → 18个COCO关键点 (x,y)
2. 躯干居中: 以肩/髋中心为原点平移
3. 尺度归一化: 以躯干长度缩放到单位长度
4. 速度特征: 帧间关键点位移 (vx, vy)
5. 拼接: [位置(36), 速度(36)] = 72维
6. 序列标准化: 按通道减去均值除以标准差
```

### 数据增强

- 高斯噪声 (σ=0.005)
- 关键点随机丢失 (10%概率, 模拟遮挡)
- 时间采样步长随机化 (1-3帧间隔)
- 时序平滑抖动

### 模型

```
BiLSTM(
  LSTM(72→128, bidirectional, num_layers=1)
  Dropout(0.3)
  Linear(256→1, sigmoid)
)
```

损失函数: BCEWithLogitsLoss + AdamW (lr=1e-3, weight_decay=1e-3)

## 限制

基于纯骨架姿态的方法无法区分运动学相似的动作（如跳舞 vs 打斗）。这是此类方法的固有局限，原始论文同样存在。对静止、行走、睡眠等场景区分效果良好。

## License

MIT
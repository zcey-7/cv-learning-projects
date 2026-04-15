# Computer Vision Learning Projects

一系列基于 YOLOv8 / MediaPipe / FastSAM 的计算机视觉实战项目，覆盖目标检测、姿态分析、实例分割、电子围栏告警等核心应用场景。

---

## 项目列表

### 1. 安全帽检测 + 电子围栏告警系统

**技术栈：** YOLOv8 · ONNX Runtime · OpenCV · ByteTrack 多目标追踪

**功能：**
- 自定义数据集训练 YOLOv8 目标检测模型
- 任意划定多边形危险区域，实时判断人员入侵
- 使用 ByteTrack 对每个目标分配唯一 ID，持续追踪
- 模型导出为 ONNX，脱离 PyTorch 环境独立部署

**相关文件：**
- `train.py` — 训练脚本
- `export_onnx.py` — 导出 ONNX
- `export_trt.py` — 导出 TensorRT（需 NVIDIA GPU）
- `app_geofence.py` — 静态图片电子围栏检测
- `app_video_geofence.py` — 实时视频流电子围栏 + 多目标追踪
- `infer_onnx.py` — ONNX Runtime 原生推理示例

---

### 2. 深蹲动作分析系统（3D 姿态估计）

**技术栈：** MediaPipe Tasks API · OpenCV · NumPy

**功能：**
- 使用 MediaPipe PoseLandmarker 提取人体 33 个关键点（含 Z 轴深度）
- 基于三维向量夹角计算膝关节角度，精度不受正面/侧面角度影响
- 有限状态机实现深蹲计数：`Stand → Squatting → Stand`（膝角阈值 100°/160°）
- 实时可视化：角度数值、计数 Counter、骨骼连线叠加渲染

**相关文件：**
- `app_pose.py` — 主程序
- `pose_landmarker.task` — MediaPipe 模型权重（需自行下载，见下方说明）

---

### 3. FastSAM 实例分割（多彩抠图）

**技术栈：** FastSAM · OpenCV · NumPy

**功能：**
- 使用 Fast Segment Anything Model 对图像中所有物体做精确像素级分割
- 每个独立物体自动分配随机颜色，半透明叠加在原图上
- 边缘轮廓加粗渲染，视觉效果清晰

**相关文件：**
- `app_segment.py` — 主程序

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型权重

| 模型 | 用途 | 下载命令 |
|---|---|---|
| `yolov8n.pt` | 训练起点 | 运行 `train.py` 时自动下载 |
| `pose_landmarker.task` | 姿态估计 | 见下方 |
| `FastSAM-s.pt` | 实例分割 | 运行 `app_segment.py` 时自动下载 |

下载 MediaPipe 姿态模型：
```python
import urllib.request
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker.task'
)
```

### 3. 运行各模块

```bash
# 训练安全帽检测模型
python train.py

# 电子围栏（图片）
python app_geofence.py

# 电子围栏（实时摄像头）
python app_video_geofence.py

# 深蹲计数（视频文件）
# 把视频文件放到项目根目录，修改 app_pose.py 第51行的文件名
python app_pose.py

# 实例分割
# 把图片命名为 test_sam.jpg 放到根目录
python app_segment.py
```

---

## 环境要求

- Python 3.10 ~ 3.13
- 推荐 GPU：任意 NVIDIA 显卡（CPU 也可运行，速度较慢）
- TensorRT 导出需要 NVIDIA 显卡 + TensorRT 安装

---

## 目录结构

```
internship_learn_yolo/
├── train.py                 # 模型训练
├── export_onnx.py           # 导出 ONNX
├── export_trt.py            # 导出 TensorRT
├── infer_onnx.py            # ONNX 推理示例
├── app_geofence.py          # 图片电子围栏
├── app_video_geofence.py    # 视频电子围栏 + 追踪
├── app_pose.py              # 深蹲姿态分析
├── app_segment.py           # 实例分割
├── data.yaml                # 训练数据集配置
├── custom_bytetrack.yaml    # ByteTrack 追踪器配置
└── requirements.txt         # 依赖列表
```

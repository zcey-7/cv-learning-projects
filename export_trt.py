# 将训练好的 YOLOv8 模型导出为 TensorRT Engine 格式（仅支持 NVIDIA GPU）
# 用途：TensorRT 模型推理速度比 ONNX 快 2~5 倍
# 运行前请确保已安装 TensorRT 并拥有 NVIDIA GPU
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.export(format='engine', half=True, workspace=4)

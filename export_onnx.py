# 将训练好的 YOLOv8 模型导出为 ONNX 格式
# 用途：脱离 PyTorch 环境，实现跨平台推理
# 运行前请确保 runs/detect/train/weights/best.pt 已经存在
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

model.export(format='onnx', imgsz=640)
print("ONNX 导出完成：runs/detect/train/weights/best.onnx")
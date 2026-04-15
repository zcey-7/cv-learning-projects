import cv2
import numpy as np
import onnxruntime as ort
import time



model_path = 'runs/detect/train/weights/best.onnx'
session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name


print("--- 正在用 OpenCV 手工处理图片 ---")
img = cv2.imread("test.jpg") 
img_resized = cv2.resize(img, (640, 640)) 
blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (640, 640), swapRB=True, crop=False)
print("开始矩阵运算！")
start_time = time.time()


outputs = session.run(None, {input_name: blob})

end_time = time.time()
print(f"推理完成！耗时: {(end_time - start_time) * 1000:.2f} 毫秒")

# 4. 揭开矩阵的真面目
raw_data = outputs[0]
print(f"📦 吐出的原始矩阵形状: {raw_data.shape}")
print("--- 🎉 恭喜！你已成功脱离 PyTorch 完成了一次纯血推理！ ---")
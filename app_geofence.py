import cv2
import numpy as np
from ultralytics import YOLO



# 1. 加载我们脱离了 PyTorch 训练环境的轻量级 ONNX 模型！
# 注意：换成你真实的 best.onnx 路径
model = YOLO('runs/detect/train/weights/best.onnx', task='detect')

# 2. 定义危险区域（多边形的顶点坐标 [x, y]）
# 这里我随便画了一个四边形，你可以根据你的图片实际情况去修改这些坐标
danger_zone = np.array([
    [100, 200],   # 左上角
    [500, 200],   # 右上角
    [600, 600],   # 右下角
    [50, 600]     # 左下角
], np.int32)

# 读取图片
img = cv2.imread('test_fence.jpg')
if img is None:
    print("❌ 找不到图片，请检查 test_fence.jpg 是否存在！")
    exit()

# 3. 呼叫 AI 大脑进行推理
print("--- 🧠 AI 视觉引擎扫描中... ---")
results = model.predict(img, conf=0.3,device='cpu') # 信心指数大于 0.3 的框才算数

# 默认状态下，电子围栏是安全的（绿色）
zone_color = (0, 255, 0) 
alarm_triggered = False

# 4. 业务逻辑层：遍历每一个被找到的目标
for box in results[0].boxes:
    # 拿到框的坐标 (左上角 x1, y1, 右下角 x2, y2)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 计算脚底板的中心点 (x, y)
    foot_x = int((x1 + x2) / 2)
    foot_y = y2 
    
    # 🌟 核心算法：判断脚丫子是否踩进了多边形！
    # 返回值 >= 0 表示在多边形内部或边缘
    is_inside = cv2.pointPolygonTest(danger_zone, (foot_x, foot_y), False)
    
    if is_inside >= 0:
        alarm_triggered = True # 触发警报！
        zone_color = (0, 0, 255) # 围栏变成红色！
        
        # 在他脚下画个红色的圈，锁定目标
        cv2.circle(img, (foot_x, foot_y), 10, (0, 0, 255), -1)
        # 给违规目标画红框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, "INTRUDER!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # 安全目标画绿框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (foot_x, foot_y), 5, (0, 255, 0), -1)

# 5. UI 渲染层：把电子围栏画在图上
# 将多边形闭合 (isClosed=True)
cv2.polylines(img, [danger_zone], isClosed=True, color=zone_color, thickness=3)

# 叠加一层半透明的红色警报滤镜
if alarm_triggered:
    overlay = img.copy()
    cv2.fillPoly(overlay, [danger_zone], (0, 0, 255)) # 涂满红色
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img) # 混合透明度
    # 屏幕打出巨大的警告
    cv2.putText(img, "WARNING: INTRUSION DETECTED!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
    print("🚨🚨🚨 警报触发！发现目标闯入危险区域！")
else:
    cv2.putText(img, "ZONE SECURE", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
    print("✅ 区域安全，未发现违规目标。")

# 6. 保存并验收成果！
cv2.imwrite("output_fence.jpg", img)
print("--- 📸 现场取证照片已保存至 output_fence.jpg ---")
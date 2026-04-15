import os
# 🚀 护身符 1：屏蔽网络源检查
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
# 🛡️ 护身符 2：防止底层多线程冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 💥 护身符 3：终极降维打击！强行关闭飞桨 3.x 存在 Bug 的 PIR 编译器和底层 MKLDNN 加速！
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

from paddleocr import PaddleOCR
import cv2
import numpy as np

print("--- 🖨️ 智能 OCR 识别引擎（底层锁死稳定版）启动 ---")

# 1. 加载模型（模型已经下好了，这次是秒开）
ocr = PaddleOCR(use_textline_orientation=True, lang='ch')

img_path = 'test_ocr.jpg'
img = cv2.imread(img_path)

if img is None:
    print("❌ 找不到图片，请检查 test_ocr.jpg 是否存在！")
    exit()

# 2. 核心：向 AI 喂入图片，榨取文字！
print("--- 🧠 正在提取图像字符... ---")
# 回归经典的 ocr.ocr() API（因为现在是 2.8.1 版本了）
results = ocr.ocr(img_path) 

# 3. 业务解析层
if results and results[0]:
    print("\n✅ 提取成功！以下是业务数据：")
    print("-" * 40)
    
    for line in results[0]:
        box = line[0]
        text = line[1][0]
        score = line[1][1]

        # 打印提取出的字符串！
        print(f"📝 [{text}] (可信度: {score:.2f})")

        # 在原图上画绿色框
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
else:
    print("⚠️ 画面中未检测到任何文字。")

# 4. 验收成果
cv2.imwrite("output_ocr.jpg", img)
print("-" * 40)
print("📸 提取完毕！请去文件夹查看划了重点的 output_ocr.jpg！")
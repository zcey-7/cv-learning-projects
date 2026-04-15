import cv2
from ultralytics import FastSAM

def main():
    print("Initializing FastSAM segmentation engine...")

    # 加载模型
    try:
        model = FastSAM('FastSAM-s.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    img_path = 'test_sam.jpg'
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Cannot find image at '{img_path}'.")
        return

    print("Running segmentation scan...")

    # 执行推理
    # conf: 置信度阈值
    # iou: NMS 交并比阈值
    results = model(img, conf=0.4, iou=0.9, verbose=False)

    # 渲染并保存结果
    if len(results) > 0 and results[0].masks is not None:
        import numpy as np
        # 提取原图并创建一个画板专门用于涂色
        annotated_frame = img.copy()
        overlay = img.copy()
        
        # 取出所有的多边形轮廓坐标
        for points in results[0].masks.xy:
            if len(points) == 0:
                continue
                
            # 给当前的这个物体随机摇一个独一无二的 BGR 颜色
            color = np.random.randint(0, 255, size=3).tolist()
            
            # FastSAM 给的坐标带小数，转为 OpenCV 要求的整型列表
            pts = np.array(points, dtype=np.int32)
            
            # 涂满区域
            cv2.fillPoly(overlay, [pts], color)
            # 加粗一下外侧边界线显得更清晰
            cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)
            
        # 把有色彩贴膜的 overlay 跟原图进行完美的 半透明混合 (各占 50%)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        
        cv2.imshow("FastSAM Segmentation", annotated_frame)
        cv2.imwrite("output_sam.jpg", annotated_frame)
        
        print("Segmentation complete. Result saved to 'output_sam.jpg'.")
    else:
        print("No objects segmented.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
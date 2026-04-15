import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO(r'D:\internship_learn_yolo\runs\detect\train\weights\best.onnx', task='detect')

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("无法打开摄像头或视频文件！")
    exit()

# 3. 定义危险区域（多边形）
# 注意：如果是摄像头画面，这里可能需要根据摄像头的分辨率微调一下坐标
danger_zone = np.array([
    [100, 100],   # 左上
    [500, 100],   # 右上
    [500, 400],   # 右下
    [100, 400]    # 左下
], np.int32)

print("--- 监控流已接通，按键盘上的 'q' 键退出系统 ---")


while True:
    # 抽取视频的每一帧 (一瞬间的画面)
    success, frame = cap.read()
    if not success:
        print("视频播放完毕或摄像头断开。")
        break
        
    # 将画面镜像翻转一下（只针对前置摄像头，如果是真实监控视频请删掉这行）
    frame = cv2.flip(frame, 1)

    # 喂给 AI 引擎推理！
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, device='cpu', verbose=False) # verbose=False 让终端别疯狂刷屏

    zone_color = (0, 255, 0)
    alarm_triggered = False

# 遍历当前画面里的所有目标
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        foot_x = int((x1 + x2) / 2)
        foot_y = y2 
        
        # 获取目标的唯一身份证号 (ID)！
        # 注意：有时候画面里刚出现目标，追踪器还没确认，id 会是 None，所以要判断一下
        target_id = int(box.id[0]) if box.id is not None else 0
        
        # 判断入侵
        is_inside = cv2.pointPolygonTest(danger_zone, (foot_x, foot_y), False)
        
        if is_inside >= 0:
            alarm_triggered = True
            cv2.circle(frame, (foot_x, foot_y), 10, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 把 ID 挂在他的头顶上！
            cv2.putText(frame, f"ID:{target_id} INTRUDER", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 安全的人也显示 ID
            cv2.putText(frame, f"ID:{target_id} SAFE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # UI 渲染：画出围栏和警报
    cv2.polylines(frame, [danger_zone], isClosed=True, color=zone_color, thickness=3)
    if alarm_triggered:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [danger_zone], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, "WARNING!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    # 5. 实时显示监控画面！
    cv2.imshow("Smart Construction Security System", frame)

    # 监听键盘，如果按下 'q' 键，就打破死循环退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 下班关机，释放资源
cap.release()
cv2.destroyAllWindows()
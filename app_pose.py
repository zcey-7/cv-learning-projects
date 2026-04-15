import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# 1. 核心数学模块：计算 3D 空间向量夹角
def calculate_3d_angle(a, b, c):
    """
    计算三维空间中三点构成的夹角
    参数: a, b, c 均为 [x, y, z] 格式的 numpy 数组，b 为顶点
    返回: 角度值 (0-180)
    """
    v1 = a - b
    v2 = c - b
    
    # 计算向量点积和模长
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    # 计算余弦值并限制在 [-1, 1] 范围内避免浮点数精度溢出
    cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# 手动连线的骨架关键点索引（用于可视化连线）
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # 手臂和肩膀
    (11, 23), (12, 24), (23, 24), # 躯干
    (23, 25), (24, 26), (25, 27), (26, 28) # 腿部
]

# 2. 初始化 MediaPipe Tasks API 配置
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# 3. 业务状态机初始化
squat_count = 0
is_squatting = False
current_status = "Stand"

# 打开视频 (或使用 0 打开摄像头)
cap = cv2.VideoCapture('test_squat_front1.mp4')

# 获取视频帧率用来手动计算递增的时间戳（以免原生视频时间戳出错）
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30 # 容错机制

# 4. 启用全新的 Pose Landmarker 模型
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # MediaPipe 需要 RGB 图像
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # 确保时间戳严格递增 (Tasks API 的硬性要求)
        timestamp_ms = int((frame_index * 1000) / fps)
        frame_index += 1
        
        # 使用 Tasks API 核心推理
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        image = frame.copy() # 用于可视化绘制渲染

        # 确保检测到了地标
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            # 拿到第一个检测到的人的世界坐标和像素坐标
            world_landmarks = results.pose_world_landmarks[0]
            pixel_landmarks = results.pose_landmarks[0]
            
            # 获取右侧胯部(24)、膝盖(26)、脚踝(28)的三维坐标 [x, y, z]
            hip = np.array([world_landmarks[24].x, world_landmarks[24].y, world_landmarks[24].z])
            knee = np.array([world_landmarks[26].x, world_landmarks[26].y, world_landmarks[26].z])
            ankle = np.array([world_landmarks[28].x, world_landmarks[28].y, world_landmarks[28].z])
            
            # 计算稳定的真实 3D 角度
            angle = calculate_3d_angle(hip, knee, ankle)
            
            # 状态机逻辑
            if angle < 100 and not is_squatting:
                is_squatting = True
                current_status = "Squatting"
            elif angle > 160 and is_squatting:
                is_squatting = False
                squat_count += 1
                current_status = "Stand"
                
            # UI 渲染数据：把角度文本标在膝盖像素位置旁边
            knee_pixel_coords = np.multiply(
                [pixel_landmarks[26].x, pixel_landmarks[26].y], 
                [image.shape[1], image.shape[0]]
            ).astype(int)
            knee_pixel_coords = (int(knee_pixel_coords[0]), int(knee_pixel_coords[1]))
            
            cv2.putText(image, str(int(angle)), knee_pixel_coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(image, f'COUNT: {squat_count}', (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.putText(image, f'STATUS: {current_status}', (30, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 绘制骨架连线图 (原生绘制法，不再依赖旧版 drawing_utils)
            for connection in POSE_CONNECTIONS:
                start_id, end_id = connection
                start_pt = pixel_landmarks[start_id]
                end_pt = pixel_landmarks[end_id]
                
                # 过滤掉视野盲区或确信度极低的点
                if start_pt.visibility > 0.5 and end_pt.visibility > 0.5:
                    pt1 = (int(start_pt.x * image.shape[1]), int(start_pt.y * image.shape[0]))
                    pt2 = (int(end_pt.x * image.shape[1]), int(end_pt.y * image.shape[0]))
                    cv2.line(image, pt1, pt2, (255, 255, 255), 2)
                    cv2.circle(image, pt1, 4, (0, 255, 0), -1)
                    cv2.circle(image, pt2, 4, (0, 255, 0), -1)
            
            # 特殊高亮计数的右侧三点(深红色)
            for idx in [24, 26, 28]:
                pt = pixel_landmarks[idx]
                if pt.visibility > 0.5:
                    c_pt = (int(pt.x * image.shape[1]), int(pt.y * image.shape[0]))
                    cv2.circle(image, c_pt, 6, (0, 0, 255), -1)

        cv2.imshow('MediaPipe 3D Pose (Tasks API)', image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import os
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data='data.yaml',
        epochs=100,      
        imgsz=640,      
        batch=8         
    )

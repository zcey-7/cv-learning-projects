from ultralytics import YOLO


model = YOLO(r'D:\internship_learn_yolo\runs\detect\train\weights\best.pt')


model.export(format='engine', half=True, workspace=4)


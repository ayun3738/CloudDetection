from ultralytics import YOLO
from roboflow import Roboflow
import os

# os.system(f'yolo task=detect mode=train model=runs/detect/yolo8n_suvel50/weights/best.pt data=datasets/Cloud_Classification-14/data.yaml epochs=50 imgsz=640') 
# os.system(f'yolo task=detect mode=train model=yolov8m_modified.pt data=datasets/Cloud_Classification-1/data.yaml epochs=50 imgsz=640') 
os.system(f'yolo task=detect mode=train model=yolov8x.pt data=datasets/Cloud_Classification-1/data.yaml epochs=50 imgsz=640 batch= 8') 
# os.system(f'yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=datasets/Cloud_Classification_suvel-1/test/images')

# project.version(dataset.version).deploy(model_type='yolov8', model_path=f'runs/detect/train/')
# os.system(f'yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=datasets/Cloud_Classification_HSV-1/data.yaml')
# HOME = ''
# {HOME}/runs/detect/train/weights/best.pt

# train

# os.system(f'yolo task=detect mode=train model=yolov8x.pt data=Cloud_Classification_zero1-1/data.yaml epochs=200 imgsz=448') 

# os.system(f'yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=Cloud_Classification_zero1-1/test/images')
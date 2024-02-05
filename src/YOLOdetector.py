from ultralytics import YOLO
import cv2
import numpy as np
import math

class object_detector():
    def __init__(self, model = YOLO('utils/best.pt')):
        self.model = model

    def detect(self, frame):
        results = self.model(frame, stream=True)
        
        centers=[]
        bbox = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                centers.append(np.array([[int((x1+x2)/2)], [int((y1+y2)/2)]]))
                
        return centers

    

    

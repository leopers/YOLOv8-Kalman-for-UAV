from ultralytics import YOLO 

model = YOLO('/Users/leonardodias/Desktop/Graduação ITA/Exame CM/Exame-CM203/best.pt')

results = model.track(source=0, show=True, tracker='bytetrack.yaml')
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt") #100 fotograma por segundo

results = model.predict(source='0', show= True, conf=0.5, stream=True)
print(results)
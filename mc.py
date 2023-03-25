from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format="onnx", imgsz=[480,640])
model.train(data='data.yaml', epochs=100, imgsz=640)
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="C:/Users/mayn_ku/Desktop/parking_ai/dataset/data.yaml", epochs=50, imgsz=640)


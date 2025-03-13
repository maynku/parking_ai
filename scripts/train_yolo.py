from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt for a smaller model

# Train model
model.train(data="../dataset/data.yaml", epochs=50, imgsz=640)

# Save the best model
model.export(format="torchscript") 

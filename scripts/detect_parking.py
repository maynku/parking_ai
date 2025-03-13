from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")  # Load trained model
cap = cv2.VideoCapture("http://192.168.240.226:8080/video")  # Replace with your IP camera URL

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = results[0].plot()  # Draw detections

    cv2.imshow("Parking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

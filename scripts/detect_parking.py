import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import matplotlib.pyplot as plt  # Import Matplotlib

# Load trained YOLO model
model = YOLO("C:/Users/mayn_ku/runs/detect/train16/weights/best.pt")  # Path to trained model

# Open video stream from IP camera
cap = cv2.VideoCapture("http://172.19.29.29:8080/video")  # Ensure the URL is correct

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0  # Initialize a counter for saving frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue  # Try again

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO detection
    results = model.predict(frame)
    detections = results[0].boxes.data

    # Check if any detections were made
    if detections is not None:
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = "Available" if cls == 0 else "Occupied"

            # Set color based on slot status
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for empty, Red for occupied
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Headless Environment Handling:
    # Option 1: Save Images
    cv2.imwrite(f"output_frame_{frame_count}.jpg", frame)  # Save each frame

    # Option 2: Use Matplotlib (if display is possible)
    try:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Parking Detection")
        plt.pause(0.01)  # Small pause to allow updates
        plt.clf()  # Clear the figure for the next frame
    except Exception as e:
        print(f"Matplotlib display error: {e}")

    frame_count += 1  # Increment the frame counter

    # Exit on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
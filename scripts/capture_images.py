import cv2
import os

# IP Camera URL (Update if changed)
ip_camera_url = "http://192.168.240.226:8080/video"

# Folder to save images
save_dir = "../dataset/images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open video stream
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("⚠️ Error: Could not open video stream. Check IP and network connection.")
    exit()

image_count = 0

print("📸 Press 's' to save an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠️ Error: Failed to grab frame. Check if IP Webcam is running.")
        break

    # Display the live feed
    cv2.imshow("Live Parking Feed - Press 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save image
        image_path = os.path.join(save_dir, f"parking_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"✅ Image saved: {image_path}")
        image_count += 1

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

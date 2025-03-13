import cv2

# Updated IP camera URL
ip_camera_url = "http://192.168.240.226:8080/video"

# Open video stream
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("⚠️ Error: Could not open video stream. Check the IP address and network connection.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠️ Error: Failed to grab frame")
        break

    # Display the live feed
    cv2.imshow("Live Parking Feed", frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

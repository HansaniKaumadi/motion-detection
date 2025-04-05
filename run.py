import cv2
import numpy as np
import time
from datetime import datetime

# Load the MobileNet SSD model
prototxt_path = "model/MobileNetSSD_deploy.prototxt"
model_path = "model/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# List of class labels that MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open webcam with resolution optimization for Raspberry Pi
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Store last detected position
last_positions = []
save_threshold = 50  # Minimum movement in pixels to save a frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Resize for better performance on Raspberry Pi
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_positions = []
    detected_human = False
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Only consider detections with confidence > 50%
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":  # Only detect humans
                detected_human = True

                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
                current_positions.append((centerX, centerY))

                # Draw bounding box and label
                label = f"Human: {confidence * 100:.1f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Check for significant movement
    if detected_human and last_positions:
        for new_pos in current_positions:
            for old_pos in last_positions:
                movement = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                if movement > save_threshold:
                    filename = f"motion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Motion detected! Frame saved as {filename}")
                    break
    
    last_positions = current_positions  # Update tracked positions
    
    # Show the live output
    cv2.imshow("Human Motion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
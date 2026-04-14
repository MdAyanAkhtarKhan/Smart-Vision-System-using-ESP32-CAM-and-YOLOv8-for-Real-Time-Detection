from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

# Check CUDA

print("CUDA available:", torch.cuda.is_available())

# IP Webcam settings

ip = "192.168.0.108"
port = "8080"

# Try this FIRST
url = f"http://{ip}:{port}/video"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(" Cannot open video stream")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load YOLOv8 LARGE model

model = YOLO("../yolo-weights/yolov8x.pt")


# ==========================
# Detection loop
# ==========================
while True:
    success, img = cap.read()
    if not success:
        print(" Frame not received")
        break

    img = cv2.resize(img, (960, 540))

    #  GPU inference
    results = model(img, device=0, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(img, label, (x1, max(35, y1)), scale=0.9,thickness=1)

    cv2.imshow("YOLOv8l | Phone Camera | GPU", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

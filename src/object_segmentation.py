from ultralytics import YOLO
import cv2
import numpy as np
import os

input_video_path = "park.mp4"
model_name = 'yolov8n-seg.pt'

if not os.path.exists(input_video_path):
    print(f"Input video file '{input_video_path}' not found.")
    exit()

model = YOLO(model_name)
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Could not open the video file '{input_video_path}'.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    results = model(frame)

    if results and results[0].masks and len(results[0].masks) > 0:
        output_frame = frame.copy()
        for mask, box in zip(results[0].masks, results[0].boxes):
            m = mask.data.cpu().numpy()[0]
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = m.astype(bool)

            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            alpha = 0.5
            for c in range(3):
                output_frame[:, :, c] = np.where(
                    mask_bool,
                    output_frame[:, :, c] * (1 - alpha) + color[c] * alpha,
                    output_frame[:, :, c]
                )

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            conf = box.conf[0]

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Real-time Object Segmentation", output_frame)
    else:
        cv2.imshow("Real-time Object Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
